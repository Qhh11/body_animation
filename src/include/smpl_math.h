#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "smpl_model.h"

// Define a standard parent map for SMPL (24 joints)
// This defines the kinematic tree structure
const int kintree_parents[24] = {
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
};

/**
 * Computes the Forward Kinematics and LBS to generate deformed vertices.
 * * Inputs:
 * beta: Shape parameters [10]
 * theta: Pose parameters [72] (24 joints * 3 axis-angle)
 * trans: Global translation [3]
 * model: The loaded SMPL model data (read-only)
 * * Output:
 * vertices: Deformed vertices [N x 3] (RowMajor)
 */
template <typename T>
void SMPLForward(
    const T* beta, 
    const T* theta, 
    const T* trans,
    const SmplModel& model, 
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& out_vertices) 
{
    // 1. Shape Blend Shapes
    // v_shaped = v_template + shapedirs * beta
    // shapedirs is (N*3, 10), so we map it to perform matrix multiplication
    // Note: To use loaded double data with T (which might be Jet), we need to cast or iterate.
    // For efficiency with Ceres, simple iteration is often safer to avoid type mismatches.
    
    // Initialize output with template vertices
    out_vertices.resize(model.N, 3);
    for (int i = 0; i < model.N; ++i) {
        out_vertices(i, 0) = T(model.v_template(i, 0));
        out_vertices(i, 1) = T(model.v_template(i, 1));
        out_vertices(i, 2) = T(model.v_template(i, 2));
    }

    // Apply shape blend shapes
    // This is a simplified loop. In production, using Eigen::Map with casting is faster but trickier.
    for (int i = 0; i < model.N; ++i) {
        for (int j = 0; j < 3; ++j) {
            T offset = T(0.0);
            for (int k = 0; k < 10; ++k) { // Assume 10 betas
                // model.shapedirs is (N*3, 10). Row index is i*3+j
                offset += T(model.shapedirs(i * 3 + j, k)) * beta[k];
            }
            out_vertices(i, j) += offset;
        }
    }

    // 2. Pose Blend Shapes (Skipped for basic Week 1/2, add later for better quality)
    // To enable this, you need to compute the rotation matrix for each joint, subtract identity,
    // and multiply by posedirs. It adds significant complexity. 
    // Let's stick to pure LBS first to get it running.

    // 3. Forward Kinematics (FK)
    // We need to compute the Global Transformation Matrix for each joint.
    // G_k = G_parent * L_k
    
    // Array to store local transforms (24 matrices of 4x4)
    Eigen::Matrix<T, 4, 4> local_transforms[24];
    // Array to store global transforms (24 matrices of 4x4)
    Eigen::Matrix<T, 4, 4> global_transforms[24];

    // Compute joint locations given the shaped vertices (J_regressor * v_shaped)
    // For simplicity now, let's use the static J_regressor * v_template + offsets
    // Or simpler: Just compute joint locations from the v_shaped we just computed.
    Eigen::Matrix<T, 24, 3> joints;
    joints.setZero();
    for (int j = 0; j < 24; ++j) {
        for (int v = 0; v < model.N; ++v) {
            T w = T(model.J_regressor(j, v));
            if (w > T(1e-5)) { // Sparse check
                joints(j, 0) += w * out_vertices(v, 0);
                joints(j, 1) += w * out_vertices(v, 1);
                joints(j, 2) += w * out_vertices(v, 2);
            }
        }
    }

    // Traverse kinematic tree
    for (int i = 0; i < 24; ++i) {
        // Convert axis-angle theta[i] to Rotation Matrix
        T angle_axis[3] = { theta[i*3], theta[i*3+1], theta[i*3+2] };
        T R[9];
        ceres::AngleAxisToRotationMatrix(angle_axis, R);

        // Construct Local Transformation Matrix (4x4)
        local_transforms[i].setIdentity();
        
        // Rotation part
        local_transforms[i](0,0) = R[0]; local_transforms[i](0,1) = R[1]; local_transforms[i](0,2) = R[2];
        local_transforms[i](1,0) = R[3]; local_transforms[i](1,1) = R[4]; local_transforms[i](1,2) = R[5];
        local_transforms[i](2,0) = R[6]; local_transforms[i](2,1) = R[7]; local_transforms[i](2,2) = R[8];

        // Translation part (Joint position relative to parent)
        // This is tricky: Local translation is (J_current - J_parent) in rest pose
        int parent = kintree_parents[i];
        if (parent == -1) {
            local_transforms[i](0,3) = joints(i, 0);
            local_transforms[i](1,3) = joints(i, 1);
            local_transforms[i](2,3) = joints(i, 2);
            global_transforms[i] = local_transforms[i];
        } else {
            local_transforms[i](0,3) = joints(i, 0) - joints(parent, 0);
            local_transforms[i](1,3) = joints(i, 1) - joints(parent, 1);
            local_transforms[i](2,3) = joints(i, 2) - joints(parent, 2);
            global_transforms[i] = global_transforms[parent] * local_transforms[i];
        }
    }

    // 4. Linear Blend Skinning (LBS)
    // v_final = Sum(weight_ij * G_j * inv(G_rest_j) * v_shaped)
    // Precompute G_j * inv(G_rest_j). G_rest_j is just the translation of the joint in rest pose.
    
    // Skinning transforms
    Eigen::Matrix<T, 4, 4> skinning_transforms[24];
    for (int i = 0; i < 24; ++i) {
        Eigen::Matrix<T, 4, 4> bind_pose_inv;
        bind_pose_inv.setIdentity();
        bind_pose_inv(0,3) = -joints(i, 0);
        bind_pose_inv(1,3) = -joints(i, 1);
        bind_pose_inv(2,3) = -joints(i, 2);
        
        skinning_transforms[i] = global_transforms[i] * bind_pose_inv;
    }

    // Apply skinning to all vertices
    // We create a copy to avoid reading and writing to same variable during update
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> v_shaped = out_vertices;
    out_vertices.setZero(); // Reset for accumulation

    for (int i = 0; i < model.N; ++i) {
        T v_tmp[3] = { v_shaped(i,0), v_shaped(i,1), v_shaped(i,2) };
        
        for (int j = 0; j < 24; ++j) {
            T w = T(model.weights(i, j));
            if (w > T(0.0)) { // Only process influencing joints
                // Transform point: T * v
                T px = skinning_transforms[j](0,0)*v_tmp[0] + skinning_transforms[j](0,1)*v_tmp[1] + skinning_transforms[j](0,2)*v_tmp[2] + skinning_transforms[j](0,3);
                T py = skinning_transforms[j](1,0)*v_tmp[0] + skinning_transforms[j](1,1)*v_tmp[1] + skinning_transforms[j](1,2)*v_tmp[2] + skinning_transforms[j](1,3);
                T pz = skinning_transforms[j](2,0)*v_tmp[0] + skinning_transforms[j](2,1)*v_tmp[1] + skinning_transforms[j](2,2)*v_tmp[2] + skinning_transforms[j](2,3);
                
                out_vertices(i, 0) += w * px;
                out_vertices(i, 1) += w * py;
                out_vertices(i, 2) += w * pz;
            }
        }
    }

    // 5. Global Translation
    for (int i = 0; i < model.N; ++i) {
        out_vertices(i, 0) += trans[0];
        out_vertices(i, 1) += trans[1];
        out_vertices(i, 2) += trans[2];
    }
}