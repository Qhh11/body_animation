#pragma once
#include <Eigen/Dense>
#include <ceres/rotation.h>
#include "smpl_model.h"

// FAST joints-only SMPL forward for optimization.
// Assumes SMPL has 24 joints (standard SMPL). If your model.J != 24, it falls back.
template <typename T>
void SMPLJointsForward(
    const T* beta,
    const T* theta,
    const T* trans,
    const SmplModel& model,
    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>& out_joints_world
) {
    const int J = model.J;

    out_joints_world.resize(J, 3);

    // --------- Preferred fast path: J==24 with joint caches ----------
    const bool has_joint_cache =
        (model.J_template_3d.rows() == J && model.J_template_3d.cols() == 3) &&
        (model.J_shapedirs_3d.rows() == J * 3 && model.J_shapedirs_3d.cols() >= 10);

    if (J == 24 && has_joint_cache) {
        // joints_shaped: (24 x 3) on stack (fixed size)
        Eigen::Matrix<T, 24, 3, Eigen::RowMajor> joints;
        // global rotations/positions on stack
        Eigen::Matrix<T, 24, 3, Eigen::RowMajor> global_p; // positions
        // store rotation matrices
        std::array<Eigen::Matrix<T,3,3>, 24> global_R;

        // 1) joints = J_template_3d + J_shapedirs_3d * beta
        for (int j = 0; j < 24; ++j) {
            joints(j,0) = T(model.J_template_3d(j,0));
            joints(j,1) = T(model.J_template_3d(j,1));
            joints(j,2) = T(model.J_template_3d(j,2));

            const int base = j * 3;
            // add 10 betas
            for (int c = 0; c < 3; ++c) {
                T off = T(0);
                for (int k = 0; k < 10; ++k) {
                    off += T(model.J_shapedirs_3d(base + c, k)) * beta[k];
                }
                joints(j,c) += off;
            }
        }

        // 2) FK recursion (R,p)
        for (int i = 0; i < 24; ++i) {
            T aa[3] = { theta[i*3+0], theta[i*3+1], theta[i*3+2] };
            T Rcolmaj[9];
            ceres::AngleAxisToRotationMatrix(aa, Rcolmaj);

            Eigen::Matrix<T,3,3> R;
            R(0,0)=Rcolmaj[0]; R(0,1)=Rcolmaj[3]; R(0,2)=Rcolmaj[6];
            R(1,0)=Rcolmaj[1]; R(1,1)=Rcolmaj[4]; R(1,2)=Rcolmaj[7];
            R(2,0)=Rcolmaj[2]; R(2,1)=Rcolmaj[5]; R(2,2)=Rcolmaj[8];

            int p = (i == 0) ? -1 : model.kintree_table(0, i);
            if (p < 0 || p >= 24) p = -1;

            if (p == -1) {
                global_R[i] = R;
                global_p(i,0) = joints(i,0);
                global_p(i,1) = joints(i,1);
                global_p(i,2) = joints(i,2);
            } else {
                global_R[i] = global_R[p] * R;

                Eigen::Matrix<T,3,1> rel;
                rel(0) = joints(i,0) - joints(p,0);
                rel(1) = joints(i,1) - joints(p,1);
                rel(2) = joints(i,2) - joints(p,2);

                Eigen::Matrix<T,3,1> parent_p;
                parent_p(0) = global_p(p,0);
                parent_p(1) = global_p(p,1);
                parent_p(2) = global_p(p,2);

                Eigen::Matrix<T,3,1> cur_p = parent_p + global_R[p] * rel;
                global_p(i,0) = cur_p(0);
                global_p(i,1) = cur_p(1);
                global_p(i,2) = cur_p(2);
            }
        }

        // 3) output + trans  (FIXED: global_p(i,*) not global_p)
        for (int i = 0; i < 24; ++i) {
            out_joints_world(i,0) = global_p(i,0) + trans[0];
            out_joints_world(i,1) = global_p(i,1) + trans[1];
            out_joints_world(i,2) = global_p(i,2) + trans[2];
        }
        return;
    }

    // --------- Fallback path (generic J): your original method (slow) ----------
    // Keep this so code still runs if you load a non-24-joint model.
    const int N = model.N;

    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> v_shaped(N, 3);
    for (int i = 0; i < N; ++i) {
        v_shaped(i,0) = T(model.v_template(i,0));
        v_shaped(i,1) = T(model.v_template(i,1));
        v_shaped(i,2) = T(model.v_template(i,2));
    }
    for (int i = 0; i < N; ++i) for (int c = 0; c < 3; ++c) {
        T off = T(0);
        for (int k = 0; k < 10; ++k) off += T(model.shapedirs(i*3+c, k)) * beta[k];
        v_shaped(i,c) += off;
    }

    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> joints(J, 3);
    joints.setZero();
    for (int j = 0; j < J; ++j) {
        const auto& nz = model.J_regressor_nz[j];
        for (const auto& it : nz) {
            int v = it.first;
            T w = T(it.second);
            joints(j,0) += w * v_shaped(v,0);
            joints(j,1) += w * v_shaped(v,1);
            joints(j,2) += w * v_shaped(v,2);
        }
    }

    // Generic FK using 4x4 (slow but safe)
    std::vector<int> parents(J, -1);
    parents[0] = -1;
    for (int i = 1; i < J; ++i) {
        int p = model.kintree_table(0, i);
        if (p < 0 || p >= J) p = -1;
        parents[i] = p;
    }

    std::vector<Eigen::Matrix<T,4,4>> global_T(J);
    for (int i = 0; i < J; ++i) {
        T aa[3] = { theta[i*3+0], theta[i*3+1], theta[i*3+2] };
        T Rcolmaj[9];
        ceres::AngleAxisToRotationMatrix(aa, Rcolmaj);

        Eigen::Matrix<T,4,4> local;
        local.setIdentity();
        local(0,0)=Rcolmaj[0]; local(0,1)=Rcolmaj[3]; local(0,2)=Rcolmaj[6];
        local(1,0)=Rcolmaj[1]; local(1,1)=Rcolmaj[4]; local(1,2)=Rcolmaj[7];
        local(2,0)=Rcolmaj[2]; local(2,1)=Rcolmaj[5]; local(2,2)=Rcolmaj[8];

        int p = parents[i];
        if (p == -1) {
            local(0,3)=joints(i,0);
            local(1,3)=joints(i,1);
            local(2,3)=joints(i,2);
            global_T[i] = local;
        } else {
            local(0,3)=joints(i,0)-joints(p,0);
            local(1,3)=joints(i,1)-joints(p,1);
            local(2,3)=joints(i,2)-joints(p,2);
            global_T[i] = global_T[p] * local;
        }
    }

    for (int i = 0; i < J; ++i) {
        out_joints_world(i,0) = global_T[i](0,3) + trans[0];
        out_joints_world(i,1) = global_T[i](1,3) + trans[1];
        out_joints_world(i,2) = global_T[i](2,3) + trans[2];
    }
}


template <typename T>
void SMPLForward(
    const T* beta,
    const T* theta,
    const T* trans,
    const SmplModel& model,
    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>& out_vertices,
    bool use_pose_blendshapes = true,
        // --- debug outputs (optional) ---
    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>* out_v_shaped = nullptr,
    Eigen::Matrix<T, Eigen::Dynamic, 1>* out_pose_feat = nullptr,   // size 207
    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>* out_v_posed = nullptr
) {
    const int N = model.N;
    const int J = model.J;
    const T eps = T(1e-12);

    // v_shaped = v_template + shapedirs * beta
    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> v_shaped(N, 3);
    for (int i = 0; i < N; ++i) {
        v_shaped(i,0) = T(model.v_template(i,0));
        v_shaped(i,1) = T(model.v_template(i,1));
        v_shaped(i,2) = T(model.v_template(i,2));
    }
    if (model.shapedirs.rows() == N*3 && model.shapedirs.cols() >= 10) {
        for (int i = 0; i < N; ++i) for (int c = 0; c < 3; ++c) {
            T off = T(0);
            for (int k = 0; k < 10; ++k) off += T(model.shapedirs(i*3+c, k)) * beta[k];
            v_shaped(i,c) += off;
        }
    }
    if (out_v_shaped) *out_v_shaped = v_shaped;

    // joints = J_regressor * v_shaped  (using cached nonzeros)
    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> joints(J, 3);
    joints.setZero();
    for (int j = 0; j < J; ++j) {
        const auto& nz = model.J_regressor_nz[j];
        for (const auto& it : nz) {
            int v = it.first;
            T w = T(it.second);
            joints(j,0) += w * v_shaped(v,0);
            joints(j,1) += w * v_shaped(v,1);
            joints(j,2) += w * v_shaped(v,2);
        }
    }

    // parents from kintree_table (assume row 0 contains parents for standard SMPL pkl->npz conversion)
    std::vector<int> parents(J, -1);
    parents[0] = -1;
    for (int i = 1; i < J; ++i) {
        int p = model.kintree_table(0, i);
        if (p < 0 || p >= J) p = -1;
        parents[i] = p;
    }

    // FK transforms
    std::vector<Eigen::Matrix<T,4,4>> local_T(J), global_T(J);
    for (int i = 0; i < J; ++i) {
        T aa[3] = { theta[i*3+0], theta[i*3+1], theta[i*3+2] };
        T R[9];
        ceres::AngleAxisToRotationMatrix(aa, R);

        local_T[i].setIdentity();
        local_T[i](0,0)=R[0]; local_T[i](0,1)=R[3]; local_T[i](0,2)=R[6];
        local_T[i](1,0)=R[1]; local_T[i](1,1)=R[4]; local_T[i](1,2)=R[7];
        local_T[i](2,0)=R[2]; local_T[i](2,1)=R[5]; local_T[i](2,2)=R[8];

        int p = parents[i];
        if (p == -1) {
            local_T[i](0,3)=joints(i,0);
            local_T[i](1,3)=joints(i,1);
            local_T[i](2,3)=joints(i,2);
            global_T[i] = local_T[i];
        } else {
            local_T[i](0,3)=joints(i,0)-joints(p,0);
            local_T[i](1,3)=joints(i,1)-joints(p,1);
            local_T[i](2,3)=joints(i,2)-joints(p,2);
            global_T[i] = global_T[p] * local_T[i];
        }
    }

    // posedirs (pose blendshapes)
    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> v_posed = v_shaped;

    if (use_pose_blendshapes && J == 24 &&
        model.posedirs.rows() == N*3 && model.posedirs.cols() >= 207) {

        Eigen::Matrix<T, 207, 1> pose_feat;
        pose_feat.setZero();

        for (int i = 1; i < 24; ++i) {
            Eigen::Matrix<T,3,3> Ri = local_T[i].template block<3,3>(0,0);
            Eigen::Matrix<T,3,3> dR = Ri - Eigen::Matrix<T,3,3>::Identity();

            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    pose_feat((i-1)*9 + r*3 + c) = dR(r,c);

        }

        // write pose_feat out
        if (out_pose_feat) {
            out_pose_feat->resize(207);
            *out_pose_feat = pose_feat;
        }

        // v_posed = v_shaped + posedirs * pose_feat
        for (int i = 0; i < N; ++i) {
            for (int c = 0; c < 3; ++c) {
                T off = T(0);
                for (int k = 0; k < 207; ++k) {
                    off += T(model.posedirs(i*3 + c, k)) * pose_feat(k);
                }
                v_posed(i,c) += off;
            }
        }
    }

    // NOW v_posed is final
    if (out_v_posed) *out_v_posed = v_posed;


    // skinning transforms Tskin = global_T * inv_bind
    std::vector<Eigen::Matrix<T,4,4>> Tskin(J);
    for (int i = 0; i < J; ++i) {
        Eigen::Matrix<T,4,4> inv_bind; inv_bind.setIdentity();
        inv_bind(0,3) = -joints(i,0);
        inv_bind(1,3) = -joints(i,1);
        inv_bind(2,3) = -joints(i,2);
        Tskin[i] = global_T[i] * inv_bind;
    }

    // LBS
    out_vertices.resize(N,3);
    out_vertices.setZero();
    for (int i = 0; i < N; ++i) {
        T vx=v_posed(i,0), vy=v_posed(i,1), vz=v_posed(i,2);
        for (int j = 0; j < J; ++j) {
            T w = T(model.weights(i,j));
            if (w <= eps) continue;
            T px = Tskin[j](0,0)*vx + Tskin[j](0,1)*vy + Tskin[j](0,2)*vz + Tskin[j](0,3);
            T py = Tskin[j](1,0)*vx + Tskin[j](1,1)*vy + Tskin[j](1,2)*vz + Tskin[j](1,3);
            T pz = Tskin[j](2,0)*vx + Tskin[j](2,1)*vy + Tskin[j](2,2)*vz + Tskin[j](2,3);
            out_vertices(i,0) += w*px;
            out_vertices(i,1) += w*py;
            out_vertices(i,2) += w*pz;
        }
    }

    // global translation
    for (int i = 0; i < N; ++i) {
        out_vertices(i,0) += trans[0];
        out_vertices(i,1) += trans[1];
        out_vertices(i,2) += trans[2];
    }
}
