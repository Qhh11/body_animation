#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "smpl_model.h"
#include "smpl_math.h"


struct Landmark2D {
    float x, y;           
    float confidence;    
    int joint_id;        
    
    Landmark2D() : x(0), y(0), confidence(0), joint_id(-1) {}
    Landmark2D(float x_, float y_, float conf, int id) 
        : x(x_), y(y_), confidence(conf), joint_id(id) {}
};

struct CameraIntrinsics {
    float fx, fy;
    float cx, cy;
    int width, height;

    CameraIntrinsics() : fx(0), fy(0), cx(0), cy(0), width(0), height(0) {}
    CameraIntrinsics(float fx_, float fy_, float cx_, float cy_, int w, int h)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_), width(w), height(h) {}
};

// Camera extrinsics: world-to-camera transform
// P_cam = R * P_world + t
struct CameraExtrinsics {
    Eigen::Matrix3d R;  // 3x3 rotation matrix
    Eigen::Vector3d t;  // 3x1 translation vector

    CameraExtrinsics() : R(Eigen::Matrix3d::Identity()), t(Eigen::Vector3d::Zero()) {}
    CameraExtrinsics(const Eigen::Matrix3d& R_, const Eigen::Vector3d& t_) : R(R_), t(t_) {}

    // Transform a point from world to camera coordinates
    template<typename T>
    void TransformPoint(const T* p_world, T* p_cam) const {
        // p_cam = R * p_world + t
        for (int i = 0; i < 3; ++i) {
            p_cam[i] = T(t(i));
            for (int j = 0; j < 3; ++j) {
                p_cam[i] += T(R(i, j)) * p_world[j];
            }
        }
    }

    bool IsIdentity() const {
        return R.isIdentity(1e-6) && t.isZero(1e-6);
    }
};



// Cost Function 1: 2D Reprojection Error
struct ReprojectionResidual {
    ReprojectionResidual(
        const Landmark2D& observed,
        int smpl_joint_id,
        const CameraIntrinsics& camera,
        const SmplModel& model
    ) : observed_landmark_(observed),
        smpl_joint_id_(smpl_joint_id),
        camera_(camera),
        model_(model) {}
    
    template <typename T>
    bool operator()(
        const T* beta,   // [10] 
        const T* theta,  // [72] 
        const T* trans,  // [3] 
        T* residual      // [2] 
    ) const {
        
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> vertices;
        SMPLForward<T>(beta, theta, trans, model_, vertices);
        
        // Step 2: 计算3D关节位置 (J_regressor × vertices)
        T joint_3d[3] = {T(0.0), T(0.0), T(0.0)};
        
        for (int v = 0; v < model_.N; ++v) {
            T weight = T(model_.J_regressor(smpl_joint_id_, v));
            if (weight > T(1e-5)) {  // 稀疏优化
                joint_3d[0] += weight * vertices(v, 0);
                joint_3d[1] += weight * vertices(v, 1);
                joint_3d[2] += weight * vertices(v, 2);
            }
        }
        
        // Step 3: 3D → 2D 投影 
        // 公式: u = fx * (X/Z) + cx
        //       v = fy * (Y/Z) + cy
        T X = joint_3d[0];
        T Y = joint_3d[1];
        T Z = joint_3d[2];
        
        if (ceres::abs(Z) < T(1e-6)) {
            Z = T(1e-6);
        }
        
        T projected_x = T(camera_.fx) * (X / Z) + T(camera_.cx);
        T projected_y = T(camera_.fy) * (Y / Z) + T(camera_.cy);
        
        // Step 4: 计算残差 = 预测 - 观测
        residual[0] = projected_x - T(observed_landmark_.x);
        residual[1] = projected_y - T(observed_landmark_.y);
        
        // Ceres会自动计算 residual[0]² + residual[1]²
        return true;
    }
    
private:
    Landmark2D observed_landmark_;
    int smpl_joint_id_;
    CameraIntrinsics camera_;
    const SmplModel& model_;
};

// Cost Function 1b: 2D Reprojection Error with Camera Extrinsics
// SMPL body is in WORLD coordinates, camera extrinsics transform to camera space
struct ReprojectionResidualWithExtrinsics {
    ReprojectionResidualWithExtrinsics(
        const Landmark2D& observed,
        int smpl_joint_id,
        const CameraIntrinsics& camera,
        const CameraExtrinsics& extrinsics,
        const SmplModel& model
    ) : observed_landmark_(observed),
        smpl_joint_id_(smpl_joint_id),
        camera_(camera),
        extrinsics_(extrinsics),
        model_(model) {}

    template <typename T>
    bool operator()(
        const T* beta,   // [10]
        const T* theta,  // [72]
        const T* trans,  // [3] - translation in WORLD coordinates
        T* residual      // [2]
    ) const {

        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> vertices;
        SMPLForward<T>(beta, theta, trans, model_, vertices);

        // Step 2: 计算3D关节位置 (J_regressor × vertices) - in WORLD coordinates
        T joint_world[3] = {T(0.0), T(0.0), T(0.0)};

        for (int v = 0; v < model_.N; ++v) {
            T weight = T(model_.J_regressor(smpl_joint_id_, v));
            if (weight > T(1e-5)) {
                joint_world[0] += weight * vertices(v, 0);
                joint_world[1] += weight * vertices(v, 1);
                joint_world[2] += weight * vertices(v, 2);
            }
        }

        // Step 3: Transform from WORLD to CAMERA coordinates
        // P_cam = R * P_world + t
        T joint_cam[3];
        for (int i = 0; i < 3; ++i) {
            joint_cam[i] = T(extrinsics_.t(i));
            for (int j = 0; j < 3; ++j) {
                joint_cam[i] += T(extrinsics_.R(i, j)) * joint_world[j];
            }
        }

        // Step 4: 3D → 2D 投影 (pinhole camera model)
        // Note: 3DPW camera has Y+ down, but we flipped Y in extrinsics
        T X = joint_cam[0];
        T Y = joint_cam[1];
        T Z = joint_cam[2];

        if (ceres::abs(Z) < T(1e-6)) {
            Z = T(1e-6);
        }

        T projected_x = T(camera_.fx) * (X / Z) + T(camera_.cx);
        T projected_y = T(camera_.fy) * (Y / Z) + T(camera_.cy);

        // Step 5: 计算残差 = 预测 - 观测
        residual[0] = projected_x - T(observed_landmark_.x);
        residual[1] = projected_y - T(observed_landmark_.y);

        return true;
    }

private:
    Landmark2D observed_landmark_;
    int smpl_joint_id_;
    CameraIntrinsics camera_;
    CameraExtrinsics extrinsics_;
    const SmplModel& model_;
};

// Cost Function 2: Shape正则化 (L2范数)
struct ShapeRegularizer {
    ShapeRegularizer(int index, double lambda) 
        : index_(index), lambda_(lambda) {}
    
    template <typename T>
    bool operator()(const T* beta, T* residual) const {
        // 惩罚项: λ * β[index]
        residual[0] = T(lambda_) * beta[index_];
        return true;
    }
    
private:
    int index_;
    double lambda_;  // 正则化权重
};

// Cost Function 3: Pose正则化 (L2范数)
struct PoseRegularizer {
    PoseRegularizer(int index, double lambda) 
        : index_(index), lambda_(lambda) {}
    
    template <typename T>
    bool operator()(const T* theta, T* residual) const {
        // 惩罚项: λ * θ[index]
        residual[0] = T(lambda_) * theta[index_];
        return true;
    }
    
private:
    int index_;
    double lambda_;
};

// Cost Function 4: 深度先验 (可选)
struct DepthPrior {
    DepthPrior(double expected_depth, double sigma) 
        : expected_depth_(expected_depth), sigma_(sigma) {}
    
    template <typename T>
    bool operator()(const T* trans, T* residual) const {
        // 希望深度接近expected_depth
        residual[0] = (trans[2] - T(expected_depth_)) / T(sigma_);
        return true;
    }
    
private:
    double expected_depth_;  
    double sigma_;          
};


struct AnatomicalLimitResidual {
    AnatomicalLimitResidual(int joint_idx, int axis, double min_angle, double max_angle, double weight)
        : joint_idx_(joint_idx), axis_(axis), min_a_(min_angle), max_a_(max_angle), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const theta, T* residual) const {
        // theta is 72-dim. Joint i starts at 3*i.
        // We focus on the specific axis (0=x, 1=y, 2=z).
        const T& angle = theta[joint_idx_ * 3 + axis_]; 

        residual[0] = T(0.0);

        // Barrier Function
        if (angle < T(min_a_)) {
            residual[0] = T(weight_) * (T(min_a_) - angle);
        } else if (angle > T(max_a_)) {
            residual[0] = T(weight_) * (angle - T(max_a_));
        }
        return true;
    }

private:
    int joint_idx_;
    int axis_;    
    double min_a_, max_a_, weight_;
};