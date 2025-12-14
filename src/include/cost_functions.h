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

// OpenPose BODY_25 to SMPL 24 Joint Mapping
// -1 means no corresponding SMPL joint exists
const int OPENPOSE_TO_SMPL[25] = {
    15,  // 0: Nose -> SMPL Head (approximate)
    12,  // 1: Neck -> SMPL Neck
    17,  // 2: RShoulder -> SMPL Right Shoulder
    19,  // 3: RElbow -> SMPL Right Elbow
    21,  // 4: RWrist -> SMPL Right Wrist
    16,  // 5: LShoulder -> SMPL Left Shoulder
    18,  // 6: LElbow -> SMPL Left Elbow
    20,  // 7: LWrist -> SMPL Left Wrist
    0,   // 8: MidHip -> SMPL Pelvis
    2,   // 9: RHip -> SMPL Right Hip
    5,   // 10: RKnee -> SMPL Right Knee
    8,   // 11: RAnkle -> SMPL Right Ankle
    1,   // 12: LHip -> SMPL Left Hip
    4,   // 13: LKnee -> SMPL Left Knee
    7,   // 14: LAnkle -> SMPL Left Ankle
    -1,  // 15: REye (not used)
    -1,  // 16: LEye (not used)
    -1,  // 17: REar (not used)
    -1,  // 18: LEar (not used)
    -1,  // 19: LBigToe (not used)
    -1,  // 20: LSmallToe (not used)
    -1,  // 21: LHeel (not used)
    -1,  // 22: RBigToe (not used)
    -1,  // 23: RSmallToe (not used)
    -1   // 24: RHeel (not used)
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
        
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vertices;
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
    double expected_depth_;  // 期望深度 (例如2.5米)
    double sigma_;           // 标准差
};
