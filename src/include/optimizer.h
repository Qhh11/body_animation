#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>

#include <vector>
#include <algorithm>
#include <iostream>

#include "smpl_model.h"
#include "smpl_math.h"

using MatX3dRM = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

// ------------------------------
// Data structs
// ------------------------------
struct Landmark2D {
    float x, y;
    float confidence;
    int joint_id;

    Landmark2D() : x(0.f), y(0.f), confidence(0.f), joint_id(-1) {}
    Landmark2D(float x_, float y_, float conf_, int id_)
        : x(x_), y(y_), confidence(conf_), joint_id(id_) {}
};

struct CameraIntrinsics {
    float fx, fy;
    float cx, cy;
    int width, height;

    CameraIntrinsics() : fx(0.f), fy(0.f), cx(0.f), cy(0.f), width(0), height(0) {}
    CameraIntrinsics(float fx_, float fy_, float cx_, float cy_, int w_, int h_)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_), width(w_), height(h_) {}
};

struct OptimizerConfig {
    bool verbose = false;
    bool amass_unit_test = false;

    // 2D reproj config (你已有的)
    float min_confidence = 0.05f;

    // 3D unit test config
    bool unit_test_exact = true;   // true: 不加正则，理论上可到 0
    double w_3d = 1.0;             // 3D residual 权重
};

struct Joints3DResidual {
    Joints3DResidual(const MatX3dRM& J_obs_, const SmplModel& model_, double w_)
        : J_obs(J_obs_), model(model_), w(w_) {}

    int NumJoints() const { return static_cast<int>(J_obs.rows()); }

    template <typename T>
    bool operator()(const T* const beta,
                    const T* const theta,
                    const T* const trans,
                    T* residuals) const {
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> J_pred;
        SMPLJointsForward<T>(beta, theta, trans, model, J_pred);

        const int K = NumJoints();
        for (int i = 0; i < K; ++i) {
            residuals[3*i+0] = T(w) * (J_pred(i,0) - T(J_obs(i,0)));
            residuals[3*i+1] = T(w) * (J_pred(i,1) - T(J_obs(i,1)));
            residuals[3*i+2] = T(w) * (J_pred(i,2) - T(J_obs(i,2)));
        }
        return true;
    }

    MatX3dRM J_obs;
    const SmplModel& model;
    double w;
};


bool OptimizeSMPL3D(const MatX3dRM& J_obs,
                    const SmplModel& model,
                    double* beta,
                    double* theta,
                    double* trans,
                    const OptimizerConfig& config);

bool OptimizeSMPL(const std::vector<Landmark2D>& observed_keypoints,
                  const CameraIntrinsics& camera,
                  const SmplModel& model,
                  double* beta,
                  double* theta,
                  double* trans,
                  const OptimizerConfig& config);

// ------------------------------
// OpenPose -> SMPL mapping
// IMPORTANT: Only declare here. Define ONCE in optimizer.cpp
// ------------------------------
extern const int OPENPOSE_TO_SMPL[25];

// ------------------------------
// Regularizers / constraints
// ------------------------------
struct ShapeRegularizer {
    ShapeRegularizer(int index, double lambda) : index_(index), lambda_(lambda) {}

    template <typename T>
    bool operator()(const T* beta, T* residual) const {
        residual[0] = T(lambda_) * beta[index_];
        return true;
    }

    int index_;
    double lambda_;
};

struct PoseRegularizer {
    PoseRegularizer(int index, double lambda) : index_(index), lambda_(lambda) {}

    template <typename T>
    bool operator()(const T* theta, T* residual) const {
        residual[0] = T(lambda_) * theta[index_];
        return true;
    }

    int index_;
    double lambda_;
};

struct AnatomicalLimitResidual {
    AnatomicalLimitResidual(int joint_idx, int axis,
                            double min_angle, double max_angle, double weight)
        : joint_idx_(joint_idx),
          axis_(axis),
          min_a_(min_angle),
          max_a_(max_angle),
          weight_(weight) {}

    template <typename T>
    bool operator()(const T* const theta, T* residual) const {
        const T angle = theta[joint_idx_ * 3 + axis_];

        residual[0] = T(0);

        if (angle < T(min_a_)) {
            residual[0] = T(weight_) * (T(min_a_) - angle);
        } else if (angle > T(max_a_)) {
            residual[0] = T(weight_) * (angle - T(max_a_));
        }
        return true;
    }

    int joint_idx_;
    int axis_;
    double min_a_, max_a_, weight_;
};

// ------------------------------
// All-keypoints reprojection residual (DynamicAutoDiff)
// Residual dimension: 2 * NumUsedKeypoints()
// Parameter blocks: beta(10), theta(72), trans(3)
// ------------------------------
struct AllKeypointsReprojectionResidual {
    AllKeypointsReprojectionResidual(
        const std::vector<Landmark2D>& observed,
        const CameraIntrinsics& cam,
        const SmplModel& model,
        bool use_pose_blendshapes,
        float min_confidence
    )
    : camera_(cam),
      model_(model),
      use_pose_blendshapes_(use_pose_blendshapes)
    {
        lms_.reserve(observed.size());
        smpl_ids_.reserve(observed.size());

        for (const auto& lm : observed) {
            if (lm.confidence < min_confidence) continue;
            if (lm.joint_id < 0 || lm.joint_id >= 25) continue;

            const int sid = OPENPOSE_TO_SMPL[lm.joint_id];
            if (sid < 0 || sid >= model_.J) continue;

            lms_.push_back(lm);
            smpl_ids_.push_back(sid);
        }
    }

    int NumUsedKeypoints() const { return static_cast<int>(lms_.size()); }

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const
    {
        const T* beta  = parameters[0]; // 10
        const T* theta = parameters[1]; // 72
        const T* trans = parameters[2]; // 3

        // (1) joints-only forward: Jw is (J x 3) in world/camera coords
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> Jw;
        SMPLJointsForward<T>(beta, theta, trans, model_, Jw);

        // (2) residuals (2 per used keypoint)
        for (int i = 0; i < static_cast<int>(lms_.size()); ++i) {
            const int jid = smpl_ids_[i];

            const T X = Jw(jid, 0);
            const T Y = Jw(jid, 1);
            T       Z = Jw(jid, 2);

            if (Z < T(1e-6)) Z = T(1e-6); 

            const T u = T(camera_.fx) * (X / Z) + T(camera_.cx);
            const T v = T(camera_.fy) * (Y / Z) + T(camera_.cy);

            const T wconf = T(std::max(0.05f, lms_[i].confidence));
            residuals[2*i + 0] = wconf * (u - T(lms_[i].x));
            residuals[2*i + 1] = wconf * (v - T(lms_[i].y));
        }

        return true;
    }


private:
    std::vector<Landmark2D> lms_;
    std::vector<int> smpl_ids_;

    CameraIntrinsics camera_;
    const SmplModel& model_;
    bool use_pose_blendshapes_;
};


CameraIntrinsics CreateDefaultCamera(int image_width, int image_height);
