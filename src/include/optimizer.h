#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>

#include <vector>
#include <array>
#include <algorithm>
#include <iostream>

#include "smpl_model.h"
#include "smpl_math.h"
#include <ceres/rotation.h>


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
    bool verbose = true;

    // ---------- 2D fitting ----------
    float min_confidence = 0.20f;     // OpenPose 建议别太低
    bool use_stable_joints_only = true; // 只用躯干+四肢（更稳）
    bool flip_y_in_projection = true;   // v = fy * (-Y/Z) + cy
    int root_free_axis = 1;             // Stage1 只放开一个root轴：0/1/2试哪个最稳（默认1）

    // ---------- 3D unit test ----------
    bool amass_unit_test = false;
    bool unit_test_exact = true;
    double w_3d = 1.0;
};

// ------------------------------
// OpenPose -> SMPL mapping
// IMPORTANT: only declare here, define ONCE in optimizer.cpp
// ------------------------------
extern const int OPENPOSE_TO_SMPL[25];

// ============================================================
// Helper (template): joints from mesh forward (slow but consistent with mesh)
// Used for 3D unit test / debug, NOT recommended for 2D optimization.
// ============================================================
template <typename T>
inline void SMPLJointsFromMeshForward(
    const T* beta,
    const T* theta,
    const T* trans,
    const SmplModel& model,
    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>& out_joints_world,
    bool use_pose_blendshapes
) {
    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> V;
    SMPLForward<T>(beta, theta, trans, model, V, use_pose_blendshapes);

    const int J = model.J;
    out_joints_world.resize(J, 3);
    out_joints_world.setZero();

    for (int j = 0; j < J; ++j) {
        const auto& nz = model.J_regressor_nz[j];
        for (const auto& it : nz) {
            const int v = it.first;
            const T w = T(it.second);
            out_joints_world(j,0) += w * V(v,0);
            out_joints_world(j,1) += w * V(v,1);
            out_joints_world(j,2) += w * V(v,2);
        }
    }
}

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
        if (angle < T(min_a_)) residual[0] = T(weight_) * (T(min_a_) - angle);
        else if (angle > T(max_a_)) residual[0] = T(weight_) * (angle - T(max_a_));
        return true;
    }
    int joint_idx_;
    int axis_;
    double min_a_, max_a_, weight_;
};

// ------------------------------
// 2D reprojection residual (FAST):
// - uses SMPLJointsForward (FK-only) for speed
// - supports stable joint filtering + y-flip projection
// Residual dim: 2 * NumUsedKeypoints()
// Parameter blocks: beta(10), theta(72), trans(3)
// ------------------------------
struct AllKeypointsReprojectionResidual {
    AllKeypointsReprojectionResidual(
        const std::vector<Landmark2D>& observed,
        const CameraIntrinsics& cam,
        const SmplModel& model,
        bool use_pose_blendshapes,
        float min_confidence,
        bool use_stable_only,
        bool flip_y
    )
    : camera_(cam),
      model_(model),
      use_pose_blendshapes_(use_pose_blendshapes),
      flip_y_(flip_y)
    {
        // stable joints: 1,2,3,4,5,6,7,8,9,10,11,12,13,14
        std::array<bool,25> allowed{};
        allowed.fill(true);
        if (use_stable_only) {
            allowed.fill(false);
            int ids[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14};
            for (int id : ids) allowed[id] = true;
        }

        lms_.reserve(observed.size());
        smpl_ids_.reserve(observed.size());

        for (const auto& lm : observed) {
            if (lm.confidence < min_confidence) continue;
            if (lm.joint_id < 0 || lm.joint_id >= 25) continue;
            if (!allowed[lm.joint_id]) continue;

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
        const T* beta  = parameters[0];
        const T* theta = parameters[1];
        const T* trans = parameters[2];

        // FAST: joints-only FK
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> Jw;
        SMPLJointsForward<T>(beta, theta, trans, model_, Jw);

        for (int i = 0; i < static_cast<int>(lms_.size()); ++i) {
            const int jid = smpl_ids_[i];

            const T X = Jw(jid, 0);
            const T Y = Jw(jid, 1);
            T       Z = Jw(jid, 2);
            if (Z < T(1e-6)) Z = T(1e-6);

            const T u = T(camera_.fx) * (X / Z) + T(camera_.cx);
            const T v = T(camera_.fy) * ((flip_y_ ? -Y : Y) / Z) + T(camera_.cy);

            const float c = std::max(1e-6f, lms_[i].confidence);
            const T wconf = ceres::sqrt(T(c)); // sqrt(conf) 比 conf 稳
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
    bool flip_y_;
};

struct RootUprightResidual {
    explicit RootUprightResidual(double w_) : w(w_) {}
    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        // theta[0:3] is root angle-axis
        const T* aa = theta; // root
        const T up[3] = {T(0), T(1), T(0)};
        T up_rot[3];
        ceres::AngleAxisRotatePoint(aa, up, up_rot);

        // want up_rot = (0,1,0) -> penalize x,z
        residuals[0] = T(w) * up_rot[0];
        residuals[1] = T(w) * up_rot[2];
        return true;
    }
    double w;
};

struct HingeAxisResidual {
    HingeAxisResidual(int joint_idx,
                      Eigen::Vector3d axis_unit,
                      double w_perp,
                      double w_range,
                      double min_a,
                      double max_a)
        : j_(joint_idx),
          u_(axis_unit.normalized()),
          w_perp_(w_perp),
          w_range_(w_range),
          min_a_(min_a),
          max_a_(max_a) {}

    template <typename T>
    bool operator()(const T* const theta, T* residual) const {
        const int base = 3 * j_;
        Eigen::Matrix<T,3,1> r(theta[base+0], theta[base+1], theta[base+2]);
        Eigen::Matrix<T,3,1> u(T(u_.x()), T(u_.y()), T(u_.z()));

        // parallel component (hinge DOF)
        const T a = u.dot(r);
        // perpendicular component (forbidden twist/swing)
        Eigen::Matrix<T,3,1> perp = r - u * a;

        // smooth norm to avoid nondiff at 0
        const T perp_norm = ceres::sqrt(perp.squaredNorm() + T(1e-12));
        residual[0] = T(w_perp_) * perp_norm;

        // range clamp on a
        T range_pen = T(0);
        if (a < T(min_a_)) range_pen = T(min_a_) - a;
        else if (a > T(max_a_)) range_pen = a - T(max_a_);
        residual[1] = T(w_range_) * range_pen;
        return true;
    }

    int j_;
    Eigen::Vector3d u_;
    double w_perp_, w_range_, min_a_, max_a_;
};


// ------------------------------
// 3D joints residual (used for AMASS unit test)
// ------------------------------
struct Joints3DResidual {
    Joints3DResidual(const MatX3dRM& J_obs_, const SmplModel& model_, double w_, bool use_pose_blendshapes_)
        : J_obs(J_obs_), model(model_), w(w_), use_pose_blendshapes(use_pose_blendshapes_) {}

    int NumJoints() const { return static_cast<int>(J_obs.rows()); }

    template <typename T>
    bool operator()(const T* const beta,
                    const T* const theta,
                    const T* const trans,
                    T* residuals) const {
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> J_pred;
        SMPLJointsFromMeshForward<T>(beta, theta, trans, model, J_pred, use_pose_blendshapes);

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
    bool use_pose_blendshapes;
};

// ------------------------------
// APIs
// ------------------------------
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

CameraIntrinsics CreateDefaultCamera(int image_width, int image_height);
