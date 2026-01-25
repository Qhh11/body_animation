#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>

#include <vector>
#include <array>
#include <algorithm>
#include <iostream>
#include <limits>

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

// Depth constraint data (Z coordinate in the same convention as SMPLJointsForward output)
struct JointDepthData {
    int smpl_joint_idx;
    double depth_z;
};

// Forward declaration for relative depth (full definition below)
struct RelativeDepthPair;

// ------------------------------
// GMM Pose Prior (from SMPLify)
// ------------------------------
struct GMMPosePrior {
    static constexpr int K = 8;    // number of Gaussian components
    static constexpr int D = 69;   // pose dimension (excluding root)

    Eigen::Matrix<double, K, 1> weights;                    // (K,)
    Eigen::Matrix<double, K, D, Eigen::RowMajor> means;     // (K, D)
    std::array<Eigen::Matrix<double, D, D, Eigen::RowMajor>, K> prec_choleskys;

    bool loaded = false;

    bool LoadFromNPZ(const std::string& npz_path);
};

struct OptimizerConfig {
    bool verbose = true;

    // ---------- 2D fitting ----------
    float min_confidence = 0.20f;
    bool use_stable_joints_only = true;
    bool flip_y_in_projection = true;

    // ---------- Depth constraint (absolute) ----------
    bool use_depth = false;
    double w_depth = 15.0;
    const std::vector<JointDepthData>* depth_data = nullptr;

    // ---------- Relative depth constraint ----------
    bool use_relative_depth = false;
    double w_relative_depth = 20.0;
    const std::vector<RelativeDepthPair>* relative_depth_pairs = nullptr;

    // ---------- Root upright regularizer ----------
    bool use_root_upright = true;
    double root_upright_w_stage2 = 1.0;
    double root_upright_w_stage3 = 0.0;

    // ---------- GMM pose prior ----------
    bool use_gmm_prior = false;
    double w_gmm = 1.0;
    const GMMPosePrior* gmm_prior = nullptr;

    // ---------- Angle/hinge priors ----------
    bool use_hinge_priors = true;
    double w_hinge_perp_knee  = 120.0;
    double w_hinge_range_knee = 60.0;
    double w_hinge_perp_elbow  = 80.0;
    double w_hinge_range_elbow = 40.0;

    // 统一 hinge range（以前你写死 -2.6~2.6，现在 main 可调）
    double hinge_min = -2.6;
    double hinge_max =  2.6;

    // Optional hyperextension exp penalty
    bool use_bend_limits = true;
    double w_bend = 1.0;

    // ---------- Shoulder / Collar / Wrist / Spine soft limits (NEW: main可调) ----------
    bool use_spine_limits = true;
    double w_spine = 10.0;
    double spine_amax = 1.2;   // radians
    // spine joints ids in your SMPL joint indexing (you之前用 {3,6,9})
    int spine_j1 = 3;
    int spine_j2 = 6;
    int spine_j3 = 9;

    bool use_shoulder_collar_limits = true;
    double w_shoulder_limit = 60.0;
    double shoulder_amax = 1.7; // radians
    double w_collar_limit = 40.0;
    double collar_amax = 0.8;   // radians
    // collar joints {13,14}, shoulder joints {16,17} (按你原来的)
    int collar_L = 13;
    int collar_R = 14;
    int shoulder_L = 16;
    int shoulder_R = 17;

    bool use_wrist_limits = true;
    double w_wrist_limit = 80.0;
    double wrist_amax = 0.6;
    int wrist_L = 20;
    int wrist_R = 21;

    // ------------- Arm–torso separation -----------
    bool use_arm_torso_sep = true;

    // Optional tuning parameters (meters and weights).
    double arm_torso_d_elbow = 0.08;
    double arm_torso_d_wrist = 0.10;
    double arm_torso_w_elbow = 80.0;
    double arm_torso_w_wrist = 120.0;

    int arm_torso_seg_a = 3;   
    int arm_torso_seg_b = 12; 

    // ---------- 3D unit test ----------
    bool amass_unit_test = false;
    bool unit_test_exact = true;
    double w_3d = 1.0;

    // ---------- temporal smoothness (tracking) ----------
    bool use_temporal = false;     // 开关
    double w_temp_pose = 10.0;     // 一阶：||theta_t - theta_prev||^2 权重
    double w_temp_trans = 2.0;     // 一阶：||trans_t - trans_prev||^2 权重
    bool use_temporal_acc = false; // 二阶加速度项
    double w_temp_acc_pose = 2.0;  // 二阶：||theta_t - 2*theta_{t-1} + theta_{t-2}||^2

    // 只对 body pose 做时序（建议默认 true）：theta[3:71]
    bool temporal_body_only = true;


    // ---------- speed / solver ----------
    int threads = -1;                 // -1: auto
    bool print_progress = false;      // printing is slow
    int max_iter_stage1 = 60;
    int max_iter_stage2 = 200;
    int max_iter_stage3 = 200;
};

// ------------------------------
// OpenPose -> SMPL mapping
// declare here, define ONCE in optimizer.cpp
// ------------------------------
extern const int OPENPOSE_TO_SMPL[25];

// ============================================================
// Helper (template): joints from mesh forward
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

// ------------------------------
// GMM Pose Prior Residual
// ------------------------------
struct GMMPosePriorResidual {
    GMMPosePriorResidual(const GMMPosePrior& gmm, double weight)
        : gmm_(gmm), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        constexpr int D = 69;
        constexpr int K = 8;

        T min_dist = T(1e10);
        int best_k = 0;

        for (int k = 0; k < K; ++k) {
            T dist = T(0);
            for (int i = 0; i < D; ++i) {
                T weighted = T(0);
                for (int j = 0; j <= i; ++j) {
                    weighted += T(gmm_.prec_choleskys[k](i, j)) * (theta[3 + j] - T(gmm_.means(k, j)));
                }
                dist += weighted * weighted;
            }
            T log_w = T(std::log(gmm_.weights(k) + 1e-10));
            T score = dist - T(2.0) * log_w;

            if (score < min_dist) { min_dist = score; best_k = k; }
        }

        for (int i = 0; i < D; ++i) {
            T weighted = T(0);
            for (int j = 0; j <= i; ++j) {
                weighted += T(gmm_.prec_choleskys[best_k](i, j)) * (theta[3 + j] - T(gmm_.means(best_k, j)));
            }
            residuals[i] = T(weight_) * weighted;
        }
        return true;
    }

    const GMMPosePrior& gmm_;
    double weight_;
};

struct RootUprightResidual {
    explicit RootUprightResidual(double w_) : w(w_) {}
    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        const T* aa = theta; // root axis-angle
        const T up[3] = {T(0), T(1), T(0)};
        T up_rot[3];
        ceres::AngleAxisRotatePoint(aa, up, up_rot);
        residuals[0] = T(w) * up_rot[0];
        residuals[1] = T(w) * up_rot[2];
        return true;
    }
    double w;
};

struct ArmSegToTorsoSegFloorResidual {
    ArmSegToTorsoSegFloorResidual(int arm_a, int arm_b,
                                  int torso_a, int torso_b,
                                  double dmin, double w,
                                  const SmplModel& model_)
        : aa(arm_a), ab(arm_b), ta(torso_a), tb(torso_b),
          dmin(dmin), w(w), model(model_) {}

    template <typename T>
    bool operator()(const T* const beta,
                    const T* const theta,
                    const T* const trans,
                    T* residual) const
    {
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> J;
        SMPLJointsForward<T>(beta, theta, trans, model, J);

        const Eigen::Matrix<T,3,1> A0 = J.row(aa).transpose();
        const Eigen::Matrix<T,3,1> A1 = J.row(ab).transpose();
        const Eigen::Matrix<T,3,1> B0 = J.row(ta).transpose();
        const Eigen::Matrix<T,3,1> B1 = J.row(tb).transpose();

        auto dist_point_to_seg = [&](const Eigen::Matrix<T,3,1>& P)->T{
            const Eigen::Matrix<T,3,1> AB = B1 - B0;
            const T ab2 = AB.squaredNorm() + T(1e-12);
            T t = (P - B0).dot(AB) / ab2;
            t = ceres::fmin(T(1), ceres::fmax(T(0), t));
            const Eigen::Matrix<T,3,1> Q = B0 + t * AB;
            return ceres::sqrt((P - Q).squaredNorm() + T(1e-12));
        };

        const T ts[3] = {T(0.2), T(0.5), T(0.8)};
        T sse = T(0);
        for (int i=0;i<3;++i){
            const Eigen::Matrix<T,3,1> P = A0 + ts[i]*(A1 - A0);
            const T dist = dist_point_to_seg(P);
            const T x = T(dmin) - dist;
            const T viol = (ceres::abs(x) + x) * T(0.5);
            sse += viol * viol;
        }
        residual[0] = T(w) * ceres::sqrt(sse + T(1e-12));
        return true;
    }

    int aa, ab, ta, tb;
    double dmin, w;
    const SmplModel& model;
};

struct PointToSegmentDistanceFloorResidual {
    PointToSegmentDistanceFloorResidual(int point_joint,
                                        int seg_a_joint,
                                        int seg_b_joint,
                                        double d_min,
                                        double w,
                                        const SmplModel& model_)
        : p(point_joint), a(seg_a_joint), b(seg_b_joint),
          dmin(d_min), w(w), model(model_) {}

    template <typename T>
    bool operator()(const T* const beta,
                    const T* const theta,
                    const T* const trans,
                    T* residual) const
    {
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> J;
        SMPLJointsForward<T>(beta, theta, trans, model, J);

        const Eigen::Matrix<T,3,1> P = J.row(p).transpose();
        const Eigen::Matrix<T,3,1> A = J.row(a).transpose();
        const Eigen::Matrix<T,3,1> B = J.row(b).transpose();

        const Eigen::Matrix<T,3,1> AB = B - A;
        const T ab2 = AB.squaredNorm() + T(1e-12);

        T t = (P - A).dot(AB) / ab2;
        t = ceres::fmin(T(1), ceres::fmax(T(0), t));

        const Eigen::Matrix<T,3,1> Q = A + t * AB;
        const T dist = ceres::sqrt((P - Q).squaredNorm() + T(1e-12));

        const T x = T(dmin) - dist;
        const T viol = (ceres::abs(x) + x) * T(0.5);

        residual[0] = T(w) * viol;
        return true;
    }

    int p, a, b;
    double dmin, w;
    const SmplModel& model;
};

// ------------------------------
// Hinge axis residual
// ------------------------------
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

        const T a = u.dot(r);
        Eigen::Matrix<T,3,1> perp = r - u * a;

        residual[0] = T(w_perp_) * ceres::sqrt(perp.squaredNorm() + T(1e-12));

        const T low  = T(min_a_) - a;
        const T high = a - T(max_a_);
        const T over_low  = (ceres::abs(low)  + low)  * T(0.5);
        const T over_high = (ceres::abs(high) + high) * T(0.5);
        residual[1] = T(w_range_) * (over_low + over_high);

        return true;
    }

    int j_;
    Eigen::Vector3d u_;
    double w_perp_, w_range_, min_a_, max_a_;
};

struct SoftAngleLimitResidual {
    SoftAngleLimitResidual(int joint_idx, double amax, double w)
        : j_(joint_idx), amax_(amax), w_(w) {}

    template <typename T>
    bool operator()(const T* const theta, T* residual) const {
        const int base = 3 * j_;
        const T rx = theta[base + 0];
        const T ry = theta[base + 1];
        const T rz = theta[base + 2];

        const T ang = ceres::sqrt(rx*rx + ry*ry + rz*rz + T(1e-12));
        const T over = ang - T(amax_);
        const T abs_over = ceres::sqrt(over * over + T(1e-12));
        const T max0 = (abs_over + over) * T(0.5);

        residual[0] = T(w_) * max0;
        return true;
    }

    int j_;
    double amax_, w_;
};

struct BendLimitResidual {
    BendLimitResidual(int joint_idx, int axis_idx, double sign, double weight)
        : joint_idx_(joint_idx), axis_idx_(axis_idx), sign_(sign), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const theta, T* residual) const {
        const int idx = 3 * joint_idx_ + axis_idx_;
        residual[0] = T(weight_) * ceres::exp(T(sign_) * theta[idx]);
        return true;
    }

    int joint_idx_;
    int axis_idx_;
    double sign_;
    double weight_;
};

struct TemporalPoseResidual {
    TemporalPoseResidual(const double* theta_prev_, double w_, bool body_only_)
        : prev(theta_prev_), w(w_), body_only(body_only_) {}

    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        int k = 0;
        if (body_only) {
            for (int i = 3; i < 72; ++i) residuals[k++] = T(w) * (theta[i] - T(prev[i]));
        } else {
            for (int i = 0; i < 72; ++i) residuals[k++] = T(w) * (theta[i] - T(prev[i]));
        }
        return true;
    }
    const double* prev;
    double w;
    bool body_only;
};

struct TemporalTransResidual {
    TemporalTransResidual(const double* trans_prev_, double w_)
        : prev(trans_prev_), w(w_) {}

    template <typename T>
    bool operator()(const T* const trans, T* residuals) const {
        residuals[0] = T(w) * (trans[0] - T(prev[0]));
        residuals[1] = T(w) * (trans[1] - T(prev[1]));
        residuals[2] = T(w) * (trans[2] - T(prev[2]));
        return true;
    }
    const double* prev;
    double w;
};

struct TemporalAccelPoseResidual {
    TemporalAccelPoseResidual(const double* theta_prev_, const double* theta_prev2_,
                              double w_, bool body_only_)
        : p1(theta_prev_), p2(theta_prev2_), w(w_), body_only(body_only_) {}

    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        int k = 0;
        if (body_only) {
            for (int i = 3; i < 72; ++i) {
                residuals[k++] = T(w) * (theta[i] - T(2.0)*T(p1[i]) + T(p2[i]));
            }
        } else {
            for (int i = 0; i < 72; ++i) {
                residuals[k++] = T(w) * (theta[i] - T(2.0)*T(p1[i]) + T(p2[i]));
            }
        }
        return true;
    }
    const double* p1;
    const double* p2;
    double w;
    bool body_only;
};

// ------------------------------
// 2D reprojection residual
// ------------------------------
struct AllKeypointsReprojectionResidual {
    AllKeypointsReprojectionResidual(
        const std::vector<Landmark2D>& observed,
        const CameraIntrinsics& cam,
        const SmplModel& model,
        float min_confidence,
        bool use_stable_only,
        bool flip_y
    )
    : camera_(cam), model_(model), flip_y_(flip_y)
    {
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
    bool operator()(const T* const beta,
                    const T* const theta,
                    const T* const trans,
                    T* residuals) const
    {
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> Jw;
        SMPLJointsForward<T>(beta, theta, trans, model_, Jw);

        for (int i = 0; i < (int)lms_.size(); ++i) {
            const int jid = smpl_ids_[i];

            const T X = Jw(jid, 0);
            const T Y = Jw(jid, 1);
            T       Z = Jw(jid, 2);
            if (Z < T(1e-6)) Z = T(1e-6);

            const T u = T(camera_.fx) * (X / Z) + T(camera_.cx);
            const T v = T(camera_.fy) * ((flip_y_ ? -Y : Y) / Z) + T(camera_.cy);

            const float c = std::max(1e-6f, lms_[i].confidence);
            const T wconf = ceres::sqrt(T(c));
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
    bool flip_y_;
};

// ------------------------------
// 3D joints residual (FAST)
// ------------------------------
struct Joints3DResidualFast {
    Joints3DResidualFast(const MatX3dRM& J_obs_, const SmplModel& model_, double w_)
        : J_obs(J_obs_), model(model_), w(w_) {}

    template <typename T>
    bool operator()(const T* const beta,
                    const T* const theta,
                    const T* const trans,
                    T* residuals) const {
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> J_pred;
        SMPLJointsForward<T>(beta, theta, trans, model, J_pred);

        for (int i = 0; i < 24; ++i) {
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

// ------------------------------
// Depth residual (absolute)
// ------------------------------
struct DepthResidual {
    DepthResidual(const std::vector<JointDepthData>& depth_data_,
                  const SmplModel& model_,
                  double w_)
        : depth_data(depth_data_), model(model_), w(w_) {}

    int NumDepthPoints() const { return static_cast<int>(depth_data.size()); }

    template <typename T>
    bool operator()(const T* const beta,
                    const T* const theta,
                    const T* const trans,
                    T* residuals) const {
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> J_pred;
        SMPLJointsForward<T>(beta, theta, trans, model, J_pred);

        for (int i = 0; i < NumDepthPoints(); ++i) {
            int j = depth_data[i].smpl_joint_idx;
            const T z_pred = J_pred(j, 2);
            const T z_gt = T(depth_data[i].depth_z);
            residuals[i] = T(w) * (z_pred - z_gt);
        }
        return true;
    }

    std::vector<JointDepthData> depth_data;
    const SmplModel& model;
    double w;
};

// ------------------------------
// Relative depth residual
// ------------------------------
struct RelativeDepthPair {
    int smpl_joint_a;
    int smpl_joint_b;
    double rel_depth_a;
    double rel_depth_b;
};

struct RelativeDepthResidual {
    RelativeDepthResidual(const std::vector<RelativeDepthPair>& pairs_,
                          const SmplModel& model_,
                          double w_)
        : pairs(pairs_), model(model_), w(w_) {}

    int NumPairs() const { return static_cast<int>(pairs.size()); }

    template <typename T>
    bool operator()(const T* const beta,
                    const T* const theta,
                    const T* const trans,
                    T* residuals) const {
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> J_pred;
        SMPLJointsForward<T>(beta, theta, trans, model, J_pred);

        for (int i = 0; i < NumPairs(); ++i) {
            const int ja = pairs[i].smpl_joint_a;
            const int jb = pairs[i].smpl_joint_b;

            const T z_a = J_pred(ja, 2);
            const T z_b = J_pred(jb, 2);
            const T z_diff = z_a - z_b;

            const double rel_diff = pairs[i].rel_depth_a - pairs[i].rel_depth_b;

            const double eps = 1e-3;
            if (std::abs(rel_diff) < eps) {
                residuals[i] = T(0);
                continue;
            }

            const T s = T(rel_diff > 0 ? 1.0 : -1.0);
            const T violation = s * z_diff; // >0 => wrong ordering
            residuals[i] = T(w) * (violation > T(0) ? violation : T(0));
        }
        return true;
    }

    std::vector<RelativeDepthPair> pairs;
    const SmplModel& model;
    double w;
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
                  
bool OptimizeSMPLTracked(const std::vector<Landmark2D>& observed_keypoints,
                         const CameraIntrinsics& camera,
                         const SmplModel& model,
                         double* beta,
                         double* theta,
                         double* trans,
                         const OptimizerConfig& cfg,
                         const double* theta_prev,
                         const double* trans_prev,
                         const double* theta_prev2 = nullptr);

CameraIntrinsics CreateDefaultCamera(int image_width, int image_height);
