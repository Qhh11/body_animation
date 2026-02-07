// optimizer.h
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

// Depth constraint data (absolute Z in SMPL world/camera convention)
struct JointDepthData {
    int smpl_joint_idx;
    double depth_z;
};

// Relative depth observation (arbitrary scale)
struct JointRelDepthData {
    int smpl_joint_idx;
    double rel_depth;
    double weight; // usually confidence (0..1)
};

// Forward declaration
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

    // ---------- Affine depth constraint ----------
    bool use_depth_affine = false;
    double w_depth_affine = 30.0;
    double depth_affine_sign =-1.0;  

    const std::vector<JointRelDepthData>* depth_affine_data = nullptr;

    // parameter block of size 2: [0]=scale, [1]=bias
    double* depth_affine_params = nullptr;

    // temporal prior for affine params
    bool use_temporal_depth_affine = true;
    double w_temp_depth_scale = 80.0;
    double w_temp_depth_bias  = 200.0;
    const double* depth_affine_prev = nullptr;

    // If true, fix the bias parameter (don't optimize it after first frame)
    bool fix_depth_affine_bias = false;

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
    double w_hinge_perp_elbow  = 10.0;
    double w_hinge_range_elbow = 5.0;

    double hinge_min = -2.6;
    double hinge_max =  2.6;

    bool use_bend_limits = true;
    double w_bend = 1.0;

    // ---------- Shoulder / Collar / Wrist / Spine soft limits ----------
    bool use_spine_limits = true;
    double w_spine = 10.0;
    double spine_amax = 1.2;        // symmetric overall angle limit
    int spine_j1 = 3;
    int spine_j2 = 6;
    int spine_j3 = 9;

    bool use_shoulder_collar_limits = true;
    double w_shoulder_limit = 60.0;
    double shoulder_amax = 3.0;
    double w_collar_limit = 40.0;
    double collar_amax = 1.4;
    int collar_L = 13;
    int collar_R = 14;
    int shoulder_L = 16;
    int shoulder_R = 17;

    bool use_wrist_limits = true;
    double w_wrist_limit = 80.0;
    double wrist_amax = 0.6;

    // pelvis-over-ankles
    bool use_pelvis_over_ankles = false;
    double w_pelvis_over_ankles = 10.0;

    // ------------- Arm-torso separation -----------
    bool use_arm_torso_sep = false;
    double arm_torso_d_elbow = 0.08;
    double arm_torso_d_wrist = 0.10;
    double arm_torso_w_elbow = 80.0;
    double arm_torso_w_wrist = 120.0;
    int arm_torso_seg_a = 0;
    int arm_torso_seg_b = 9;

    // ---------- 3D unit test ----------
    bool amass_unit_test = false;
    bool unit_test_exact = true;
    double w_3d = 1.0;

    // ---------- temporal smoothness (tracking) ----------
    bool use_temporal = false;
    double w_temp_pose  = 10.0;
    double w_temp_trans = 2.0;
    bool use_temporal_acc = false;
    double w_temp_acc_pose = 1.5;
    bool temporal_body_only = true;

    // independent root-orientation temporal weight (stronger than w_temp_pose)
    double w_temp_root_orient = 0.0;

    // root flip clamp threshold (degrees)
    double root_jump_max_deg = 0.0;

    // ---------- extremity locks ----------
    bool lock_wrists = true;
    bool lock_hands  = true;
    bool lock_feet   = true;

    // lock neck + head
    bool lock_neck_head = false;

    // SMPL joint indices
    int hand_L = 22;
    int hand_R = 23;
    int foot_L = 10;
    int foot_R = 11;
    int wrist_L = 20;
    int wrist_R = 21;

    // neck/head joint ids (SMPL)
    int neck_j = 12;
    int head_j = 15;

    // ---------- foot contact / flatness ----------
    bool use_foot_ground = false;
    double ground_y = 0.0;
    double foot_contact_L = 0.0;
    double foot_contact_R = 0.0;
    double w_foot_ground = 100.0;
    double w_foot_flat   = 50.0;

    // ---------- per-frame evaluation print ----------
    bool print_eval_per_frame = false;

    // ---------- Joint importance (per-joint weighting) ----------
    bool use_joint_importance = false;

    // ---------- Capsule collision / separation ----------
    bool use_capsule_collision = false;
    double w_capsule_collision = 200.0;
    bool use_arm_leg_sep = false;
    bool use_leg_leg_sep = false;

    // ---------- Cross-limb depth disambiguation ----------
    bool use_cross_limb_depth = false;
    double w_cross_limb_depth = 40.0;
    const std::vector<RelativeDepthPair>* cross_limb_pairs = nullptr;

    // ---------- Translation Z prior ----------
    bool use_trans_z_prior = false;
    double w_trans_z_prior = 80.0;
    double trans_z_prior_target = 3.0;   // target Z value for the prior

    // ---------- Neck / Head soft limits ----------
    bool use_neck_head_limits = false;
    double w_neck_limit = 40.0;
    double neck_amax = 0.5;
    double w_head_limit = 50.0;
    double head_amax = 0.4;

    // ---------- Temporal arm scale ----------
    double temporal_arm_scale = 0.2;

    // ---------- Graduated energy schedule ----------
    // Scale factors for depth/prior weights per stage (1.0 = full weight)
    // Stage 1: mostly reproj, light depth → avoids fighting bad Z init
    // Stage 2: balanced → depth starts pulling
    // Stage 3: full weights → final 3D refinement
    bool use_graduated_energy = false;
    double grad_stage1_depth_scale = 0.1;   // scale for w_depth_affine, w_depth, w_relative_depth
    double grad_stage1_prior_scale = 0.2;   // scale for w_trans_z_prior, w_cross_limb_depth
    double grad_stage2_depth_scale = 0.5;
    double grad_stage2_prior_scale = 0.5;
    // stage 3 = 1.0 (full)

    // ---------- Multi-hypothesis optimization ----------
    bool use_multi_hypothesis = false;
    int multi_hyp_quick_iter_s1 = 15;
    int multi_hyp_quick_iter_s2 = 40;
    double multi_hyp_z_perturb = 0.3;       // Z perturbation amount (meters)
    double multi_hyp_max_z_shift = 0.5;     // max depth-guided Z correction

    // Combined score weights for hypothesis selection
    double multi_hyp_w_reproj  = 1.0;
    double multi_hyp_w_depth   = 30.0;
    double multi_hyp_w_temporal = 5.0;

    // ---------- speed / solver ----------
    int threads = -1;
    bool print_progress = false;
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
// Regularizers / constraints
// ============================================================
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
        const T* aa = theta;
        const T up[3] = {T(0), T(1), T(0)};
        T up_rot[3];
        ceres::AngleAxisRotatePoint(aa, up, up_rot);
        residuals[0] = T(w) * up_rot[0];
        residuals[1] = T(w) * up_rot[2];
        return true;
    }
    double w;
};

struct TransZPriorResidual {
  TransZPriorResidual(double z0_, double w_) : z0(z0_), w(w_) {}
  template<typename T> bool operator()(const T* const trans, T* r) const {
    r[0] = T(w) * (trans[2] - T(z0));
    return true;
  }
  double z0, w;
};

struct PelvisOverAnklesResidual {
  PelvisOverAnklesResidual(int pelvis_, int lank_, int rank_, double w_, const SmplModel& m)
    : p(pelvis_), la(lank_), ra(rank_), w(w_), model(m) {}
  template<typename T>
  bool operator()(const T* const beta, const T* const theta, const T* const trans, T* r) const {
    Eigen::Matrix<T,Eigen::Dynamic,3,Eigen::RowMajor> J;
    SMPLJointsForward<T>(beta, theta, trans, model, J);

    const T midx = (J(la,0) + J(ra,0)) * T(0.5);
    const T midz = (J(la,2) + J(ra,2)) * T(0.5);

    r[0] = T(w) * (J(p,0) - midx);
    r[1] = T(w) * (J(p,2) - midz);
    return true;
  }
  int p, la, ra; double w; const SmplModel& model;
};

struct FootGroundFlatResidual {
    FootGroundFlatResidual(int ankle_j_, int foot_j_,
                           double ground_y_,
                           double w_ground_, double w_flat_,
                           double contact_w_,
                           const SmplModel& model_)
        : ankle_j(ankle_j_), foot_j(foot_j_),
          ground_y(ground_y_),
          w_ground(w_ground_), w_flat(w_flat_),
          contact_w(contact_w_),
          model(model_) {}

    template <typename T>
    bool operator()(const T* const beta,
                    const T* const theta,
                    const T* const trans,
                    T* r) const
    {
        const T cw = T(contact_w);
        if (cw < T(1e-6)) { r[0]=T(0); r[1]=T(0); return true; }

        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> J;
        SMPLJointsForward<T>(beta, theta, trans, model, J);

        if (ankle_j < 0 || ankle_j >= J.rows() || foot_j < 0 || foot_j >= J.rows()) {
            r[0]=T(0); r[1]=T(0); return true;
        }

        const T yA = J(ankle_j, 1);
        const T yF = J(foot_j,  1);

        r[0] = T(w_ground) * cw * (yA - T(ground_y));
        r[1] = T(w_flat)   * cw * (yF - yA);
        return true;
    }

    int ankle_j, foot_j;
    double ground_y;
    double w_ground, w_flat;
    double contact_w;
    const SmplModel& model;
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
    TemporalPoseResidual(const double* theta_prev_, double w_, bool body_only_,
                         double arm_scale_ = 1.0)
        : prev(theta_prev_), w(w_), body_only(body_only_), arm_scale(arm_scale_) {}

    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        auto is_arm = [](int i) -> bool {
            int j = i / 3;
            return j == 13 || j == 14 || j == 16 || j == 17 ||
                   j == 18 || j == 19 || j == 20 || j == 21 ||
                   j == 22 || j == 23;
        };
        int k = 0;
        int start = body_only ? 3 : 0;
        for (int i = start; i < 72; ++i) {
            double s = is_arm(i) ? arm_scale : 1.0;
            residuals[k++] = T(w * s) * (theta[i] - T(prev[i]));
        }
        return true;
    }
    const double* prev;
    double w;
    bool body_only;
    double arm_scale;
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

struct TemporalRootResidual {
    TemporalRootResidual(const double* theta_prev_, double w_)
        : prev(theta_prev_), w(w_) {}

    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        residuals[0] = T(w) * (theta[0] - T(prev[0]));
        residuals[1] = T(w) * (theta[1] - T(prev[1]));
        residuals[2] = T(w) * (theta[2] - T(prev[2]));
        return true;
    }
    const double* prev;
    double w;
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
            if (std::abs(rel_diff) < eps) { residuals[i] = T(0); continue; }

            const T s = T(rel_diff > 0 ? 1.0 : -1.0);
            const T violation = s * z_diff;
            residuals[i] = T(w) * (violation > T(0) ? violation : T(0));
        }
        return true;
    }

    std::vector<RelativeDepthPair> pairs;
    const SmplModel& model;
    double w;
};

// ------------------------------
// Depth affine residual
// z_pred(j) ~= bias + scale * rel_depth(j)
// scale allowed negative
// ------------------------------
struct DepthAffineResidual {
    DepthAffineResidual(const std::vector<JointRelDepthData>& obs_,
                        const SmplModel& model_,
                        double w_)
        : obs(obs_), model(model_), w(w_) {}

    int NumPoints() const { return static_cast<int>(obs.size()); }

    template <typename T>
    bool operator()(const T* const beta,
                    const T* const theta,
                    const T* const trans,
                    const T* const affine, // [scale, bias]
                    T* residuals) const
    {
        const T scale = affine[0];
        const T bias  = affine[1];

        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> J_pred;
        SMPLJointsForward<T>(beta, theta, trans, model, J_pred);

        for (int i = 0; i < NumPoints(); ++i) {
            const int j = obs[i].smpl_joint_idx;
            const T z_pred = J_pred(j, 2);
            const T z_tgt  = bias + scale * T(obs[i].rel_depth);

            const T wi = ceres::sqrt(T(std::max(1e-6, obs[i].weight)));
            residuals[i] = T(w) * wi * (z_pred - z_tgt);
        }
        return true;
    }

    std::vector<JointRelDepthData> obs;
    const SmplModel& model;
    double w;
};

// ------------------------------
// Temporal prior for affine depth params
// ------------------------------
struct TemporalDepthAffineResidual {
    TemporalDepthAffineResidual(const double* prev_, double w_s_, double w_b_)
        : prev(prev_), w_s(w_s_), w_b(w_b_) {}

    template <typename T>
    bool operator()(const T* const affine, T* r) const {
        r[0] = T(w_s) * (affine[0] - T(prev[0])); // scale
        r[1] = T(w_b) * (affine[1] - T(prev[1])); // bias
        return true;
    }

    const double* prev;
    double w_s, w_b;
};

// ------------------------------
// Capsule Collision Residual
// ------------------------------
struct CapsuleCollisionResidual {
    int jA1, jA2, jB1, jB2;
    double rA, rB, w;
    const SmplModel& model;

    CapsuleCollisionResidual(int a1, int a2, double ra,
                             int b1, int b2, double rb,
                             double w_, const SmplModel& m)
        : jA1(a1), jA2(a2), rA(ra), jB1(b1), jB2(b2), rB(rb), w(w_), model(m) {}

    template <typename T>
    bool operator()(const T* const beta, const T* const theta,
                    const T* const trans, T* residuals) const {
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> J;
        SMPLJointsForward(beta, theta, trans, model, J);

        Eigen::Matrix<T,3,1> a0 = J.row(jA1).transpose();
        Eigen::Matrix<T,3,1> a1 = J.row(jA2).transpose();
        Eigen::Matrix<T,3,1> b0 = J.row(jB1).transpose();
        Eigen::Matrix<T,3,1> b1 = J.row(jB2).transpose();

        Eigen::Matrix<T,3,1> d  = a0 - b0;
        Eigen::Matrix<T,3,1> u  = a1 - a0;
        Eigen::Matrix<T,3,1> v  = b1 - b0;

        T uu = u.dot(u);
        T vv = v.dot(v);
        T uv = u.dot(v);
        T ud = u.dot(d);
        T vd = v.dot(d);

        T denom = uu * vv - uv * uv;
        T eps = T(1e-8);

        T sN = T(0), tN = T(0);
        T sD = denom, tD = denom;

        if (denom < eps) {
            sN = T(0); sD = T(1);
            tN = vd;   tD = vv;
        } else {
            sN = uv * vd - vv * ud;
            tN = uu * vd - uv * ud;
        }

        if (sN < T(0))      { sN = T(0); tN = vd;           tD = vv; }
        else if (sN > sD)   { sN = sD;   tN = vd + uv;      tD = vv; }

        if (tN < T(0)) {
            tN = T(0);
            if (-ud < T(0))      sN = T(0);
            else if (-ud > uu)   sN = sD;
            else                 { sN = -ud; sD = uu; }
        } else if (tN > tD) {
            tN = tD;
            if ((-ud + uv) < T(0))    sN = T(0);
            else if ((-ud + uv) > uu)  sN = sD;
            else                       { sN = -ud + uv; sD = uu; }
        }

        T s = (abs(sN) < eps) ? T(0) : sN / sD;
        T t = (abs(tN) < eps) ? T(0) : tN / tD;

        Eigen::Matrix<T,3,1> closest = d + s * u - t * v;
        T dist = closest.norm() + eps;

        T sum_r = T(rA + rB);
        T penetration = sum_r - dist;

        if (penetration > T(0)) {
            residuals[0] = T(w) * penetration * penetration;
        } else {
            residuals[0] = T(0);
        }
        return true;
    }
};

// ------------------------------
// Cross-Limb Depth Disambiguation
// ------------------------------

struct CrossLimbDepthResidual {
    const std::vector<RelativeDepthPair>& pairs;
    const SmplModel& model;
    double w;
    double depth_sign;  // +1 if larger depth = farther, -1 if disparity-like

    CrossLimbDepthResidual(const std::vector<RelativeDepthPair>& p,
                           const SmplModel& m, double w_, double sign_)
        : pairs(p), model(m), w(w_), depth_sign(sign_) {}

    template <typename T>
    bool operator()(const T* const beta, const T* const theta,
                    const T* const trans, T* residuals) const {
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> J;
        SMPLJointsForward(beta, theta, trans, model, J);

        for (size_t i = 0; i < pairs.size(); ++i) {
            const auto& p = pairs[i];
            T z_L = J(p.smpl_joint_a, 2);
            T z_R = J(p.smpl_joint_b, 2);
            T pred_diff = z_L - z_R;
            T obs_diff = T(depth_sign) * T(p.rel_depth_a - p.rel_depth_b);
            T violation = pred_diff * T(-1.0) * obs_diff;
            if (violation > T(0)) {
                residuals[i] = T(w) * violation;
            } else {
                residuals[i] = T(0);
            }
        }
        return true;
    }
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
                         const double* theta_prev2 = nullptr,
                         const double* trans_prev2 = nullptr);

CameraIntrinsics CreateDefaultCamera(int image_width, int image_height);