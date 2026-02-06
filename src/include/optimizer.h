#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>

#include <vector>
#include <array>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cstring>

#include "smpl_model.h"
#include "smpl_math.h"

using MatX3dRM = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

// ============================================================
// Data structs
// ============================================================
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

// Camera extrinsics: world-to-camera transform
// P_cam = R * P_world + t
//
// *** IMPORTANT DESIGN NOTE ***
// Extrinsics are NEVER passed into the optimizer.
// All optimization happens in CAMERA SPACE.
// Extrinsics are used ONLY for:
//   1) Warm-start: transform prev-frame cam-space params → current cam space
//   2) Temporal prior targets: same transform for prev-frame reference values
//   3) Evaluation: transform cam-space results → world space for MPJPE/PA-MPJPE
struct CameraExtrinsics {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    CameraExtrinsics() : R(Eigen::Matrix3d::Identity()), t(Eigen::Vector3d::Zero()) {}
    CameraExtrinsics(const Eigen::Matrix3d& R_, const Eigen::Vector3d& t_) : R(R_), t(t_) {}
};

struct JointDepthData {
    int smpl_joint_idx;
    double depth_z;
};

struct JointRelDepthData {
    int smpl_joint_idx;
    double rel_depth;
    double weight;
};

struct RelativeDepthPair {
    int smpl_joint_a;
    int smpl_joint_b;
    double rel_depth_a;
    double rel_depth_b;
};

// ============================================================
// GMM Pose Prior
// ============================================================
struct GMMPosePrior {
    static constexpr int K = 8;
    static constexpr int D = 69;

    Eigen::Matrix<double, K, 1> weights;
    Eigen::Matrix<double, K, D, Eigen::RowMajor> means;
    std::array<Eigen::Matrix<double, D, D, Eigen::RowMajor>, K> prec_choleskys;

    bool loaded = false;
    bool LoadFromNPZ(const std::string& npz_path);
};

// ============================================================
// OptimizerConfig — ALL optimization in CAMERA SPACE
// ============================================================
struct OptimizerConfig {
    bool verbose = true;

    // ---------- 2D fitting ----------
    float min_confidence = 0.20f;
    bool use_stable_joints_only = true;
    bool flip_y_in_projection = true;

    // ---------- Depth (absolute) ----------
    bool use_depth = false;
    double w_depth = 15.0;
    const std::vector<JointDepthData>* depth_data = nullptr;

    // ---------- Relative depth ----------
    bool use_relative_depth = false;
    double w_relative_depth = 20.0;
    const std::vector<RelativeDepthPair>* relative_depth_pairs = nullptr;

    // ---------- Affine depth  ----------
    bool use_depth_affine = false;
    double w_depth_affine = 30.0;
    double depth_affine_sign = 1.0;
    const std::vector<JointRelDepthData>* depth_affine_data = nullptr;
    double* depth_affine_params = nullptr;

    bool use_temporal_depth_affine = false;
    double w_temp_depth_scale = 80.0;
    double w_temp_depth_bias  = 200.0;
    const double* depth_affine_prev = nullptr;
    bool fix_depth_affine_bias = false;

    // ---------- Camera-space Z prior ----------
    // Soft prior on trans[2] to prevent Z-drift.
    // In camera space this is trivial and highly effective.
    bool use_trans_z_prior = false;
    double trans_z_prior_target = 3.0;
    double w_trans_z_prior = 100.0;

    // ---------- Root upright ----------
    bool use_root_upright = true;
    double root_upright_w_stage2 = 1.0;
    double root_upright_w_stage3 = 0.0;

    // ---------- GMM pose prior ----------
    bool use_gmm_prior = false;
    double w_gmm = 1.0;
    const GMMPosePrior* gmm_prior = nullptr;

    // ---------- Hinge priors ----------
    bool use_hinge_priors = true;
    double w_hinge_perp_knee  = 120.0;
    double w_hinge_range_knee = 60.0;
    double w_hinge_perp_elbow  = 80.0;
    double w_hinge_range_elbow = 40.0;
    double hinge_min = -2.6;
    double hinge_max =  2.6;

    bool use_bend_limits = true;
    double w_bend = 1.0;

    // ---------- Spine ----------
    bool use_spine_limits = true;
    double w_spine = 10.0;
    double spine_amax = 1.2;
    int spine_j1 = 3, spine_j2 = 6, spine_j3 = 9;

    // ---------- Neck / Head ----------
    bool use_neck_head_limits = true;
    double w_neck_limit = 40.0;
    double neck_amax = 0.6;
    double w_head_limit = 40.0;
    double head_amax = 0.5;
    int neck_j = 12, head_j = 15;

    // ---------- Shoulder / Collar ----------
    bool use_shoulder_collar_limits = true;
    double w_shoulder_limit = 60.0;
    double shoulder_amax = 1.7;
    double w_collar_limit = 40.0;
    double collar_amax = 0.8;
    int collar_L = 13, collar_R = 14;
    int shoulder_L = 16, shoulder_R = 17;

    // ---------- Wrist ----------
    bool use_wrist_limits = true;
    double w_wrist_limit = 80.0;
    double wrist_amax = 0.6;
    int wrist_L = 20, wrist_R = 21;

    // ---------- Pelvis over ankles ----------
    bool use_pelvis_over_ankles = true;
    double w_pelvis_over_ankles = 10.0;

    // ---------- Arm-torso separation ----------
    bool use_arm_torso_sep = true;
    double arm_torso_d_elbow = 0.08;
    double arm_torso_d_wrist = 0.10;
    double arm_torso_w_elbow = 80.0;
    double arm_torso_w_wrist = 120.0;
    int arm_torso_seg_a = 3, arm_torso_seg_b = 12;

    // ---------- 3D unit test ----------
    bool amass_unit_test = false;
    bool unit_test_exact = true;
    double w_3d = 1.0;

    // ---------- Temporal (camera-space) ----------
    bool use_temporal = false;
    double w_temp_pose  = 10.0;
    double w_temp_trans = 2.0;
    bool use_temporal_acc = false;
    double w_temp_acc_pose = 2.0;
    bool temporal_body_only = true;

    // Per-joint temporal scaling: reduce for arms so they can move faster.
    // Applies to collar(13,14), shoulder(16,17), elbow(18,19), wrist(20,21).
    double temporal_arm_scale = 1.0;    // set to 0.2~0.4 for arm freedom

    // ---------- Extremity locks ----------
    bool lock_wrists = false;   // NOTE: was true, unlock for arm articulation
    bool lock_hands  = true;
    bool lock_feet   = true;
    int hand_L = 22, hand_R = 23;
    int foot_L = 10, foot_R = 11;

    // ---------- Foot ground ----------
    bool use_foot_ground = false;
    double ground_y = 0.0;
    double foot_contact_L = 0.0;
    double foot_contact_R = 0.0;
    double w_foot_ground = 400.0;
    double w_foot_flat   = 200.0;

    // ---------- Solver ----------
    int threads = -1;
    bool print_progress = false;
    int max_iter_stage1 = 60;
    int max_iter_stage2 = 200;
    int max_iter_stage3 = 200;
};

// ============================================================
// OpenPose -> SMPL mapping
// ============================================================
extern const int OPENPOSE_TO_SMPL[25];

// ============================================================
// Camera-space transform helpers for moving cameras
// ============================================================

// Relative camera transform from frame (t-1) to frame (t).
// P_cam_t = R_rel * P_cam_{t-1} + t_rel
inline void ComputeRelativeCameraTransform(
    const CameraExtrinsics& ext_prev,
    const CameraExtrinsics& ext_curr,
    Eigen::Matrix3d& R_rel,
    Eigen::Vector3d& t_rel)
{
    R_rel = ext_curr.R * ext_prev.R.transpose();
    t_rel = ext_curr.t - R_rel * ext_prev.t;
}

// Transform previous frame's camera-space SMPL params to current camera space.
// Body pose theta[3:72] is unchanged (relative rotations are frame-invariant).
// Root orientation theta[0:3] and translation are transformed.
inline void TransformPrevParamsToCurrCam(
    const double* theta_prev_cam,
    const double* trans_prev_cam,
    const Eigen::Matrix3d& R_rel,
    const Eigen::Vector3d& t_rel,
    double* theta_out,
    double* trans_out)
{
    // Copy all pose params (body pose stays the same)
    std::memcpy(theta_out, theta_prev_cam, 72 * sizeof(double));

    // Transform root orientation: R_new = R_rel * R_old
    {
        double aa[3] = {theta_prev_cam[0], theta_prev_cam[1], theta_prev_cam[2]};
        double R_arr[9];
        ceres::AngleAxisToRotationMatrix(aa, R_arr);

        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> R_old(R_arr);
        Eigen::Matrix3d R_new = R_rel * R_old;

        double R_new_arr[9];
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> R_new_map(R_new_arr);
        R_new_map = R_new;

        double aa_new[3];
        ceres::RotationMatrixToAngleAxis(R_new_arr, aa_new);
        theta_out[0] = aa_new[0];
        theta_out[1] = aa_new[1];
        theta_out[2] = aa_new[2];
    }

    // Transform translation: t_curr = R_rel * t_prev + t_rel
    Eigen::Vector3d tp(trans_prev_cam[0], trans_prev_cam[1], trans_prev_cam[2]);
    Eigen::Vector3d tc = R_rel * tp + t_rel;
    trans_out[0] = tc(0);
    trans_out[1] = tc(1);
    trans_out[2] = tc(2);
}

// Transform camera-space joints to world space for evaluation
inline void TransformJointsCamToWorld(
    const MatX3dRM& J_cam,
    const CameraExtrinsics& ext,
    MatX3dRM& J_world)
{
    const int N = J_cam.rows();
    J_world.resize(N, 3);
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d p_cam(J_cam(i,0), J_cam(i,1), J_cam(i,2));
        Eigen::Vector3d p_world = ext.R.transpose() * (p_cam - ext.t);
        J_world(i,0) = p_world(0);
        J_world(i,1) = p_world(1);
        J_world(i,2) = p_world(2);
    }
}

// Transform camera-space vertices to world space
inline void TransformVerticesCamToWorld(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V_cam,
    const CameraExtrinsics& ext,
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V_world)
{
    const int N = V_cam.rows();
    V_world.resize(N, 3);
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d p_cam(V_cam(i,0), V_cam(i,1), V_cam(i,2));
        Eigen::Vector3d p_world = ext.R.transpose() * (p_cam - ext.t);
        V_world(i,0) = p_world(0);
        V_world(i,1) = p_world(1);
        V_world(i,2) = p_world(2);
    }
}

// ============================================================
// SMPLJointsFromMeshForward
// ============================================================
template <typename T>
inline void SMPLJointsFromMeshForward(
    const T* beta, const T* theta, const T* trans,
    const SmplModel& model,
    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>& out_joints,
    bool use_pose_blendshapes)
{
    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> V;
    SMPLForward<T>(beta, theta, trans, model, V, use_pose_blendshapes);
    const int J = model.J;
    out_joints.resize(J, 3);
    out_joints.setZero();
    for (int j = 0; j < J; ++j) {
        for (const auto& it : model.J_regressor_nz[j]) {
            const T w = T(it.second);
            out_joints(j,0) += w * V(it.first,0);
            out_joints(j,1) += w * V(it.first,1);
            out_joints(j,2) += w * V(it.first,2);
        }
    }
}

// ============================================================
// Residual structs
// ============================================================

struct ShapeRegularizer {
    ShapeRegularizer(int index, double lambda) : index_(index), lambda_(lambda) {}
    template <typename T>
    bool operator()(const T* beta, T* residual) const {
        residual[0] = T(lambda_) * beta[index_];
        return true;
    }
    int index_; double lambda_;
};

struct PoseRegularizer {
    PoseRegularizer(int index, double lambda) : index_(index), lambda_(lambda) {}
    template <typename T>
    bool operator()(const T* theta, T* residual) const {
        residual[0] = T(lambda_) * theta[index_];
        return true;
    }
    int index_; double lambda_;
};

struct GMMPosePriorResidual {
    GMMPosePriorResidual(const GMMPosePrior& gmm, double weight)
        : gmm_(gmm), weight_(weight) {}
    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        constexpr int D = 69, K = 8;
        T min_dist = T(1e10);
        int best_k = 0;
        for (int k = 0; k < K; ++k) {
            T dist = T(0);
            for (int i = 0; i < D; ++i) {
                T weighted = T(0);
                for (int j = 0; j <= i; ++j)
                    weighted += T(gmm_.prec_choleskys[k](i,j)) * (theta[3+j] - T(gmm_.means(k,j)));
                dist += weighted * weighted;
            }
            T score = dist - T(2.0) * T(std::log(gmm_.weights(k) + 1e-10));
            if (score < min_dist) { min_dist = score; best_k = k; }
        }
        for (int i = 0; i < D; ++i) {
            T weighted = T(0);
            for (int j = 0; j <= i; ++j)
                weighted += T(gmm_.prec_choleskys[best_k](i,j)) * (theta[3+j] - T(gmm_.means(best_k,j)));
            residuals[i] = T(weight_) * weighted;
        }
        return true;
    }
    const GMMPosePrior& gmm_; double weight_;
};

struct RootUprightResidual {
    explicit RootUprightResidual(double w_) : w(w_) {}
    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        const T up[3] = {T(0), T(1), T(0)};
        T up_rot[3];
        ceres::AngleAxisRotatePoint(theta, up, up_rot);
        residuals[0] = T(w) * up_rot[0];
        residuals[1] = T(w) * up_rot[2];
        return true;
    }
    double w;
};

struct PointToSegmentDistanceFloorResidual {
    PointToSegmentDistanceFloorResidual(int point_joint, int seg_a, int seg_b,
                                        double d_min, double w, const SmplModel& m)
        : p(point_joint), a(seg_a), b(seg_b), dmin(d_min), w(w), model(m) {}
    template <typename T>
    bool operator()(const T* const beta, const T* const theta,
                    const T* const trans, T* residual) const {
        Eigen::Matrix<T,Eigen::Dynamic,3,Eigen::RowMajor> J;
        SMPLJointsForward<T>(beta, theta, trans, model, J);
        const Eigen::Matrix<T,3,1> P = J.row(p).transpose();
        const Eigen::Matrix<T,3,1> A = J.row(a).transpose();
        const Eigen::Matrix<T,3,1> B = J.row(b).transpose();
        const Eigen::Matrix<T,3,1> AB = B - A;
        const T ab2 = AB.squaredNorm() + T(1e-12);
        T t = (P-A).dot(AB) / ab2;
        t = ceres::fmin(T(1), ceres::fmax(T(0), t));
        const T dist = ceres::sqrt((P - A - t*AB).squaredNorm() + T(1e-12));
        const T x = T(dmin) - dist;
        residual[0] = T(w) * (ceres::abs(x) + x) * T(0.5);
        return true;
    }
    int p, a, b; double dmin, w; const SmplModel& model;
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
    bool operator()(const T* const beta, const T* const theta,
                    const T* const trans, T* r) const {
        Eigen::Matrix<T,Eigen::Dynamic,3,Eigen::RowMajor> J;
        SMPLJointsForward<T>(beta, theta, trans, model, J);
        r[0] = T(w) * (J(p,0) - (J(la,0)+J(ra,0))*T(0.5));
        r[1] = T(w) * (J(p,2) - (J(la,2)+J(ra,2))*T(0.5));
        return true;
    }
    int p, la, ra; double w; const SmplModel& model;
};

struct FootGroundFlatResidual {
    FootGroundFlatResidual(int ankle_j_, int foot_j_, double ground_y_,
                           double w_ground_, double w_flat_, double contact_w_,
                           const SmplModel& m)
        : ankle_j(ankle_j_), foot_j(foot_j_), ground_y(ground_y_),
          w_ground(w_ground_), w_flat(w_flat_), contact_w(contact_w_), model(m) {}
    template <typename T>
    bool operator()(const T* const beta, const T* const theta,
                    const T* const trans, T* r) const {
        const T cw = T(contact_w);
        if (cw < T(1e-6)) { r[0]=T(0); r[1]=T(0); return true; }
        Eigen::Matrix<T,Eigen::Dynamic,3,Eigen::RowMajor> J;
        SMPLJointsForward<T>(beta, theta, trans, model, J);
        if (ankle_j<0||ankle_j>=J.rows()||foot_j<0||foot_j>=J.rows()) {
            r[0]=T(0); r[1]=T(0); return true;
        }
        r[0] = T(w_ground) * cw * (J(ankle_j,1) - T(ground_y));
        r[1] = T(w_flat)   * cw * (J(foot_j,1)  - J(ankle_j,1));
        return true;
    }
    int ankle_j, foot_j; double ground_y, w_ground, w_flat, contact_w;
    const SmplModel& model;
};

struct HingeAxisResidual {
    HingeAxisResidual(int joint, Eigen::Vector3d axis, double w_perp, double w_range,
                      double min_a, double max_a)
        : j_(joint), u_(axis.normalized()), w_perp_(w_perp), w_range_(w_range),
          min_a_(min_a), max_a_(max_a) {}
    template <typename T>
    bool operator()(const T* const theta, T* residual) const {
        const int base = 3*j_;
        Eigen::Matrix<T,3,1> r(theta[base], theta[base+1], theta[base+2]);
        Eigen::Matrix<T,3,1> u(T(u_.x()), T(u_.y()), T(u_.z()));
        const T a = u.dot(r);
        residual[0] = T(w_perp_) * ceres::sqrt((r - u*a).squaredNorm() + T(1e-12));
        const T low  = T(min_a_)-a, high = a-T(max_a_);
        residual[1] = T(w_range_) * ((ceres::abs(low)+low + ceres::abs(high)+high) * T(0.5));
        return true;
    }
    int j_; Eigen::Vector3d u_; double w_perp_, w_range_, min_a_, max_a_;
};

struct SoftAngleLimitResidual {
    SoftAngleLimitResidual(int joint, double amax, double w)
        : j_(joint), amax_(amax), w_(w) {}
    template <typename T>
    bool operator()(const T* const theta, T* residual) const {
        const int b = 3*j_;
        const T ang = ceres::sqrt(theta[b]*theta[b]+theta[b+1]*theta[b+1]+theta[b+2]*theta[b+2]+T(1e-12));
        const T over = ang - T(amax_);
        residual[0] = T(w_) * (ceres::abs(over)+over) * T(0.5);
        return true;
    }
    int j_; double amax_, w_;
};

struct BendLimitResidual {
    BendLimitResidual(int joint, int axis, double sign, double weight)
        : joint_(joint), axis_(axis), sign_(sign), w_(weight) {}
    template <typename T>
    bool operator()(const T* const theta, T* residual) const {
        residual[0] = T(w_) * ceres::exp(T(sign_) * theta[3*joint_+axis_]);
        return true;
    }
    int joint_, axis_; double sign_, w_;
};

struct TemporalPoseResidual {
    TemporalPoseResidual(const double* prev_, double w_, bool body_only_)
        : prev(prev_), w(w_), body_only(body_only_) {}
    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        int k=0; for (int i=(body_only?3:0); i<72; ++i) residuals[k++] = T(w)*(theta[i]-T(prev[i]));
        return true;
    }
    const double* prev; double w; bool body_only;
};

// Weighted temporal: per-joint weight array for differentiated smoothing
// (lower weight on arms → faster response to 2D evidence)
struct TemporalPoseWeightedResidual {
    TemporalPoseWeightedResidual(const double* prev_, const double* jw24_, bool body_only_)
        : prev(prev_), body_only(body_only_) {
        for (int j = 0; j < 24; ++j) jw[j] = jw24_[j];
    }
    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        int k = 0;
        int start_j = body_only ? 1 : 0;
        for (int j = start_j; j < 24; ++j) {
            T w = T(jw[j]);
            for (int a = 0; a < 3; ++a) {
                int i = 3*j + a;
                residuals[k++] = w * (theta[i] - T(prev[i]));
            }
        }
        return true;
    }
    const double* prev; double jw[24]; bool body_only;
};

struct TemporalTransResidual {
    TemporalTransResidual(const double* prev_, double w_) : prev(prev_), w(w_) {}
    template <typename T>
    bool operator()(const T* const trans, T* r) const {
        for (int i=0;i<3;++i) r[i]=T(w)*(trans[i]-T(prev[i]));
        return true;
    }
    const double* prev; double w;
};

struct TransConstVelPriorResidual {
    TransConstVelPriorResidual(const double* prev_, const double* prevprev_, double w_)
        : t1(prev_), t2(prevprev_), w(w_) {}
    template <typename T>
    bool operator()(const T* const trans, T* r) const {
        for (int i=0;i<3;++i) r[i]=T(w)*(trans[i] - (T(2)*T(t1[i])-T(t2[i])));
        return true;
    }
    const double* t1; const double* t2; double w;
};

struct TemporalRootResidual {
    TemporalRootResidual(const double* prev_, double w_) : prev(prev_), w(w_) {}
    template <typename T>
    bool operator()(const T* const theta, T* r) const {
        for (int i=0;i<3;++i) r[i]=T(w)*(theta[i]-T(prev[i]));
        return true;
    }
    const double* prev; double w;
};

struct TemporalAccelPoseResidual {
    TemporalAccelPoseResidual(const double* p1_, const double* p2_, double w_, bool body_only_)
        : p1(p1_), p2(p2_), w(w_), body_only(body_only_) {}
    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        int k=0; for (int i=(body_only?3:0); i<72; ++i)
            residuals[k++] = T(w)*(theta[i]-T(2)*T(p1[i])+T(p2[i]));
        return true;
    }
    const double* p1; const double* p2; double w; bool body_only;
};

struct TemporalAccelPoseWeightedResidual {
    TemporalAccelPoseWeightedResidual(const double* p1_, const double* p2_,
                                      const double* jw24_, bool body_only_)
        : p1(p1_), p2(p2_), body_only(body_only_) {
        for (int j = 0; j < 24; ++j) jw[j] = jw24_[j];
    }
    template <typename T>
    bool operator()(const T* const theta, T* residuals) const {
        int k = 0;
        int start_j = body_only ? 1 : 0;
        for (int j = start_j; j < 24; ++j) {
            T w = T(jw[j]);
            for (int a = 0; a < 3; ++a) {
                int i = 3*j + a;
                residuals[k++] = w * (theta[i] - T(2)*T(p1[i]) + T(p2[i]));
            }
        }
        return true;
    }
    const double* p1; const double* p2; double jw[24]; bool body_only;
};

struct TemporalDepthAffineResidual {
    TemporalDepthAffineResidual(const double* prev_, double ws_, double wb_)
        : prev(prev_), w_s(ws_), w_b(wb_) {}
    template <typename T>
    bool operator()(const T* const affine, T* r) const {
        r[0]=T(w_s)*(affine[0]-T(prev[0]));
        r[1]=T(w_b)*(affine[1]-T(prev[1]));
        return true;
    }
    const double* prev; double w_s, w_b;
};

// 2D reprojection — CAMERA SPACE ONLY, no extrinsics
struct AllKeypointsReprojectionResidual {
    AllKeypointsReprojectionResidual(
        const std::vector<Landmark2D>& observed,
        const CameraIntrinsics& cam,
        const SmplModel& model,
        float min_confidence,
        bool use_stable_only,
        bool flip_y)
    : camera_(cam), model_(model), flip_y_(flip_y)
    {
        std::array<bool,25> allowed{};
        allowed.fill(true);
        if (use_stable_only) {
            allowed.fill(false);
            for (int id : {1,2,3,4,5,6,7,8,9,10,11,12,13,14}) allowed[id]=true;
        }
        for (const auto& lm : observed) {
            if (lm.confidence < min_confidence) continue;
            if (lm.joint_id < 0 || lm.joint_id >= 25) continue;
            if (!allowed[lm.joint_id]) continue;
            int sid = OPENPOSE_TO_SMPL[lm.joint_id];
            if (sid < 0 || sid >= model_.J) continue;
            lms_.push_back(lm);
            smpl_ids_.push_back(sid);
        }
    }

    int NumUsedKeypoints() const { return (int)lms_.size(); }

    template <typename T>
    bool operator()(const T* const beta, const T* const theta,
                    const T* const trans, T* residuals) const
    {
        Eigen::Matrix<T,Eigen::Dynamic,3,Eigen::RowMajor> J;
        SMPLJointsForward<T>(beta, theta, trans, model_, J);

        for (int i=0; i<(int)lms_.size(); ++i) {
            T X=J(smpl_ids_[i],0), Y=J(smpl_ids_[i],1), Z=J(smpl_ids_[i],2);
            if (Z < T(1e-6)) Z = T(1e-6);
            const T u = T(camera_.fx)*(X/Z) + T(camera_.cx);
            const T v = T(camera_.fy)*((flip_y_?-Y:Y)/Z) + T(camera_.cy);
            const T wc = ceres::sqrt(T(std::max(1e-6f, lms_[i].confidence)));
            residuals[2*i]   = wc*(u-T(lms_[i].x));
            residuals[2*i+1] = wc*(v-T(lms_[i].y));
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

struct Joints3DResidualFast {
    Joints3DResidualFast(const MatX3dRM& J_obs_, const SmplModel& m, double w_)
        : J_obs(J_obs_), model(m), w(w_) {}
    template <typename T>
    bool operator()(const T* const beta, const T* const theta,
                    const T* const trans, T* residuals) const {
        Eigen::Matrix<T,Eigen::Dynamic,3,Eigen::RowMajor> J;
        SMPLJointsForward<T>(beta, theta, trans, model, J);
        for (int i=0;i<24;++i) for (int c=0;c<3;++c)
            residuals[3*i+c]=T(w)*(J(i,c)-T(J_obs(i,c)));
        return true;
    }
    MatX3dRM J_obs; const SmplModel& model; double w;
};

struct DepthResidual {
    DepthResidual(const std::vector<JointDepthData>& d, const SmplModel& m, double w_)
        : data(d), model(m), w(w_) {}
    int NumDepthPoints() const { return (int)data.size(); }
    template <typename T>
    bool operator()(const T* const beta, const T* const theta,
                    const T* const trans, T* residuals) const {
        Eigen::Matrix<T,Eigen::Dynamic,3,Eigen::RowMajor> J;
        SMPLJointsForward<T>(beta, theta, trans, model, J);
        for (int i=0;i<(int)data.size();++i)
            residuals[i]=T(w)*(J(data[i].smpl_joint_idx,2)-T(data[i].depth_z));
        return true;
    }
    std::vector<JointDepthData> data; const SmplModel& model; double w;
};

struct RelativeDepthResidual {
    RelativeDepthResidual(const std::vector<RelativeDepthPair>& p, const SmplModel& m, double w_)
        : pairs(p), model(m), w(w_) {}
    int NumPairs() const { return (int)pairs.size(); }
    template <typename T>
    bool operator()(const T* const beta, const T* const theta,
                    const T* const trans, T* residuals) const {
        Eigen::Matrix<T,Eigen::Dynamic,3,Eigen::RowMajor> J;
        SMPLJointsForward<T>(beta, theta, trans, model, J);
        for (int i=0;i<(int)pairs.size();++i) {
            const T zd = J(pairs[i].smpl_joint_a,2)-J(pairs[i].smpl_joint_b,2);
            double rd = pairs[i].rel_depth_a-pairs[i].rel_depth_b;
            if (std::abs(rd)<1e-3) { residuals[i]=T(0); continue; }
            const T s = T(rd>0?1.0:-1.0);
            const T v = s*zd;
            residuals[i]=T(w)*(v>T(0)?v:T(0));
        }
        return true;
    }
    std::vector<RelativeDepthPair> pairs; const SmplModel& model; double w;
};

struct DepthAffineResidual {
    DepthAffineResidual(const std::vector<JointRelDepthData>& obs_,
                        const SmplModel& m, double w_, double sign_)
        : obs(obs_), model(m), w(w_), sign(sign_) {}
    int NumPoints() const { return (int)obs.size(); }
    template <typename T>
    bool operator()(const T* const beta, const T* const theta,
                    const T* const trans, const T* const affine, T* residuals) const {
        Eigen::Matrix<T,Eigen::Dynamic,3,Eigen::RowMajor> J;
        SMPLJointsForward<T>(beta, theta, trans, model, J);
        for (int i=0;i<(int)obs.size();++i) {
            const T z_pred = J(obs[i].smpl_joint_idx, 2);  // camera-space Z = depth
            const T z_tgt = affine[1] + T(sign)*affine[0]*T(obs[i].rel_depth);
            residuals[i] = T(w)*ceres::sqrt(T(std::max(1e-6,obs[i].weight)))*(z_pred-z_tgt);
        }
        return true;
    }
    std::vector<JointRelDepthData> obs; const SmplModel& model; double w, sign;
};

// ============================================================
// API
// ============================================================
bool OptimizeSMPL(const std::vector<Landmark2D>& observed_keypoints,
                  const CameraIntrinsics& camera,
                  const SmplModel& model,
                  double* beta, double* theta, double* trans,
                  const OptimizerConfig& config);

bool OptimizeSMPLTracked(const std::vector<Landmark2D>& observed_keypoints,
                         const CameraIntrinsics& camera,
                         const SmplModel& model,
                         double* beta, double* theta, double* trans,
                         const OptimizerConfig& cfg,
                         const double* theta_prev,
                         const double* trans_prev,
                         const double* theta_prev2 = nullptr,
                         const double* trans_prev2 = nullptr);

bool OptimizeSMPL3D(const MatX3dRM& J_obs, const SmplModel& model,
                    double* beta, double* theta, double* trans,
                    const OptimizerConfig& config);

std::vector<CameraExtrinsics> LoadCameraPoses(const std::string& path);
CameraIntrinsics CreateDefaultCamera(int image_width, int image_height);