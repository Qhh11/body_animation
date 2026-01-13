#pragma once

// Increase Eigen stack allocation limit for Ceres autodiff with large Jet types
#ifndef EIGEN_STACK_ALLOCATION_LIMIT
#define EIGEN_STACK_ALLOCATION_LIMIT 1000000
#endif

#include <ceres/ceres.h>
#include <Eigen/Dense>

#include <vector>
#include <array>
#include <algorithm>
#include <iostream>
#include <map>

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

    // ---------- Depth constraint ----------
    bool use_depth_ordering = true;
    double depth_ordering_weight = 100.0;
    bool higher_depth_is_closer = false;  // Sapiens NPY: smaller depth = closer

    // ---------- 3D unit test ----------
    bool amass_unit_test = false;
    bool unit_test_exact = true;
    double w_3d = 1.0;

    // ---------- Interpenetration (capsule collision) ----------
    bool use_interpenetration = true;
    double interpenetration_weight = 100.0;
    double thigh_capsule_radius = 0.08;  // 大腿半径 (m)
    double calf_capsule_radius = 0.06;   // 小腿半径 (m)
    double foot_capsule_radius = 0.05;   // 脚部半径 (m)
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
// Depth data struct (from Sapiens depth map)
// ------------------------------
struct KeypointDepth {
    int joint_id;        // OpenPose joint id (0-24)
    float depth;         // relative depth value from Sapiens
    float confidence;    // confidence from OpenPose

    KeypointDepth() : joint_id(-1), depth(0.f), confidence(0.f) {}
    KeypointDepth(int id, float d, float c) : joint_id(id), depth(d), confidence(c) {}
};

// ------------------------------
// Symmetric limb depth ordering residual
// Constrains left-right symmetric pairs based on depth
// ------------------------------
struct DepthOrderingResidual {
    DepthOrderingResidual(
        const std::vector<KeypointDepth>& depth_data,
        const SmplModel& model,
        double weight,
        bool higher_depth_is_closer = false  // Our NPY: smaller depth = closer
    )
    : model_(model), weight_(weight), higher_is_closer_(higher_depth_is_closer)
    {
        // Build a map from OpenPose joint_id to depth/conf
        std::map<int, std::pair<float, float>> depth_map;  // joint_id -> (depth, conf)
        for (const auto& kd : depth_data) {
            if (kd.joint_id >= 0 && kd.joint_id < 25 && kd.confidence > 0.3f) {
                depth_map[kd.joint_id] = {kd.depth, kd.confidence};
            }
        }

        // Define symmetric pairs: (right_openpose_id, left_openpose_id)
        const std::vector<std::pair<int, int>> symmetric_pairs = {
            {9, 12},   // Hip
            {10, 13},  // Knee
            {11, 14},  // Ankle
            {2, 5},    // Shoulder
            {3, 6},    // Elbow
            {4, 7},    // Wrist
        };

        for (const auto& [right_op, left_op] : symmetric_pairs) {
            if (depth_map.find(right_op) == depth_map.end()) continue;
            if (depth_map.find(left_op) == depth_map.end()) continue;

            float depth_right = depth_map[right_op].first;
            float depth_left = depth_map[left_op].first;
            float conf = std::min(depth_map[right_op].second, depth_map[left_op].second);

            float depth_diff = std::abs(depth_right - depth_left);
            if (depth_diff < 0.05f) continue;

            int smpl_right = OPENPOSE_TO_SMPL[right_op];
            int smpl_left = OPENPOSE_TO_SMPL[left_op];
            if (smpl_right < 0 || smpl_left < 0) continue;

            int closer_smpl, farther_smpl;
            if (higher_is_closer_) {
                if (depth_right > depth_left) {
                    closer_smpl = smpl_right;
                    farther_smpl = smpl_left;
                } else {
                    closer_smpl = smpl_left;
                    farther_smpl = smpl_right;
                }
            } else {
                if (depth_right < depth_left) {
                    closer_smpl = smpl_right;
                    farther_smpl = smpl_left;
                } else {
                    closer_smpl = smpl_left;
                    farther_smpl = smpl_right;
                }
            }

            pairs_.push_back({closer_smpl, farther_smpl, conf, depth_diff});

            std::cout << "[Depth] Pair: OpenPose (" << right_op << " vs " << left_op << ") -> SMPL ("
                      << closer_smpl << " closer, " << farther_smpl << " farther)"
                      << ", depth_L=" << depth_left << ", depth_R=" << depth_right
                      << ", diff=" << depth_diff << std::endl;
        }
    }

    int NumPairs() const { return static_cast<int>(pairs_.size()); }

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        const T* beta  = parameters[0];
        const T* theta = parameters[1];
        const T* trans = parameters[2];

        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> Jw;
        SMPLJointsForward<T>(beta, theta, trans, model_, Jw);

        for (int p = 0; p < static_cast<int>(pairs_.size()); ++p) {
            int smpl_closer = pairs_[p].closer_smpl;
            int smpl_farther = pairs_[p].farther_smpl;

            T z_closer = Jw(smpl_closer, 2);
            T z_farther = Jw(smpl_farther, 2);

            // We want: z_closer > z_farther (closer to camera = larger Z)
            const T margin = T(0.10);  // 10cm margin
            T violation = z_farther - z_closer + margin;

            T pair_weight = T(weight_) * T(pairs_[p].conf);
            residuals[p] = pair_weight * ceres::fmax(T(0.0), violation);
        }

        return true;
    }

private:
    struct DepthPair {
        int closer_smpl;
        int farther_smpl;
        float conf;
        float depth_diff;
    };

    std::vector<DepthPair> pairs_;
    const SmplModel& model_;
    double weight_;
    bool higher_is_closer_;
};

// ------------------------------
// Segment-to-segment distance (for capsule collision)
// AutoDiff compatible template function
// ------------------------------
template <typename T>
T DistSegmentToSegment(
    const Eigen::Matrix<T, 1, 3>& P1, const Eigen::Matrix<T, 1, 3>& P2,
    const Eigen::Matrix<T, 1, 3>& Q1, const Eigen::Matrix<T, 1, 3>& Q2)
{
    Eigen::Matrix<T, 1, 3> d1 = P2 - P1;  // 线段1方向向量
    Eigen::Matrix<T, 1, 3> d2 = Q2 - Q1;  // 线段2方向向量
    Eigen::Matrix<T, 1, 3> r = P1 - Q1;

    T a = d1.dot(d1);  // 线段1长度平方
    T e = d2.dot(d2);  // 线段2长度平方
    T f = d2.dot(r);

    const T eps = T(1e-8);
    T s, t;

    // 处理退化情况（点或平行）
    if (a <= eps && e <= eps) {
        // 两线段都退化为点
        return (P1 - Q1).norm();
    }
    if (a <= eps) {
        // 线段1退化为点
        s = T(0.0);
        t = ceres::fmax(T(0.0), ceres::fmin(f / e, T(1.0)));
    } else {
        T c = d1.dot(r);
        if (e <= eps) {
            // 线段2退化为点
            t = T(0.0);
            s = ceres::fmax(T(0.0), ceres::fmin(-c / a, T(1.0)));
        } else {
            T b = d1.dot(d2);
            T denom = a * e - b * b;

            if (denom > eps) {
                s = ceres::fmax(T(0.0), ceres::fmin((b * f - c * e) / denom, T(1.0)));
            } else {
                s = T(0.0);  // 平行情况
            }

            t = (b * s + f) / e;

            if (t < T(0.0)) {
                t = T(0.0);
                s = ceres::fmax(T(0.0), ceres::fmin(-c / a, T(1.0)));
            } else if (t > T(1.0)) {
                t = T(1.0);
                s = ceres::fmax(T(0.0), ceres::fmin((b - c) / a, T(1.0)));
            }
        }
    }

    Eigen::Matrix<T, 1, 3> closestP = P1 + s * d1;
    Eigen::Matrix<T, 1, 3> closestQ = Q1 + t * d2;

    // 添加 eps 防止 sqrt(0) 的梯度问题
    return ceres::sqrt((closestP - closestQ).squaredNorm() + eps);
}

// ------------------------------
// Interpenetration Residual (capsule collision penalty)
// Penalizes when two limb capsules overlap
// ------------------------------
struct InterpenetrationResidual {
    // 构造函数：传入肢体两端关节索引和半径
    InterpenetrationResidual(const SmplModel& model,
                             int j1_start, int j1_end,
                             int j2_start, int j2_end,
                             double r1, double r2, double weight)
        : model_(model), s1_(j1_start), e1_(j1_end),
          s2_(j2_start), e2_(j2_end),
          radius_sum_(r1 + r2), weight_(weight) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residual) const {
        const T* beta  = parameters[0];
        const T* theta = parameters[1];
        const T* trans = parameters[2];

        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> Jw;
        SMPLJointsForward<T>(beta, theta, trans, model_, Jw);

        // 计算两线段（肢体骨骼）之间的最短距离
        T dist = DistSegmentToSegment<T>(
            Jw.row(s1_), Jw.row(e1_),
            Jw.row(s2_), Jw.row(e2_));

        // Hinge Loss: 仅在距离小于半径之和时产生惩罚
        // penetration = radius_sum - dist
        // residual = weight * max(0, penetration)
        T penetration = T(radius_sum_) - dist;
        residual[0] = T(weight_) * ceres::fmax(T(0.0), penetration);

        return true;
    }

private:
    const SmplModel& model_;
    int s1_, e1_;  // 肢体1的起始/结束关节索引
    int s2_, e2_;  // 肢体2的起始/结束关节索引
    double radius_sum_;  // 两胶囊半径之和
    double weight_;
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

// Overload with depth constraints
bool OptimizeSMPL(const std::vector<Landmark2D>& observed_keypoints,
                  const std::vector<KeypointDepth>& depth_data,
                  const CameraIntrinsics& camera,
                  const SmplModel& model,
                  double* beta,
                  double* theta,
                  double* trans,
                  const OptimizerConfig& config);

CameraIntrinsics CreateDefaultCamera(int image_width, int image_height);

// Load depth data from JSON file
bool LoadKeypointDepthsFromJSON(const std::string& json_path,
                                 std::vector<KeypointDepth>& depths);
