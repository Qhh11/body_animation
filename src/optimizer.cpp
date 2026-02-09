// optimizer.cpp
#define EIGEN_STACK_ALLOCATION_LIMIT 1000000

#include "optimizer.h"

#include <thread>
#include <fstream>
#include <sstream>
#include <cstring>
#include <memory>

#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <cnpy.h>

// -----------------------------------------
// OpenPose BODY_25 to SMPL mapping
// -----------------------------------------
const int OPENPOSE_TO_SMPL[25] = {
    15,  // 0: Nose -> Head (rough)
    12,  // 1: Neck -> Neck
    17,  // 2: RShoulder
    19,  // 3: RElbow
    21,  // 4: RWrist
    16,  // 5: LShoulder
    18,  // 6: LElbow
    20,  // 7: LWrist
    0,   // 8: MidHip -> Pelvis
    2,   // 9: RHip
    5,   // 10: RKnee
    8,   // 11: RAnkle
    1,   // 12: LHip
    4,   // 13: LKnee
    7,   // 14: LAnkle
    -1,  // 15: REye
    -1,  // 16: LEye
    -1,  // 17: REar
    -1,  // 18: LEar
    -1,  // 19: LBigToe
    -1,  // 20: LSmallToe
    -1,  // 21: LHeel
    -1,  // 22: RBigToe
    -1,  // 23: RSmallToe
    -1   // 24: RHeel
};

// -----------------------------------------
// GMM Pose Prior Loading
// -----------------------------------------
bool GMMPosePrior::LoadFromNPZ(const std::string& npz_path) {
    try {
        cnpy::npz_t npz = cnpy::npz_load(npz_path);

        cnpy::NpyArray arr_w = npz["weights"];
        if (arr_w.shape.size() != 1 || (int)arr_w.shape[0] != K) {
            std::cerr << "[GMM] Invalid weights shape\n";
            return false;
        }
        double* w_data = arr_w.data<double>();
        for (int k = 0; k < K; ++k) weights(k) = w_data[k];

        cnpy::NpyArray arr_m = npz["means"];
        if (arr_m.shape.size() != 2 || (int)arr_m.shape[0] != K || (int)arr_m.shape[1] != D) {
            std::cerr << "[GMM] Invalid means shape\n";
            return false;
        }
        double* m_data = arr_m.data<double>();
        for (int k = 0; k < K; ++k)
            for (int d = 0; d < D; ++d)
                means(k, d) = m_data[k * D + d];

        cnpy::NpyArray arr_pc = npz["prec_choleskys"];
        if (arr_pc.shape.size() != 3 || (int)arr_pc.shape[0] != K ||
            (int)arr_pc.shape[1] != D || (int)arr_pc.shape[2] != D) {
            std::cerr << "[GMM] Invalid prec_choleskys shape\n";
            return false;
        }
        double* pc_data = arr_pc.data<double>();
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j)
                    prec_choleskys[k](i, j) = pc_data[k * D * D + i * D + j];

        loaded = true;
        std::cout << "[GMM] Loaded pose prior from " << npz_path << "\n";
        std::cout << "[GMM] K=" << K << ", D=" << D << "\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[GMM] Failed to load: " << e.what() << "\n";
        return false;
    }
}

// -----------------------------------------
// Solver helpers
// -----------------------------------------
static int AutoThreads(int cfg_threads) {
    if (cfg_threads > 0) return cfg_threads;
    if (cfg_threads == 0) return 1;
    unsigned hc = std::thread::hardware_concurrency();
    return std::max(1u, hc);
}

static void SetupSolverOptions(ceres::Solver::Options& options,
                               const OptimizerConfig& cfg,
                               int max_iter) {
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = AutoThreads(cfg.threads);
    options.max_num_iterations = max_iter;

    options.minimizer_progress_to_stdout = cfg.print_progress;
    options.function_tolerance  = 1e-6;
    options.gradient_tolerance  = 1e-10;
    options.parameter_tolerance = 1e-8;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
}

// -----------------------------------------
// Data term adders
// -----------------------------------------
static void AddAllKeypointReprojBlock(
    ceres::Problem& problem,
    const std::vector<Landmark2D>& observed,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans,
    const OptimizerConfig& cfg,
    ceres::LossFunction* loss
) {
    auto* functor = new AllKeypointsReprojectionResidual(
        observed, camera, model,
        cfg.min_confidence,
        cfg.use_stable_joints_only,
        cfg.flip_y_in_projection
    );

    const int K = functor->NumUsedKeypoints();
    if (K <= 0) {
        std::cerr << "[Warn] No valid keypoints for reprojection.\n";
        delete functor;
        return;
    }

    auto* cost = new ceres::AutoDiffCostFunction<
        AllKeypointsReprojectionResidual, ceres::DYNAMIC, 10, 72, 3>(
            functor, 2 * K);

    problem.AddResidualBlock(cost, loss, beta, theta, trans);
}

static void AddDepthBlock(
    ceres::Problem& problem,
    const OptimizerConfig& cfg,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans
) {
    if (!cfg.use_depth || cfg.depth_data == nullptr || cfg.depth_data->empty()) return;

    auto* functor = new DepthResidual(*cfg.depth_data, model, cfg.w_depth);
    auto* cost = new ceres::AutoDiffCostFunction<
        DepthResidual, ceres::DYNAMIC, 10, 72, 3>(
            functor, (int)cfg.depth_data->size());

    problem.AddResidualBlock(cost, nullptr, beta, theta, trans);
}

static void AddRelativeDepthBlock(
    ceres::Problem& problem,
    const OptimizerConfig& cfg,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans
) {
    if (!cfg.use_relative_depth || cfg.relative_depth_pairs == nullptr || cfg.relative_depth_pairs->empty()) return;

    auto* functor = new RelativeDepthResidual(*cfg.relative_depth_pairs, model, cfg.w_relative_depth);
    auto* cost = new ceres::AutoDiffCostFunction<
        RelativeDepthResidual, ceres::DYNAMIC, 10, 72, 3>(
            functor, (int)cfg.relative_depth_pairs->size());

    problem.AddResidualBlock(cost, nullptr, beta, theta, trans);
}

// Depth affine block (scale can be negative)
static void AddDepthAffineBlock(
    ceres::Problem& problem,
    const OptimizerConfig& cfg,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans
) {
    if (!cfg.use_depth_affine ||
        cfg.depth_affine_data == nullptr || cfg.depth_affine_data->empty() ||
        cfg.depth_affine_params == nullptr) return;

    auto* functor = new DepthAffineResidual(*cfg.depth_affine_data, model, cfg.w_depth_affine);

    auto* cost = new ceres::AutoDiffCostFunction<
        DepthAffineResidual, ceres::DYNAMIC, 10, 72, 3, 2>(
            functor, (int)cfg.depth_affine_data->size());

    ceres::LossFunction* loss = new ceres::CauchyLoss(0.5);
    problem.AddResidualBlock(cost, loss, beta, theta, trans, cfg.depth_affine_params);

    // scale in [-1e3, 1e3], bias in [0.1, 50]
    problem.SetParameterLowerBound(cfg.depth_affine_params, 0, -1e3);
    problem.SetParameterUpperBound(cfg.depth_affine_params, 0,  1e3);
    problem.SetParameterLowerBound(cfg.depth_affine_params, 1,  0.1);
    problem.SetParameterUpperBound(cfg.depth_affine_params, 1,  50.0);

    if (cfg.fix_depth_affine_bias) {
        // SubsetManifold: index 1 (bias) is held constant
        problem.SetManifold(cfg.depth_affine_params,
                            new ceres::SubsetManifold(2, {1}));
    }
}

static void AddTemporalDepthAffinePrior(
    ceres::Problem& problem,
    const OptimizerConfig& cfg
) {
    if (!cfg.use_temporal_depth_affine ||
        cfg.depth_affine_prev == nullptr ||
        cfg.depth_affine_params == nullptr) return;

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<TemporalDepthAffineResidual, 2, 2>(
            new TemporalDepthAffineResidual(cfg.depth_affine_prev,
                                            cfg.w_temp_depth_scale,
                                            cfg.w_temp_depth_bias)),
        nullptr,
        cfg.depth_affine_params
    );
}

static void AddTransZPriorBlock(
    ceres::Problem& problem,
    const OptimizerConfig& cfg,
    double* trans
) {
    if (!cfg.use_trans_z_prior || cfg.w_trans_z_prior <= 0.0) return;

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<TransZPriorResidual, 1, 3>(
            new TransZPriorResidual(cfg.trans_z_prior_target, cfg.w_trans_z_prior)),
        nullptr, trans
    );
}

// -----------------------------------------
// Cross-limb depth block
// -----------------------------------------

static void AddCrossLimbDepthBlock(
    ceres::Problem& problem,
    const OptimizerConfig& cfg,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans
) {
    if (!cfg.use_cross_limb_depth ||
        cfg.cross_limb_pairs == nullptr || cfg.cross_limb_pairs->empty()) return;

    double depth_sign = cfg.depth_affine_sign;
    if (cfg.depth_affine_params && cfg.depth_affine_params[0] != 0.0) {
        depth_sign = (cfg.depth_affine_params[0] > 0.0) ? 1.0 : -1.0;
    }

    auto* functor = new CrossLimbDepthResidual(*cfg.cross_limb_pairs, model,
                                                cfg.w_cross_limb_depth, depth_sign);
    auto* cost = new ceres::AutoDiffCostFunction<
        CrossLimbDepthResidual, ceres::DYNAMIC, 10, 72, 3>(
            functor, (int)cfg.cross_limb_pairs->size());

    problem.AddResidualBlock(cost, nullptr, beta, theta, trans);
}

// -----------------------------------------
// Capsule collision adder
// -----------------------------------------
static void AddCapsuleCollisions(
    ceres::Problem& problem,
    const OptimizerConfig& cfg,
    const SmplModel& model,
    double* beta, double* theta, double* trans
) {
    if (!cfg.use_capsule_collision) return;

    struct CapsuleDef { int j1, j2; double radius; };
    CapsuleDef L_upper_arm = {16, 18, 0.045};
    CapsuleDef R_upper_arm = {17, 19, 0.045};
    CapsuleDef L_forearm   = {18, 20, 0.035};
    CapsuleDef R_forearm   = {19, 21, 0.035};
    CapsuleDef L_thigh = {1, 4, 0.10};  
    CapsuleDef R_thigh = {2, 5, 0.10};  
    CapsuleDef L_shin      = {4,  7,  0.045};
    CapsuleDef R_shin      = {5,  8,  0.045};
    CapsuleDef Torso_low  = {0,  9,  0.11};
    CapsuleDef Torso_high = {9, 12,  0.10};


    double w = cfg.w_capsule_collision;

    auto add = [&](const CapsuleDef& a, const CapsuleDef& b, double w_override = 0.0) {
        double w_use = (w_override > 0.0) ? w_override : w;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CapsuleCollisionResidual, 1, 10, 72, 3>(
                new CapsuleCollisionResidual(a.j1, a.j2, a.radius,
                                            b.j1, b.j2, b.radius, w_use, model)),
            nullptr, beta, theta, trans);
    };

    double w_leg = w * 6.0;  // heavy for leg

    // Arm vs leg
    add(L_forearm, R_thigh); add(L_forearm, L_thigh);
    add(R_forearm, R_thigh); add(R_forearm, L_thigh);
    add(L_upper_arm, R_thigh); add(R_upper_arm, L_thigh);

    // Leg vs leg 
    add(L_shin, R_shin, w_leg); add(L_shin, R_thigh, w_leg);
    add(R_shin, L_thigh, w_leg); add(L_thigh, R_thigh, w_leg);

    // Arm vs torso 
    add(L_forearm, Torso_low);  add(R_forearm, Torso_low);
    add(L_upper_arm, Torso_low); add(R_upper_arm, Torso_low);
    add(L_forearm, Torso_high);  add(R_forearm, Torso_high);
    add(L_upper_arm, Torso_high); add(R_upper_arm, Torso_high);
}

// -----------------------------------------
// Priors helpers
// -----------------------------------------
static Eigen::Vector3d AxisBasis(int axis) {
    if (axis == 0) return Eigen::Vector3d(1,0,0);
    if (axis == 1) return Eigen::Vector3d(0,1,0);
    return Eigen::Vector3d(0,0,1);
}

static int DetectDominantAxisForChild(const SmplModel& model,
                                      const double* beta,
                                      double* theta,
                                      const double* trans,
                                      int joint,
                                      int child)
{
    MatX3dRM J0, J1;
    SMPLJointsForward<double>(beta, theta, trans, model, J0);

    int best_axis = 0;
    double best_d = -1.0;
    const double eps = 1e-2;

    for (int a = 0; a < 3; ++a) {
        int idx = 3 * joint + a;
        theta[idx] += eps;
        SMPLJointsForward<double>(beta, theta, trans, model, J1);
        theta[idx] -= eps;

        double d = (J1.row(child) - J0.row(child)).norm();
        if (d > best_d) { best_d = d; best_axis = a; }
    }
    return best_axis;
}

struct BendAxes {
    int axis_L_knee, axis_R_knee, axis_L_elb, axis_R_elb;
};

static BendAxes ComputeBendAxes(const SmplModel& model, const double* beta, double* theta, const double* trans) {
    BendAxes ax;
    ax.axis_L_knee = DetectDominantAxisForChild(model, beta, theta, trans, 4, 7);
    ax.axis_R_knee = DetectDominantAxisForChild(model, beta, theta, trans, 5, 8);
    ax.axis_L_elb  = DetectDominantAxisForChild(model, beta, theta, trans, 18, 20);
    ax.axis_R_elb  = DetectDominantAxisForChild(model, beta, theta, trans, 19, 21);
    return ax;
}

static void AddGMMPrior(ceres::Problem& problem, const OptimizerConfig& cfg, double* theta) {
    if (!cfg.use_gmm_prior || !cfg.gmm_prior || !cfg.gmm_prior->loaded) return;

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<GMMPosePriorResidual, 69, 72>(
            new GMMPosePriorResidual(*cfg.gmm_prior, cfg.w_gmm)),
        nullptr, theta
    );
}

static void AddHingePriors(ceres::Problem& problem,
                           const OptimizerConfig& cfg,
                           const BendAxes& ax,
                           double* theta)
{
    if (!cfg.use_hinge_priors) return;

    // Knees
    {
        Eigen::Vector3d uL = AxisBasis(ax.axis_L_knee);
        Eigen::Vector3d uR = AxisBasis(ax.axis_R_knee);

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
                new HingeAxisResidual(4, uL,
                                      cfg.w_hinge_perp_knee,
                                      cfg.w_hinge_range_knee,
                                      cfg.hinge_min, cfg.hinge_max)),
            nullptr, theta
        );

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
                new HingeAxisResidual(5, uR,
                                      cfg.w_hinge_perp_knee,
                                      cfg.w_hinge_range_knee,
                                      cfg.hinge_min, cfg.hinge_max)),
            nullptr, theta
        );
    }

    // Elbows
    {
        Eigen::Vector3d uL = AxisBasis(ax.axis_L_elb);
        Eigen::Vector3d uR = AxisBasis(ax.axis_R_elb);

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
                new HingeAxisResidual(18, uL,
                                      cfg.w_hinge_perp_elbow,
                                      cfg.w_hinge_range_elbow,
                                      cfg.hinge_min, cfg.hinge_max)),
            nullptr, theta
        );

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
                new HingeAxisResidual(19, uR,
                                      cfg.w_hinge_perp_elbow,
                                      cfg.w_hinge_range_elbow,
                                      cfg.hinge_min, cfg.hinge_max)),
            nullptr, theta
        );
    }
}

static void AddBendLimitPriors(ceres::Problem& problem,
                               const OptimizerConfig& cfg,
                               const BendAxes& ax,
                               double* theta)
{
    if (!cfg.use_bend_limits) return;

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BendLimitResidual, 1, 72>(
            new BendLimitResidual(4, ax.axis_L_knee, 1.0, cfg.w_bend)),
        nullptr, theta
    );
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BendLimitResidual, 1, 72>(
            new BendLimitResidual(5, ax.axis_R_knee, 1.0, cfg.w_bend)),
        nullptr, theta
    );

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BendLimitResidual, 1, 72>(
            new BendLimitResidual(18, ax.axis_L_elb, -1.0, cfg.w_bend)),
        nullptr, theta
    );
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BendLimitResidual, 1, 72>(
            new BendLimitResidual(19, ax.axis_R_elb, -1.0, cfg.w_bend)),
        nullptr, theta
    );
}

static void AddSoftAngleLimitGroup(ceres::Problem& problem,
                                  double* theta,
                                  const std::vector<int>& joints,
                                  double amax,
                                  double w)
{
    for (int j : joints) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SoftAngleLimitResidual, 1, 72>(
                new SoftAngleLimitResidual(j, amax, w)),
            nullptr, theta
        );
    }
}

// Point-to-segment distance floor (arm-torso separation) -- kept for backward compat
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

static void AddArmTorsoSeparation(ceres::Problem& problem,
                                 const OptimizerConfig& cfg,
                                 const SmplModel& model,
                                 double* beta, double* theta, double* trans)
{
    if (!cfg.use_arm_torso_sep) return;

    const int segA = cfg.arm_torso_seg_a;
    const int segB = cfg.arm_torso_seg_b;

    auto add_point = [&](int joint_point, double dmin, double w){
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PointToSegmentDistanceFloorResidual, 1, 10, 72, 3>(
                new PointToSegmentDistanceFloorResidual(joint_point, segA, segB, dmin, w, model)),
            nullptr, beta, theta, trans
        );
    };

    add_point(18, cfg.arm_torso_d_elbow, cfg.arm_torso_w_elbow);
    add_point(19, cfg.arm_torso_d_elbow, cfg.arm_torso_w_elbow);
    add_point(20, cfg.arm_torso_d_wrist, cfg.arm_torso_w_wrist);
    add_point(21, cfg.arm_torso_d_wrist, cfg.arm_torso_w_wrist);
}

// Build a SubsetManifold for pose locks
static void ApplyPoseLocksIfEnabled(
    ceres::Problem& problem,
    const OptimizerConfig& cfg,
    double* theta
) {
    std::vector<int> const_idx;
    const_idx.reserve(64);

    auto lock_joint = [&](int j){
        if (j < 0) return;
        const_idx.push_back(3*j + 0);
        const_idx.push_back(3*j + 1);
        const_idx.push_back(3*j + 2);
    };

    if (cfg.lock_wrists) { lock_joint(cfg.wrist_L); lock_joint(cfg.wrist_R); }
    if (cfg.lock_hands)  { lock_joint(cfg.hand_L);  lock_joint(cfg.hand_R);  }
    if (cfg.lock_feet)   { lock_joint(cfg.foot_L);  lock_joint(cfg.foot_R);  }

    if (cfg.lock_neck_head) {
        lock_joint(cfg.neck_j);
        lock_joint(cfg.head_j);
    }

    if (const_idx.empty()) return;

    std::sort(const_idx.begin(), const_idx.end());
    const_idx.erase(std::unique(const_idx.begin(), const_idx.end()), const_idx.end());

    problem.SetManifold(theta, new ceres::SubsetManifold(72, const_idx));
}


enum class PriorStage { Stage2, Stage3 };

static void AddPosePriorsUnified(ceres::Problem& problem,
                                 const OptimizerConfig& cfg,
                                 const SmplModel& model,
                                 double* beta, double* theta, double* trans,
                                 PriorStage stage)
{
    AddGMMPrior(problem, cfg, theta);

    BendAxes ax = ComputeBendAxes(model, beta, theta, trans);
    AddHingePriors(problem, cfg, ax, theta);
    AddBendLimitPriors(problem, cfg, ax, theta);

    if (cfg.use_shoulder_collar_limits) {
        AddSoftAngleLimitGroup(problem, theta, {cfg.collar_L, cfg.collar_R}, cfg.collar_amax, cfg.w_collar_limit);
        AddSoftAngleLimitGroup(problem, theta, {cfg.shoulder_L, cfg.shoulder_R}, cfg.shoulder_amax, cfg.w_shoulder_limit);
    }
    if (cfg.use_wrist_limits) {
        AddSoftAngleLimitGroup(problem, theta, {cfg.wrist_L, cfg.wrist_R}, cfg.wrist_amax, cfg.w_wrist_limit);
    }
    if (cfg.use_spine_limits) {
        AddSoftAngleLimitGroup(problem, theta, {cfg.spine_j1, cfg.spine_j2, cfg.spine_j3}, cfg.spine_amax, cfg.w_spine);
    }
    if (cfg.use_neck_head_limits) {
        AddSoftAngleLimitGroup(problem, theta, {cfg.neck_j}, cfg.neck_amax, cfg.w_neck_limit);
        AddSoftAngleLimitGroup(problem, theta, {cfg.head_j}, cfg.head_amax, cfg.w_head_limit);
    }
   
    if (cfg.use_arm_in_front) {
        double w = cfg.w_arm_in_front;
        double m = cfg.arm_in_front_margin;

        // Wrists
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ArmInFrontOfTorsoResidual, 1, 10, 72, 3>(
                new ArmInFrontOfTorsoResidual(20, m, w, model)),
            nullptr, beta, theta, trans);
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ArmInFrontOfTorsoResidual, 1, 10, 72, 3>(
                new ArmInFrontOfTorsoResidual(21, m, w, model)),
            nullptr, beta, theta, trans);

        // Elbows
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ArmInFrontOfTorsoResidual, 1, 10, 72, 3>(
                new ArmInFrontOfTorsoResidual(18, m, w * 0.7, model)),
            nullptr, beta, theta, trans);
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ArmInFrontOfTorsoResidual, 1, 10, 72, 3>(
                new ArmInFrontOfTorsoResidual(19, m, w * 0.7, model)),
            nullptr, beta, theta, trans);
    }

    AddArmTorsoSeparation(problem, cfg, model, beta, theta, trans);

    // Capsule-based self-collision
    AddCapsuleCollisions(problem, cfg, model, beta, theta, trans);

    // Cross-limb depth
    AddCrossLimbDepthBlock(problem, cfg, model, beta, theta, trans);

    if (cfg.use_pelvis_over_ankles && cfg.w_pelvis_over_ankles > 0.0) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PelvisOverAnklesResidual, 2, 10, 72, 3>(
                new PelvisOverAnklesResidual(0, 7, 8, cfg.w_pelvis_over_ankles, model)),
            nullptr, beta, theta, trans
        );
    }

    if (cfg.use_root_upright) {
            double wr = cfg.root_upright_w_roll;
            double wp = cfg.root_upright_w_pitch;
            if (stage == PriorStage::Stage3) {
                wr *= 0.3;
                wp *= 0.3;
            }
            if (wr > 0.0 || wp > 0.0) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<RootUprightResidual, 2, 72>(
                        new RootUprightResidual(wr, wp)),
                    nullptr, theta
                );
            }
        }

    if (cfg.use_foot_ground) {
        if (cfg.foot_contact_L > 1e-6) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<FootGroundFlatResidual, 2, 10, 72, 3>(
                    new FootGroundFlatResidual(
                        /*ankle=*/7, /*foot=*/cfg.foot_L,
                        cfg.ground_y,
                        cfg.w_foot_ground, cfg.w_foot_flat,
                        cfg.foot_contact_L,
                        model)),
                nullptr, beta, theta, trans
            );
        }
        if (cfg.foot_contact_R > 1e-6) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<FootGroundFlatResidual, 2, 10, 72, 3>(
                    new FootGroundFlatResidual(
                        /*ankle=*/8, /*foot=*/cfg.foot_R,
                        cfg.ground_y,
                        cfg.w_foot_ground, cfg.w_foot_flat,
                        cfg.foot_contact_R,
                        model)),
                nullptr, beta, theta, trans
            );
        }
    }
}

// -----------------------------------------
// Temporal priors with arm scaling
// -----------------------------------------
static void AddTemporalPriors(ceres::Problem& problem,
                              const OptimizerConfig& cfg,
                              double* theta, double* trans,
                              const double* theta_prev,
                              const double* trans_prev,
                              const double* theta_prev2)
{
    if (!cfg.use_temporal) return;

    if (theta_prev) {
        const int dim = cfg.temporal_body_only ? 69 : 72;

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TemporalPoseResidual, ceres::DYNAMIC, 72>(
                new TemporalPoseResidual(theta_prev, cfg.w_temp_pose,
                                         cfg.temporal_body_only, cfg.temporal_arm_scale),
                dim),
            nullptr, theta
        );

        const double w_root = (cfg.w_temp_root_orient > 0.0) ? cfg.w_temp_root_orient : cfg.w_temp_pose;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TemporalRootResidual, 3, 72>(
                new TemporalRootResidual(theta_prev, w_root)),
            nullptr, theta
        );
    }

    if (trans_prev) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TemporalTransResidual, 3, 3>(
                new TemporalTransResidual(trans_prev, cfg.w_temp_trans)),
            nullptr, trans
        );
    }

    if (cfg.use_temporal_acc && theta_prev && theta_prev2) {
        const int dim = cfg.temporal_body_only ? 69 : 72;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TemporalAccelPoseResidual, ceres::DYNAMIC, 72>(
                new TemporalAccelPoseResidual(theta_prev, theta_prev2, cfg.w_temp_acc_pose, cfg.temporal_body_only),
                dim),
            nullptr, theta
        );
    }
}

// -----------------------------------------
// OptimizeSMPL (first frame, no temporal)
// -----------------------------------------
bool OptimizeSMPL(
    const std::vector<Landmark2D>& observed_keypoints,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans,
    const OptimizerConfig& cfg
) {
    std::cout << "[Optimizer] Starting 2D Optimization...\n";

    // Stage 1: trans + root only
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 1: trans + full root (3DoF)\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock(problem, observed_keypoints, camera, model,
                                  beta, theta, trans, cfg,
                                  new ceres::HuberLoss(20.0));

        AddDepthBlock(problem, cfg, model, beta, theta, trans);
        AddRelativeDepthBlock(problem, cfg, model, beta, theta, trans);
        AddDepthAffineBlock(problem, cfg, model, beta, theta, trans);

        // [FIX] Z prior was missing in first-frame optimization
        AddTransZPriorBlock(problem, cfg, trans);

        problem.SetParameterBlockConstant(beta);

        std::vector<int> const_idx;
        const_idx.reserve(69);
        for (int i = 3; i < 72; ++i) const_idx.push_back(i);
        problem.SetManifold(theta, new ceres::SubsetManifold(72, const_idx));

        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 20.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, cfg, cfg.max_iter_stage1);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (cfg.verbose) std::cout << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    // Stage 2: full pose (beta fixed)
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 2: full pose (beta fixed)\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock(problem, observed_keypoints, camera, model,
                                  beta, theta, trans, cfg,
                                  new ceres::HuberLoss(5.0));

        problem.SetParameterBlockConstant(beta);

        ApplyPoseLocksIfEnabled(problem, cfg, theta);

        AddDepthBlock(problem, cfg, model, beta, theta, trans);
        AddRelativeDepthBlock(problem, cfg, model, beta, theta, trans);
        AddDepthAffineBlock(problem, cfg, model, beta, theta, trans);

        // [FIX] Z prior
        AddTransZPriorBlock(problem, cfg, trans);

        AddPosePriorsUnified(problem, cfg, model, beta, theta, trans, PriorStage::Stage2);

        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 20.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, cfg, cfg.max_iter_stage2);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (cfg.verbose) std::cout << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    // Stage 3: refine beta + pose + trans
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 3: refine beta + pose + trans\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock(problem, observed_keypoints, camera, model,
                                  beta, theta, trans, cfg,
                                  new ceres::HuberLoss(5.0));

        AddDepthBlock(problem, cfg, model, beta, theta, trans);
        AddRelativeDepthBlock(problem, cfg, model, beta, theta, trans);
        AddDepthAffineBlock(problem, cfg, model, beta, theta, trans);

        // [FIX] Z prior
        AddTransZPriorBlock(problem, cfg, trans);

        const double shape_reg_weight = 25.0;
        for (int i = 0; i < 10; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ShapeRegularizer, 1, 10>(
                    new ShapeRegularizer(i, shape_reg_weight)),
                nullptr, beta
            );
        }

        AddPosePriorsUnified(problem, cfg, model, beta, theta, trans, PriorStage::Stage3);

        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 20.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, cfg, cfg.max_iter_stage3);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (cfg.verbose) std::cout << "[Final] " << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    return true;
}

// -----------------------------------------
// Graduated energy: create a scaled config for a given stage
// Stage 1: light depth → reproj finds rough position
// Stage 2: balanced → depth starts correcting Z
// Stage 3: full → final 3D refinement
// -----------------------------------------
static OptimizerConfig ScaleDepthWeights(const OptimizerConfig& cfg,
                                          double depth_scale,
                                          double prior_scale)
{
    OptimizerConfig scaled = cfg;
    scaled.w_depth          *= depth_scale;
    scaled.w_relative_depth *= depth_scale;
    scaled.w_depth_affine   *= depth_scale;
    scaled.w_trans_z_prior  *= prior_scale;
    scaled.w_cross_limb_depth *= prior_scale;
    return scaled;
}

// -----------------------------------------
// OptimizeSMPLTracked (subsequent frames, with temporal)
// -----------------------------------------
bool OptimizeSMPLTracked(
    const std::vector<Landmark2D>& observed_keypoints,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans,
    const OptimizerConfig& cfg,
    const double* theta_prev,
    const double* trans_prev,
    const double* theta_prev2,
    const double* trans_prev2
) {
    std::cout << "[Optimizer] Starting 2D Optimization (Tracked)...\n";

    // Stage 1: trans + root only
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 1: trans + full root (lightweight)\n";
        ceres::Problem problem;

        // Graduated energy: scale depth/prior weights for stage 1
        const OptimizerConfig cfg_s1 = cfg.use_graduated_energy
            ? ScaleDepthWeights(cfg, cfg.grad_stage1_depth_scale, cfg.grad_stage1_prior_scale)
            : cfg;

        AddAllKeypointReprojBlock(problem, observed_keypoints, camera, model,
                                  beta, theta, trans, cfg,
                                  new ceres::HuberLoss(20.0));

        AddDepthBlock(problem, cfg_s1, model, beta, theta, trans);
        AddRelativeDepthBlock(problem, cfg_s1, model, beta, theta, trans);
        AddDepthAffineBlock(problem, cfg_s1, model, beta, theta, trans);
        AddTemporalDepthAffinePrior(problem, cfg);
            
        problem.SetParameterLowerBound(theta, 0, -0.6);  
        problem.SetParameterUpperBound(theta, 0,  0.6);

        problem.SetParameterBlockConstant(beta);

        std::vector<int> const_idx;
        const_idx.reserve(69);
        for (int i = 3; i < 72; ++i) const_idx.push_back(i);
        problem.SetManifold(theta, new ceres::SubsetManifold(72, const_idx));

        AddTemporalPriors(problem, cfg, theta, trans, theta_prev, trans_prev, theta_prev2);

        AddTransZPriorBlock(problem, cfg_s1, trans);

        ceres::Solver::Options options;
        SetupSolverOptions(options, cfg, cfg.max_iter_stage1);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (cfg.verbose) std::cout << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    // Stage 2: full pose (beta fixed)
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 2: full pose (beta fixed)\n";
        ceres::Problem problem;

        // Graduated energy: scale depth/prior weights for stage 2
        const OptimizerConfig cfg_s2 = cfg.use_graduated_energy
            ? ScaleDepthWeights(cfg, cfg.grad_stage2_depth_scale, cfg.grad_stage2_prior_scale)
            : cfg;

        AddAllKeypointReprojBlock(problem, observed_keypoints, camera, model,
                                  beta, theta, trans, cfg,
                                  new ceres::HuberLoss(5.0));

        problem.SetParameterBlockConstant(beta);

        ApplyPoseLocksIfEnabled(problem, cfg, theta);

        AddDepthBlock(problem, cfg_s2, model, beta, theta, trans);
        AddRelativeDepthBlock(problem, cfg_s2, model, beta, theta, trans);
        AddDepthAffineBlock(problem, cfg_s2, model, beta, theta, trans);
        AddTemporalDepthAffinePrior(problem, cfg);

        AddPosePriorsUnified(problem, cfg, model, beta, theta, trans, PriorStage::Stage2);
        AddTemporalPriors(problem, cfg, theta, trans, theta_prev, trans_prev, theta_prev2);

        // Graduated energy scaled Z prior
        AddTransZPriorBlock(problem, cfg_s2, trans);

        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 20.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, cfg, cfg.max_iter_stage2);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (cfg.verbose) std::cout << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    if (cfg.max_iter_stage3 <= 0) return true;

    // Stage 3: refine beta + pose + trans
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 3: refine beta + pose + trans\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock(problem, observed_keypoints, camera, model,
                                  beta, theta, trans, cfg,
                                  new ceres::HuberLoss(5.0));

        ApplyPoseLocksIfEnabled(problem, cfg, theta);

        AddDepthBlock(problem, cfg, model, beta, theta, trans);
        AddRelativeDepthBlock(problem, cfg, model, beta, theta, trans);
        AddDepthAffineBlock(problem, cfg, model, beta, theta, trans);
        AddTemporalDepthAffinePrior(problem, cfg);

        // [FIX] Use config Z prior
        AddTransZPriorBlock(problem, cfg, trans);

        const double shape_reg_weight = 25.0;
        for (int i = 0; i < 10; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ShapeRegularizer, 1, 10>(
                    new ShapeRegularizer(i, shape_reg_weight)),
                nullptr, beta
            );
        }

        AddPosePriorsUnified(problem, cfg, model, beta, theta, trans, PriorStage::Stage3);
        AddTemporalPriors(problem, cfg, theta, trans, theta_prev, trans_prev, theta_prev2);

        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 20.0);

        problem.SetParameterBlockConstant(beta);

        ceres::Solver::Options options;
        SetupSolverOptions(options, cfg, cfg.max_iter_stage3);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (cfg.verbose) std::cout << "[Final] " << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    return true;
}

// -----------------------------------------
// OptimizeSMPL3D (unit test) + Default camera
// -----------------------------------------
static void AddJoints3DBlockFast(
    ceres::Problem& problem,
    const MatX3dRM& J_obs,
    const SmplModel& model,
    double* beta, double* theta, double* trans,
    double w,
    ceres::LossFunction* loss
) {
    auto* functor = new Joints3DResidualFast(J_obs, model, w);
    auto* cost = new ceres::AutoDiffCostFunction<Joints3DResidualFast, 72, 10, 72, 3>(functor);
    problem.AddResidualBlock(cost, loss, beta, theta, trans);
}

bool OptimizeSMPL3D(const MatX3dRM& J_obs,
                    const SmplModel& model,
                    double* beta,
                    double* theta,
                    double* trans,
                    const OptimizerConfig& cfg) {
    {
        ceres::Problem problem;
        AddJoints3DBlockFast(problem, J_obs, model, beta, theta, trans, cfg.w_3d, nullptr);
        problem.SetParameterBlockConstant(beta);

        std::vector<int> const_idx;
        for (int i = 3; i < 72; ++i) const_idx.push_back(i);
        problem.SetManifold(theta, new ceres::SubsetManifold(72, const_idx));

        ceres::Solver::Options options;
        SetupSolverOptions(options, cfg, 50);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (cfg.verbose) std::cout << "[3D Stage1] " << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    {
        ceres::Problem problem;
        AddJoints3DBlockFast(problem, J_obs, model, beta, theta, trans, cfg.w_3d, nullptr);
        problem.SetParameterBlockConstant(beta);

        AddPosePriorsUnified(problem, cfg, model, beta, theta, trans, PriorStage::Stage2);

        ceres::Solver::Options options;
        SetupSolverOptions(options, cfg, 80);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (cfg.verbose) std::cout << "[3D Stage2] " << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    return true;
}


CameraIntrinsics CreateDefaultCamera(int image_width, int image_height) {
    float focal = image_width * 1.2f;
    float cx = image_width * 0.5f;
    float cy = image_height * 0.5f;
    return CameraIntrinsics(focal, focal, cx, cy, image_width, image_height);
}