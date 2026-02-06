#define EIGEN_STACK_ALLOCATION_LIMIT 1000000

#include "optimizer.h"

#include <thread>
#include <fstream>
#include <sstream>
#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <cnpy.h>

// ============================================================
// OpenPose BODY_25 → SMPL mapping (DEFINE ONCE)
// ============================================================
const int OPENPOSE_TO_SMPL[25] = {
    15,  // 0: Nose  → Head
    12,  // 1: Neck  → Neck
    17,  // 2: RShoulder
    19,  // 3: RElbow
    21,  // 4: RWrist
    16,  // 5: LShoulder
    18,  // 6: LElbow
    20,  // 7: LWrist
    0,   // 8: MidHip → Pelvis
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

// ============================================================
// GMM Prior Loading
// ============================================================
bool GMMPosePrior::LoadFromNPZ(const std::string& npz_path) {
    try {
        cnpy::npz_t npz = cnpy::npz_load(npz_path);

        cnpy::NpyArray arr_w = npz["weights"];
        if (arr_w.shape.size() != 1 || (int)arr_w.shape[0] != K) {
            std::cerr << "[GMM] Invalid weights shape\n"; return false;
        }
        double* w_data = arr_w.data<double>();
        for (int k = 0; k < K; ++k) weights(k) = w_data[k];

        cnpy::NpyArray arr_m = npz["means"];
        if (arr_m.shape.size() != 2 || (int)arr_m.shape[0] != K || (int)arr_m.shape[1] != D) {
            std::cerr << "[GMM] Invalid means shape\n"; return false;
        }
        double* m_data = arr_m.data<double>();
        for (int k = 0; k < K; ++k)
            for (int d = 0; d < D; ++d)
                means(k, d) = m_data[k * D + d];

        cnpy::NpyArray arr_pc = npz["prec_choleskys"];
        if (arr_pc.shape.size() != 3 || (int)arr_pc.shape[0] != K ||
            (int)arr_pc.shape[1] != D || (int)arr_pc.shape[2] != D) {
            std::cerr << "[GMM] Invalid prec_choleskys shape\n"; return false;
        }
        double* pc_data = arr_pc.data<double>();
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j)
                    prec_choleskys[k](i, j) = pc_data[k * D * D + i * D + j];

        loaded = true;
        std::cout << "[GMM] Loaded pose prior from " << npz_path
                  << " (K=" << K << ", D=" << D << ")\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[GMM] Failed to load: " << e.what() << "\n";
        return false;
    }
}

// ============================================================
// Solver helpers
// ============================================================
static int AutoThreads(int cfg_threads) {
    if (cfg_threads > 0) return cfg_threads;
    if (cfg_threads == 0) return 1;
    return std::max(1u, std::thread::hardware_concurrency());
}

static void SetupSolverOptions(ceres::Solver::Options& opts,
                               const OptimizerConfig& cfg,
                               int max_iter) {
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.num_threads = AutoThreads(cfg.threads);
    opts.max_num_iterations = max_iter;
    opts.minimizer_progress_to_stdout = cfg.print_progress;
    opts.function_tolerance  = 1e-6;
    opts.gradient_tolerance  = 1e-10;
    opts.parameter_tolerance = 1e-8;
    opts.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
}

// ============================================================
// Data term adders — ALL CAMERA SPACE
// ============================================================
static void AddKeypointReprojBlock(
    ceres::Problem& problem,
    const std::vector<Landmark2D>& observed,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    double* beta, double* theta, double* trans,
    const OptimizerConfig& cfg,
    ceres::LossFunction* loss)
{
    auto* functor = new AllKeypointsReprojectionResidual(
        observed, camera, model,
        cfg.min_confidence,
        cfg.use_stable_joints_only,
        cfg.flip_y_in_projection);

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

static void AddDepthBlock(ceres::Problem& problem, const OptimizerConfig& cfg,
                          const SmplModel& model,
                          double* beta, double* theta, double* trans)
{
    if (!cfg.use_depth || !cfg.depth_data || cfg.depth_data->empty()) return;
    auto* f = new DepthResidual(*cfg.depth_data, model, cfg.w_depth);
    auto* c = new ceres::AutoDiffCostFunction<DepthResidual, ceres::DYNAMIC, 10, 72, 3>(
        f, (int)cfg.depth_data->size());
    problem.AddResidualBlock(c, nullptr, beta, theta, trans);
}

static void AddRelativeDepthBlock(ceres::Problem& problem, const OptimizerConfig& cfg,
                                  const SmplModel& model,
                                  double* beta, double* theta, double* trans)
{
    if (!cfg.use_relative_depth || !cfg.relative_depth_pairs || cfg.relative_depth_pairs->empty()) return;
    auto* f = new RelativeDepthResidual(*cfg.relative_depth_pairs, model, cfg.w_relative_depth);
    auto* c = new ceres::AutoDiffCostFunction<RelativeDepthResidual, ceres::DYNAMIC, 10, 72, 3>(
        f, (int)cfg.relative_depth_pairs->size());
    problem.AddResidualBlock(c, nullptr, beta, theta, trans);
}

static void AddDepthAffineBlock(ceres::Problem& problem, const OptimizerConfig& cfg,
                                const SmplModel& model,
                                double* beta, double* theta, double* trans)
{
    if (!cfg.use_depth_affine || !cfg.depth_affine_data ||
        cfg.depth_affine_data->empty() || !cfg.depth_affine_params) return;

    auto* f = new DepthAffineResidual(*cfg.depth_affine_data, model,
                                      cfg.w_depth_affine, cfg.depth_affine_sign);
    auto* c = new ceres::AutoDiffCostFunction<
        DepthAffineResidual, ceres::DYNAMIC, 10, 72, 3, 2>(
            f, (int)cfg.depth_affine_data->size());

    problem.AddResidualBlock(c, new ceres::CauchyLoss(0.15),
                             beta, theta, trans, cfg.depth_affine_params);

    problem.SetParameterLowerBound(cfg.depth_affine_params, 0, 0.5);
    problem.SetParameterUpperBound(cfg.depth_affine_params, 0, 20.0);
    problem.SetParameterLowerBound(cfg.depth_affine_params, 1, 0.5);
    problem.SetParameterUpperBound(cfg.depth_affine_params, 1, 20.0);

    if (cfg.fix_depth_affine_bias) {
        problem.SetManifold(cfg.depth_affine_params, new ceres::SubsetManifold(2, {1}));
    }
}

static void AddTemporalDepthAffinePrior(ceres::Problem& problem, const OptimizerConfig& cfg)
{
    if (!cfg.use_temporal_depth_affine || !cfg.depth_affine_prev || !cfg.depth_affine_params) return;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<TemporalDepthAffineResidual, 2, 2>(
            new TemporalDepthAffineResidual(cfg.depth_affine_prev,
                                            cfg.w_temp_depth_scale,
                                            cfg.w_temp_depth_bias)),
        nullptr, cfg.depth_affine_params);
}

// Camera-space Z prior: soft constraint trans[2] ~ target
static void AddTransZPrior(ceres::Problem& problem, const OptimizerConfig& cfg,
                           double* trans)
{
    if (!cfg.use_trans_z_prior) return;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<TransZPriorResidual, 1, 3>(
            new TransZPriorResidual(cfg.trans_z_prior_target, cfg.w_trans_z_prior)),
        new ceres::HuberLoss(0.3),
        trans);
}

// ============================================================
// Prior helpers
// ============================================================
static Eigen::Vector3d AxisBasis(int axis) {
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    v(axis) = 1.0;
    return v;
}

static int DetectDominantAxisForChild(const SmplModel& model,
                                      const double* beta, double* theta,
                                      const double* trans,
                                      int joint, int child)
{
    MatX3dRM J0, J1;
    SMPLJointsForward<double>(beta, theta, trans, model, J0);
    int best = 0;
    double best_d = -1.0;
    const double eps = 1e-2;
    for (int a = 0; a < 3; ++a) {
        int idx = 3*joint + a;
        theta[idx] += eps;
        SMPLJointsForward<double>(beta, theta, trans, model, J1);
        theta[idx] -= eps;
        double d = (J1.row(child) - J0.row(child)).norm();
        if (d > best_d) { best_d = d; best = a; }
    }
    return best;
}

struct BendAxes {
    int axis_L_knee, axis_R_knee, axis_L_elb, axis_R_elb;
};

static BendAxes ComputeBendAxes(const SmplModel& model, const double* beta,
                                double* theta, const double* trans)
{
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
        nullptr, theta);
}

static void AddHingePriors(ceres::Problem& problem, const OptimizerConfig& cfg,
                           const BendAxes& ax, double* theta)
{
    if (!cfg.use_hinge_priors) return;
    auto add = [&](int j, const Eigen::Vector3d& u, double wp, double wr) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
                new HingeAxisResidual(j, u, wp, wr, cfg.hinge_min, cfg.hinge_max)),
            nullptr, theta);
    };
    add(4,  AxisBasis(ax.axis_L_knee), cfg.w_hinge_perp_knee,  cfg.w_hinge_range_knee);
    add(5,  AxisBasis(ax.axis_R_knee), cfg.w_hinge_perp_knee,  cfg.w_hinge_range_knee);
    add(18, AxisBasis(ax.axis_L_elb),  cfg.w_hinge_perp_elbow, cfg.w_hinge_range_elbow);
    add(19, AxisBasis(ax.axis_R_elb),  cfg.w_hinge_perp_elbow, cfg.w_hinge_range_elbow);
}

static void AddBendLimitPriors(ceres::Problem& problem, const OptimizerConfig& cfg,
                               const BendAxes& ax, double* theta)
{
    if (!cfg.use_bend_limits) return;
    auto add = [&](int j, int a, double s) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<BendLimitResidual, 1, 72>(
                new BendLimitResidual(j, a, s, cfg.w_bend)),
            nullptr, theta);
    };
    add(4,  ax.axis_L_knee, +1.0);
    add(5,  ax.axis_R_knee, +1.0);
    add(18, ax.axis_L_elb,  -1.0);
    add(19, ax.axis_R_elb,  -1.0);
}

static void AddSoftAngleLimitGroup(ceres::Problem& problem, double* theta,
                                   const std::vector<int>& joints,
                                   double amax, double w)
{
    for (int j : joints)
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SoftAngleLimitResidual, 1, 72>(
                new SoftAngleLimitResidual(j, amax, w)),
            nullptr, theta);
}

static void AddArmTorsoSeparation(ceres::Problem& problem, const OptimizerConfig& cfg,
                                  const SmplModel& model,
                                  double* beta, double* theta, double* trans)
{
    if (!cfg.use_arm_torso_sep) return;
    auto add = [&](int jp, double d, double w) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PointToSegmentDistanceFloorResidual, 1, 10, 72, 3>(
                new PointToSegmentDistanceFloorResidual(jp, cfg.arm_torso_seg_a, cfg.arm_torso_seg_b, d, w, model)),
            nullptr, beta, theta, trans);
    };
    add(18, cfg.arm_torso_d_elbow, cfg.arm_torso_w_elbow);
    add(19, cfg.arm_torso_d_elbow, cfg.arm_torso_w_elbow);
    add(20, cfg.arm_torso_d_wrist, cfg.arm_torso_w_wrist);
    add(21, cfg.arm_torso_d_wrist, cfg.arm_torso_w_wrist);
}

static void ApplyExtremityLocks(ceres::Problem& problem, const OptimizerConfig& cfg,
                                double* theta)
{
    std::vector<int> const_idx;
    auto lock = [&](int j) {
        if (j < 0) return;
        const_idx.push_back(3*j); const_idx.push_back(3*j+1); const_idx.push_back(3*j+2);
    };
    if (cfg.lock_wrists) { lock(cfg.wrist_L); lock(cfg.wrist_R); }
    if (cfg.lock_hands)  { lock(cfg.hand_L);  lock(cfg.hand_R);  }
    if (cfg.lock_feet)   { lock(cfg.foot_L);  lock(cfg.foot_R);  }
    if (!const_idx.empty())
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
        AddSoftAngleLimitGroup(problem, theta, {cfg.collar_L, cfg.collar_R},
                               cfg.collar_amax, cfg.w_collar_limit);
        AddSoftAngleLimitGroup(problem, theta, {cfg.shoulder_L, cfg.shoulder_R},
                               cfg.shoulder_amax, cfg.w_shoulder_limit);
    }
    if (cfg.use_wrist_limits)
        AddSoftAngleLimitGroup(problem, theta, {cfg.wrist_L, cfg.wrist_R},
                               cfg.wrist_amax, cfg.w_wrist_limit);
    if (cfg.use_spine_limits)
        AddSoftAngleLimitGroup(problem, theta, {cfg.spine_j1, cfg.spine_j2, cfg.spine_j3},
                               cfg.spine_amax, cfg.w_spine);
    if (cfg.use_neck_head_limits) {
        AddSoftAngleLimitGroup(problem, theta, {cfg.neck_j}, cfg.neck_amax, cfg.w_neck_limit);
        AddSoftAngleLimitGroup(problem, theta, {cfg.head_j}, cfg.head_amax, cfg.w_head_limit);
    }

    AddArmTorsoSeparation(problem, cfg, model, beta, theta, trans);

    if (cfg.use_pelvis_over_ankles && cfg.w_pelvis_over_ankles > 0.0)
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PelvisOverAnklesResidual, 2, 10, 72, 3>(
                new PelvisOverAnklesResidual(0, 7, 8, cfg.w_pelvis_over_ankles, model)),
            nullptr, beta, theta, trans);

    if (cfg.use_root_upright) {
        const double w = (stage == PriorStage::Stage2) ? cfg.root_upright_w_stage2
                                                       : cfg.root_upright_w_stage3;
        if (w > 0.0)
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<RootUprightResidual, 2, 72>(
                    new RootUprightResidual(w)),
                nullptr, theta);
    }

    if (cfg.use_foot_ground) {
        if (cfg.foot_contact_L > 1e-6)
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<FootGroundFlatResidual, 2, 10, 72, 3>(
                    new FootGroundFlatResidual(7, cfg.foot_L, cfg.ground_y,
                                              cfg.w_foot_ground, cfg.w_foot_flat,
                                              cfg.foot_contact_L, model)),
                nullptr, beta, theta, trans);
        if (cfg.foot_contact_R > 1e-6)
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<FootGroundFlatResidual, 2, 10, 72, 3>(
                    new FootGroundFlatResidual(8, cfg.foot_R, cfg.ground_y,
                                              cfg.w_foot_ground, cfg.w_foot_flat,
                                              cfg.foot_contact_R, model)),
                nullptr, beta, theta, trans);
    }
}

static void AddTemporalPriors(ceres::Problem& problem,
                              const OptimizerConfig& cfg,
                              double* theta, double* trans,
                              const double* theta_prev,
                              const double* trans_prev,
                              const double* theta_prev2,
                              const double* trans_prev2)
{
    if (!cfg.use_temporal) return;

    if (theta_prev) {
        const int dim = cfg.temporal_body_only ? 69 : 72;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TemporalPoseResidual, ceres::DYNAMIC, 72>(
                new TemporalPoseResidual(theta_prev, cfg.w_temp_pose, cfg.temporal_body_only), dim),
            nullptr, theta);

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TemporalRootResidual, 3, 72>(
                new TemporalRootResidual(theta_prev, cfg.w_temp_pose)),
            nullptr, theta);
    }

    // Constant-velocity trans prior when 2 previous frames available
    if (trans_prev && trans_prev2) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TransConstVelPriorResidual, 3, 3>(
                new TransConstVelPriorResidual(trans_prev, trans_prev2, cfg.w_temp_trans)),
            nullptr, trans);
    } else if (trans_prev) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TemporalTransResidual, 3, 3>(
                new TemporalTransResidual(trans_prev, cfg.w_temp_trans)),
            nullptr, trans);
    }

    if (cfg.use_temporal_acc && theta_prev && theta_prev2) {
        const int dim = cfg.temporal_body_only ? 69 : 72;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TemporalAccelPoseResidual, ceres::DYNAMIC, 72>(
                new TemporalAccelPoseResidual(theta_prev, theta_prev2,
                                             cfg.w_temp_acc_pose, cfg.temporal_body_only), dim),
            nullptr, theta);
    }
}

// ============================================================
// Camera-space Z bounds — applies to ALL stages
// ============================================================
static void ApplyCamSpaceZBounds(ceres::Problem& problem, double* trans) {
    problem.SetParameterLowerBound(trans, 2, 0.3);
    problem.SetParameterUpperBound(trans, 2, 25.0);
}

// ============================================================
// OptimizeSMPL — first frame (camera space)
// ============================================================
bool OptimizeSMPL(
    const std::vector<Landmark2D>& observed_keypoints,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    double* beta, double* theta, double* trans,
    const OptimizerConfig& cfg)
{
    if (cfg.verbose) std::cout << "[Optimizer] Camera-space optimization (first frame)...\n";

    // Stage 1: trans + root only
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 1: trans + root (3DoF)\n";
        ceres::Problem problem;

        AddKeypointReprojBlock(problem, observed_keypoints, camera, model,
                               beta, theta, trans, cfg, new ceres::HuberLoss(20.0));

        problem.SetParameterBlockConstant(beta);

        std::vector<int> const_idx;
        for (int i = 3; i < 72; ++i) const_idx.push_back(i);
        problem.SetManifold(theta, new ceres::SubsetManifold(72, const_idx));

        ApplyCamSpaceZBounds(problem, trans);
        AddTransZPrior(problem, cfg, trans);

        ceres::Solver::Options opts;
        SetupSolverOptions(opts, cfg, cfg.max_iter_stage1);
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        if (cfg.verbose) std::cout << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    // Stage 2: full pose (beta fixed)
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 2: full pose (beta fixed)\n";
        ceres::Problem problem;

        AddKeypointReprojBlock(problem, observed_keypoints, camera, model,
                               beta, theta, trans, cfg, new ceres::HuberLoss(5.0));

        problem.SetParameterBlockConstant(beta);
        ApplyExtremityLocks(problem, cfg, theta);
        AddDepthBlock(problem, cfg, model, beta, theta, trans);
        AddRelativeDepthBlock(problem, cfg, model, beta, theta, trans);
        AddDepthAffineBlock(problem, cfg, model, beta, theta, trans);
        AddPosePriorsUnified(problem, cfg, model, beta, theta, trans, PriorStage::Stage2);
        ApplyCamSpaceZBounds(problem, trans);
        AddTransZPrior(problem, cfg, trans);

        ceres::Solver::Options opts;
        SetupSolverOptions(opts, cfg, cfg.max_iter_stage2);
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        if (cfg.verbose) std::cout << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    // Stage 3: refine beta + pose + trans
    if (cfg.max_iter_stage3 > 0)
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 3: refine beta + pose + trans\n";
        ceres::Problem problem;

        AddKeypointReprojBlock(problem, observed_keypoints, camera, model,
                               beta, theta, trans, cfg, new ceres::HuberLoss(5.0));

        AddDepthBlock(problem, cfg, model, beta, theta, trans);
        AddRelativeDepthBlock(problem, cfg, model, beta, theta, trans);
        AddDepthAffineBlock(problem, cfg, model, beta, theta, trans);

        const double shape_reg = 25.0;
        for (int i = 0; i < 10; ++i)
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ShapeRegularizer, 1, 10>(
                    new ShapeRegularizer(i, shape_reg)),
                nullptr, beta);

        AddPosePriorsUnified(problem, cfg, model, beta, theta, trans, PriorStage::Stage3);
        ApplyCamSpaceZBounds(problem, trans);
        AddTransZPrior(problem, cfg, trans);

        ceres::Solver::Options opts;
        SetupSolverOptions(opts, cfg, cfg.max_iter_stage3);
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        if (cfg.verbose) std::cout << "[Final] " << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    return true;
}

// ============================================================
// OptimizeSMPLTracked — subsequent frames (camera space)
// Temporal prior targets are already in CURRENT camera space
// (caller transforms them before calling this function)
// ============================================================
bool OptimizeSMPLTracked(
    const std::vector<Landmark2D>& observed_keypoints,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    double* beta, double* theta, double* trans,
    const OptimizerConfig& cfg,
    const double* theta_prev,
    const double* trans_prev,
    const double* theta_prev2,
    const double* trans_prev2)
{
    if (cfg.verbose) std::cout << "[Optimizer] Camera-space tracked optimization...\n";

    // Stage 1: trans + root
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 1: trans + root (lightweight)\n";
        ceres::Problem problem;

        AddKeypointReprojBlock(problem, observed_keypoints, camera, model,
                               beta, theta, trans, cfg, new ceres::HuberLoss(20.0));

        problem.SetParameterBlockConstant(beta);

        std::vector<int> const_idx;
        for (int i = 3; i < 72; ++i) const_idx.push_back(i);
        problem.SetManifold(theta, new ceres::SubsetManifold(72, const_idx));

        AddTemporalPriors(problem, cfg, theta, trans,
                          theta_prev, trans_prev, theta_prev2, trans_prev2);

        // Strong Z prior in Stage 1 (fewer DoF, needs more guidance)
        if (trans_prev) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<TransZPriorResidual, 1, 3>(
                    new TransZPriorResidual(trans_prev[2], cfg.w_trans_z_prior * 1.5)),
                new ceres::HuberLoss(0.3),
                trans);
        }

        ApplyCamSpaceZBounds(problem, trans);

        ceres::Solver::Options opts;
        SetupSolverOptions(opts, cfg, cfg.max_iter_stage1);
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        if (cfg.verbose) std::cout << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    // Stage 2: full pose (beta fixed)
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 2: full pose (beta fixed)\n";
        ceres::Problem problem;

        AddKeypointReprojBlock(problem, observed_keypoints, camera, model,
                               beta, theta, trans, cfg, new ceres::HuberLoss(5.0));

        problem.SetParameterBlockConstant(beta);
        ApplyExtremityLocks(problem, cfg, theta);
        AddDepthBlock(problem, cfg, model, beta, theta, trans);
        AddRelativeDepthBlock(problem, cfg, model, beta, theta, trans);
        AddDepthAffineBlock(problem, cfg, model, beta, theta, trans);
        AddTemporalDepthAffinePrior(problem, cfg);
        AddPosePriorsUnified(problem, cfg, model, beta, theta, trans, PriorStage::Stage2);
        AddTemporalPriors(problem, cfg, theta, trans,
                          theta_prev, trans_prev, theta_prev2, trans_prev2);

        // Z prior (softer in stage 2 since we have more constraints)
        if (trans_prev) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<TransZPriorResidual, 1, 3>(
                    new TransZPriorResidual(trans_prev[2], cfg.w_trans_z_prior)),
                new ceres::HuberLoss(0.3),
                trans);
        }

        ApplyCamSpaceZBounds(problem, trans);

        ceres::Solver::Options opts;
        SetupSolverOptions(opts, cfg, cfg.max_iter_stage2);
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        if (cfg.verbose) std::cout << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    if (cfg.max_iter_stage3 <= 0) return true;

    // Stage 3: refine beta + pose + trans
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 3: refine beta + pose + trans\n";
        ceres::Problem problem;

        AddKeypointReprojBlock(problem, observed_keypoints, camera, model,
                               beta, theta, trans, cfg, new ceres::HuberLoss(5.0));

        ApplyExtremityLocks(problem, cfg, theta);
        AddDepthBlock(problem, cfg, model, beta, theta, trans);
        AddRelativeDepthBlock(problem, cfg, model, beta, theta, trans);
        AddDepthAffineBlock(problem, cfg, model, beta, theta, trans);
        AddTemporalDepthAffinePrior(problem, cfg);

        const double shape_reg = 25.0;
        for (int i = 0; i < 10; ++i)
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ShapeRegularizer, 1, 10>(
                    new ShapeRegularizer(i, shape_reg)),
                nullptr, beta);

        AddPosePriorsUnified(problem, cfg, model, beta, theta, trans, PriorStage::Stage3);
        AddTemporalPriors(problem, cfg, theta, trans,
                          theta_prev, trans_prev, theta_prev2, trans_prev2);

        if (trans_prev) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<TransZPriorResidual, 1, 3>(
                    new TransZPriorResidual(trans_prev[2], cfg.w_trans_z_prior)),
                new ceres::HuberLoss(0.3),
                trans);
        }

        ApplyCamSpaceZBounds(problem, trans);
        problem.SetParameterBlockConstant(beta);

        ceres::Solver::Options opts;
        SetupSolverOptions(opts, cfg, cfg.max_iter_stage3);
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        if (cfg.verbose) std::cout << "[Final] " << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    return true;
}

// ============================================================
// OptimizeSMPL3D (unit test)
// ============================================================
static void AddJoints3DBlockFast(ceres::Problem& problem,
                                 const MatX3dRM& J_obs, const SmplModel& model,
                                 double* beta, double* theta, double* trans,
                                 double w, ceres::LossFunction* loss)
{
    auto* c = new ceres::AutoDiffCostFunction<Joints3DResidualFast, 72, 10, 72, 3>(
        new Joints3DResidualFast(J_obs, model, w));
    problem.AddResidualBlock(c, loss, beta, theta, trans);
}

bool OptimizeSMPL3D(const MatX3dRM& J_obs, const SmplModel& model,
                    double* beta, double* theta, double* trans,
                    const OptimizerConfig& cfg)
{
    {
        ceres::Problem problem;
        AddJoints3DBlockFast(problem, J_obs, model, beta, theta, trans, cfg.w_3d, nullptr);
        problem.SetParameterBlockConstant(beta);
        std::vector<int> ci; for(int i=3;i<72;++i) ci.push_back(i);
        problem.SetManifold(theta, new ceres::SubsetManifold(72, ci));
        ceres::Solver::Options o; SetupSolverOptions(o, cfg, 50);
        ceres::Solver::Summary s; ceres::Solve(o, &problem, &s);
        if (cfg.verbose) std::cout << "[3D S1] " << s.BriefReport() << "\n";
        if (!s.IsSolutionUsable()) return false;
    }
    {
        ceres::Problem problem;
        AddJoints3DBlockFast(problem, J_obs, model, beta, theta, trans, cfg.w_3d, nullptr);
        problem.SetParameterBlockConstant(beta);
        AddPosePriorsUnified(problem, cfg, model, beta, theta, trans, PriorStage::Stage2);
        ceres::Solver::Options o; SetupSolverOptions(o, cfg, 80);
        ceres::Solver::Summary s; ceres::Solve(o, &problem, &s);
        if (cfg.verbose) std::cout << "[3D S2] " << s.BriefReport() << "\n";
        if (!s.IsSolutionUsable()) return false;
    }
    return true;
}

// ============================================================
// Default camera
// ============================================================
CameraIntrinsics CreateDefaultCamera(int w, int h) {
    float focal = w * 1.2f;
    return CameraIntrinsics(focal, focal, w*0.5f, h*0.5f, w, h);
}

// ============================================================
// Load camera poses
// ============================================================
std::vector<CameraExtrinsics> LoadCameraPoses(const std::string& path) {
    std::vector<CameraExtrinsics> result;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[LoadCameraPoses] Failed to open: " << path << std::endl;
        return result;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double v[12];
        for (int i = 0; i < 12; ++i)
            if (!(iss >> v[i])) { std::cerr << "[LoadCameraPoses] Parse error\n"; return result; }

        CameraExtrinsics e;
        e.R << v[0],v[1],v[2], v[4],v[5],v[6], v[8],v[9],v[10];
        e.t << v[3], v[7], v[11];
        result.push_back(e);
    }
    std::cout << "[LoadCameraPoses] Loaded " << result.size() << " camera poses\n";
    return result;
}