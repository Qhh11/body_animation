#define EIGEN_STACK_ALLOCATION_LIMIT 1000000

#include "optimizer.h"

#include <thread>
#include <fstream>
#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <cnpy.h>

// -----------------------------------------
// OpenPose BODY_25 to SMPL mapping (DEFINE ONCE here)
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
// Data term adders (统一风格)
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

// -----------------------------------------
// Priors helpers (统一风格)
// -----------------------------------------
static Eigen::Vector3d AxisBasis(int axis) {
    if (axis == 0) return Eigen::Vector3d(1,0,0);
    if (axis == 1) return Eigen::Vector3d(0,1,0);
    return Eigen::Vector3d(0,0,1);
}

// Detect which axis-angle component moves child most.
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

    // Knees: joint 4(LKnee) 5(RKnee)
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

    // Elbows: joint 18(LElb) 19(RElb)
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

    // Knees: hyperextension barrier
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BendLimitResidual, 1, 72>(
            new BendLimitResidual(4, ax.axis_L_knee, +1.0, cfg.w_bend)),
        nullptr, theta
    );
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BendLimitResidual, 1, 72>(
            new BendLimitResidual(5, ax.axis_R_knee, +1.0, cfg.w_bend)),
        nullptr, theta
    );

    // Elbows: sign keeps your old convention (-1.0). If elbow bends wrong, flip sign.
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

    // elbows & wrists
    add_point(18, cfg.arm_torso_d_elbow, cfg.arm_torso_w_elbow);
    add_point(19, cfg.arm_torso_d_elbow, cfg.arm_torso_w_elbow);
    add_point(20, cfg.arm_torso_d_wrist, cfg.arm_torso_w_wrist);
    add_point(21, cfg.arm_torso_d_wrist, cfg.arm_torso_w_wrist);
}

enum class PriorStage { Stage2, Stage3 };

static void AddPosePriorsUnified(ceres::Problem& problem,
                                 const OptimizerConfig& cfg,
                                 const SmplModel& model,
                                 double* beta, double* theta, double* trans,
                                 PriorStage stage)
{
    // GMM
    AddGMMPrior(problem, cfg, theta);

    // hinge + bend
    BendAxes ax = ComputeBendAxes(model, beta, theta, trans);
    AddHingePriors(problem, cfg, ax, theta);
    AddBendLimitPriors(problem, cfg, ax, theta);

    // shoulder/collar
    if (cfg.use_shoulder_collar_limits) {
        AddSoftAngleLimitGroup(problem, theta, {cfg.collar_L, cfg.collar_R}, cfg.collar_amax, cfg.w_collar_limit);
        AddSoftAngleLimitGroup(problem, theta, {cfg.shoulder_L, cfg.shoulder_R}, cfg.shoulder_amax, cfg.w_shoulder_limit);
    }

    // wrist
    if (cfg.use_wrist_limits) {
        AddSoftAngleLimitGroup(problem, theta, {cfg.wrist_L, cfg.wrist_R}, cfg.wrist_amax, cfg.w_wrist_limit);
    }

    // spine
    if (cfg.use_spine_limits) {
        AddSoftAngleLimitGroup(problem, theta, {cfg.spine_j1, cfg.spine_j2, cfg.spine_j3}, cfg.spine_amax, cfg.w_spine);
    }

    // arm-torso separation
    AddArmTorsoSeparation(problem, cfg, model, beta, theta, trans);

    // root upright (stage-wise)
    if (cfg.use_root_upright) {
        const double w = (stage == PriorStage::Stage2) ? cfg.root_upright_w_stage2 : cfg.root_upright_w_stage3;
        if (w > 0.0) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<RootUprightResidual, 2, 72>(
                    new RootUprightResidual(w)),
                nullptr, theta
            );
        }
    }
}

// -----------------------------------------
// OptimizeSMPL (2D)
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

    // ----------------------------
    // Stage 1: trans + root only
    // ----------------------------
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 1: trans + full root (3DoF)\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock(problem, observed_keypoints, camera, model,
                                  beta, theta, trans, cfg,
                                  new ceres::HuberLoss(20.0));

        problem.SetParameterBlockConstant(beta);

        // Lock all joints except the root (theta[0:3]).
        std::vector<int> const_idx;
        const_idx.reserve(69);
        for (int i = 3; i < 72; ++i) const_idx.push_back(i);
        problem.SetManifold(theta, new ceres::SubsetManifold(72, const_idx));

        // simple Z bounds
        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 20.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, cfg, cfg.max_iter_stage1);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (cfg.verbose) std::cout << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    // ----------------------------
    // Stage 2: full pose (beta fixed)
    // ----------------------------
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 2: full pose (beta fixed)\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock(problem, observed_keypoints, camera, model,
                                  beta, theta, trans, cfg,
                                  new ceres::HuberLoss(5.0));

        problem.SetParameterBlockConstant(beta);

        AddDepthBlock(problem, cfg, model, beta, theta, trans);
        AddRelativeDepthBlock(problem, cfg, model, beta, theta, trans);

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

    // ----------------------------
    // Stage 3: refine beta + pose + trans
    // ----------------------------
    {
        if (cfg.verbose) std::cout << "\n>>> Stage 3: refine beta + pose + trans\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock(problem, observed_keypoints, camera, model,
                                  beta, theta, trans, cfg,
                                  new ceres::HuberLoss(5.0));

        AddDepthBlock(problem, cfg, model, beta, theta, trans);
        AddRelativeDepthBlock(problem, cfg, model, beta, theta, trans);

        // shape regularization (保持你原来的)
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
// OptimizeSMPL3D (unit test)
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
    // Stage1: trans + root only
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

    // Stage2: full pose + priors
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

// -----------------------------------------
// Default camera
// -----------------------------------------
CameraIntrinsics CreateDefaultCamera(int image_width, int image_height) {
    float focal = image_width * 1.2f;
    float cx = image_width * 0.5f;
    float cy = image_height * 0.5f;
    return CameraIntrinsics(focal, focal, cx, cy, image_width, image_height);
}
