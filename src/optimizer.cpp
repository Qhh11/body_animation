#define EIGEN_STACK_ALLOCATION_LIMIT 1000000

#include "optimizer.h"

#include <thread>
#include <fstream>
#include <ceres/ceres.h>
#include <ceres/manifold.h>

// OpenPose BODY_25 to SMPL mapping (define ONCE here)
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

static int AutoThreads(int cfg_threads) {
    if (cfg_threads > 0) return cfg_threads;
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

static void AddAllKeypointReprojBlock_AutoDiff(
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
        observed,
        camera,
        model,
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

    auto* cost = new ceres::AutoDiffCostFunction<AllKeypointsReprojectionResidual, ceres::DYNAMIC, 10, 72, 3>(
        functor,
        2 * K
    );

    problem.AddResidualBlock(cost, loss, beta, theta, trans);
}

static void AddDepthBlock_AutoDiff(
    ceres::Problem& problem,
    const OptimizerConfig& cfg,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans
) {
    if (!cfg.use_depth || cfg.depth_data == nullptr || cfg.depth_data->empty()) return;

    auto* functor = new DepthResidual(*cfg.depth_data, model, cfg.w_depth);

    auto* cost = new ceres::AutoDiffCostFunction<DepthResidual, ceres::DYNAMIC, 10, 72, 3>(
        functor,
        (int)cfg.depth_data->size()
    );

    problem.AddResidualBlock(cost, nullptr, beta, theta, trans);
}

static void AddDiagonalPosePriorBlock_AutoDiff(
    ceres::Problem& problem,
    const OptimizerConfig& cfg,
    double* theta
) {
    if (!cfg.use_pose_prior || cfg.pose_prior == nullptr || !cfg.pose_prior->ready) return;

    auto* functor = new DiagonalPosePriorResidual(*cfg.pose_prior, cfg.w_pose_prior);
    auto* cost = new ceres::AutoDiffCostFunction<DiagonalPosePriorResidual, 69, 72>(functor);
    problem.AddResidualBlock(cost, nullptr, theta);
}

// Detect which axis-angle component (0/1/2) of 'joint' moves 'child' the most.
static int DetectDominantAxisForChild(const SmplModel& model,
                                      const double* beta,
                                      double* theta,
                                      const double* trans,
                                      int joint,
                                      int child)
{
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J0, J1;
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

static Eigen::Vector3d AxisBasis(int axis) {
    if (axis == 0) return Eigen::Vector3d(1,0,0);
    if (axis == 1) return Eigen::Vector3d(0,1,0);
    return Eigen::Vector3d(0,0,1);
}

static void AddHingePriors(
    ceres::Problem& problem,
    const OptimizerConfig& cfg,
    const SmplModel& model,
    const double* beta,
    double* theta,
    const double* trans
) {
    if (!cfg.use_hinge_priors) return;

    // Knees: choose dominant axis that moves ankles the most.
    int axis_L_knee = DetectDominantAxisForChild(model, beta, theta, trans, 4, 7);
    int axis_R_knee = DetectDominantAxisForChild(model, beta, theta, trans, 5, 8);
    Eigen::Vector3d uLk = AxisBasis(axis_L_knee);
    Eigen::Vector3d uRk = AxisBasis(axis_R_knee);

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
            new HingeAxisResidual(4, uLk,
                                  cfg.w_hinge_perp_knee,
                                  cfg.w_hinge_range_knee,
                                  -2.6, 2.6)), // keep wide first
        nullptr, theta
    );
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
            new HingeAxisResidual(5, uRk,
                                  cfg.w_hinge_perp_knee,
                                  cfg.w_hinge_range_knee,
                                  -2.6, 2.6)),
        nullptr, theta
    );

    // Elbows: choose dominant axis that moves wrists the most.
    int axis_L_elb = DetectDominantAxisForChild(model, beta, theta, trans, 18, 20);
    int axis_R_elb = DetectDominantAxisForChild(model, beta, theta, trans, 19, 21);
    Eigen::Vector3d uLe = AxisBasis(axis_L_elb);
    Eigen::Vector3d uRe = AxisBasis(axis_R_elb);

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
            new HingeAxisResidual(18, uLe,
                                  cfg.w_hinge_perp_elbow,
                                  cfg.w_hinge_range_elbow,
                                  -2.6, 2.6)),
        nullptr, theta
    );
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
            new HingeAxisResidual(19, uRe,
                                  cfg.w_hinge_perp_elbow,
                                  cfg.w_hinge_range_elbow,
                                  -2.6, 2.6)),
        nullptr, theta
    );

    std::cout << "[Dbg] hinge axis: knee(L/R)=" << axis_L_knee << "/" << axis_R_knee
              << " elbow(L/R)=" << axis_L_elb << "/" << axis_R_elb << "\n";
}

static void AddBendLimitExpPriors(
    ceres::Problem& problem,
    const OptimizerConfig& cfg,
    double* theta,
    int axis_L_knee,
    int axis_R_knee,
    int axis_L_elb,
    int axis_R_elb
) {
    if (!cfg.use_bend_limits) return;

    // Knees: penalize hyperextension using exp(+angle_on_bend_axis).
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BendLimitResidual, 1, 72>(
            new BendLimitResidual(4, axis_L_knee, +1.0, cfg.w_bend)),
        nullptr, theta
    );
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BendLimitResidual, 1, 72>(
            new BendLimitResidual(5, axis_R_knee, +1.0, cfg.w_bend)),
        nullptr, theta
    );

    // Elbows: hyperextension direction is model-dependent.
    // Keep your sign convention for now; verify with GT if needed.
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BendLimitResidual, 1, 72>(
            new BendLimitResidual(18, axis_L_elb, -1.0, cfg.w_bend)),
        nullptr, theta
    );
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BendLimitResidual, 1, 72>(
            new BendLimitResidual(19, axis_R_elb, -1.0, cfg.w_bend)),
        nullptr, theta
    );
}

static void AddArmTorsoSeparationCapsule(
    ceres::Problem& problem,
    const SmplModel& model,
    double* beta, double* theta, double* trans,
    const OptimizerConfig& cfg
){
    if (!cfg.use_arm_torso_sep) return;

    const int torsoA = 3;   // spine1
    const int torsoB = 12;  // neck

    // 这才是“胶囊厚度”的关键：dmin = r_torso + r_arm + margin
    const double r_torso = 0.12;
    const double r_uarm  = 0.06;
    const double r_farm  = 0.05;
    const double margin  = 0.01;

    const double w = 300.0; // 先大一点，能明显顶出来

    auto add = [&](int a0,int a1,double dmin){
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ArmSegToTorsoSegFloorResidual, 1, 10, 72, 3>(
                new ArmSegToTorsoSegFloorResidual(a0,a1, torsoA,torsoB, dmin, w, model)),
            nullptr, beta, theta, trans
        );
    };

    add(16,18, r_torso + r_uarm + margin);
    add(18,20, r_torso + r_farm + margin);
    add(17,19, r_torso + r_uarm + margin);
    add(19,21, r_torso + r_farm + margin);
}



static void AddSpineSoftLimits(ceres::Problem& problem, double* theta) {
    // Softly discourage extreme spine rotations that often show up as "candy-wrapper" compensation.
    const double w_spine = 10.0;
    const double spine_amax = 1.2; // radians
    for (int j : {3, 6, 9}) { // spine1, spine2, spine3
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SoftAngleLimitResidual, 1, 72>(
                new SoftAngleLimitResidual(j, spine_amax, w_spine)),
            nullptr, theta
        );
    }
}

// ------------------------------
// OptimizeSMPL (2D)
// ------------------------------
bool OptimizeSMPL(
    const std::vector<Landmark2D>& observed_keypoints,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans,
    const OptimizerConfig& config
) {
    std::cout << "[Optimizer] Starting 2D Optimization...\n";

    // ----------------------------
    // Stage 1: trans + full 3-DoF root
    // ----------------------------
    {
        std::cout << "\n>>> Stage 1: trans + full root (3DoF)\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock_AutoDiff(
            problem, observed_keypoints, camera, model,
            beta, theta, trans,
            config,
            new ceres::HuberLoss(20.0)
        );

        // Keep beta fixed in Stage 1
        problem.SetParameterBlockConstant(beta);

        // Lock all joints except the root (theta[0:3]).
        std::vector<int> const_idx;
        const_idx.reserve(69);
        for (int i = 3; i < 72; ++i) const_idx.push_back(i);
        problem.SetManifold(theta, new ceres::SubsetManifold(72, const_idx));

        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 20.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, config, config.max_iter_stage1);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (config.verbose) std::cout << summary.BriefReport() << "\n";
    }

    // ----------------------------
    // Stage 2: full pose (beta fixed)
    // ----------------------------
    {
        std::cout << "\n>>> Stage 2: full pose (beta fixed)\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock_AutoDiff(
            problem, observed_keypoints, camera, model,
            beta, theta, trans,
            config,
            new ceres::HuberLoss(5.0)
        );

        // Keep beta fixed in Stage 2
        problem.SetParameterBlockConstant(beta);
        // Detect dominant bend axes for knees and elbows.
        const int axis_L_knee = DetectDominantAxisForChild(model, beta, theta, trans, 4, 7);
        const int axis_R_knee = DetectDominantAxisForChild(model, beta, theta, trans, 5, 8);
        const int axis_L_elb  = DetectDominantAxisForChild(model, beta, theta, trans, 18, 20);
        const int axis_R_elb  = DetectDominantAxisForChild(model, beta, theta, trans, 19, 21);

        std::cout << "[Dbg] bend axes: knee(L/R)=" << axis_L_knee << "/" << axis_R_knee
                << " elbow(L/R)=" << axis_L_elb << "/" << axis_R_elb << "\n";

        // Depth constraint (optional)
        AddDepthBlock_AutoDiff(problem, config, model, beta, theta, trans);

        // Our replacement pose prior (diagonal Gaussian)
        AddDiagonalPosePriorBlock_AutoDiff(problem, config, theta);

        // Hinge priors for elbows/knees (critical to suppress twist ambiguity)
        AddHingePriors(problem, config, model, beta, theta, trans);
        AddBendLimitExpPriors(problem, config, theta, axis_L_knee, axis_R_knee,axis_L_elb,  axis_R_elb);

        AddArmTorsoSeparationCapsule(problem, model, beta, theta, trans, config);

        // Shoulder / collar soft limits
        const double w_shoulder_limit = 60.0;
        const double w_collar_limit   = 40.0;
        const double shoulder_amax = 1.7;
        const double collar_amax   = 0.8;

        for (int j : {13, 14}) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SoftAngleLimitResidual, 1, 72>(
                    new SoftAngleLimitResidual(j, collar_amax, w_collar_limit)),
                nullptr, theta
            );
        }
        for (int j : {16, 17}) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SoftAngleLimitResidual, 1, 72>(
                    new SoftAngleLimitResidual(j, shoulder_amax, w_shoulder_limit)),
                nullptr, theta
            );
        }

        // Mild spine soft limits to reduce "torso twisting as explanation"
        AddSpineSoftLimits(problem, theta);

        // Root upright (optional)
        if (config.use_root_upright && config.root_upright_w_stage2 > 0.0) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<RootUprightResidual, 2, 72>(
                    new RootUprightResidual(config.root_upright_w_stage2)),
                nullptr, theta
            );
        }

        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 20.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, config, config.max_iter_stage2);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (config.verbose) std::cout << summary.BriefReport() << "\n";
    }

    // ----------------------------
    // Stage 3: refine beta + pose + trans
    // ----------------------------
    {
        std::cout << "\n>>> Stage 3: refine beta + pose + trans\n";
        ceres::Problem problem;
        const int axis_L_knee = DetectDominantAxisForChild(model, beta, theta, trans, 4, 7);
        const int axis_R_knee = DetectDominantAxisForChild(model, beta, theta, trans, 5, 8);
        const int axis_L_elb  = DetectDominantAxisForChild(model, beta, theta, trans, 18, 20);
        const int axis_R_elb  = DetectDominantAxisForChild(model, beta, theta, trans, 19, 21);


        AddAllKeypointReprojBlock_AutoDiff(
            problem, observed_keypoints, camera, model,
            beta, theta, trans,
            config,
            new ceres::HuberLoss(5.0)
        );

        // Depth constraint (optional)
        AddDepthBlock_AutoDiff(problem, config, model, beta, theta, trans);

        // Shape regularization
        const double shape_reg_weight = 25.0;
        for (int i = 0; i < 10; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ShapeRegularizer, 1, 10>(
                    new ShapeRegularizer(i, shape_reg_weight)),
                nullptr, beta
            );
        }

        // Our pose prior
        AddDiagonalPosePriorBlock_AutoDiff(problem, config, theta);

        // Hinge priors
        AddHingePriors(problem, config, model, beta, theta, trans);
        AddBendLimitExpPriors(problem, config, theta, axis_L_knee, axis_R_knee,axis_L_elb,  axis_R_elb);

        // Shoulder/collar limits
        const double w_shoulder_limit = 30.0;
        const double w_collar_limit   = 20.0;
        const double shoulder_amax = 2.0;
        const double collar_amax   = 1.0;

        for (int j : {13, 14}) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SoftAngleLimitResidual, 1, 72>(
                    new SoftAngleLimitResidual(j, collar_amax, w_collar_limit)),
                nullptr, theta
            );
        }
        for (int j : {16, 17}) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SoftAngleLimitResidual, 1, 72>(
                    new SoftAngleLimitResidual(j, shoulder_amax, w_shoulder_limit)),
                nullptr, theta
            );
        }

        AddSpineSoftLimits(problem, theta);

        // Root upright in Stage3 is often better disabled (let data drive)
        if (config.use_root_upright && config.root_upright_w_stage3 > 0.0) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<RootUprightResidual, 2, 72>(
                    new RootUprightResidual(config.root_upright_w_stage3)),
                nullptr, theta
            );
        }

        AddArmTorsoSeparationCapsule(problem, model, beta, theta, trans, config);

        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 20.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, config, config.max_iter_stage3);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (config.verbose) std::cout << "[Final] " << summary.BriefReport() << "\n";
    }

    return true;
}

// ------------------------------
// OptimizeSMPL3D (unit test)
// ------------------------------
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
                    const OptimizerConfig& config) {
    // Stage1: trans + root only
    {
        ceres::Problem problem;
        AddJoints3DBlockFast(problem, J_obs, model, beta, theta, trans, config.w_3d, nullptr);
        problem.SetParameterBlockConstant(beta);

        std::vector<int> const_idx;
        for (int i = 3; i < 72; ++i) const_idx.push_back(i);
        problem.SetManifold(theta, new ceres::SubsetManifold(72, const_idx));

        ceres::Solver::Options options;
        SetupSolverOptions(options, config, 50);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (!summary.IsSolutionUsable()) return false;
        if (config.verbose) std::cout << "[3D Stage1] " << summary.BriefReport() << "\n";
    }

    // Stage2: full pose (+ priors)
    {
        ceres::Problem problem;
        AddJoints3DBlockFast(problem, J_obs, model, beta, theta, trans, config.w_3d, nullptr);
        problem.SetParameterBlockConstant(beta);
        // Detect dominant bend axes for knees and elbows.
        const int axis_L_knee = DetectDominantAxisForChild(model, beta, theta, trans, 4, 7);
        const int axis_R_knee = DetectDominantAxisForChild(model, beta, theta, trans, 5, 8);
        const int axis_L_elb  = DetectDominantAxisForChild(model, beta, theta, trans, 18, 20);
        const int axis_R_elb  = DetectDominantAxisForChild(model, beta, theta, trans, 19, 21);

        std::cout << "[Dbg] bend axes: knee(L/R)=" << axis_L_knee << "/" << axis_R_knee
                << " elbow(L/R)=" << axis_L_elb << "/" << axis_R_elb << "\n";
        AddHingePriors(problem, config, model, beta, theta, trans);
        AddBendLimitExpPriors(problem, config, theta, axis_L_knee, axis_R_knee,axis_L_elb,  axis_R_elb);
        AddSpineSoftLimits(problem, theta);

        ceres::Solver::Options options;
        SetupSolverOptions(options, config, 80);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (!summary.IsSolutionUsable()) return false;
        if (config.verbose) std::cout << "[3D Stage2] " << summary.BriefReport() << "\n";
    }

    return true;
}

CameraIntrinsics CreateDefaultCamera(int image_width, int image_height) {
    float focal = image_width * 1.2f;
    float cx = image_width * 0.5f;
    float cy = image_height * 0.5f;
    return CameraIntrinsics(focal, focal, cx, cy, image_width, image_height);
}
