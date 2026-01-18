#define EIGEN_STACK_ALLOCATION_LIMIT 1000000

#include "optimizer.h"

#include <thread>
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

    // printing is expensive (especially on Windows)
    options.minimizer_progress_to_stdout = cfg.print_progress;

    // Reasonable termination for speed
    options.function_tolerance  = 1e-6;
    options.gradient_tolerance  = 1e-10;
    options.parameter_tolerance = 1e-8;

    // LM is usually stable for reprojection
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    // If you want maximum speed:
    // options.num_linear_solver_threads = options.num_threads; // only for some solvers
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

    // Key speed change:
    // Use AutoDiffCostFunction with DYNAMIC residual count (cheaper than DynamicAutoDiffCostFunction).
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
    // Stage 1: trans + ONE root axis
    // ----------------------------
    {
        std::cout << "\n>>> Stage 1: trans + single-axis root\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock_AutoDiff(
            problem, observed_keypoints, camera, model,
            beta, theta, trans,
            config,
            new ceres::HuberLoss(20.0)
        );

        problem.SetParameterBlockConstant(beta);

        // lock all theta except one axis
        std::vector<int> const_idx;
        const_idx.reserve(71);
        for (int i = 0; i < 72; ++i) {
            if (i == config.root_free_axis) continue;
            const_idx.push_back(i);
        }
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

        problem.SetParameterBlockConstant(beta);

        // Pose regularization
        const double pose_reg_weight = 2.0;
        for (int i = 3; i < 72; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                    new PoseRegularizer(i, pose_reg_weight)),
                nullptr, theta
            );
        }

        // Depth constraint (optional)
        AddDepthBlock_AutoDiff(problem, config, model, beta, theta, trans);

        // Shoulder / collar soft limits
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

        // Stronger local reg for collar/shoulder
        auto add_pose_reg_joint = [&](int j, double w) {
            for (int k = 0; k < 3; ++k) {
                const int idx = 3 * j + k;
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                        new PoseRegularizer(idx, w)),
                    nullptr, theta
                );
            }
        };
        for (int j : {13,14,16,17}) add_pose_reg_joint(j, 5.0);

        // Root upright
        const double root_upright_w = 1.0;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RootUprightResidual, 2, 72>(
                new RootUprightResidual(root_upright_w)),
            nullptr, theta
        );

        // Hinge constraints
        auto add_hinge = [&](int j, Eigen::Vector3d axis, double w_perp, double w_range, double min_a, double max_a) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
                    new HingeAxisResidual(j, axis, w_perp, w_range, min_a, max_a)),
                nullptr, theta
            );
        };

        Eigen::Vector3d HINGE_X(1,0,0);

        // knees
        for (int k : {4, 5}) {
            add_hinge(k, HINGE_X, 120.0, 60.0, 0.0, 2.6);
        }
        // elbows
        for (int e : {18, 19}) {
            add_hinge(e, HINGE_X, 60.0, 30.0, -2.6, 0.1);
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

        AddAllKeypointReprojBlock_AutoDiff(
            problem, observed_keypoints, camera, model,
            beta, theta, trans,
            config,
            new ceres::HuberLoss(2.0)
        );

        // Depth constraint (optional)
        AddDepthBlock_AutoDiff(problem, config, model, beta, theta, trans);

        // Shape reg
        const double shape_reg_weight = 80.0;
        for (int i = 0; i < 10; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ShapeRegularizer, 1, 10>(
                    new ShapeRegularizer(i, shape_reg_weight)),
                nullptr, beta
            );
        }


        // Shoulder / collar soft limits
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

        // Stronger local reg for collar/shoulder
        auto add_pose_reg_joint = [&](int j, double w) {
            for (int k = 0; k < 3; ++k) {
                const int idx = 3 * j + k;
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                        new PoseRegularizer(idx, w)),
                    nullptr, theta
                );
            }
        };
        for (int j : {13,14,16,17}) add_pose_reg_joint(j, 5.0);

        // global pose reg
        for (int i = 3; i < 72; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                    new PoseRegularizer(i, 5.0)),
                nullptr, theta
            );
        }

        // weaker hinge
        auto add_hinge = [&](int j, Eigen::Vector3d axis, double w_perp, double w_range, double min_a, double max_a) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
                    new HingeAxisResidual(j, axis, w_perp, w_range, min_a, max_a)),
                nullptr, theta
            );
        };

        Eigen::Vector3d HINGE_X(1,0,0);
        for (int k : {4,5}) add_hinge(k, HINGE_X, 60.0, 30.0, 0.0, 2.6);
        for (int e : {18,19}) add_hinge(e, HINGE_X, 60.0, 30.0, -2.6, 0.1);

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

    // Stage2: full pose
    {
        ceres::Problem problem;
        AddJoints3DBlockFast(problem, J_obs, model, beta, theta, trans, config.w_3d, nullptr);
        problem.SetParameterBlockConstant(beta);

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

        const double pose_reg_w = 1e-2;
        for (int i = 3; i < 72; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                    new PoseRegularizer(i, pose_reg_w)),
                nullptr, theta
            );
        }

        ceres::Solver::Options options;
        SetupSolverOptions(options, config, 60);
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
