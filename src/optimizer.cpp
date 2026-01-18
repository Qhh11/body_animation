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

static void SetupSolverOptions(ceres::Solver::Options& options, int max_iter) {
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = std::max(1u, std::thread::hardware_concurrency());
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = max_iter;
}

static void AddAllKeypointReprojBlock(
    ceres::Problem& problem,
    const std::vector<Landmark2D>& observed,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans,
    bool use_pose_blendshapes,
    const OptimizerConfig& cfg,
    ceres::LossFunction* loss
) {
    auto* functor = new AllKeypointsReprojectionResidual(
        observed,
        camera,
        model,
        use_pose_blendshapes,
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

    auto* cost = new ceres::DynamicAutoDiffCostFunction<AllKeypointsReprojectionResidual>(functor);
    cost->AddParameterBlock(10);
    cost->AddParameterBlock(72);
    cost->AddParameterBlock(3);
    cost->SetNumResiduals(2 * K);

    problem.AddResidualBlock(cost, loss, beta, theta, trans);
}

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
    // Stage 1: ONLY trans + ONE root axis (avoid flip /倒立)
    // ----------------------------
    {
        std::cout << "\n>>> Stage 1: trans + single-axis root (FAST & stable)\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock(
            problem, observed_keypoints, camera, model,
            beta, theta, trans,
            /*use_pose_blendshapes=*/false,
            config,
            new ceres::HuberLoss(20.0)
        );

        // fix shape
        problem.SetParameterBlockConstant(beta);

        // lock all theta except one root axis
        std::vector<int> const_idx;
        const_idx.reserve(71);

        // lock theta[0..71] except index = config.root_free_axis
        for (int i = 0; i < 72; ++i) {
            if (i == config.root_free_axis) continue; // free only one axis
            const_idx.push_back(i);
        }
        problem.SetManifold(theta, new ceres::SubsetManifold(72, const_idx));

        // bounds on depth
        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 20.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, 30);
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

        AddAllKeypointReprojBlock(
            problem, observed_keypoints, camera, model,
            beta, theta, trans,
            /*use_pose_blendshapes=*/true,
            config,
            new ceres::HuberLoss(5.0)
        );

        problem.SetParameterBlockConstant(beta);

        // stronger pose reg for 2D 
        const double pose_reg_weight = 2.0;
        for (int i = 3; i < 72; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                    new PoseRegularizer(i, pose_reg_weight)),
                nullptr, theta
            );
        }
        // Depth constraint (if enabled)
        if (config.use_depth && config.depth_data != nullptr) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<DepthResidual, ceres::DYNAMIC, 10, 72, 3>(
                    new DepthResidual(*config.depth_data, model, config.w_depth, true),
                    static_cast<int>(config.depth_data->size())
                ),
                nullptr, beta, theta, trans
            );
        }

        // ---- Shoulder / collar soft limits (energy) ----
        const double w_shoulder_limit = 30.0;
        const double w_collar_limit   = 20.0;

        const double shoulder_amax = 2.0; // ~115 deg
        const double collar_amax   = 1.0; // ~57 deg

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

        // ---- Stronger local pose prior for shoulder/collar (energy, not hard) ----
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
        // collars + shoulders stronger than generic pose reg
        for (int j : {13,14,16,17}) add_pose_reg_joint(j, 5.0);

        // hinge constraints (keep)
        const double hinge_lock_weight = 100.0; 
        const double limit_weight = 50.0;

        const double root_upright_w = 10.0; 

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RootUprightResidual, 2, 72>(
                new RootUprightResidual(root_upright_w)),
            nullptr, theta
        );

        auto add_hinge = [&](int j, Eigen::Vector3d axis, double w_perp, double w_range, double min_a, double max_a) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
                    new HingeAxisResidual(j, axis, w_perp, w_range, min_a, max_a)),
                nullptr, theta
            );
        };

        // 先假设 hinge 轴是 X（你得试 1~2 次确认哪个轴对应 flexion）
        Eigen::Vector3d HINGE_X(1,0,0);
        // knees (joint ids 4,5) 
        for (int k : {4, 5}) {
            add_hinge(k, HINGE_X,
                    /*w_perp=*/120.0, /*w_range=*/60.0,
                    /*min_a=*/0.0, /*max_a=*/2.6);
        }

        // elbows (joint ids 18,19) - 降低权重让手肘更自由
        for (int e : {18, 19}) {
            add_hinge(e, HINGE_X,
                    /*w_perp=*/60.0, /*w_range=*/30.0,
                    /*min_a=*/-2.6, /*max_a=*/0.1);
        }

        

        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 20.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, 40);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (config.verbose) std::cout << summary.BriefReport() << "\n";
    }

    // ----------------------------
    // Stage 3: allow beta refine (with strong shape prior)
    // ----------------------------
    {
        std::cout << "\n>>> Stage 3: refine beta + pose + trans\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock(
            problem, observed_keypoints, camera, model,
            beta, theta, trans,
            /*use_pose_blendshapes=*/true,
            config,
            new ceres::HuberLoss(2.0)
        );

        // Depth constraint (if enabled)
        if (config.use_depth && config.depth_data != nullptr) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<DepthResidual, ceres::DYNAMIC, 10, 72, 3>(
                    new DepthResidual(*config.depth_data, model, config.w_depth, true),
                    static_cast<int>(config.depth_data->size())
                ),
                nullptr, beta, theta, trans
            );
        }

        // shape reg
        const double shape_reg_weight = 80.0;
        for (int i = 0; i < 10; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ShapeRegularizer, 1, 10>(
                    new ShapeRegularizer(i, shape_reg_weight)),
                nullptr, beta
            );
        }

        const double root_upright_w = 10.0; 
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RootUprightResidual, 2, 72>(
                new RootUprightResidual(root_upright_w)),
            nullptr, theta
        );

         // ---- Shoulder / collar soft limits (energy) ----
        const double w_shoulder_limit = 30.0;
        const double w_collar_limit   = 20.0;

        const double shoulder_amax = 2.0; // ~115 deg
        const double collar_amax   = 1.0; // ~57 deg

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

        // ---- Stronger local pose prior for shoulder/collar (energy, not hard) ----
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
        // collars + shoulders stronger than generic pose reg
        for (int j : {13,14,16,17}) add_pose_reg_joint(j, 5.0);

        // keep pose reg
        for (int i = 3; i < 72; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                    new PoseRegularizer(i, 5.0)),
                nullptr, theta
            );
        }

        auto add_hinge = [&](int j, Eigen::Vector3d axis, double w_perp, double w_range, double min_a, double max_a) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<HingeAxisResidual, 2, 72>(
                    new HingeAxisResidual(j, axis, w_perp, w_range, min_a, max_a)),
                nullptr, theta
            );
        };


        // Stage3: weaker hinge than stage2
        Eigen::Vector3d HINGE_X(1,0,0);
        for (int k : {4,5}) add_hinge(k, HINGE_X, 60.0, 30.0, 0.0, 2.6);
        for (int e : {18,19}) add_hinge(e, HINGE_X, 60.0, 30.0, -2.6, 0.1);

        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 20.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, 40);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "[Final] " << summary.BriefReport() << "\n";
    }

    return true;
}

// ------------------------------
// 3D unit test
// ------------------------------
static void AddJoints3DBlock(
    ceres::Problem& problem,
    const MatX3dRM& J_obs,
    const SmplModel& model,
    bool use_pose_blendshapes,
    double* beta, double* theta, double* trans,
    double w,
    ceres::LossFunction* loss
) {
    auto* functor = new Joints3DResidual(J_obs, model, w, use_pose_blendshapes);
    auto* cost = new ceres::AutoDiffCostFunction<Joints3DResidual, ceres::DYNAMIC, 10, 72, 3>(
        functor,
        3 * static_cast<int>(J_obs.rows())
    );
    problem.AddResidualBlock(cost, loss, beta, theta, trans);
}

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
        SetupSolverOptions(options, 50);
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

        // ---- Shoulder / collar soft limits (energy) ----
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

        // ---- Stronger pose prior for collar/shoulder to reduce twist ambiguity ----
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
        for (int j : {13,14,16,17}) add_pose_reg_joint(j, 5e-2);  // 3D里权重比2D小很多

        const double pose_reg_w =1e-2;
        for (int i = 3; i < 72; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                    new PoseRegularizer(i, pose_reg_w)),
                nullptr, theta
            );
        }

        ceres::Solver::Options options;
        SetupSolverOptions(options, 200);
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
