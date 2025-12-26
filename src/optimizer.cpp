#include "optimizer.h"
#include <thread>
#include <ceres/rotation.h>
#include <ceres/ceres.h>
#include <ceres/manifold.h>


// OpenPose BODY_25 to SMPL 24 Joint Mapping
// -1 means no corresponding SMPL joint exists
const int OPENPOSE_TO_SMPL[25] = {
    15,  // 0: Nose -> SMPL Head (approximate)
    12,  // 1: Neck -> SMPL Neck
    17,  // 2: RShoulder -> SMPL Right Shoulder
    19,  // 3: RElbow -> SMPL Right Elbow
    21,  // 4: RWrist -> SMPL Right Wrist
    16,  // 5: LShoulder -> SMPL Left Shoulder
    18,  // 6: LElbow -> SMPL Left Elbow
    20,  // 7: LWrist -> SMPL Left Wrist
    0,   // 8: MidHip -> SMPL Pelvis
    2,   // 9: RHip -> SMPL Right Hip
    5,   // 10: RKnee -> SMPL Right Knee
    8,   // 11: RAnkle -> SMPL Right Ankle
    1,   // 12: LHip -> SMPL Left Hip
    4,   // 13: LKnee -> SMPL Left Knee
    7,   // 14: LAnkle -> SMPL Left Ankle
    -1,  // 15: REye (not used)
    -1,  // 16: LEye (not used)
    -1,  // 17: REar (not used)
    -1,  // 18: LEar (not used)
    -1,  // 19: LBigToe (not used)
    -1,  // 20: LSmallToe (not used)
    -1,  // 21: LHeel (not used)
    -1,  // 22: RBigToe (not used)
    -1,  // 23: RSmallToe (not used)
    -1   // 24: RHeel (not used)
};

static void SetupSolverOptions(ceres::Solver::Options& options, int max_iter) {
    // For this problem size, DENSE_QR is often OK.
    // If you later add many residuals / priors, try DENSE_SCHUR.
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
    float min_conf,
    ceres::LossFunction* loss
) {
    auto* functor = new AllKeypointsReprojectionResidual(
        observed, camera, model, use_pose_blendshapes, min_conf
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

    if (config.amass_unit_test) {
        ceres::Problem problem;

        AddAllKeypointReprojBlock(
            problem, observed_keypoints, camera, model,
            beta, theta, trans,
            /*use_pose_blendshapes=*/true,
            config.min_confidence,
            nullptr
        );

        problem.SetParameterBlockConstant(beta);

        // z bound
        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 50.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, 80);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (config.verbose) std::cout << summary.FullReport() << "\n";
        return summary.IsSolutionUsable();
    }

    std::cout << "[Optimizer] Starting Optimization...\n";

    // ========================================================================
    // STAGE 1: rigid alignment (ONLY trans + root rot), posedirs OFF
    // ========================================================================
    {
        std::cout << "\n>>> Stage 1: Global Translation & Root Orientation (FAST)...\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock(
            problem, observed_keypoints, camera, model,
            beta, theta, trans,
            /*use_pose_blendshapes=*/false,
            config.min_confidence,
            new ceres::HuberLoss(20.0)
        );

        // keep shape fixed in stage 1
        problem.SetParameterBlockConstant(beta);

        // lock theta[3..71] using SubsetManifold
        std::vector<int> const_idx;
        const_idx.reserve(72 - 3);
        for (int i = 3; i < 72; ++i) const_idx.push_back(i);
        problem.SetManifold(theta, new ceres::SubsetManifold(72, const_idx));

        // bounds on depth
        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 50.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, 30);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "Stage 1 Time: " << summary.total_time_in_seconds << "s\n";
    }

    // ========================================================================
    // STAGE 2: articulated pose (posedirs ON)
    // ========================================================================
    {
        std::cout << "\n>>> Stage 2: Articulated Pose Fitting (posedirs ON)...\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock(
            problem, observed_keypoints, camera, model,
            beta, theta, trans,
            /*use_pose_blendshapes=*/true,
            config.min_confidence,
            new ceres::HuberLoss(5.0)
        );

        // keep shape fixed in stage 2
        problem.SetParameterBlockConstant(beta);

        // light L2 pose reg (avoid exploding)
        const double pose_reg_weight = 1.0;
        for (int i = 3; i < 72; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(new PoseRegularizer(i, pose_reg_weight)),
                nullptr, theta
            );
        }

        // hinge constraints
        const double hinge_lock_weight = 1000.0;
        const double limit_weight = 100.0;

        // knees (SMPL joint indices: adjust if your indexing differs)
        int knees[] = {4, 5};
        for (int k : knees) {
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AnatomicalLimitResidual,1,72>(
                new AnatomicalLimitResidual(k,0,0.0,2.6,limit_weight)), nullptr, theta);
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AnatomicalLimitResidual,1,72>(
                new AnatomicalLimitResidual(k,1,-0.1,0.1,hinge_lock_weight)), nullptr, theta);
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AnatomicalLimitResidual,1,72>(
                new AnatomicalLimitResidual(k,2,-0.1,0.1,hinge_lock_weight)), nullptr, theta);
        }

        // elbows
        int elbows[] = {18, 19};
        for (int e : elbows) {
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AnatomicalLimitResidual,1,72>(
                new AnatomicalLimitResidual(e,0,-2.6,0.1,limit_weight)), nullptr, theta);
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AnatomicalLimitResidual,1,72>(
                new AnatomicalLimitResidual(e,1,-0.1,0.1,hinge_lock_weight)), nullptr, theta);
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AnatomicalLimitResidual,1,72>(
                new AnatomicalLimitResidual(e,2,-0.1,0.1,hinge_lock_weight)), nullptr, theta);
        }

        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 50.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, 50);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "Stage 2 Time: " << summary.total_time_in_seconds << "s\n";
    }

    // ========================================================================
    // STAGE 3: shape refine (posedirs ON)
    // ========================================================================
    {
        std::cout << "\n>>> Stage 3: Shape & Fine Tuning...\n";
        ceres::Problem problem;

        AddAllKeypointReprojBlock(
            problem, observed_keypoints, camera, model,
            beta, theta, trans,
            /*use_pose_blendshapes=*/true,
            config.min_confidence,
            new ceres::HuberLoss(2.0)
        );

        // shape reg
        const double shape_reg_weight = 50.0;
        for (int i = 0; i < 10; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ShapeRegularizer,1,10>(new ShapeRegularizer(i, shape_reg_weight)),
                nullptr, beta
            );
        }

        // pose reg (slightly stronger)
        for (int i = 3; i < 72; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PoseRegularizer,1,72>(new PoseRegularizer(i, 2.0)),
                nullptr, theta
            );
        }

        problem.SetParameterLowerBound(trans, 2, 0.5);
        problem.SetParameterUpperBound(trans, 2, 50.0);

        ceres::Solver::Options options;
        SetupSolverOptions(options, 30);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "Stage 3 Time: " << summary.total_time_in_seconds << "s\n";
        std::cout << "[Final] Cost: " << summary.final_cost << "\n";
    }

    return true;
}


static void AddJoints3DBlock(ceres::Problem& problem,
                             const MatX3dRM& J_obs,
                             const SmplModel& model,
                             double* beta, double* theta, double* trans,
                             double w,
                             ceres::LossFunction* loss) {
    auto* functor = new Joints3DResidual(J_obs, model, w);
    auto* cost = new ceres::AutoDiffCostFunction<Joints3DResidual, ceres::DYNAMIC, 10, 72, 3>(
        functor,
        3 * static_cast<int>(J_obs.rows())  // residual dim = 72
    );
    problem.AddResidualBlock(cost, loss, beta, theta, trans);
}


bool OptimizeSMPL3D(const MatX3dRM& J_obs,
                    const SmplModel& model,
                    double* beta,
                    double* theta,
                    double* trans,
                    const OptimizerConfig& config) {

    // ---------------------------
    // Stage 1: trans + root rot only
    // ---------------------------
    {
        ceres::Problem problem;
        AddJoints3DBlock(problem, J_obs, model, beta, theta, trans,
                         config.w_3d, nullptr);

        problem.SetParameterBlockConstant(beta);

        std::vector<int> const_idx;
        const_idx.reserve(69);
        for (int i = 3; i < 72; ++i) const_idx.push_back(i);
        problem.SetManifold(theta, new ceres::SubsetManifold(72, const_idx));

        ceres::Solver::Options options;
        SetupSolverOptions(options, 50);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (config.verbose) std::cout << "[3D Stage1] " << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    // ---------------------------
    // Stage 2: full pose (beta fixed)
    // ---------------------------
    {
        ceres::Problem problem;
        AddJoints3DBlock(problem, J_obs, model, beta, theta, trans,
                         config.w_3d, nullptr);

        problem.SetParameterBlockConstant(beta);

        if (!config.unit_test_exact) {
            const double pose_reg_w = 1e-4;
            for (int i = 3; i < 72; ++i) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                        new PoseRegularizer(i, pose_reg_w)),
                    nullptr, theta);
            }
        }

        ceres::Solver::Options options;
        SetupSolverOptions(options, 200);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (config.verbose) std::cout << "[3D Stage2] " << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    // ---------------------------
    // Stage 3: optional beta refine
    // ---------------------------
    if (!config.unit_test_exact) {
        ceres::Problem problem;
        AddJoints3DBlock(problem, J_obs, model, beta, theta, trans,
                         config.w_3d, nullptr);

        const double shape_reg_w = 1e-4;
        for (int i = 0; i < 10; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ShapeRegularizer, 1, 10>(
                    new ShapeRegularizer(i, shape_reg_w)),
                nullptr, beta);
        }

        ceres::Solver::Options options;
        SetupSolverOptions(options, 100);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (config.verbose) std::cout << "[3D Stage3] " << summary.BriefReport() << "\n";
        if (!summary.IsSolutionUsable()) return false;
    }

    return true;
}


CameraIntrinsics CreateDefaultCamera(int image_width, int image_height) {
    float focal = image_width * 1.2f;
    float cx = image_width / 2.0f;
    float cy = image_height / 2.0f;
    return CameraIntrinsics(focal, focal, cx, cy, image_width, image_height);
}
