#include "optimizer.h"
#include <thread>
#include <ceres/ceres.h>
#include <ceres/manifold.h>

// OpenPose BODY_25 to SMPL mapping (define ONCE here)
// SMPL joints: 0=Pelvis, 1=LHip, 2=RHip, 4=LKnee, 5=RKnee, 7=LAnkle, 8=RAnkle, 10=LFoot, 11=RFoot
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
    10,  // 19: LBigToe -> L_Foot
    -1,  // 20: LSmallToe (skip - use BigToe only)
    -1,  // 21: LHeel (skip - use BigToe only)
    11,  // 22: RBigToe -> R_Foot
    -1,  // 23: RSmallToe (skip - use BigToe only)
    -1   // 24: RHeel (skip - use BigToe only)
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

// ------------------------------
// Helper: Add depth ordering constraint
// ------------------------------
static void AddDepthOrderingBlock(
    ceres::Problem& problem,
    const std::vector<KeypointDepth>& depth_data,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans,
    const OptimizerConfig& cfg
) {
    if (!cfg.use_depth_ordering || depth_data.empty()) return;

    auto* functor = new DepthOrderingResidual(
        depth_data, model, cfg.depth_ordering_weight, cfg.higher_depth_is_closer
    );

    const int num_pairs = functor->NumPairs();
    if (num_pairs <= 0) {
        std::cerr << "[Warn] No valid depth pairs for ordering constraint.\n";
        delete functor;
        return;
    }

    std::cout << "[Depth] Adding " << num_pairs << " depth ordering constraints\n";

    auto* cost = new ceres::DynamicAutoDiffCostFunction<DepthOrderingResidual>(functor);
    cost->AddParameterBlock(10);   // beta
    cost->AddParameterBlock(72);   // theta
    cost->AddParameterBlock(3);    // trans
    cost->SetNumResiduals(num_pairs);

    problem.AddResidualBlock(cost, nullptr, beta, theta, trans);
}

// ------------------------------
// Helper: Add interpenetration (capsule collision) constraint
// ------------------------------
static void AddInterpenetrationBlock(
    ceres::Problem& problem,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans,
    const OptimizerConfig& cfg
) {
    if (!cfg.use_interpenetration) return;

    const double w = cfg.interpenetration_weight;
    const double r_thigh = cfg.thigh_capsule_radius;
    const double r_calf = cfg.calf_capsule_radius;
    const double r_foot = cfg.foot_capsule_radius;

    struct CollisionPair {
        int s1, e1, s2, e2;
        double r1, r2;
        const char* name;
    };

    std::vector<CollisionPair> pairs = {
        {1, 4, 2, 5, r_thigh, r_thigh, "L_Thigh vs R_Thigh"},
        {1, 4, 5, 8, r_thigh, r_calf,  "L_Thigh vs R_Calf"},
        {2, 5, 4, 7, r_thigh, r_calf,  "R_Thigh vs L_Calf"},
        {4, 7, 5, 8, r_calf,  r_calf,  "L_Calf vs R_Calf"},
        {4, 7, 8, 11, r_calf, r_foot,  "L_Calf vs R_Foot"},
        {5, 8, 7, 10, r_calf, r_foot,  "R_Calf vs L_Foot"},
        {7, 10, 8, 11, r_foot, r_foot, "L_Foot vs R_Foot"},
    };

    std::cout << "[Collision] Adding " << pairs.size() << " interpenetration constraints:\n";

    for (const auto& p : pairs) {
        std::cout << "  - " << p.name << " (r_sum=" << (p.r1 + p.r2) << "m)\n";

        auto* functor = new InterpenetrationResidual(
            model, p.s1, p.e1, p.s2, p.e2, p.r1, p.r2, w
        );

        auto* cost = new ceres::DynamicAutoDiffCostFunction<InterpenetrationResidual>(functor);
        cost->AddParameterBlock(10);
        cost->AddParameterBlock(72);
        cost->AddParameterBlock(3);
        cost->SetNumResiduals(1);

        problem.AddResidualBlock(cost, nullptr, beta, theta, trans);
    }
}

// OptimizeSMPL without depth (calls version with empty depth_data)
bool OptimizeSMPL(
    const std::vector<Landmark2D>& observed_keypoints,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans,
    const OptimizerConfig& config
) {
    std::vector<KeypointDepth> empty_depth;
    return OptimizeSMPL(observed_keypoints, empty_depth, camera, model, beta, theta, trans, config);
}

// OptimizeSMPL with depth data
bool OptimizeSMPL(
    const std::vector<Landmark2D>& observed_keypoints,
    const std::vector<KeypointDepth>& depth_data,
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

        // elbows (joint ids 18,19) 
        for (int e : {18, 19}) {
            add_hinge(e, HINGE_X,
                    /*w_perp=*/300.0, /*w_range=*/80.0,
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

        // Add depth ordering constraint (if enabled)
        AddDepthOrderingBlock(problem, depth_data, model, beta, theta, trans, config);

        // Add interpenetration (capsule collision) constraint (if enabled)
        AddInterpenetrationBlock(problem, model, beta, theta, trans, config);

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

        // keep pose reg
        for (int i = 3; i < 72; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                    new PoseRegularizer(i, 5.0)),
                nullptr, theta
            );
        }

        // Stage3: HingeAxisResidual
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

        // Stage3: AnatomicalLimitResidual (commented out)
        // const double hinge_lock_weight = 100.0;
        // const double limit_weight = 50.0;
        // // Knees: flex [0, 2.6], hinge lock [-0.1, 0.1]
        // int knees[] = {4, 5};
        // for (int k : knees) {
        //     problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AnatomicalLimitResidual,1,72>(
        //         new AnatomicalLimitResidual(k,0,0.0,2.6,limit_weight)), nullptr, theta);
        //     problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AnatomicalLimitResidual,1,72>(
        //         new AnatomicalLimitResidual(k,1,-0.1,0.1,hinge_lock_weight)), nullptr, theta);
        //     problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AnatomicalLimitResidual,1,72>(
        //         new AnatomicalLimitResidual(k,2,-0.1,0.1,hinge_lock_weight)), nullptr, theta);
        // }
        // // Elbows: flex [-2.6, 0.1], hinge lock [-0.1, 0.1]
        // int elbows[] = {18, 19};
        // for (int e : elbows) {
        //     problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AnatomicalLimitResidual,1,72>(
        //         new AnatomicalLimitResidual(e,0,-2.6,0.1,limit_weight)), nullptr, theta);
        //     problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AnatomicalLimitResidual,1,72>(
        //         new AnatomicalLimitResidual(e,1,-0.1,0.1,hinge_lock_weight)), nullptr, theta);
        //     problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AnatomicalLimitResidual,1,72>(
        //         new AnatomicalLimitResidual(e,2,-0.1,0.1,hinge_lock_weight)), nullptr, theta);
        // }

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

bool OptimizeSMPL3D(const MatX3dRM& J_obs,
                    const SmplModel& model,
                    double* beta,
                    double* theta,
                    double* trans,
                    const OptimizerConfig& config) {
    // Stage1: trans + root only
    {
        ceres::Problem problem;
        AddJoints3DBlock(problem, J_obs, model, true, beta, theta, trans, config.w_3d, nullptr);
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
        AddJoints3DBlock(problem, J_obs, model, true, beta, theta, trans, config.w_3d, nullptr);
        problem.SetParameterBlockConstant(beta);

        const double pose_reg_w = config.unit_test_exact ? 1e-2 : 1e-3;
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

// ------------------------------
// Load depth data from JSON file
// ------------------------------
#include <fstream>
#include <nlohmann/json.hpp>

bool LoadKeypointDepthsFromJSON(const std::string& json_path,
                                 std::vector<KeypointDepth>& depths) {
    depths.clear();

    std::ifstream infile(json_path);
    if (!infile.is_open()) {
        std::cerr << "[Depth] Failed to open: " << json_path << "\n";
        return false;
    }

    nlohmann::json j;
    try {
        infile >> j;
    } catch (const std::exception& e) {
        std::cerr << "[Depth] JSON parse error: " << e.what() << "\n";
        return false;
    }

    if (!j.contains("keypoint_depths")) {
        std::cerr << "[Depth] Missing 'keypoint_depths' field\n";
        return false;
    }

    for (const auto& kd : j["keypoint_depths"]) {
        int joint_id = kd.value("joint_id", -1);
        float depth = kd.value("depth", 0.0f);
        float conf = kd.value("confidence", 0.0f);

        if (joint_id >= 0 && joint_id < 25) {
            depths.emplace_back(joint_id, depth, conf);
        }
    }

    std::cout << "[Depth] Loaded " << depths.size() << " keypoint depths from " << json_path << "\n";
    return !depths.empty();
}
