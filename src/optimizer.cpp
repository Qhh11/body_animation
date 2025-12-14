#include "optimizer.h"
#include <iostream>
#include <fstream>
#include <ceres/ceres.h>

// Main Optimization Function
bool OptimizeSMPL(
    const std::vector<Landmark2D>& observed_keypoints,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    double* beta,
    double* theta,
    double* trans,
    const OptimizerConfig& config
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting SMPL Optimization..." << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total landmarks: " << observed_keypoints.size() << std::endl;

    // Step 1: Create Ceres Problem
    ceres::Problem problem;

    // Step 2: Add 2D Reprojection Residuals
    int num_valid_landmarks = 0;

    for (const auto& landmark : observed_keypoints) {
        // Skip low-confidence detections
        if (landmark.confidence < config.min_confidence) {
            continue;
        }

        // Check if this OpenPose joint has a valid SMPL mapping
        if (landmark.joint_id < 0 || landmark.joint_id >= 25) {
            std::cerr << "[Warning] Invalid joint_id: " << landmark.joint_id << std::endl;
            continue;
        }

        int smpl_joint_id = OPENPOSE_TO_SMPL[landmark.joint_id];
        if (smpl_joint_id < 0) {
            // This joint (e.g., eyes, ears, toes) has no SMPL equivalent
            continue;
        }

        // Create cost function
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 10, 72, 3>(
                new ReprojectionResidual(landmark, smpl_joint_id, camera, model)
            );

        // Add residual block (no loss function = L2 norm)
        problem.AddResidualBlock(cost_function, nullptr, beta, theta, trans);
        num_valid_landmarks++;
    }

    std::cout << "Valid landmarks used: " << num_valid_landmarks << std::endl;

    if (num_valid_landmarks == 0) {
        std::cerr << "[Error] No valid landmarks to optimize!" << std::endl;
        return false;
    }

    // Step 3: Add Shape Regularization (for each beta parameter)
    for (int i = 0; i < 10; ++i) {
        ceres::CostFunction* shape_cost =
            new ceres::AutoDiffCostFunction<ShapeRegularizer, 1, 10>(
                new ShapeRegularizer(i, config.lambda_shape)
            );
        problem.AddResidualBlock(shape_cost, nullptr, beta);
    }

    // Step 4: Add Pose Regularization (for each theta parameter)
    for (int i = 0; i < 72; ++i) {
        ceres::CostFunction* pose_cost =
            new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                new PoseRegularizer(i, config.lambda_pose)
            );
        problem.AddResidualBlock(pose_cost, nullptr, theta);
    }

    // Step 5: (Disabled) Depth prior used to constrain global Z translation.
    // if (config.lambda_depth > 0) {
    //     ceres::CostFunction* depth_cost =
    //         new ceres::AutoDiffCostFunction<DepthPrior, 1, 3>(
    //             new DepthPrior(config.expected_depth, 0.5)  // sigma = 0.5m
    //         );
    //     problem.AddResidualBlock(depth_cost, nullptr, trans);
    // }

    // Step 6: Configure Solver Options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = config.verbose;
    options.max_num_iterations = config.max_iterations;
    options.function_tolerance = config.function_tolerance;
    options.gradient_tolerance = config.gradient_tolerance;
    options.parameter_tolerance = config.parameter_tolerance;

    // Step 7: Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Step 8: Report Results
    std::cout << "\n========================================" << std::endl;
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "========================================\n" << std::endl;

    if (config.verbose) {
        std::cout << "Optimized Parameters:" << std::endl;
        std::cout << "  Beta[0:3]: [" << beta[0] << ", " << beta[1] << ", " << beta[2] << "]" << std::endl;
        std::cout << "  Theta[0:3]: [" << theta[0] << ", " << theta[1] << ", " << theta[2] << "]" << std::endl;
        std::cout << "  Trans: [" << trans[0] << ", " << trans[1] << ", " << trans[2] << "]" << std::endl;
    }

    return summary.IsSolutionUsable();
}


CameraIntrinsics CreateDefaultCamera(int image_width, int image_height) {
    // Typical camera intrinsics for a standard webcam or phone camera
    float focal_length = image_width * 1.2f;  // Rule of thumb: ~1.2x image width
    float cx = image_width / 2.0f;
    float cy = image_height / 2.0f;

    return CameraIntrinsics(focal_length, focal_length, cx, cy, image_width, image_height);
}


bool LoadLandmarksFromJSON(
    const std::string& json_path,
    std::vector<Landmark2D>& landmarks
) {
    // TODO: Implement JSON parsing using a library like nlohmann/json or RapidJSON
    // For now, this is a placeholder
    std::cerr << "[Warning] LoadLandmarksFromJSON not yet implemented!" << std::endl;
    std::cerr << "Please manually create Landmark2D vectors for now." << std::endl;
    return false;
}
