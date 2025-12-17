#include "optimizer.h"
#include <iostream>
#include <fstream>
#include <ceres/ceres.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

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
    std::cout << "[Optimizer] Starting Multi-Stage Optimization Strategy..." << std::endl;

    // ====================================================================
    // STAGE 1: Camera Translation & Global Orientation Only
    // ====================================================================
    {
        std::cout << ">>> Stage 1: Global Translation & Orientation..." << std::endl;
        ceres::Problem problem;

        for (const auto& lm : observed_keypoints) {
            if (lm.confidence < config.min_confidence) continue;
            int smpl_id = OPENPOSE_TO_SMPL[lm.joint_id];
            if (smpl_id < 0) continue;

            ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 10, 72, 3>(
                new ReprojectionResidual(lm, smpl_id, camera, model)
            );
            
            // 使用 HuberLoss 增加鲁棒性
            problem.AddResidualBlock(cost, new ceres::HuberLoss(5.0), beta, theta, trans);
        }

        // 核心技巧：设置 Constant (锁死变量)
        problem.SetParameterBlockConstant(beta); 
        
        // 添加强力正则化，迫使身体保持静止 (naive approach)
        for(int i=3; i<72; ++i) { 
             ceres::CostFunction* rigid_body = new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                new PoseRegularizer(i, 1000.0) // 巨大的权重，锁死身体
             );
             problem.AddResidualBlock(rigid_body, nullptr, theta);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 20; 
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // std::cout << summary.BriefReport() << std::endl;
    }

    // ====================================================================
    // STAGE 2: Pose Alignment (Body Joints)
    // ====================================================================
    {
        std::cout << ">>> Stage 2: Articulated Pose Fitting..." << std::endl;
        ceres::Problem problem;

        // 1. Data Term (Reprojection)
        for (const auto& lm : observed_keypoints) {
            if (lm.confidence < config.min_confidence) continue;
            int smpl_id = OPENPOSE_TO_SMPL[lm.joint_id];
            if (smpl_id < 0) continue;

            ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 10, 72, 3>(
                new ReprojectionResidual(lm, smpl_id, camera, model)
            );
            problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), beta, theta, trans);
        }

        // 2. Regularization 
        problem.SetParameterBlockConstant(beta);

        double pose_reg_weight = 5.0; 
        for (int i = 0; i < 72; ++i) {
            ceres::CostFunction* pose_cost = new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                new PoseRegularizer(i, pose_reg_weight)
            );
            problem.AddResidualBlock(pose_cost, nullptr, theta);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 50;
        options.minimizer_progress_to_stdout = config.verbose;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
    }

    // ====================================================================
    // STAGE 3: Shape Refinement (Optional)
    // ====================================================================
    {
        std::cout << ">>> Stage 3: Shape Refinement..." << std::endl;
        ceres::Problem problem;

        // Data Term
        for (const auto& lm : observed_keypoints) {
            if (lm.confidence < config.min_confidence) continue;
            int smpl_id = OPENPOSE_TO_SMPL[lm.joint_id];
            if (smpl_id < 0) continue;
            ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 10, 72, 3>(
                new ReprojectionResidual(lm, smpl_id, camera, model)
            );
            problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), beta, theta, trans);
        }

        // Regularization
        // 1. Pose Regularizer (Keep consistent)
        for (int i = 0; i < 72; ++i) {
            ceres::CostFunction* pose_cost = new ceres::AutoDiffCostFunction<PoseRegularizer, 1, 72>(
                new PoseRegularizer(i, 1.0)
            );
            problem.AddResidualBlock(pose_cost, nullptr, theta);
        }

        // 2. Shape Regularizer (Very Strong!)
        double shape_reg_weight = 100.0; 
        for (int i = 0; i < 10; ++i) {
            ceres::CostFunction* shape_cost = new ceres::AutoDiffCostFunction<ShapeRegularizer, 1, 10>(
                new ShapeRegularizer(i, shape_reg_weight)
            );
            problem.AddResidualBlock(shape_cost, nullptr, beta);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 30;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    return true;
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
    landmarks.clear();

    // 1. 打开 JSON 文件
    std::ifstream infile(json_path);
    if (!infile.is_open()) {
        std::cerr << "[Error] Cannot open OpenPose JSON file: "
                  << json_path << std::endl;
        return false;
    }

    // 2. 解析 JSON
    json j;
    try {
        infile >> j;
    } catch (const std::exception& e) {
        std::cerr << "[Error] Failed to parse JSON: " << e.what() << std::endl;
        return false;
    }

    // 3. 检查是否有人
    if (!j.contains("people") || j["people"].empty()) {
        std::cerr << "[Warning] No people detected in JSON: "
                  << json_path << std::endl;
        return false;
    }

    // 👉 简化假设：只取第一个人
    const json& person = j["people"][0];

    // 4. 检查 BODY_25 关键点
    if (!person.contains("pose_keypoints_2d")) {
        std::cerr << "[Error] JSON does not contain pose_keypoints_2d"
                  << std::endl;
        return false;
    }

    const auto& keypoints = person["pose_keypoints_2d"];

    if (!keypoints.is_array() || keypoints.size() != 25 * 3) {
        std::cerr << "[Error] pose_keypoints_2d has invalid size: "
                  << keypoints.size() << std::endl;
        return false;
    }

    // 5. 读取 25 个关键点
    for (int joint_id = 0; joint_id < 25; ++joint_id) {
        float x = keypoints[3 * joint_id + 0].get<float>();
        float y = keypoints[3 * joint_id + 1].get<float>();
        float c = keypoints[3 * joint_id + 2].get<float>();

        landmarks.emplace_back(x, y, c, joint_id);
    }

    std::cout << "[Info] Loaded " << landmarks.size()
              << " OpenPose BODY_25 landmarks from "
              << json_path << std::endl;

    return true;
    // For now, this is a placeholder
    std::cerr << "[Warning] LoadLandmarksFromJSON not yet implemented!" << std::endl;
    std::cerr << "Please manually create Landmark2D vectors for now." << std::endl;
    return false;
}
