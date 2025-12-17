#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include "smpl_model.h"
#include "smpl_math.h"
#include "optimizer.h"
#include <ceres/ceres.h>

// Forward declaration
void SaveOBJ(const std::string& filename,
             const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& vertices,
             const Eigen::MatrixXi& faces);

namespace {

Eigen::Vector3d ComputeSmplJointPosition(
    const SmplModel& model,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& vertices,
    int smpl_joint_id) {
    Eigen::Vector3d joint = Eigen::Vector3d::Zero();
    for (int v = 0; v < model.N; ++v) {
        double weight = model.J_regressor(smpl_joint_id, v);
        if (weight > 1e-8) {
            joint.x() += weight * vertices(v, 0);
            joint.y() += weight * vertices(v, 1);
            joint.z() += weight * vertices(v, 2);
        }
    }
    return joint;
}

// [ONLY FOR TEST] 从GT参数生成带噪声的假2D关键点
std::vector<Landmark2D> GenerateSyntheticLandmarks(
    const SmplModel& model,
    const CameraIntrinsics& camera,
    const double* beta,
    const double* theta,
    const double* trans) {
    std::vector<Landmark2D> landmarks;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vertices;
    SMPLForward<double>(beta, theta, trans, model, vertices);

    for (int joint_id = 0; joint_id < 25; ++joint_id) {
        int smpl_joint_id = OPENPOSE_TO_SMPL[joint_id];
        if (smpl_joint_id < 0) {
            continue;
        }

        Eigen::Vector3d joint = ComputeSmplJointPosition(model, vertices, smpl_joint_id);
        double Z = std::max(joint.z(), 1e-6);
        float projected_x = static_cast<float>(camera.fx * (joint.x() / Z) + camera.cx);
        float projected_y = static_cast<float>(camera.fy * (joint.y() / Z) + camera.cy);

        // Add deterministic pixel offsets to mimic detector noise
        float noise_u = (joint_id % 2 == 0) ? 1.5f : -1.5f;
        float noise_v = (joint_id % 3 == 0) ? -1.0f : 1.0f;

        landmarks.emplace_back(projected_x + noise_u, projected_y + noise_v, 0.95f, joint_id);
    }
    return landmarks;
}

void SaveLandmarksToFile(const std::string& filename, const std::vector<Landmark2D>& landmarks) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    for (const auto& lm : landmarks) {
        outfile << lm.joint_id << " " << lm.x << " " << lm.y << " " << lm.confidence << "\n";
    }
    outfile.close();
    std::cout << "Saved landmarks to " << filename << std::endl;
}

// [ONLY FOR TEST] 合成测试Demo：验证优化器能否从假数据恢复GT参数
void RunSyntheticOptimizationDemo(const SmplModel& model) {
    std::cout << "\n=== Synthetic optimization demo ===" << std::endl;

    CameraIntrinsics camera = CreateDefaultCamera(640, 480);

    // [ONLY FOR TEST] 人工设定Ground Truth参数用于生成假数据
    double beta_gt[10] = {0.0};
    beta_gt[1] = 0.2;
    beta_gt[4] = -0.1;

    double theta_gt[72] = {0.0};
    theta_gt[0] = 0.15;
    theta_gt[1] = -0.2;
    theta_gt[20] = 0.3;

    double trans_gt[3] = {0.05, -0.03, 3.0};

    // [ONLY FOR TEST] 从GT参数生成带噪声的假2D关键点
    // 真实数据时替换为: LoadLandmarksFromJSON() 或 LoadLandmarksFromAMASS()
    // auto synthetic_landmarks = GenerateSyntheticLandmarks(model, camera, beta_gt, theta_gt, trans_gt);
    auto synthetic_landmarks = std::vector<Landmark2D>();
    LoadLandmarksFromJSON("../../Data/OpenPose/output_jsons/video_000000000000_keypoints.json", synthetic_landmarks);
    SaveLandmarksToFile("landmarks_gt.txt", synthetic_landmarks);

    // 初始化优化参数
    double beta_init[10] = {0.0};
    double theta_init[72] = {0.0};
    double trans_init[3] = {0.0, 0.0, 2.2};

    // 保存优化前的mesh（可选）
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> verts_before;
    SMPLForward<double>(beta_init, theta_init, trans_init, model, verts_before);
    SaveOBJ("mesh_before_opt.obj", verts_before, model.faces);

    // 配置优化器
    OptimizerConfig config;
    // config.lambda_depth = 0.05;     // 真实数据时建议启用depth prior
    // config.expected_depth = 2.5;
    config.max_iterations = 20;
    config.verbose = true;

    // 运行Ceres优化
    bool success = OptimizeSMPL(synthetic_landmarks, camera, model, beta_init, theta_init, trans_init, config);

    if (!success) {
        std::cerr << "[Demo] Optimization failed." << std::endl;
        return;
    }

    // [ONLY FOR TEST] 打印优化结果与GT的对比
    auto print_vec = [](const std::string& name, const double* est, const double* gt, int length) {
        std::cout << name << ": ";
        for (int i = 0; i < length; ++i) {
            std::cout << "[" << est[i] << " | " << gt[i] << "] ";
        }
        std::cout << std::endl;
    };

    print_vec("Beta[0:3]  (est | gt)", beta_init, beta_gt, 3);
    print_vec("Theta[0:3] (est | gt)", theta_init, theta_gt, 3);
    print_vec("Trans      (est | gt)", trans_init, trans_gt, 3);

    // 保存优化后的mesh
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> verts_after;
    SMPLForward<double>(beta_init, theta_init, trans_init, model, verts_after);
    SaveOBJ("mesh_after_opt.obj", verts_after, model.faces);

    // [ONLY FOR TEST] 保存GT mesh用于可视化对比
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> verts_gt;
    SMPLForward<double>(beta_gt, theta_gt, trans_gt, model, verts_gt);
    SaveOBJ("mesh_ground_truth.obj", verts_gt, model.faces);

    // 生成优化后的2D重投影，评估拟合质量
    auto optimized_landmarks = GenerateSyntheticLandmarks(model, camera, beta_init, theta_init, trans_init);
    SaveLandmarksToFile("landmarks_optimized.txt", optimized_landmarks);

    // 保存相机参数供可视化脚本使用
    std::ofstream cam_file("camera_params.txt");
    cam_file << camera.fx << " " << camera.fy << " " << camera.cx << " " << camera.cy << "\n";
    cam_file << camera.width << " " << camera.height << "\n";
    cam_file.close();

    std::cout << "\n=== Saved visualization files ===" << std::endl;
    std::cout << "- landmarks_gt.txt (ground truth 2D landmarks)" << std::endl;
    std::cout << "- landmarks_optimized.txt (optimized 2D landmarks)" << std::endl;
    std::cout << "- mesh_before_opt.obj (initial mesh)" << std::endl;
    std::cout << "- mesh_after_opt.obj (optimized mesh)" << std::endl;
    std::cout << "- mesh_ground_truth.obj (ground truth mesh)" << std::endl;
    std::cout << "- camera_params.txt (camera intrinsics)" << std::endl;
}

}  // namespace

void SaveOBJ(const std::string& filename, 
             const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& vertices, 
             const Eigen::MatrixXi& faces) {
    
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    std::cout << "Saving OBJ to " << filename << "..." << std::endl;

    // 1. Write Vertices
    // OBJ format: "v x y z"
    for (int i = 0; i < vertices.rows(); ++i) {
        outfile << "v " << vertices(i, 0) << " " << vertices(i, 1) << " " << vertices(i, 2) << "\n";
    }

    // 2. Write Faces
    // OBJ format: "f v1 v2 v3"
    // Note: OBJ indices start at 1, but C++ arrays start at 0. So we add +1.
    for (int i = 0; i < faces.rows(); ++i) {
        outfile << "f " 
                << faces(i, 0) + 1 << " " 
                << faces(i, 1) + 1 << " " 
                << faces(i, 2) + 1 << "\n";
    }

    outfile.close();
    std::cout << "Saved successfully!" << std::endl;
}

int main() {
    google::InitGoogleLogging("SMPL_Optimizer");
    std::cout << ">>> Start SMPL Project..." << std::endl;

    // 1. Load Model
    SmplModel model;
    std::string npz_path = "../../Data/star_1_1/neutral/model.npz"; 
    
    if (!model.LoadFromNPZ(npz_path)) {
        return -1;
    }

    // 2. Prepare Inputs (T-Pose)
    double beta[10] = {0}; 
    double theta[72] = {0};
    double trans[3] = {0, 0, 0};

    // 3. Run Forward Pass
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out_verts;
    SMPLForward<double>(beta, theta, trans, model, out_verts);

    // 4. Save Result for Visualization
    // Save to the build folder (or any accessible path)
    SaveOBJ("output_tpose.obj", out_verts, model.faces);

    // 5. Run a synthetic fitting demo to verify the optimizer end-to-end
    RunSyntheticOptimizationDemo(model);

    return 0;
}
