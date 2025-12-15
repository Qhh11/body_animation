#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include "smpl_model.h"
#include "smpl_math.h"
#include "optimizer.h"
#include "amass_loader.h" 
#include <ceres/ceres.h>

// Forward declaration
void SaveOBJ(const std::string& filename,
             const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& vertices,
             const Eigen::MatrixXi& faces);

namespace {

// Compute 3D joint position from vertices
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

// Generate synthetic 2D landmarks from 3D model parameters
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
        
        // Safety check: ensure point is in front of camera
        double Z = joint.z();
        if (Z <= 0.1) continue; // Skip points behind camera

        float projected_x = static_cast<float>(camera.fx * (joint.x() / Z) + camera.cx);
        float projected_y = static_cast<float>(camera.fy * (joint.y() / Z) + camera.cy);

        // Check if point is roughly within image bounds (optional)
        // if (projected_x < 0 || projected_x > camera.width || projected_y < 0 || projected_y > camera.height) continue;

        // Add small deterministic noise to simulate real detection error
        float noise_u = (joint_id % 2 == 0) ? 0.5f : -0.5f;
        float noise_v = (joint_id % 3 == 0) ? -0.5f : 0.5f;

        landmarks.emplace_back(projected_x + noise_u, projected_y + noise_v, 0.95f, joint_id);
    }
    return landmarks;
}

void SaveLandmarksToFile(const std::string& filename, const std::vector<Landmark2D>& landmarks) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) return;
    for (const auto& lm : landmarks) {
        outfile << lm.joint_id << " " << lm.x << " " << lm.y << " " << lm.confidence << "\n";
    }
    outfile.close();
}

// [NEW] Helper to copy Eigen vector to C-array
void EigenToArray(const Eigen::VectorXd& src, double* dst, int size) {
    for(int i=0; i<size; ++i) dst[i] = src[i];
}

void EigenToArray3(const Eigen::Vector3d& src, double* dst) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}

// [NEW] Run Optimization using AMASS data as Ground Truth
void RunAMASSValidation(const SmplModel& model) {
    std::cout << "\n=== Running AMASS Validation Pipeline ===" << std::endl;

    // 1. Load AMASS Data
    // Update this path to your actual downloaded AMASS .npz file
    std::string amass_path = "../../Data/CMU/01/01_01_poses_formatted.txt";
    std::vector<AmassFrame> amass_frames;
    
    if (!AmassLoader::Load(amass_path, amass_frames)) {
        std::cerr << "[Error] Failed to load AMASS file. Skipping demo." << std::endl;
        return;
    }

    if (amass_frames.empty()) return;

    // 2. Pick a frame (e.g., frame 0)
    const AmassFrame& frame_gt = amass_frames[0];

    // 3. Convert GT data to arrays
    double beta_gt[10];
    double theta_gt[72];
    double trans_gt[3];

    EigenToArray(frame_gt.beta, beta_gt, 10);
    EigenToArray(frame_gt.theta, theta_gt, 72);
    EigenToArray3(frame_gt.trans, trans_gt);

    // [CRITICAL TRICK]
    // AMASS data usually puts the person at (0,0,0) or on the ground.
    // If our camera is at (0,0,0), the person might be invisible or behind the camera.
    // We forcefully move the person 3 meters away so they are visible in our virtual camera.
    trans_gt[0] = 0.0; // Center X
    trans_gt[1] = 0.0; // Center Y
    trans_gt[2] = 3.0; // Move to Z = 3.0m

    std::cout << "[GT] Using AMASS Frame 0 (translated to Z=3.0m)" << std::endl;

    // 4. Setup Virtual Camera
    CameraIntrinsics camera = CreateDefaultCamera(800, 800);

    // 5. Generate "Observed" 2D Landmarks from AMASS GT
    auto observed_landmarks = GenerateSyntheticLandmarks(model, camera, beta_gt, theta_gt, trans_gt);
    SaveLandmarksToFile("landmarks_gt.txt", observed_landmarks);

    // 6. Save Ground Truth Mesh
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> verts_gt;
    SMPLForward<double>(beta_gt, theta_gt, trans_gt, model, verts_gt);
    SaveOBJ("mesh_ground_truth.obj", verts_gt, model.faces);

    // 7. Initialize Optimization (Reset to Zero/Mean Pose)
    double beta_opt[10] = {0.0};
    double theta_opt[72] = {0.0};
    double trans_opt[3] = {0.0, 0.0, 2.5}; // Init guess: 2.5m depth

    // Save Initial Mesh
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> verts_init;
    SMPLForward<double>(beta_opt, theta_opt, trans_opt, model, verts_init);
    SaveOBJ("mesh_initial.obj", verts_init, model.faces);

    // 8. Run Optimization
    OptimizerConfig config;
    config.verbose = true;
    config.max_iterations = 50;

    bool success = OptimizeSMPL(observed_landmarks, camera, model, beta_opt, theta_opt, trans_opt, config);

    if (success) {
        std::cout << "[Success] Optimization converged!" << std::endl;
        
        // Save Optimized Mesh
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> verts_opt;
        SMPLForward<double>(beta_opt, theta_opt, trans_opt, model, verts_opt);
        SaveOBJ("mesh_optimized.obj", verts_opt, model.faces);
        
        // Generate projected 2D points for visualization
        auto final_landmarks = GenerateSyntheticLandmarks(model, camera, beta_opt, theta_opt, trans_opt);
        SaveLandmarksToFile("landmarks_optimized.txt", final_landmarks);
        
        // Save Camera Params
        std::ofstream cam_file("camera_params.txt");
        cam_file << camera.fx << " " << camera.fy << " " << camera.cx << " " << camera.cy << "\n";
        cam_file << camera.width << " " << camera.height << "\n";
        cam_file.close();
    } else {
        std::cerr << "[Failure] Optimization did not converge." << std::endl;
    }
}

} // namespace

void SaveOBJ(const std::string& filename, 
             const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& vertices, 
             const Eigen::MatrixXi& faces) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return;
    }
    std::cout << "Saving OBJ to " << filename << "..." << std::endl;
    for (int i = 0; i < vertices.rows(); ++i) {
        outfile << "v " << vertices(i, 0) << " " << vertices(i, 1) << " " << vertices(i, 2) << "\n";
    }
    for (int i = 0; i < faces.rows(); ++i) {
        outfile << "f " << faces(i, 0) + 1 << " " << faces(i, 1) + 1 << " " << faces(i, 2) + 1 << "\n";
    }
    outfile.close();
}

int main() {
    google::InitGoogleLogging("SMPL_Optimizer");
    std::cout << ">>> Start SMPL Project..." << std::endl;

    // 1. Load SMPL Model
    SmplModel model;
    // Ensure this path matches your docker setup
    std::string npz_path = "../../Data/star_1_1/neutral/model.npz"; 
    
    if (!model.LoadFromNPZ(npz_path)) {
        return -1;
    }

    // 2. Run AMASS Validation Pipeline
    // This replaces the old RunSyntheticOptimizationDemo
    RunAMASSValidation(model);

    return 0;
}