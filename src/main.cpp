#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath> // For M_PI
#include "smpl_model.h"
#include "smpl_math.h"
#include "optimizer.h"
#include "amass_loader.h" 
#include <ceres/ceres.h>
#include <nlohmann/json.hpp> 
#include <cnpy.h>
#include <thread>

using json = nlohmann::json;

// Forward declarations
void SaveOBJ(const std::string& filename,
             const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vertices,
             const Eigen::MatrixXi& faces);

namespace {

// =========================================================
// [HELPER] Estimate Initial Z from Bounding Box Height
// =========================================================
void EstimateGlobalTranslation(
    const std::vector<Landmark2D>& landmarks,
    const CameraIntrinsics& camera,
    double* trans_init
) {
    if (landmarks.empty()) return;

    float min_y = 1e5, max_y = -1e5;
    float min_x = 1e5, max_x = -1e5;
    
    for (const auto& lm : landmarks) {
        if (lm.confidence > 0) {
            if (lm.y < min_y) min_y = lm.y;
            if (lm.y > max_y) max_y = lm.y;
            if (lm.x < min_x) min_x = lm.x;
            if (lm.x > max_x) max_x = lm.x;
        }
    }

    float pixel_height = max_y - min_y;
    float center_x = (min_x + max_x) / 2.0f;
    float center_y = (min_y + max_y) / 2.0f;

    // Safety check for tiny boxes
    if (pixel_height < 20.0f) {
        std::cerr << "[Init] Warning: Box too small. Defaulting to 3.0m" << std::endl;
        trans_init[2] = 3.0; 
        return; 
    }

    // Assume average human height ~1.6m - 1.7m
    float avg_human_height = 1.65f; 
    float estimated_z = (camera.fy * avg_human_height) / pixel_height;

    // Clamp Z to reasonable human interaction range (0.5m to 10m)
    if (estimated_z < 0.5) estimated_z = 0.5;
    if (estimated_z > 10.0) estimated_z = 10.0;

    // Back-project center
    float estimated_x = (center_x - camera.cx) * estimated_z / camera.fx;
    float estimated_y = (center_y - camera.cy) * estimated_z / camera.fy;

    trans_init[0] = estimated_x;
    trans_init[1] = estimated_y; 
    trans_init[2] = estimated_z;

    std::cout << "[Init] Auto-Estimated Translation: " 
              << "Z=" << estimated_z << "m (based on pixel height " << pixel_height << ")" << std::endl;
}

// [HELPER] Load Real OpenPose JSON
bool LoadLandmarksFromJSON(
    const std::string& json_path,
    std::vector<Landmark2D>& landmarks
) {
    landmarks.clear();
    std::ifstream infile(json_path);
    if (!infile.is_open()) return false;
    json j;
    try { infile >> j; } catch(...) { return false; }

    if (!j.contains("people") || j["people"].empty()) return false;
    const auto& keypoints = j["people"][0]["pose_keypoints_2d"];
    if (keypoints.size() < 75) return false;

    for (int joint_id = 0; joint_id < 25; ++joint_id) {
        float x = keypoints[3 * joint_id + 0].get<float>();
        float y = keypoints[3 * joint_id + 1].get<float>();
        float c = keypoints[3 * joint_id + 2].get<float>();
        if (c > 0.05f) landmarks.emplace_back(x, y, c, joint_id);
    }
    return true;
}

void FlipLandmarks(std::vector<Landmark2D>& landmarks, int width) {
        for (auto& lm : landmarks) {
            lm.x = width - 1 - lm.x;
        }
        std::cout << "[Info] Flipped landmarks horizontally." << std::endl;
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

// Helper: Synthetic Landmarks
static std::vector<Landmark2D> GenerateSyntheticLandmarks(
    const SmplModel& model,
    const CameraIntrinsics& camera,
    const double* beta,
    const double* theta,
    const double* trans
) {
    std::vector<Landmark2D> landmarks;
    landmarks.reserve(25);

    // 1) joints-only forward (same as residual)
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Jw;
    SMPLJointsForward<double>(beta, theta, trans, model, Jw);

    // 2) project OpenPose 25 joints
    for (int joint_id = 0; joint_id < 25; ++joint_id) {
        int smpl_joint_id = OPENPOSE_TO_SMPL[joint_id];
        if (smpl_joint_id < 0 || smpl_joint_id >= model.J) continue;

        const double X = Jw(smpl_joint_id, 0);
        const double Y = Jw(smpl_joint_id, 1);
        const double Z = Jw(smpl_joint_id, 2);

        // behind camera 
       double Zc = Z;
        if (Zc < 1e-6) Zc = 1e-6;  
        float px = static_cast<float>(camera.fx * (X / Zc) + camera.cx);
        float py = static_cast<float>(camera.fy * (Y / Zc) + camera.cy);
        landmarks.emplace_back(px, py, 1.0f, joint_id);
    }

    return landmarks;
}



void EigenToArray(const Eigen::VectorXd& src, double* dst, int size) { for(int i=0; i<size; ++i) dst[i] = src[i]; }
void EigenToArray3(const Eigen::Vector3d& src, double* dst) { dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; }

// =========================================================
// MODE 1: AMASS Validation (Analysis-by-Synthesis)
// =========================================================
void RunAMASSValidation(const SmplModel& model) {
    std::cout << "\n=== MODE 1: Running AMASS Validation (Real AMASS Data) ===\n";


    std::string amass_path = "../../Data/CMU/01/01_01_poses_formatted.txt";
    std::vector<AmassFrame> amass_frames;
    if (!AmassLoader::Load(amass_path, amass_frames)) return;
    if (amass_frames.empty()) return;

    int frame_idx = 100;
    if ((int)amass_frames.size() <= frame_idx) frame_idx = 0;

    const AmassFrame& frame_gt = amass_frames[frame_idx];

    double beta_gt[10];
    double theta_gt[72];
    double trans_gt[3];

    for (int i = 0; i < 10; ++i) beta_gt[i] = frame_gt.beta[i];
    for (int i = 0; i < 72; ++i) theta_gt[i] = frame_gt.theta[i];

    trans_gt[0] = 0.0;
    trans_gt[1] = 0.0;
    trans_gt[2] = 3.0;

    Eigen::Matrix<double, 72, 1> pose72;   // axis-angle, 72
    Eigen::Matrix<double, 10, 1> betas10;  // 10
    Eigen::Vector3d trans3;                // 3

    std::cout << "[GT] Using AMASS Frame " << frame_idx << " as Ground Truth.\n";

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_gt;
    SMPLJointsForward<double>(beta_gt, theta_gt, trans_gt, model, J_gt);
    std::cout << "[GT] J_gt = " << J_gt.rows() << "x" << J_gt.cols() << "\n";

    // Camera + synthetic observations
    CameraIntrinsics camera = CreateDefaultCamera(800, 800);

    auto observed_landmarks = GenerateSyntheticLandmarks(model, camera, beta_gt, theta_gt, trans_gt);
    if (observed_landmarks.empty()) {
        std::cerr << "[Error] No landmarks generated (likely behind camera / bad forward).\n";
        return;
    }
    SaveLandmarksToFile("landmarks_gt.txt", observed_landmarks); 

    // Save GT mesh
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> verts_gt;
    SMPLForward<double>(beta_gt, theta_gt, trans_gt, model, verts_gt);
    SaveOBJ("mesh_ground_truth.obj", verts_gt, model.faces);

    // init 
    double beta_opt[10]  = {0};
    double theta_opt[72] = {0};
    double trans_opt[3]  = {0.0, 0.0, 2.5};


    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V0;
    SMPLForward<double>(beta_opt, theta_opt, trans_opt, model, V0);
    SaveOBJ("mesh_before_opt.obj", V0, model.faces);

    // 3D optimizer config
    OptimizerConfig cfg;
    cfg.verbose = true;
    cfg.amass_unit_test = true;
    cfg.unit_test_exact = true;  
    cfg.w_3d = 1.0;

    bool ok = OptimizeSMPL3D(J_gt, model, beta_opt, theta_opt, trans_opt, cfg);
    if (!ok) {
        std::cerr << "[Error] OptimizeSMPL3D failed.\n";
        return;
    }

    // after
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V1;
    SMPLForward<double>(beta_opt, theta_opt, trans_opt, model, V1);
    SaveOBJ("mesh_after_opt.obj", V1, model.faces);

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_fit;
    SMPLJointsForward<double>(beta_opt, theta_opt, trans_opt, model, J_fit);
    std::cout << "[AMASS] mean joint L2 error = "
            << (J_fit - J_gt).rowwise().norm().mean() << " m\n";

    auto final_landmarks = GenerateSyntheticLandmarks(model, camera, beta_opt, theta_opt, trans_opt);
    SaveLandmarksToFile("landmarks_optimized.txt", final_landmarks);

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> verts_opt;
    SMPLForward<double>(beta_opt, theta_opt, trans_opt, model, verts_opt);
    SaveOBJ("mesh_after_opt.obj", verts_opt, model.faces);

    std::ofstream cam_file("camera_params.txt");
    cam_file << camera.fx << " " << camera.fy << " " << camera.cx << " " << camera.cy << "\n";
    cam_file << camera.width << " " << camera.height << "\n";
    cam_file.close();

    std::cout << "[Success] AMASS validation finished.\n";
}

// =========================================================
// MODE 2: Real OpenPose Video
// =========================================================
void RunOpenPoseInference(const SmplModel& model) {
    std::cout << "\n=== MODE 2: Running Real OpenPose Data (Fine-Tuned) ===" << std::endl;

    std::string json_path = "../../Data/OpenPose/output_jsons/video_000000000000_keypoints.json";
    
    std::vector<Landmark2D> real_landmarks;
    if (!LoadLandmarksFromJSON(json_path, real_landmarks)) {
        std::cerr << "Failed to load JSON." << std::endl;
        return;
    }

    // 1. Setup Camera
    int img_w = 1280;
    int img_h = 720;
    CameraIntrinsics camera = CreateDefaultCamera(img_w, img_h); 

    // [Step 1] Flip 设为 false 
    bool FLIP_INPUT = false; 
    if (FLIP_INPUT) {
        FlipLandmarks(real_landmarks, img_w);
    }
    SaveLandmarksToFile("landmarks_gt.txt", real_landmarks); 

    // 2. Initialization
    double beta_opt[10] = {0.0};
    double theta_opt[72] = {0.0}; // Default T-Pose
    double trans_opt[3] = {0.0, 0.0, 3.0}; // [Step 2] Hardcode Start Z = 3.0m

    // [Step 3] 更好的姿态初始化 (Smart Initialization)
    theta_opt[0] = 3.1415926; 
    theta_opt[16*3 + 2] = -0.8; 
    theta_opt[17*3 + 2] = 0.8;

    // [Step 4] Run Auto-Estimator just to center X/Y, but FORCE Z back to 3.0
    EstimateGlobalTranslation(real_landmarks, camera, trans_opt);
    std::cout << "[FineTune] Overriding Z-Estimate. Forcing Start Z = 3.0m" << std::endl;
    trans_opt[2] = 3.0; 

    // Save Initial Mesh
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> verts_init;
    SMPLForward<double>(beta_opt, theta_opt, trans_opt, model, verts_init);
    SaveOBJ("mesh_before_opt.obj", verts_init, model.faces);


    // 3. Optimize
    OptimizerConfig config;
    config.verbose = true;
    
    if(OptimizeSMPL(real_landmarks, camera, model, beta_opt, theta_opt, trans_opt, config)) {
        // Save Result
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> verts_opt;
        SMPLForward<double>(beta_opt, theta_opt, trans_opt, model, verts_opt);
        SaveOBJ("mesh_after_opt.obj", verts_opt, model.faces);
        SaveOBJ("mesh_ground_truth.obj", verts_opt, model.faces); 

        auto final_landmarks = GenerateSyntheticLandmarks(model, camera, beta_opt, theta_opt, trans_opt);
        SaveLandmarksToFile("landmarks_optimized.txt", final_landmarks);
    }
}

} // namespace

void SaveOBJ(const std::string& filename,
             const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vertices,
             const Eigen::MatrixXi& faces) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) return;

    for (int i = 0; i < vertices.rows(); ++i) {
        outfile << "v " << vertices(i, 0) << " " << vertices(i, 1) << " " << vertices(i, 2) << "\n";
    }
    for (int i = 0; i < faces.rows(); ++i) {
        outfile << "f " << faces(i, 0) + 1 << " " << faces(i, 1) + 1 << " " << faces(i, 2) + 1 << "\n";
    }
}

int main() {
    google::InitGoogleLogging("SMPL_Optimizer");
    std::cout << ">>> Start SMPL Project..." << std::endl;

    SmplModel model;
    model.LoadFromNPZ("../../Data/models/smpl_neutral.npz");

    // Mode 1: Validation (Check if algorithm works on AMASS)
    RunAMASSValidation(model);

    // Mode 2: Real Data (Uncomment to run)
    //RunOpenPoseInference(model);

    return 0;
}