#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <cmath> // For M_PI
#include <cstdlib>
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

// Global GMM pose prior (loaded once, used by all functions)
static GMMPosePrior g_gmm_prior;

namespace {

static void EstimateGlobalTranslationXY(
    const std::vector<Landmark2D>& landmarks,
    const CameraIntrinsics& camera,
    double z_assumed,
    double* trans_init
) {
    float min_y = 1e9f, max_y = -1e9f;
    float min_x = 1e9f, max_x = -1e9f;

    for (const auto& lm : landmarks) {
        if (lm.confidence < 0.2f) continue;
        min_x = std::min(min_x, lm.x);
        max_x = std::max(max_x, lm.x);
        min_y = std::min(min_y, lm.y);
        max_y = std::max(max_y, lm.y);
    }

    const float center_x = 0.5f * (min_x + max_x);
    const float center_y = 0.5f * (min_y + max_y);

    // backproject center using assumed z
    trans_init[0] = (center_x - camera.cx) * z_assumed / camera.fx;

    trans_init[1] = -(center_y - camera.cy) * z_assumed / camera.fy;

    trans_init[2] = z_assumed;

    std::cout << "[Init] trans init = (" << trans_init[0] << ", " << trans_init[1] << ", " << trans_init[2] << ")\n";
}


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

    const auto& people = j["people"];

    // ---- pick best person ----
    int best_idx = -1;
    double best_score = -1.0;

    for (int i = 0; i < (int)people.size(); ++i) {
        if (!people[i].contains("pose_keypoints_2d")) continue;
        const auto& kp = people[i]["pose_keypoints_2d"];
        if (kp.size() < 75) continue;

        float min_x = 1e9f, max_x = -1e9f;
        float min_y = 1e9f, max_y = -1e9f;
        double sum_c = 0.0;
        int cnt = 0;

        // Use a slightly higher threshold for scoring robustness
        const float conf_thr = 0.20f;

        for (int jnt = 0; jnt < 25; ++jnt) {
            float x = kp[3 * jnt + 0].get<float>();
            float y = kp[3 * jnt + 1].get<float>();
            float c = kp[3 * jnt + 2].get<float>();
            if (c < conf_thr) continue;

            min_x = std::min(min_x, x); max_x = std::max(max_x, x);
            min_y = std::min(min_y, y); max_y = std::max(max_y, y);

            sum_c += c;
            cnt++;
        }

        if (cnt < 6) continue; // too few joints => likely noise / partial

        const float w = std::max(1.0f, max_x - min_x);
        const float h = std::max(1.0f, max_y - min_y);
        const double area = (double)w * (double)h;
        const double mean_c = sum_c / (double)cnt;

        // score: prefer large bbox (full body), many joints, and high confidence
        const double score = area * mean_c * (double)cnt;

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx < 0) {
        // fallback: use the first person if all scores failed
        best_idx = 0;
        if (!people[best_idx].contains("pose_keypoints_2d")) return false;
        if (people[best_idx]["pose_keypoints_2d"].size() < 75) return false;
    }

    const auto& keypoints = people[best_idx]["pose_keypoints_2d"];

    // ---- output landmarks (keep your original low threshold for optimization) ----
    for (int joint_id = 0; joint_id < 25; ++joint_id) {
        float x = keypoints[3 * joint_id + 0].get<float>();
        float y = keypoints[3 * joint_id + 1].get<float>();
        float c = keypoints[3 * joint_id + 2].get<float>();

        // Keep as you had it (0.05) so optimizer can decide to ignore low conf by cfg.min_confidence
        if (c > 0.05f) landmarks.emplace_back(x, y, c, joint_id);
    }

    std::cout << "[OpenPose] picked person index = " << best_idx
              << " / " << people.size()
              << " (score=" << best_score << ")\n";

    return !landmarks.empty();
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

static std::vector<Landmark2D> GenerateSyntheticLandmarks(
    const SmplModel& model,
    const CameraIntrinsics& camera,
    const double* beta,
    const double* theta,
    const double* trans,
    bool use_pose_blendshapes
) {
    std::vector<Landmark2D> landmarks;
    landmarks.reserve(25);

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Jw;
    SMPLJointsForward<double>(beta, theta, trans, model, Jw);

    for (int joint_id = 0; joint_id < 25; ++joint_id) {
        int smpl_joint_id = OPENPOSE_TO_SMPL[joint_id];
        if (smpl_joint_id < 0 || smpl_joint_id >= model.J) continue;

        const double X = Jw(smpl_joint_id, 0);
        const double Y = Jw(smpl_joint_id, 1);
        double Zc = Jw(smpl_joint_id, 2);
        if (Zc < 1e-6) Zc = 1e-6;

        float px = float(camera.fx * (X / Zc) + camera.cx);
        float py = float(camera.fy * (-Y / Zc) + camera.cy);  // flip Y to match optimizer

        landmarks.emplace_back(px, py, 1.0f, joint_id);
    }
    return landmarks;
}

void EigenToArray(const Eigen::VectorXd& src, double* dst, int size) { for(int i=0; i<size; ++i) dst[i] = src[i]; }
void EigenToArray3(const Eigen::Vector3d& src, double* dst) { dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; }

// =========================================================
// MODE 1: AMASS Validation (Analysis-by-Synthesis)
// Returns mean joint error in meters, or -1 on failure
// =========================================================
double RunAMASSValidation(const SmplModel& model, const std::string& amass_path, int frame_idx = -1, bool save_mesh = true) {
    std::cout << "\n=== AMASS Validation: " << amass_path << " ===\n";

    std::vector<AmassFrame> amass_frames;
    if (!AmassLoader::Load(amass_path, amass_frames)) return -1.0;
    if (amass_frames.empty()) return -1.0;

    // frame_idx = -1 表示自动选取 70% 位置的帧（动作更丰富）
    if (frame_idx < 0) {
        frame_idx = (int)(amass_frames.size() * 0.7);
    }
    // 边界检查
    if (frame_idx >= (int)amass_frames.size()) {
        frame_idx = (int)amass_frames.size() - 1;
    }

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

    // Generate GT depth data (Z coordinate of each joint)
    std::vector<JointDepthData> depth_gt;
    for (int i = 0; i < 24; i++) {
        depth_gt.push_back({i, J_gt(i, 2)});
    }
    std::cout << "[GT] Generated depth data for " << depth_gt.size() << " joints.\n";

    // Camera + synthetic observations
    CameraIntrinsics camera = CreateDefaultCamera(800, 800);

    auto observed_landmarks = GenerateSyntheticLandmarks(model, camera, beta_gt, theta_gt, trans_gt , false);
    if (observed_landmarks.empty()) {
        std::cerr << "[Error] No landmarks generated.\n";
        return -1.0;
    }

    // 从路径提取文件名作为前缀 (e.g. "CMU_91_91_33_poses")
    std::string base_name;
    {
        size_t last_slash = amass_path.find_last_of("/\\");
        std::string filename = (last_slash != std::string::npos) ? amass_path.substr(last_slash + 1) : amass_path;
        size_t dot_pos = filename.find_last_of('.');
        base_name = (dot_pos != std::string::npos) ? filename.substr(0, dot_pos) : filename;
    }

    // Save GT mesh (optional)
    if (save_mesh) {
        SaveLandmarksToFile(base_name + "_landmarks_gt.txt", observed_landmarks);
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> verts_gt;
        SMPLForward<double>(beta_gt, theta_gt, trans_gt, model, verts_gt);
        SaveOBJ(base_name + "_mesh_gt.obj", verts_gt, model.faces);
    }

    // init
    double beta_opt[10]  = {0};
    double theta_opt[72] = {0};
    double trans_opt[3]  = {0.0, 0.0, 2.5};

    if (save_mesh) {
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V0;
        SMPLForward<double>(beta_opt, theta_opt, trans_opt, model, V0);
        SaveOBJ(base_name + "_mesh_init.obj", V0, model.faces);
    }

    // Optimizer config
    OptimizerConfig cfg;
    cfg.verbose = false;  // 批量测试时关闭详细输出
    cfg.amass_unit_test = true;
    cfg.unit_test_exact = true;
    cfg.w_3d = 1.0;

    // Enable depth constraint for 2D optimization
    cfg.use_depth = true;
    cfg.w_depth = 50.0;
    cfg.depth_data = &depth_gt;

    // Enable GMM pose prior
    if (g_gmm_prior.loaded) {
        cfg.use_gmm_prior = true;
        cfg.w_gmm = 1.0;
        cfg.gmm_prior = &g_gmm_prior;
        std::cout << "[Config] Using GMM pose prior\n";
    } else {
        std::cout << "[Config] Using L2 pose prior (GMM not loaded)\n";
    }

    bool use_3d_optimization = false;

    if (use_3d_optimization) {
        bool ok = OptimizeSMPL3D(J_gt, model, beta_opt, theta_opt, trans_opt, cfg);
        if (!ok) {
            std::cerr << "[Error] OptimizeSMPL3D failed.\n";
            return -1.0;
        }
    } else {
        bool ok = OptimizeSMPL(observed_landmarks, camera, model, beta_opt, theta_opt, trans_opt, cfg);
        if (!ok) {
            std::cerr << "[Error] OptimizeSMPL failed.\n";
            return -1.0;
        }
    }

    // Compute error
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_fit;
    SMPLJointsForward<double>(beta_opt, theta_opt, trans_opt, model, J_fit);

    double mean_error = (J_fit - J_gt).rowwise().norm().mean();
    std::cout << "[Result] " << base_name << " frame " << frame_idx << " -> error = " << mean_error << " m\n";

    if (save_mesh) {
        auto final_landmarks = GenerateSyntheticLandmarks(model, camera, beta_opt, theta_opt, trans_opt, true);
        SaveLandmarksToFile(base_name + "_landmarks_opt.txt", final_landmarks);

        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> verts_opt;
        SMPLForward<double>(beta_opt, theta_opt, trans_opt, model, verts_opt);
        SaveOBJ(base_name + "_mesh_opt.obj", verts_opt, model.faces);

        std::ofstream cam_file(base_name + "_camera.txt");
        cam_file << camera.fx << " " << camera.fy << " " << camera.cx << " " << camera.cy << "\n";
        cam_file << camera.width << " " << camera.height << "\n";
        cam_file.close();
    }

    return mean_error;
}

// =========================================================
// MODE 2: Real OpenPose Video
// =========================================================
void RunOpenPoseInference(const SmplModel& model) {
    std::cout << "\n=== MODE 2: Running Real OpenPose Data (Fine-Tuned) ===" << std::endl;

    std::string json_path = "../../Data/OpenPose/my_openpose/output_jsons/dance_000000000120_keypoints.json";
    std::vector<Landmark2D> lms;
    if (!LoadLandmarksFromJSON(json_path, lms)) {
        std::cerr << "Failed to load json.\n";
        return;
    }
    SaveLandmarksToFile("landmarks_input.txt", lms);

    // -------- Camera --------
    const int img_w = 1280;
    const int img_h = 720;
    CameraIntrinsics camera = CreateDefaultCamera(img_w, img_h);

    // -------- Init params --------
    double beta[10]  = {0.0};
    double theta[72] = {0.0};
    double trans[3]  = {0.0, 0.0, 3.0};

    // 轻微初值（可选）：让手臂不要完全T pose（你也可以先注释）
    // theta[16*3 + 2] = -0.4;
    // theta[17*3 + 2] =  0.4;

    // init trans x/y with bbox center, keep z=3.0
    EstimateGlobalTranslationXY(lms, camera, /*z_assumed=*/3.0, trans);

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_init;
    SMPLForward<double>(beta, theta, trans, model, V_init);
    SaveOBJ("mesh_before_opt.obj", V_init, model.faces);

    // -------- Optimize --------
    OptimizerConfig cfg;
    cfg.verbose = true;
    cfg.min_confidence = 0.20f;
    cfg.use_stable_joints_only = true;    
    cfg.flip_y_in_projection = true;     
    cfg.root_free_axis = 1;               

    const bool ok = OptimizeSMPL(lms, camera, model, beta, theta, trans, cfg);
    if (!ok) {
        std::cerr << "OptimizeSMPL failed.\n";
        return;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_opt;
    SMPLForward<double>(beta, theta, trans, model, V_opt);
    SaveOBJ("mesh_after_opt.obj", V_opt, model.faces);

    std::cout << "[Done] saved mesh_after_opt.obj\n";
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

void RunBatchValidation(const SmplModel& model) {
    std::cout << "\n========== BATCH AMASS VALIDATION (Most Complex Frames) ==========\n";

    std::vector<std::pair<std::string, int>> files = {
        {"../../Data/sampled/CMU_80_80_33_poses.txt", 1303},  // complexity: 3.71
        {"../../Data/sampled/CMU_80_80_40_poses.txt", 504},   // complexity: 3.42
        {"../../Data/sampled/CMU_82_82_06_poses.txt", 250},   // complexity: 3.81
        {"../../Data/sampled/CMU_82_82_12_poses.txt", 117},   // complexity: 2.41
        {"../../Data/sampled/CMU_83_83_19_poses.txt", 342},   // complexity: 2.36
        // {"../../Data/sampled/CMU_83_83_21_poses.txt", 316},   // complexity: 2.53
        // {"../../Data/sampled/CMU_83_83_26_poses.txt", 200},   // complexity: 2.81
        // {"../../Data/sampled/CMU_83_83_31_poses.txt", 658},   // complexity: 2.85
        // {"../../Data/sampled/CMU_83_83_34_poses.txt", 578},   // complexity: 2.84
        // {"../../Data/sampled/CMU_83_83_51_poses.txt", 295},   // complexity: 2.72
        // {"../../Data/sampled/CMU_83_83_63_poses.txt", 456},   // complexity: 2.57
        // {"../../Data/sampled/CMU_83_83_67_poses.txt", 371},   // complexity: 2.33
        // {"../../Data/sampled/CMU_83_83_68_poses.txt", 383},   // complexity: 2.26
        // {"../../Data/sampled/CMU_85_85_07_poses.txt", 689},   // complexity: 4.65 (最复杂!)
        // {"../../Data/sampled/CMU_91_91_01_poses.txt", 1954},  // complexity: 2.32
        // {"../../Data/sampled/CMU_91_91_06_poses.txt", 65},    // complexity: 2.77
        // {"../../Data/sampled/CMU_91_91_13_poses.txt", 984},   // complexity: 2.34
        // {"../../Data/sampled/CMU_91_91_33_poses.txt", 1159},  // complexity: 3.16
        // {"../../Data/sampled/CMU_91_91_60_poses.txt", 149},   // complexity: 2.93
        // {"../../Data/sampled/CMU_91_91_62_poses.txt", 1015}   // complexity: 2.23
    };

    std::vector<double> errors;
    int success = 0, fail = 0;

    for (const auto& item : files) {
        double err = RunAMASSValidation(model, item.first, item.second, /*save_mesh=*/true);
        if (err >= 0) {
            errors.push_back(err);
            success++;
        } else {
            fail++;
        }
    }

    std::cout << "\n========== BATCH RESULTS ==========\n";
    std::cout << "Success: " << success << " / " << files.size() << "\n";
    std::cout << "Failed:  " << fail << "\n";

    if (!errors.empty()) {
        double sum = 0, min_e = errors[0], max_e = errors[0];
        for (double e : errors) {
            sum += e;
            min_e = std::min(min_e, e);
            max_e = std::max(max_e, e);
        }
        double mean = sum / errors.size();

        std::cout << "Mean error: " << mean << " m\n";
        std::cout << "Min error:  " << min_e << " m\n";
        std::cout << "Max error:  " << max_e << " m\n";
    }
}

int main(int argc, char** argv) {
    google::InitGoogleLogging("SMPL_Optimizer");
    std::cout << ">>> Start SMPL Project..." << std::endl;

    SmplModel model;
    model.LoadFromNPZ("../../Data/models/smpl_neutral.npz");
    g_gmm_prior.LoadFromNPZ("../../Data/models/gmm_08.npz");

    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "--batch") {
            RunBatchValidation(model);
        } else {
            RunAMASSValidation(model, arg1, /*frame_idx=*/100, /*save_mesh=*/true);
        }
    } else {
        RunBatchValidation(model);
    }

    // Mode 2: Real Data (Uncomment to run)
    //RunOpenPoseInference(model);

    return 0;
}
