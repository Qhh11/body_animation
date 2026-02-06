#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>
#include <thread>

#include <nlohmann/json.hpp>
#include <cnpy.h>
#include <ceres/ceres.h>

#include "smpl_model.h"
#include "smpl_math.h"
#include "optimizer.h"
#include "amass_loader.h"
#include "evaluation.h"
#include <filesystem>
namespace fs = std::filesystem;

using json = nlohmann::json;

// Forward declarations
void SaveOBJ(const std::string& filename,
             const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vertices,
             const Eigen::MatrixXi& faces);

// Global GMM pose prior
static GMMPosePrior g_gmm_prior;

// ============================================================
// 3DPW Ground Truth Loader (Full version with poses, betas, vertices)
// ============================================================
struct ThreeDPW_GT {
    std::vector<Eigen::Matrix<double, 24, 3, Eigen::RowMajor>> joint_positions;  // per-frame GT joints (camera space)
    std::vector<Eigen::Matrix<double, 24, 3, Eigen::RowMajor>> joint_positions_world;  // per-frame GT joints (world space)
    std::vector<Eigen::Matrix<double, 72, 1>> poses;   // per-frame GT theta (72 params, camera space)
    std::vector<Eigen::Matrix<double, 72, 1>> poses_world;   // per-frame GT theta (world space)
    std::vector<Eigen::Vector3d> trans_world;  // per-frame GT translation (world space)
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> vertices;  // per-frame GT vertices (dynamic to avoid stack overflow)
    Eigen::Matrix<double, 10, 1> betas;  // shared shape params
    bool has_poses = false;
    bool has_betas = false;
    bool has_vertices = false;
    bool has_world_data = false;  // true if world-space data is loaded
    bool loaded = false;
    int num_frames = 0;

    bool LoadFromTxt(const std::string& txt_path) {
        std::ifstream f(txt_path);
        if (!f.is_open()) return false;

        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            Eigen::Matrix<double, 24, 3, Eigen::RowMajor> joints;
            for (int j = 0; j < 24; ++j) {
                iss >> joints(j, 0) >> joints(j, 1) >> joints(j, 2);
            }
            joint_positions.push_back(joints);
        }
        num_frames = (int)joint_positions.size();
        loaded = num_frames > 0;
        return loaded;
    }

    // Load full GT from directory (joints, poses, betas, vertices)
    bool LoadFromDir(const std::string& gt_dir) {
        // Load joints (required)
        std::string joints_path = gt_dir + "/gt_joints.txt";
        if (!LoadFromTxt(joints_path)) {
            // Fallback to single file path
            if (!LoadFromTxt(gt_dir)) return false;
        }

        // Load world-space joints (for world coordinate optimization)
        std::string joints_world_path = gt_dir + "/gt_joints_world.txt";
        std::ifstream jwf(joints_world_path);
        if (jwf.is_open()) {
            std::string line;
            while (std::getline(jwf, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                Eigen::Matrix<double, 24, 3, Eigen::RowMajor> joints;
                for (int j = 0; j < 24; ++j) {
                    iss >> joints(j, 0) >> joints(j, 1) >> joints(j, 2);
                }
                joint_positions_world.push_back(joints);
            }
            if (joint_positions_world.size() == (size_t)num_frames) {
                has_world_data = true;
                std::cout << "[GT] Loaded world-space joints\n";
            }
        }

        // Load poses (optional)
        std::string poses_path = gt_dir + "/gt_poses.txt";
        std::ifstream pf(poses_path);
        if (pf.is_open()) {
            std::string line;
            while (std::getline(pf, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                Eigen::Matrix<double, 72, 1> pose;
                for (int i = 0; i < 72; ++i) iss >> pose(i);
                poses.push_back(pose);
            }
            has_poses = (poses.size() == (size_t)num_frames);
            if (has_poses) std::cout << "[GT] Loaded poses\n";
        }

        // Load world-space poses (optional)
        std::string poses_world_path = gt_dir + "/gt_poses_world.txt";
        std::ifstream pwf(poses_world_path);
        if (pwf.is_open()) {
            std::string line;
            while (std::getline(pwf, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                Eigen::Matrix<double, 72, 1> pose;
                for (int i = 0; i < 72; ++i) iss >> pose(i);
                poses_world.push_back(pose);
            }
            if (poses_world.size() == (size_t)num_frames) {
                std::cout << "[GT] Loaded world-space poses\n";
            }
        }

        // Load world-space translation (optional)
        std::string trans_world_path = gt_dir + "/gt_trans_world.txt";
        std::ifstream twf(trans_world_path);
        if (twf.is_open()) {
            std::string line;
            while (std::getline(twf, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                Eigen::Vector3d t;
                iss >> t(0) >> t(1) >> t(2);
                trans_world.push_back(t);
            }
            if (trans_world.size() == (size_t)num_frames) {
                std::cout << "[GT] Loaded world-space translation\n";
            }
        }

        // Load betas (optional)
        std::string betas_path = gt_dir + "/gt_betas.txt";
        std::ifstream bf(betas_path);
        if (bf.is_open()) {
            for (int i = 0; i < 10; ++i) bf >> betas(i);
            has_betas = true;
            std::cout << "[GT] Loaded betas\n";
        }

        // Load vertices (optional, can be large)
        std::string verts_path = gt_dir + "/gt_vertices.txt";
        std::ifstream vf(verts_path);
        if (vf.is_open()) {
            std::string line;
            while (std::getline(vf, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> verts(6890, 3);
                for (int i = 0; i < 6890; ++i) {
                    iss >> verts(i, 0) >> verts(i, 1) >> verts(i, 2);
                }
                vertices.push_back(std::move(verts));
            }
            has_vertices = (vertices.size() == (size_t)num_frames);
            if (has_vertices) std::cout << "[GT] Loaded vertices\n";
        }

        return loaded;
    }

    // Get GT joints for a specific frame
    bool GetJoints(int frame_idx, Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt) const {
        if (!loaded || frame_idx < 0 || frame_idx >= num_frames) return false;
        J_gt = joint_positions[frame_idx];
        return true;
    }

    // Get GT pose for a specific frame
    bool GetPose(int frame_idx, double* theta_gt) const {
        if (!has_poses || frame_idx < 0 || frame_idx >= (int)poses.size()) return false;
        for (int i = 0; i < 72; ++i) theta_gt[i] = poses[frame_idx](i);
        return true;
    }

    // Get GT betas
    bool GetBetas(double* beta_gt) const {
        if (!has_betas) return false;
        for (int i = 0; i < 10; ++i) beta_gt[i] = betas(i);
        return true;
    }

    // Get GT vertices for a specific frame
    bool GetVertices(int frame_idx, Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V_gt) const {
        if (!has_vertices || frame_idx < 0 || frame_idx >= (int)vertices.size()) return false;
        V_gt = vertices[frame_idx];
        return true;
    }

    // Get GT joints in world space for a specific frame
    bool GetJointsWorld(int frame_idx, Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt) const {
        if (!has_world_data || frame_idx < 0 || frame_idx >= (int)joint_positions_world.size()) return false;
        J_gt = joint_positions_world[frame_idx];
        return true;
    }

    // Get GT pose in world space for a specific frame
    bool GetPoseWorld(int frame_idx, double* theta_gt) const {
        if (frame_idx < 0 || frame_idx >= (int)poses_world.size()) return false;
        for (int i = 0; i < 72; ++i) theta_gt[i] = poses_world[frame_idx](i);
        return true;
    }

    // Get GT translation in world space for a specific frame
    bool GetTransWorld(int frame_idx, double* trans_gt) const {
        if (frame_idx < 0 || frame_idx >= (int)trans_world.size()) return false;
        trans_gt[0] = trans_world[frame_idx](0);
        trans_gt[1] = trans_world[frame_idx](1);
        trans_gt[2] = trans_world[frame_idx](2);
        return true;
    }
};

namespace {

// ------------------------------
// Init helper: estimate trans from bbox center with assumed z
// ------------------------------
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

    trans_init[0] = (center_x - camera.cx) * z_assumed / camera.fx;
    trans_init[1] = -(center_y - camera.cy) * z_assumed / camera.fy;
    trans_init[2] = z_assumed;

    std::cout << "[Init] trans init = (" << trans_init[0] << ", " << trans_init[1] << ", " << trans_init[2] << ")\n";
}

// ------------------------------
// OpenPose json loader 
// ------------------------------
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

        if (cnt < 6) continue;

        const float w = std::max(1.0f, max_x - min_x);
        const float h = std::max(1.0f, max_y - min_y);
        const double area = (double)w * (double)h;
        const double mean_c = sum_c / (double)cnt;

        const double score = area * mean_c * (double)cnt;

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx < 0) {
        best_idx = 0;
        if (!people[best_idx].contains("pose_keypoints_2d")) return false;
        if (people[best_idx]["pose_keypoints_2d"].size() < 75) return false;
    }

    const auto& keypoints = people[best_idx]["pose_keypoints_2d"];

    for (int joint_id = 0; joint_id < 25; ++joint_id) {
        float x = keypoints[3 * joint_id + 0].get<float>();
        float y = keypoints[3 * joint_id + 1].get<float>();
        float c = keypoints[3 * joint_id + 2].get<float>();
        if (c > 0.05f) landmarks.emplace_back(x, y, c, joint_id);
    }

    //std::cout << "[OpenPose] picked person index = " << best_idx<< " / " << people.size()<< " (score=" << best_score << ")\n";

    return !landmarks.empty();
}

static std::string MakeOpenPoseJsonPath(const std::string& dir, int frame) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/image_%05d_keypoints.json", dir.c_str(), frame);
    return std::string(buf);
}
static std::string MakeOpenPoseDepthPath(const std::string& dir, int frame) {
    char buf1[512], buf2[512];
    std::snprintf(buf1, sizeof(buf1), "%s/image_%05d_depth.txt", dir.c_str(), frame);
    std::snprintf(buf2, sizeof(buf2), "%s/image_%05d.txt", dir.c_str(), frame);

    if (fs::exists(buf1)) return std::string(buf1);
    if (fs::exists(buf2)) return std::string(buf2);
    return std::string(buf1); 
}

static std::string MakeOutObj(const std::string& out_dir, int frame) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/mesh_%05d.obj", out_dir.c_str(), frame);
    return std::string(buf);
}

void SaveLandmarksToFile(const std::string& filename, const std::vector<Landmark2D>& landmarks) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    for (const auto& lm : landmarks) {
        outfile << lm.joint_id << " " << lm.x << " " << lm.y << " " << lm.confidence << "\n";
    }
    outfile.close();
    std::cout << "Saved landmarks to " << filename << "\n";
}

// ------------------------------
// Synthetic landmarks for AMASS validation
// ------------------------------
static std::vector<Landmark2D> GenerateSyntheticLandmarks(
    const SmplModel& model,
    const CameraIntrinsics& camera,
    const double* beta,
    const double* theta,
    const double* trans
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
        float py = float(camera.fy * (-Y / Zc) + camera.cy);

        landmarks.emplace_back(px, py, 1.0f, joint_id);
    }
    return landmarks;
}

// ------------------------------
// Config builders 
// ------------------------------
static OptimizerConfig MakeCfgForOpenPose() {
    OptimizerConfig cfg;
    cfg.verbose = true;

    cfg.min_confidence = 0.20f;
    cfg.use_stable_joints_only = true;
    cfg.flip_y_in_projection = true;

    // GMM
    if (g_gmm_prior.loaded) {
        cfg.use_gmm_prior = true;
        cfg.w_gmm = 1.0;
        cfg.gmm_prior = &g_gmm_prior;
        std::cout << "[Config] Using GMM pose prior\n";
    }

    // hinge/bend
    cfg.use_hinge_priors = true;
    cfg.use_bend_limits  = true;
    cfg.w_bend           = 0.05;

    cfg.w_hinge_perp_elbow  = 40.0;
    cfg.w_hinge_range_elbow = 20.0;
    cfg.w_hinge_perp_knee   = 120.0;
    cfg.w_hinge_range_knee  = 60.0;
    cfg.hinge_min = -2.6;
    cfg.hinge_max =  2.6;

    // root upright
    cfg.use_root_upright = true;
    cfg.root_upright_w_stage2 = 5.0;
    cfg.root_upright_w_stage3 = 1.0;

    // shoulder/collar/wrist/spine 
    cfg.use_shoulder_collar_limits = true;
    cfg.w_collar_limit = 40.0;
    cfg.collar_amax    = 0.8;
    cfg.w_shoulder_limit = 60.0;
    cfg.shoulder_amax    = 1.7;

    cfg.use_wrist_limits = true;
    cfg.w_wrist_limit = 80.0;
    cfg.wrist_amax    = 0.6;

    cfg.use_spine_limits = true;
    cfg.w_spine     = 10.0;
    cfg.spine_amax  = 1.2;

    cfg.use_pelvis_over_ankles = true;
    
    // arm-torso sep
    cfg.use_arm_torso_sep = true;
    cfg.arm_torso_seg_a = 3;
    cfg.arm_torso_seg_b = 12;
    cfg.arm_torso_d_elbow = 0.08;
    cfg.arm_torso_d_wrist = 0.10;
    cfg.arm_torso_w_elbow = 80.0;
    cfg.arm_torso_w_wrist = 120.0;

    cfg.lock_wrists = true;
    cfg.lock_hands  = true;
    cfg.lock_feet   = true;

    // foot constraints: enable in sequence after ground_y known
    cfg.use_foot_ground = false;

    // solver
    cfg.threads = -1;
    cfg.print_progress = false;
    cfg.max_iter_stage1 = 60;
    cfg.max_iter_stage2 = 200;
    cfg.max_iter_stage3 = 200;

    return cfg;
}

static OptimizerConfig MakeCfgForAMASS(const std::vector<JointDepthData>* depth_gt) {
    OptimizerConfig cfg;
    cfg.verbose = true;
    cfg.amass_unit_test = true;
    cfg.unit_test_exact = true;

    // For AMASS synthetic data, use all joints (including head)
    // since the synthetic landmarks are accurate
    cfg.use_stable_joints_only = false;

    // depth
    cfg.use_depth = true;
    cfg.w_depth = 50.0;
    cfg.depth_data = depth_gt;

    // GMM
    if (g_gmm_prior.loaded) {
        cfg.use_gmm_prior = true;
        cfg.w_gmm = 1.0;
        cfg.gmm_prior = &g_gmm_prior;
        std::cout << "[Config] Using GMM pose prior\n";
    } else {
        std::cout << "[Config] GMM not loaded\n";
    }

    // hinge/bend
    cfg.use_hinge_priors = true;
    cfg.use_bend_limits  = true;
    cfg.w_bend           = 0.05;

    cfg.w_hinge_perp_elbow  = 30.0;
    cfg.w_hinge_range_elbow = 10.0;

    // root upright
    cfg.use_root_upright = true;
    cfg.root_upright_w_stage2 = 0.5;
    cfg.root_upright_w_stage3 = 0.0;

    // soft limits
    cfg.use_shoulder_collar_limits = true;
    cfg.w_collar_limit = 20.0;
    cfg.collar_amax    = 1.0;
    cfg.w_shoulder_limit = 30.0;
    cfg.shoulder_amax    = 2.0;

    cfg.use_wrist_limits = true;
    cfg.w_wrist_limit = 60.0;
    cfg.wrist_amax    = 0.8;

    cfg.use_spine_limits = true;
    cfg.w_spine    = 5.0;
    cfg.spine_amax = 1.5;

    // neck/head limits - prevent backward head tilt
    cfg.use_neck_head_limits = true;
    cfg.w_neck_limit = 40.0;
    cfg.neck_amax = 0.5;     // ~30 degrees
    cfg.w_head_limit = 50.0;
    cfg.head_amax = 0.4;     // ~23 degrees

    // arm-torso sep
    cfg.use_arm_torso_sep = true;

    // Lock extremities (same as OpenPose config)
    cfg.lock_wrists = true;
    cfg.lock_hands  = true;
    cfg.lock_feet   = true;

    return cfg;
}

// =========================================================
// MODE 1: AMASS Validation
// =========================================================
double RunAMASSValidation(const SmplModel& model, const std::string& amass_path, int frame_idx = -1, bool save_mesh = true) {
    std::cout << "\n=== AMASS Validation: " << amass_path << " ===\n";

    std::vector<AmassFrame> amass_frames;
    if (!AmassLoader::Load(amass_path, amass_frames)) return -1.0;
    if (amass_frames.empty()) return -1.0;

    if (frame_idx < 0) frame_idx = (int)(amass_frames.size() * 0.7);
    if (frame_idx >= (int)amass_frames.size()) frame_idx = (int)amass_frames.size() - 1;

    const AmassFrame& frame_gt = amass_frames[frame_idx];

    double beta_gt[10];
    double theta_gt[72];
    double trans_gt[3];

    for (int i = 0; i < 10; ++i) beta_gt[i] = frame_gt.beta[i];
    for (int i = 0; i < 72; ++i) theta_gt[i] = frame_gt.theta[i];

    trans_gt[0] = 0.0;
    trans_gt[1] = 0.0;
    trans_gt[2] = 3.0;

    std::cout << "[GT] Using AMASS Frame " << frame_idx << " as Ground Truth.\n";

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_gt;
    SMPLJointsForward<double>(beta_gt, theta_gt, trans_gt, model, J_gt);

    // depth gt
    std::vector<JointDepthData> depth_gt;
    depth_gt.reserve(24);
    for (int i = 0; i < 24; i++) depth_gt.push_back({i, J_gt(i, 2)});

    CameraIntrinsics camera = CreateDefaultCamera(800, 800);
    auto observed_landmarks = GenerateSyntheticLandmarks(model, camera, beta_gt, theta_gt, trans_gt);
    if (observed_landmarks.empty()) return -1.0;

    // name prefix
    std::string base_name;
    {
        size_t last_slash = amass_path.find_last_of("/\\");
        std::string filename = (last_slash != std::string::npos) ? amass_path.substr(last_slash + 1) : amass_path;
        size_t dot_pos = filename.find_last_of('.');
        base_name = (dot_pos != std::string::npos) ? filename.substr(0, dot_pos) : filename.substr(0);
    }

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

    OptimizerConfig cfg = MakeCfgForAMASS(&depth_gt);

    bool use_3d_optimization = false;
    bool ok = false;
    if (use_3d_optimization) {
        ok = OptimizeSMPL3D(J_gt, model, beta_opt, theta_opt, trans_opt, cfg);
    } else {
        ok = OptimizeSMPL(observed_landmarks, camera, model, beta_opt, theta_opt, trans_opt, cfg);
    }
    if (!ok) return -1.0;

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_fit;
    SMPLJointsForward<double>(beta_opt, theta_opt, trans_opt, model, J_fit);

    double mean_error = (J_fit - J_gt).rowwise().norm().mean();
    std::cout << "[Result] " << base_name << " frame " << frame_idx << " -> error = " << mean_error << " m\n";

    if (save_mesh) {
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> verts_opt;
        SMPLForward<double>(beta_opt, theta_opt, trans_opt, model, verts_opt);
        SaveOBJ(base_name + "_mesh_opt.obj", verts_opt, model.faces);
    }

    return mean_error;
}

// =========================================================
// AMASS Sequence Validation - Validate frames 0-50 with full evaluation
// =========================================================
void RunAmassValidationSequence(const SmplModel& model, const std::string& pose_file_path) {
    std::cout << "\n========== AMASS SEQUENCE VALIDATION (Frames 0-50) ==========\n";
    std::cout << "File: " << pose_file_path << "\n";

    // Extract base filename for naming
    std::string base_name = fs::path(pose_file_path).stem().string();
    
    // Create output directories
    const std::string mesh_dir = "./meshes/" + base_name;
    const std::string gt_mesh_dir = mesh_dir + "/gt";
    const std::string opt_mesh_dir = mesh_dir + "/optimized";
    
    fs::create_directories(gt_mesh_dir);
    fs::create_directories(opt_mesh_dir);
    
    std::cout << "Output directories created:\n";
    std::cout << "  GT meshes:   " << gt_mesh_dir << "\n";
    std::cout << "  Opt meshes:  " << opt_mesh_dir << "\n";

    // Load all frames from the file
    std::vector<AmassFrame> amass_frames;
    if (!AmassLoader::Load(pose_file_path, amass_frames)) {
        std::cerr << "Failed to load file: " << pose_file_path << "\n";
        return;
    }

    if (amass_frames.empty()) {
        std::cerr << "No frames in file\n";
        return;
    }

    std::cout << "Total frames in file: " << amass_frames.size() << "\n";

    // Determine validation range: frames 0-50 (or less if file is shorter)
    const int start_frame = 0;
    const int end_frame = std::min(50, static_cast<int>(amass_frames.size()) - 1);
    const int num_frames = end_frame - start_frame + 1;

    std::cout << "Validating frames " << start_frame << " to " << end_frame 
              << " (" << num_frames << " frames)\n\n";

    // Setup camera for synthetic landmark generation
    CameraIntrinsics camera = CreateDefaultCamera(800, 800);

    // Evaluation accumulators
    std::vector<double> all_mpjpe;
    std::vector<double> all_pa_mpjpe;
    std::vector<double> all_pve;         // Per Vertex Error
    std::vector<double> all_pose_error;  // Pose parameter error
    std::vector<double> all_shape_error; // Shape parameter error
    std::vector<double> all_reproj_2d;   // 2D reprojection error
    std::vector<std::vector<double>> per_joint_errors(24);  // 24 SMPL joints

    all_mpjpe.reserve(num_frames);
    all_pa_mpjpe.reserve(num_frames);
    all_pve.reserve(num_frames);
    all_pose_error.reserve(num_frames);
    all_shape_error.reserve(num_frames);
    all_reproj_2d.reserve(num_frames);

    // Process each frame
    int success_count = 0;
    int fail_count = 0;

    for (int frame_idx = start_frame; frame_idx <= end_frame; ++frame_idx) {
        std::cout << "[Frame " << frame_idx << "/" << end_frame << "] ";

        // Get ground truth for this frame
        const AmassFrame& frame_gt = amass_frames[frame_idx];

        double beta_gt[10];
        double theta_gt[72];
        double trans_gt[3];

        for (int i = 0; i < 10; ++i) beta_gt[i] = frame_gt.beta[i];
        for (int i = 0; i < 72; ++i) theta_gt[i] = frame_gt.theta[i];
        trans_gt[0] = 0.0;
        trans_gt[1] = 0.0;
        trans_gt[2] = 3.0;

        // Compute and save ground truth mesh and joints
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_gt;
        SMPLForward<double>(beta_gt, theta_gt, trans_gt, model, V_gt);
        
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_gt;
        SMPLJointsForward<double>(beta_gt, theta_gt, trans_gt, model, J_gt);

        // Save GT mesh
        char gt_filename[512];
        std::snprintf(gt_filename, sizeof(gt_filename), "%s/frame_%05d_gt.obj", 
                     gt_mesh_dir.c_str(), frame_idx);
        SaveOBJ(gt_filename, V_gt, model.faces);

        // Generate synthetic 2D landmarks from ground truth
        auto observed_landmarks = GenerateSyntheticLandmarks(model, camera, beta_gt, theta_gt, trans_gt);
        if (observed_landmarks.empty()) {
            fail_count++;
            std::cout << "✗ FAILED (no landmarks)\n";
            continue;
        }

        // Create depth ground truth
        std::vector<JointDepthData> depth_gt;
        depth_gt.reserve(24);
        for (int i = 0; i < 24; i++) {
            depth_gt.push_back({i, J_gt(i, 2)});
        }

        // Initialize optimization parameters
        double beta_opt[10]  = {0};
        double theta_opt[72] = {0};
        double trans_opt[3]  = {0.0, 0.0, 2.5};

        // Setup optimizer config
        OptimizerConfig cfg = MakeCfgForAMASS(&depth_gt);
        cfg.verbose = false;  // Suppress per-frame verbose output

        // Run optimization
        bool ok = OptimizeSMPL(observed_landmarks, camera, model, beta_opt, theta_opt, trans_opt, cfg);

        if (!ok) {
            fail_count++;
            std::cout << "✗ FAILED (optimization)\n";
            continue;
        }

        // Compute optimized mesh and joints
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_opt;
        SMPLForward<double>(beta_opt, theta_opt, trans_opt, model, V_opt);
        
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_opt;
        SMPLJointsForward<double>(beta_opt, theta_opt, trans_opt, model, J_opt);

        // Save optimized mesh
        char opt_filename[512];
        std::snprintf(opt_filename, sizeof(opt_filename), "%s/frame_%05d_opt.obj", 
                     opt_mesh_dir.c_str(), frame_idx);
        SaveOBJ(opt_filename, V_opt, model.faces);

        // Compute evaluation metrics
        double mpjpe = ComputeMPJPE(J_opt, J_gt);
        double pa_mpjpe = ComputePAMPJPE(J_opt, J_gt);

        all_mpjpe.push_back(mpjpe);
        all_pa_mpjpe.push_back(pa_mpjpe);

        // PVE (Per Vertex Error) - compare vertices directly
        double pve = ComputePVE(V_opt, V_gt);
        all_pve.push_back(pve);

        // Pose Error (theta parameters, excluding root)
        double pose_error = ComputePoseError(theta_opt, theta_gt, false);
        all_pose_error.push_back(pose_error);

        // Shape Error (beta parameters)
        double shape_error = ComputeShapeError(beta_opt, beta_gt);
        all_shape_error.push_back(shape_error);

        // 2D Reprojection Error - project optimized joints and compare with input landmarks
        double reproj_err = 0.0;
        int reproj_count = 0;
        for (const auto& lm : observed_landmarks) {
            if (lm.confidence < 0.3f) continue;
            int op_idx = lm.joint_id;  // OpenPose joint ID
            if (op_idx < 0 || op_idx >= 25) continue;
            int smpl_idx = OPENPOSE_TO_SMPL[op_idx];  // Convert to SMPL ID
            if (smpl_idx < 0 || smpl_idx >= J_opt.rows()) continue;

            Eigen::Vector3d p = J_opt.row(smpl_idx).transpose();
            if (p(2) <= 0) continue;

            double u_pred = camera.fx * p(0) / p(2) + camera.cx;
            double v_pred = camera.fy * (-p(1)) / p(2) + camera.cy;

            double err = std::sqrt((u_pred - lm.x) * (u_pred - lm.x) +
                                   (v_pred - lm.y) * (v_pred - lm.y));
            reproj_err += err;
            reproj_count++;
        }
        if (reproj_count > 0) {
            all_reproj_2d.push_back(reproj_err / reproj_count);
        }

        // Compute per-joint errors
        for (int j = 0; j < 24; ++j) {
            double joint_error = (J_opt.row(j) - J_gt.row(j)).norm();
            per_joint_errors[j].push_back(joint_error);
        }

        success_count++;
        std::cout << "✓ MPJPE=" << mpjpe * 1000.0 << "mm, PA-MPJPE="
                  << pa_mpjpe * 1000.0 << "mm, PVE=" << pve * 1000.0 << "mm\n";
    }

    // ========== Print and save evaluation summary ==========
    if (!all_mpjpe.empty()) {
        // Helper to compute mean
        auto compute_mean = [](const std::vector<double>& v) -> double {
            if (v.empty()) return -1.0;
            double sum = 0.0;
            for (double x : v) sum += x;
            return sum / v.size();
        };

        double mean_mpjpe = compute_mean(all_mpjpe);
        double mean_pa_mpjpe = compute_mean(all_pa_mpjpe);
        double mean_pve = compute_mean(all_pve);
        double mean_pose_error = compute_mean(all_pose_error);
        double mean_shape_error = compute_mean(all_shape_error);
        double mean_reproj_2d = compute_mean(all_reproj_2d);

        // Compute min/max
        double min_mpjpe = *std::min_element(all_mpjpe.begin(), all_mpjpe.end());
        double max_mpjpe = *std::max_element(all_mpjpe.begin(), all_mpjpe.end());
        double min_pa_mpjpe = *std::min_element(all_pa_mpjpe.begin(), all_pa_mpjpe.end());
        double max_pa_mpjpe = *std::max_element(all_pa_mpjpe.begin(), all_pa_mpjpe.end());

        // Compute standard deviation
        double var_mpjpe = 0.0, var_pa_mpjpe = 0.0;
        for (size_t i = 0; i < all_mpjpe.size(); ++i) {
            var_mpjpe += (all_mpjpe[i] - mean_mpjpe) * (all_mpjpe[i] - mean_mpjpe);
            var_pa_mpjpe += (all_pa_mpjpe[i] - mean_pa_mpjpe) * (all_pa_mpjpe[i] - mean_pa_mpjpe);
        }
        double std_mpjpe = std::sqrt(var_mpjpe / all_mpjpe.size());
        double std_pa_mpjpe = std::sqrt(var_pa_mpjpe / all_pa_mpjpe.size());

        std::cout << "\n========================================\n";
        std::cout << "EVALUATION SUMMARY\n";
        std::cout << "========================================\n";
        std::cout << "File: " << base_name << "\n";
        std::cout << "Frames evaluated: " << all_mpjpe.size() << " ("
                  << start_frame << "-" << end_frame << ")\n";
        std::cout << "Successful: " << success_count << "\n";
        std::cout << "Failed: " << fail_count << "\n\n";

        std::cout << "=== MPJPE (3D Joint Error) ===\n";
        std::cout << "  Mean:    " << mean_mpjpe * 1000.0 << " mm\n";
        std::cout << "  Min:     " << min_mpjpe * 1000.0 << " mm\n";
        std::cout << "  Max:     " << max_mpjpe * 1000.0 << " mm\n";
        std::cout << "  Std Dev: " << std_mpjpe * 1000.0 << " mm\n\n";

        std::cout << "=== PA-MPJPE (Procrustes Aligned) ===\n";
        std::cout << "  Mean:    " << mean_pa_mpjpe * 1000.0 << " mm\n";
        std::cout << "  Min:     " << min_pa_mpjpe * 1000.0 << " mm\n";
        std::cout << "  Max:     " << max_pa_mpjpe * 1000.0 << " mm\n";
        std::cout << "  Std Dev: " << std_pa_mpjpe * 1000.0 << " mm\n\n";

        std::cout << "=== PVE (Per Vertex Error) ===\n";
        std::cout << "  Mean:    " << mean_pve * 1000.0 << " mm\n\n";

        std::cout << "=== Pose Error (theta) ===\n";
        std::cout << "  Mean:    " << mean_pose_error << " rad\n\n";

        std::cout << "=== Shape Error (beta) ===\n";
        std::cout << "  Mean:    " << mean_shape_error << "\n\n";

        std::cout << "=== 2D Reprojection Error ===\n";
        std::cout << "  Mean:    " << mean_reproj_2d << " px\n";

        // Compute per-joint statistics
        std::cout << "\n=== Per-Joint Average Error (mm) ===\n";
        const char* joint_names[24] = {
            "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle",
            "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head",
            "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
        };

        for (int j = 0; j < 24; ++j) {
            if (per_joint_errors[j].empty()) continue;
            
            double sum = 0.0;
            for (double err : per_joint_errors[j]) sum += err;
            double mean = sum / per_joint_errors[j].size();
            
            std::cout << "  Joint " << std::setw(2) << j << " (" 
                      << std::setw(12) << std::left << joint_names[j] << "): "
                      << std::fixed << std::setprecision(2) << mean * 1000.0 << " mm\n";
        }
        std::cout << std::right;  // Reset alignment

        // Save evaluation results to file
        std::string eval_file = mesh_dir + "/evaluation.txt";
        std::ofstream outfile(eval_file);
        
        if (outfile.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            
            outfile << "AMASS Sequence Validation Results\n";
            outfile << "==================================\n";
            outfile << "File: " << pose_file_path << "\n";
            outfile << "Date: " << std::ctime(&time);
            outfile << "Frames evaluated: " << all_mpjpe.size() 
                   << " (" << start_frame << "-" << end_frame << ")\n";
            outfile << "Successful: " << success_count << "\n";
            outfile << "Failed: " << fail_count << "\n\n";

            outfile << "MPJPE (3D Joint Error):\n";
            outfile << "  Mean (mm): " << mean_mpjpe * 1000.0 << "\n";
            outfile << "  Min (mm):  " << min_mpjpe * 1000.0 << "\n";
            outfile << "  Max (mm):  " << max_mpjpe * 1000.0 << "\n";
            outfile << "  Std (mm):  " << std_mpjpe * 1000.0 << "\n\n";

            outfile << "PA-MPJPE (Procrustes Aligned):\n";
            outfile << "  Mean (mm): " << mean_pa_mpjpe * 1000.0 << "\n";
            outfile << "  Min (mm):  " << min_pa_mpjpe * 1000.0 << "\n";
            outfile << "  Max (mm):  " << max_pa_mpjpe * 1000.0 << "\n";
            outfile << "  Std (mm):  " << std_pa_mpjpe * 1000.0 << "\n\n";

            outfile << "PVE (Per Vertex Error):\n";
            outfile << "  Mean (mm): " << mean_pve * 1000.0 << "\n\n";

            outfile << "Pose Error (theta):\n";
            outfile << "  Mean (rad): " << mean_pose_error << "\n\n";

            outfile << "Shape Error (beta):\n";
            outfile << "  Mean: " << mean_shape_error << "\n\n";

            outfile << "2D Reprojection Error:\n";
            outfile << "  Mean (px): " << mean_reproj_2d << "\n\n";

            outfile << "Per-Joint Average Error (mm):\n";
            for (int j = 0; j < 24; ++j) {
                if (per_joint_errors[j].empty()) continue;
                double sum = 0.0;
                for (double err : per_joint_errors[j]) sum += err;
                double mean = sum / per_joint_errors[j].size();
                outfile << "  Joint " << j << " (" << joint_names[j] << "): "
                       << mean * 1000.0 << "\n";
            }

            outfile << "\nPer-Frame Results:\n";
            outfile << "Frame\tMPJPE(mm)\tPA-MPJPE(mm)\tPVE(mm)\tPoseErr(rad)\tShapeErr\t2DReproj(px)\n";
            for (size_t i = 0; i < all_mpjpe.size(); ++i) {
                outfile << (start_frame + i) << "\t"
                       << all_mpjpe[i] * 1000.0 << "\t"
                       << all_pa_mpjpe[i] * 1000.0 << "\t"
                       << (i < all_pve.size() ? all_pve[i] * 1000.0 : -1.0) << "\t"
                       << (i < all_pose_error.size() ? all_pose_error[i] : -1.0) << "\t"
                       << (i < all_shape_error.size() ? all_shape_error[i] : -1.0) << "\t"
                       << (i < all_reproj_2d.size() ? all_reproj_2d[i] : -1.0) << "\n";
            }

            outfile.close();
            std::cout << "\n[Saved] Evaluation results: " << eval_file << "\n";
        } else {
            std::cerr << "Warning: Could not save evaluation file\n";
        }

        std::cout << "========================================\n";
        std::cout << "\n[Done] Meshes saved to: " << mesh_dir << "\n";
        std::cout << "  GT meshes:        " << gt_mesh_dir << "\n";
        std::cout << "  Optimized meshes: " << opt_mesh_dir << "\n";
    } else {
        std::cout << "\n[Error] No successful frames to evaluate\n";
    }
}

// =========================================================
// MODE 2: Real OpenPose
// =========================================================
void RunOpenPoseInference(const SmplModel& model) {
    std::cout << "\n=== MODE 2: Running Real OpenPose Data ===\n";

    std::string json_path = "../../Data/openpose_output/courtyard_basketball_01/image_00025_keypoints.json";
    std::vector<Landmark2D> lms;
    if (!LoadLandmarksFromJSON(json_path, lms)) {
        std::cerr << "Failed to load json.\n";
        return;
    }
    SaveLandmarksToFile("landmarks_input.txt", lms);

    const int img_w = 1080;
    const int img_h = 1920;
    CameraIntrinsics camera(1962.0f, 1969.0f, 540.0f, 960.0f, img_w, img_h);

    double beta[10]  = {0.0};
    double theta[72] = {0.0};
    double trans[3]  = {0.0, 0.0, 3.0};

    EstimateGlobalTranslationXY(lms, camera, 3.0, trans);

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_init;
    SMPLForward<double>(beta, theta, trans, model, V_init);
    SaveOBJ("mesh_before_opt.obj", V_init, model.faces);

    OptimizerConfig cfg = MakeCfgForOpenPose();

    // -------- Load Sapiens relative depth and create pairs --------
    std::vector<RelativeDepthPair> relative_depth_pairs;
    std::string depth_path = "../../Data/sapiens_depth/courtyard_basketball_01/image_00047_depth.txt";
    std::ifstream depth_file(depth_path);
    if (depth_file.is_open()) {
        std::vector<std::pair<int, double>> kp_depths;  // (smpl_idx, depth)
        
        int op_idx;
        double depth_val;
        while (depth_file >> op_idx >> depth_val) {
            int smpl_idx = OPENPOSE_TO_SMPL[op_idx];
            if (smpl_idx >= 0) kp_depths.push_back({smpl_idx, depth_val});
        }
        std::cout << "depth size: " << kp_depths.size() << "\n";
        depth_file.close();

        for (size_t i = 0; i < kp_depths.size(); ++i) {
            for (size_t j = i + 1; j < kp_depths.size(); ++j) {
                relative_depth_pairs.push_back({kp_depths[i].first, kp_depths[j].first,
                                                kp_depths[i].second, kp_depths[j].second});
            }
        }
        std::cout << "[Depth] Created " << relative_depth_pairs.size() << " relative depth pairs\n";
    } else {
        std::cerr << "[Depth] Failed to open " << depth_path << "\n";
    }

    if (!relative_depth_pairs.empty()) {
        cfg.use_relative_depth = true; 
        cfg.w_relative_depth = 20.0;
        cfg.relative_depth_pairs = &relative_depth_pairs;
        std::cout << "[Config] Relative depth currently ENABLED\n";
    }

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

void RunOpenPoseSequenceInference(const SmplModel& model) {
    std::cout << "\n=== MODE 3: OpenPose SEQUENCE ===\n";

    const std::string seq_dir   = "../../Data/openpose_output/courtyard_basketball_01";
    const std::string depth_dir = "../../Data/sapiens_depth/courtyard_basketball_01";
    const std::string gt_dir    = "../../Data/3dpw_gt/courtyard_basketball_01";
    const std::string out_dir   = "./seq_out";
    fs::create_directories(out_dir);

    // Load 3DPW ground truth for evaluation
    ThreeDPW_GT gt_data;
    bool has_gt = gt_data.LoadFromDir(gt_dir);
    if (has_gt) {
        std::cout << "[GT] Loaded " << gt_data.num_frames << " frames of ground truth\n";
        if (gt_data.has_poses) std::cout << "[GT] Has pose parameters (theta)\n";
        if (gt_data.has_betas) std::cout << "[GT] Has shape parameters (beta)\n";
    } else {
        std::cout << "[GT] No ground truth available, skipping evaluation\n";
    }

    // Load camera extrinsics (per-frame world-to-camera transforms)
    std::vector<CameraExtrinsics> cam_poses = LoadCameraPoses(gt_dir + "/cam_poses.txt");
    bool has_cam_poses = !cam_poses.empty();
    if (has_cam_poses) {
        std::cout << "[Camera] Loaded " << cam_poses.size() << " camera poses (moving camera)\n";
    } else {
        std::cout << "[Camera] No camera poses found, assuming fixed camera at origin\n";
    }

    // World coordinate optimization mode
    // When enabled: optimize trans/theta in world coordinates, use extrinsics for reprojection
    const bool use_world_coords = has_cam_poses;
    if (use_world_coords) {
        std::cout << "[World] World-coordinate optimization ENABLED\n";
    } else {
        std::cout << "[World] Camera-coordinate optimization (no extrinsics)\n";
    }

    // Helper: convert camera-space point to world-space using inverse extrinsics
    // p_cam = R * p_world + t  =>  p_world = R^T * (p_cam - t)
    auto cam_to_world = [](const Eigen::Vector3d& p_cam, const CameraExtrinsics& ext) -> Eigen::Vector3d {
        return ext.R.transpose() * (p_cam - ext.t);
    };

    // Evaluation accumulators
    std::vector<double> all_mpjpe;
    std::vector<double> all_pa_mpjpe;
    std::vector<double> all_pve;
    std::vector<double> all_pose_error;
    std::vector<double> all_reproj_2d;

    const int start = 0;
    const int end   = 957;

    const int img_w = 1080;
    const int img_h = 1920;
    CameraIntrinsics camera(1962.0f, 1969.0f, 540.0f, 960.0f, img_w, img_h);

    OptimizerConfig cfg = MakeCfgForOpenPose();

    cfg.use_temporal = true;
    cfg.w_temp_pose  = 12.0;
    cfg.w_temp_trans = 30.0;
    cfg.use_temporal_acc = true;
    cfg.w_temp_acc_pose  = 2.0;
    cfg.temporal_body_only = true;

    cfg.use_foot_ground = true;
    cfg.w_foot_ground   = 140.0;   
    cfg.w_foot_flat     = 80.0;  
    cfg.ground_y        = 0.0;     

    double beta[10]  = {0.0};
    double theta[72] = {0.0};
    double trans[3]  = {0.0, 0.0, 3.0};

    double theta_prev[72]  = {0.0};
    double theta_prev2[72] = {0.0};
    double trans_prev[3]   = {0.0, 0.0, 3.0};
    double trans_prev2[3]  = {0.0, 0.0, 3.0};  // will be properly set when has_prev2=true

    bool has_prev  = false;
    bool has_prev2 = false;

    // Affine depth model: Z = bias + sign * scale * d
    // Sapiens depth: larger d = farther from camera (sign = +1)
    // scale ~ slope, bias ~ Z_min (~3m)
    double depth_affine[2]      = {3.0, 3.0};  // [scale, bias]
    double depth_affine_prev[2] = {3.0, 3.0};

    // Camera-space depth tracking for world-coordinate optimization
    // Z_cam = R[2,:] * trans_world + t[2]
    double Z_cam_prev  = 3.0;
    double Z_cam_prev2 = 3.0;

    static bool kp_ema_init = false;
    static double kp_x[25], kp_y[25], kp_seen[25];

    auto smooth_landmarks_inplace = [&](std::vector<Landmark2D>& lms) {
        if (!kp_ema_init) {
            for (int i=0;i<25;++i){ kp_x[i]=0; kp_y[i]=0; kp_seen[i]=0; }
            kp_ema_init = true;
        }
        for (auto& lm : lms) {
            int id = lm.joint_id;
            if (id < 0 || id >= 25) continue;
            if (lm.confidence < 0.20f) continue;

            // adaptive EMA: higher conf -> more trust current
            double a = 0.45 + 0.45 * std::min(1.0, std::max(0.0, (double)lm.confidence));
            if (kp_seen[id] < 0.5) {
                kp_x[id] = lm.x;
                kp_y[id] = lm.y;
                kp_seen[id] = 1.0;
            } else {
                kp_x[id] = a * lm.x + (1.0 - a) * kp_x[id];
                kp_y[id] = a * lm.y + (1.0 - a) * kp_y[id];
            }
            lm.x = (float)kp_x[id];
            lm.y = (float)kp_y[id];
        }
    };

    auto get_conf = [&](const std::vector<Landmark2D>& lms, int op_id)->double{
        for (auto& lm: lms) if (lm.joint_id == op_id) return lm.confidence;
        return 0.0;
    };

    auto get_xy = [&](const std::vector<Landmark2D>& lms, int op_id, double& x, double& y)->bool{
        for (auto& lm: lms) if (lm.joint_id == op_id) { x = lm.x; y = lm.y; return true; }
        return false;
    };

    auto is_stable_op = [&](int op_id)->bool{
        // Torso-only for depth affine (exclude legs for more stable fitting)
        // 1: Neck, 2: RShoulder, 5: LShoulder, 8: MidHip
        // Also include elbows: 3: RElbow, 6: LElbow
        int ids[] = {1, 2, 3, 5, 6, 8};
        //int ids[] = {1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14};

        for (int x: ids) if (x == op_id) return true;
        return false;
    };

    bool ground_init = false;
    double ground_y_est = 0.0;
    bool ankle_prev_valid = false;
    double prev_LA_x=0, prev_LA_y=0, prev_RA_x=0, prev_RA_y=0;

    auto compute_contact = [&](const std::vector<Landmark2D>& lms, bool is_left)->double{
        int op_ank = is_left ? 14 : 11;
        int op_kne = is_left ? 13 : 10; // LKnee / RKnee

        const double cA = get_conf(lms, op_ank);
        const double cK = get_conf(lms, op_kne);
        if (cA < 0.35 || cK < 0.35) return 0.0;

        double ax, ay;
        if (!get_xy(lms, op_ank, ax, ay)) return 0.0;

        // near bottom of image => likely on ground
        double y_norm = ay / double(img_h); // 0 top, 1 bottom
        double ground_like = 0.0;
        if (y_norm > 0.80) ground_like = 1.0;
        else if (y_norm > 0.72) ground_like = (y_norm - 0.72) / (0.80 - 0.72);
        else ground_like = 0.0;

        // low 2D speed => likely planted
        double speed_like = 1.0;
        if (ankle_prev_valid) {
            double px = is_left ? prev_LA_x : prev_RA_x;
            double py = is_left ? prev_LA_y : prev_RA_y;
            double sp = std::sqrt((ax-px)*(ax-px) + (ay-py)*(ay-py)); // px/frame
            if (sp > 18.0) speed_like = 0.0;
            else if (sp > 6.0) speed_like = (18.0 - sp) / (18.0 - 6.0);
            else speed_like = 1.0;
        }

        // combine + confidence
        double conf_like = std::min(1.0, 0.5*(cA+cK));
        double contact = ground_like * speed_like * conf_like;

        // avoid too binary
        contact = std::min(1.0, std::max(0.0, contact));
        return contact;
    };

    for (int f = start; f <= end; ++f) {
        std::string json_path = MakeOpenPoseJsonPath(seq_dir, f);
        std::vector<Landmark2D> lms;

        // reset depth flags per frame
        cfg.use_relative_depth = false;
        cfg.relative_depth_pairs = nullptr;

        cfg.use_depth = false;
        cfg.depth_data = nullptr;

        cfg.use_depth_affine = false;
        cfg.depth_affine_data = nullptr;
        cfg.depth_affine_params = nullptr;
        cfg.depth_affine_prev = nullptr;

        if (!LoadLandmarksFromJSON(json_path, lms)) {
            std::cerr << "[Frame " << f << "] failed to load: " << json_path << "\n";
            continue;
        }

        // 2D smoothing
        smooth_landmarks_inplace(lms);

        if (has_prev) {
            std::memcpy(theta, theta_prev, sizeof(theta));
            std::memcpy(trans, trans_prev, sizeof(trans));
            depth_affine[0] = depth_affine_prev[0];
            depth_affine[1] = depth_affine_prev[1];
        } else {
            // First frame: estimate initial translation
            // 1. Estimate trans in camera space from 2D bbox
            double trans_cam[3];
            EstimateGlobalTranslationXY(lms, camera, 3.0, trans_cam);

            if (use_world_coords && f < (int)cam_poses.size()) {
                // Convert camera-space trans to world-space: p_world = R^T * (p_cam - t)
                Eigen::Vector3d p_cam(trans_cam[0], trans_cam[1], trans_cam[2]);
                Eigen::Vector3d p_world = cam_to_world(p_cam, cam_poses[f]);
                trans[0] = p_world(0);
                trans[1] = p_world(1);
                trans[2] = p_world(2);
                std::cout << "[Init] trans_world = (" << trans[0] << ", " << trans[1] << ", " << trans[2] << ")\n";
            } else {
                trans[0] = trans_cam[0];
                trans[1] = trans_cam[1];
                trans[2] = trans_cam[2];
            }
            // Affine depth model: Z = bias + sign * scale * d
            // bias ~ Z_min, scale ~ slope
            depth_affine[0] = 3.0;           // scale (slope)
            depth_affine[1] = trans_cam[2];  // bias ~ Z_min
        }

        {
            auto damp_joint = [&](int j, double s){
                theta[3*j+0] *= s;
                theta[3*j+1] *= s;
                theta[3*j+2] *= s;
            };
            // s=0 -> fully fixed; s in (0,1) -> gentle damping
            damp_joint(20, 0.10);
            damp_joint(21, 0.10);
        }


        std::vector<JointRelDepthData> depth_pts;
        std::string depth_path = MakeOpenPoseDepthPath(depth_dir, f);
        std::ifstream depth_file(depth_path);

        if (depth_file.is_open()) {
            int op_idx;
            double depth_val;
            while (depth_file >> op_idx >> depth_val) {
                if (op_idx < 0 || op_idx >= 25) continue;
                if (!is_stable_op(op_idx)) continue;

                const double c2d = get_conf(lms, op_idx);
                if (c2d < 0.35) continue;

                int smpl_idx = OPENPOSE_TO_SMPL[op_idx];
                if (smpl_idx < 0) continue;

                // Downweight leg joints (9-14: Hip, Knee, Ankle)
                //double weight = c2d;
                //if (op_idx >= 9 && op_idx <= 14) {
                //    weight *= 0.3;  // legs get 30% weight
                //}

                depth_pts.push_back({smpl_idx, depth_val, c2d});
            }
            depth_file.close();
        } else {
            // it's ok: depth may be missing for some frames
        }

        if ((int)depth_pts.size() >= 6) {
            cfg.use_depth_affine = true;
            cfg.w_depth_affine = 60.0;
            cfg.depth_affine_sign = 1.0;  // Sapiens: larger d = farther = larger Z
            cfg.depth_affine_data   = &depth_pts;
            cfg.depth_affine_params = depth_affine;

            cfg.use_temporal_depth_affine = has_prev;
            cfg.depth_affine_prev = has_prev ? depth_affine_prev : nullptr;
            cfg.w_temp_depth_scale = 20.0;
            cfg.w_temp_depth_bias  = 50.0;

            cfg.fix_depth_affine_bias = has_prev;  // lock bias after first frame to prevent drift
        }

        // Camera-space depth prior (world-coord mode): prevent drift along optical axis
        if (use_world_coords && has_prev) {
            cfg.use_cam_depth_prior = true;
            cfg.w_cam_depth_prior = 200.0;
            cfg.Z_cam_prev = Z_cam_prev;
            cfg.Z_cam_prev2 = Z_cam_prev2;
            cfg.has_Z_cam_prev2 = has_prev2;
        }

        double LAx=0, LAy=0, RAx=0, RAy=0;
        bool hasLA = get_xy(lms, 14, LAx, LAy); // LAnkle
        bool hasRA = get_xy(lms, 11, RAx, RAy); // RAnkle

        // contact in [0,1]
        cfg.foot_contact_L = compute_contact(lms, /*is_left=*/true);
        cfg.foot_contact_R = compute_contact(lms, /*is_left=*/false);

  
        if (hasLA && hasRA) {
            prev_LA_x = LAx; prev_LA_y = LAy;
            prev_RA_x = RAx; prev_RA_y = RAy;
            ankle_prev_valid = true;
        } else if (hasLA) {
            prev_LA_x = LAx; prev_LA_y = LAy;
            ankle_prev_valid = true;
        } else if (hasRA) {
            prev_RA_x = RAx; prev_RA_y = RAy;
            ankle_prev_valid = true;
        }

        if (has_prev) {
            cfg.max_iter_stage1 = 20;
            cfg.max_iter_stage2 = 40;
            cfg.max_iter_stage3 = 0;   // fast tracking
        } else {
            cfg.max_iter_stage1 = 60;
            cfg.max_iter_stage2 = 120;
            cfg.max_iter_stage3 = 80;
        }

        // Get camera extrinsics for this frame (if using world coordinate optimization)
        const CameraExtrinsics* frame_extrinsics = nullptr;
        if (use_world_coords && f < (int)cam_poses.size()) {
            frame_extrinsics = &cam_poses[f];
        }

        // Debug: print depth_affine BEFORE optimization
        double affine_before[2] = {depth_affine[0], depth_affine[1]};

        bool ok = false;
        if (!has_prev) {
            ok = OptimizeSMPL(lms, camera, model, beta, theta, trans, cfg, frame_extrinsics);
        } else {
            ok = OptimizeSMPLTracked(lms, camera, model, beta, theta, trans, cfg,
                                     theta_prev, trans_prev,
                                     has_prev2 ? theta_prev2 : nullptr,
                                     has_prev2 ? trans_prev2 : nullptr,
                                     frame_extrinsics);
        }

        // Debug: depth constraint statistics
        if (cfg.use_depth_affine) {
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_debug;
            SMPLJointsForward<double>(beta, theta, trans, model, J_debug);

            // Compute depth residual stats
            double sum_abs_err = 0.0;
            int count = 0;
            for (const auto& pt : depth_pts) {
                int j = pt.smpl_joint_idx;
                double d = pt.rel_depth;

                double Z_pred;
                if (frame_extrinsics) {
                    Z_pred = frame_extrinsics->t(2);
                    for (int k = 0; k < 3; ++k) {
                        Z_pred += frame_extrinsics->R(2, k) * J_debug(j, k);
                    }
                } else {
                    Z_pred = J_debug(j, 2);
                }

                // Affine model: Z = bias + sign * scale * d
                double Z_tgt = depth_affine[1] + cfg.depth_affine_sign * depth_affine[0] * d;
                sum_abs_err += std::abs(Z_pred - Z_tgt);
                count++;
            }
            double mean_depth_err = count > 0 ? sum_abs_err / count : 0.0;

            // Compute trans_z delta
            double delta_trans_z = has_prev ? (trans[2] - trans_prev[2]) : 0.0;

            // Print compact stats every frame
            std::cout << "[F" << f << "] scale=" << depth_affine[0]
                      << " | mean|Z_err|=" << mean_depth_err
                      << " | Δtrans_z=" << delta_trans_z << "\n";
        }

        if (!ok) {
            std::cerr << "[Frame " << f << "] Optimize failed\n";
            continue;
        }

        // ---------------------------
        // update online ground estimate using fitted joints
        // ---------------------------
        {
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Jw;
            SMPLJointsForward<double>(beta, theta, trans, model, Jw);

            // use ankles & feet
            double y_min = 1e9;
            int ids[] = {7, 8, 10, 11};
            for (int id : ids) {
                if (id >= 0 && id < Jw.rows()) y_min = std::min(y_min, Jw(id, 1));
            }

            if (!ground_init) {
                ground_y_est = y_min;
                ground_init = true;
            } else if (!use_world_coords) {
                // Only update ground_y in camera coordinate mode
                // In world coordinates, ground is fixed (doesn't move with camera)
                // allow downward update faster, upward update slower
                if (y_min < ground_y_est) {
                    ground_y_est = 0.7 * y_min + 0.3 * ground_y_est;
                } else {
                    ground_y_est = 0.98 * ground_y_est + 0.02 * y_min;
                }
            }
            // In world coords, ground_y_est stays fixed from frame 0

            cfg.ground_y = ground_y_est; // for next frame
        }
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V;
        SMPLForward<double>(beta, theta, trans, model, V);
        SaveOBJ(MakeOutObj(out_dir, f), V, model.faces);

        // shift prev buffers
        if (has_prev) {
            std::memcpy(theta_prev2, theta_prev, sizeof(theta_prev2));
            std::memcpy(trans_prev2, trans_prev, sizeof(trans_prev2));
            has_prev2 = true;
        }
        std::memcpy(theta_prev, theta, sizeof(theta_prev));
        std::memcpy(trans_prev, trans, sizeof(trans_prev));
        has_prev = true;

        // update affine prev
        depth_affine_prev[0] = depth_affine[0];
        depth_affine_prev[1] = depth_affine[1];

        // Update Z_cam for camera-space depth prior (world-coord mode)
        if (use_world_coords && f < (int)cam_poses.size()) {
            const auto& ext = cam_poses[f];
            double Z_cam_cur = ext.R(2,0)*trans[0] + ext.R(2,1)*trans[1] + ext.R(2,2)*trans[2] + ext.t(2);
            Z_cam_prev2 = Z_cam_prev;
            Z_cam_prev = Z_cam_cur;
        }

        // Compute Z_cam for drift monitoring
        double Z_cam_cur = 0.0;
        if (use_world_coords && f < (int)cam_poses.size()) {
            const auto& ext = cam_poses[f];
            Z_cam_cur = ext.R(2,0)*trans[0] + ext.R(2,1)*trans[1] + ext.R(2,2)*trans[2] + ext.t(2);
        } else {
            Z_cam_cur = trans[2];
        }

        // Key drift indicators:
        // - ΔZ_cam: change in camera-space depth (should be small for static person)
        // - Z_cam: absolute depth (should stay ~3m for typical scene)
        static double Z_cam_first = 0.0;
        if (f == start) Z_cam_first = Z_cam_cur;
        double Z_cam_drift = Z_cam_cur - Z_cam_first;  // accumulated drift from first frame

        std::cout << "[F" << f << "] Z_cam=" << std::fixed << std::setprecision(3) << Z_cam_cur
                  << " | ΔZ_cam=" << std::showpos << (Z_cam_cur - Z_cam_prev) << std::noshowpos
                  << " | drift=" << std::showpos << Z_cam_drift << std::noshowpos
                  << " | scale=" << depth_affine[0]
                  << "\n";

        // ---------------------------
        // Evaluation against GT (silent per-frame)
        // ---------------------------
        if (has_gt && f < gt_data.num_frames) {
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_pred;
            SMPLJointsForward<double>(beta, theta, trans, model, J_pred);

            // Choose appropriate GT based on coordinate system:
            // - World coords: J_pred is in world space, compare with gt_joints_world.txt
            // - Camera coords: J_pred is in camera space, compare with gt_joints.txt
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_gt;
            bool got_gt = false;
            if (use_world_coords && gt_data.has_world_data) {
                got_gt = gt_data.GetJointsWorld(f, J_gt);
            } else {
                got_gt = gt_data.GetJoints(f, J_gt);
            }

            if (got_gt) {
                double mpjpe = ComputeMPJPE(J_pred, J_gt);
                double pa_mpjpe = ComputePAMPJPE(J_pred, J_gt);
                all_mpjpe.push_back(mpjpe);
                all_pa_mpjpe.push_back(pa_mpjpe);

                // Debug: print joint comparison every 10 frames
                if (f % 10 == 0) {
                    std::cout << "\n=== DEBUG: Joint comparison (frame " << f << ", "
                              << (use_world_coords ? "world" : "camera") << " coords) ===\n";
                    std::cout << "trans: (" << trans[0] << ", " << trans[1] << ", " << trans[2] << ")\n";
                    std::cout << "Joint 0 (pelvis): pred=(" << J_pred(0,0) << ", " << J_pred(0,1) << ", " << J_pred(0,2)
                              << ") gt=(" << J_gt(0,0) << ", " << J_gt(0,1) << ", " << J_gt(0,2) << ")\n";
                    std::cout << "Joint 12 (neck): pred=(" << J_pred(12,0) << ", " << J_pred(12,1) << ", " << J_pred(12,2)
                              << ") gt=(" << J_gt(12,0) << ", " << J_gt(12,1) << ", " << J_gt(12,2) << ")\n";
                    std::cout << "MPJPE=" << mpjpe*1000 << "mm, PA-MPJPE=" << pa_mpjpe*1000 << "mm\n";
                    std::cout << "===========================================\n\n";
                }
            }

            // PVE: 3DPW doesn't have GT vertices, skip PVE computation
            // (would need to run SMPL forward with GT theta/beta to get GT vertices)

            // Pose Error (theta parameters) - use world-space pose if available
            double theta_gt[72];
            bool got_pose = false;
            if (use_world_coords && !gt_data.poses_world.empty()) {
                got_pose = gt_data.GetPoseWorld(f, theta_gt);
            } else {
                got_pose = gt_data.GetPose(f, theta_gt);
            }
            if (got_pose) {
                all_pose_error.push_back(ComputePoseError(theta, theta_gt, false));
            }

            // 2D Reprojection Error - project joints and compare with 2D keypoints
            // For world-space optimization, need to transform to camera space first
            double reproj_err = 0.0;
            int reproj_count = 0;
            for (const auto& lm : lms) {
                if (lm.confidence < 0.3f) continue;
                int op_idx = lm.joint_id;  // OpenPose joint ID
                if (op_idx < 0 || op_idx >= 25) continue;
                int smpl_idx = OPENPOSE_TO_SMPL[op_idx];  // Convert to SMPL ID
                if (smpl_idx < 0 || smpl_idx >= J_pred.rows()) continue;

                double px, py, pz;
                if (use_world_coords && f < (int)cam_poses.size()) {
                    // Transform world to camera: p_cam = R * p_world + t
                    Eigen::Vector3d p_world(J_pred(smpl_idx, 0), J_pred(smpl_idx, 1), J_pred(smpl_idx, 2));
                    Eigen::Vector3d p_cam = cam_poses[f].R * p_world + cam_poses[f].t;
                    px = p_cam(0);
                    py = p_cam(1);
                    pz = p_cam(2);
                } else {
                    // Already in camera space
                    px = J_pred(smpl_idx, 0);
                    py = J_pred(smpl_idx, 1);
                    pz = J_pred(smpl_idx, 2);
                }
                if (pz <= 0) continue;

                // Note: 3DPW camera has Y+ down, so no flip needed after transform
                // But our preprocessing flipped Y in gt_joints.txt, so we need to be consistent
                double u_pred = camera.fx * px / pz + camera.cx;
                double v_pred = camera.fy * py / pz + camera.cy;  // 3DPW camera Y+ down, no flip

                double err = std::sqrt((u_pred - lm.x) * (u_pred - lm.x) +
                                       (v_pred - lm.y) * (v_pred - lm.y));
                reproj_err += err;
                reproj_count++;
            }
            if (reproj_count > 0) {
                all_reproj_2d.push_back(reproj_err / reproj_count);
            }
        }

        std::cout << "[Frame " << f << "] done\n";

        // Print intermediate evaluation every 10 frames
        if (!all_mpjpe.empty() && all_mpjpe.size() % 10 == 0) {
            double mean_mpjpe = 0.0, mean_pa_mpjpe = 0.0;
            for (size_t i = 0; i < all_mpjpe.size(); ++i) {
                mean_mpjpe += all_mpjpe[i];
                mean_pa_mpjpe += all_pa_mpjpe[i];
            }
            mean_mpjpe /= all_mpjpe.size();
            mean_pa_mpjpe /= all_pa_mpjpe.size();
            std::cout << "[Eval @ " << all_mpjpe.size() << " frames] MPJPE="
                      << mean_mpjpe * 1000.0 << "mm, PA-MPJPE="
                      << mean_pa_mpjpe * 1000.0 << "mm\n";
        }
    }

    // ---------------------------
    // Print and save evaluation summary
    // ---------------------------
    if (!all_mpjpe.empty()) {
        // Compute means
        auto compute_mean = [](const std::vector<double>& v) -> double {
            if (v.empty()) return -1.0;
            double sum = 0.0;
            for (double x : v) sum += x;
            return sum / v.size();
        };

        double mean_mpjpe = compute_mean(all_mpjpe);
        double mean_pa_mpjpe = compute_mean(all_pa_mpjpe);
        double mean_pve = compute_mean(all_pve);
        double mean_pose_error = compute_mean(all_pose_error);
        double mean_reproj_2d = compute_mean(all_reproj_2d);

        // Also compute shape error (single value across all frames)
        double shape_error = -1.0;
        if (gt_data.has_betas) {
            double beta_gt[10];
            if (gt_data.GetBetas(beta_gt)) {
                shape_error = ComputeShapeError(beta, beta_gt);
            }
        }

        std::cout << "\n========== EVALUATION SUMMARY ==========\n";
        std::cout << "  Frames evaluated: " << all_mpjpe.size() << "\n";
        std::cout << "  Mean MPJPE:       " << mean_mpjpe * 1000.0 << " mm\n";
        std::cout << "  Mean PA-MPJPE:    " << mean_pa_mpjpe * 1000.0 << " mm\n";
        if (mean_pve >= 0)
            std::cout << "  Mean PVE:         " << mean_pve * 1000.0 << " mm\n";
        if (mean_pose_error >= 0)
            std::cout << "  Mean Pose Error:  " << mean_pose_error << " rad\n";
        if (shape_error >= 0)
            std::cout << "  Shape Error:      " << shape_error << "\n";
        if (mean_reproj_2d >= 0)
            std::cout << "  Mean 2D Reproj:   " << mean_reproj_2d << " px\n";
        std::cout << "=========================================\n";

        // Save evaluation results to file
        std::ofstream eval_file(out_dir + "/evaluation.txt");
        if (eval_file.is_open()) {
            eval_file << "# Evaluation Summary\n";
            eval_file << "Frames evaluated: " << all_mpjpe.size() << "\n\n";

            eval_file << "# Mean Metrics\n";
            eval_file << "Mean MPJPE (mm): " << mean_mpjpe * 1000.0 << "\n";
            eval_file << "Mean PA-MPJPE (mm): " << mean_pa_mpjpe * 1000.0 << "\n";
            if (mean_pve >= 0)
                eval_file << "Mean PVE (mm): " << mean_pve * 1000.0 << "\n";
            if (mean_pose_error >= 0)
                eval_file << "Mean Pose Error (rad): " << mean_pose_error << "\n";
            if (shape_error >= 0)
                eval_file << "Shape Error: " << shape_error << "\n";
            if (mean_reproj_2d >= 0)
                eval_file << "Mean 2D Reprojection (px): " << mean_reproj_2d << "\n";

            eval_file << "\n# Per-frame Metrics\n";
            eval_file << "# frame_idx MPJPE(mm) PA-MPJPE(mm) PVE(mm) PoseErr(rad) 2DReproj(px)\n";
            for (size_t i = 0; i < all_mpjpe.size(); ++i) {
                eval_file << i << " "
                          << all_mpjpe[i] * 1000.0 << " "
                          << all_pa_mpjpe[i] * 1000.0 << " "
                          << (i < all_pve.size() ? all_pve[i] * 1000.0 : -1.0) << " "
                          << (i < all_pose_error.size() ? all_pose_error[i] : -1.0) << " "
                          << (i < all_reproj_2d.size() ? all_reproj_2d[i] : -1.0) << "\n";
            }
            eval_file.close();
            std::cout << "[Saved] " << out_dir << "/evaluation.txt\n";
        }
    }

    std::cout << "[Done] sequence saved under " << out_dir << "\n";
}


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

// ------------------------------
// Batch validation 
// ------------------------------
void RunBatchValidation(const SmplModel& model) {
    std::cout << "\n========== BATCH AMASS VALIDATION (Most Complex Frames) ==========\n";

    std::vector<std::pair<std::string, int>> files = {
        {"../../Data/sampled/CMU_80_80_33_poses.txt", 1303},
        {"../../Data/sampled/CMU_80_80_40_poses.txt", 504},
        {"../../Data/sampled/CMU_82_82_06_poses.txt", 250},
        {"../../Data/sampled/CMU_82_82_12_poses.txt", 117},
        {"../../Data/sampled/CMU_83_83_19_poses.txt", 342},
    };

    std::vector<double> errors;
    int success = 0, fail = 0;

    for (const auto& item : files) {
        double err = RunAMASSValidation(model, item.first, item.second, true);
        if (err >= 0) { errors.push_back(err); success++; }
        else { fail++; }
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
    std::cout << ">>> Start SMPL Project...\n";

    SmplModel model;
    model.LoadFromNPZ("../../Data/models/smpl_neutral.npz");

    // Load GMM pose prior
    g_gmm_prior.LoadFromNPZ("../../Data/models/gmm_08.npz");

    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "--batch") {
            RunBatchValidation(model);
        } else if (arg1 == "--openpose") {
            RunOpenPoseInference(model);
        } else if (arg1 == "--seq-val") {
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0] << " --seq-val <pose_file.txt>\n";
                return 1;
            }
            RunAmassValidationSequence(model, argv[2]);
        } else {
            RunAMASSValidation(model, arg1, 100, true);
        }
        
    } else {
        RunOpenPoseSequenceInference(model);
        //RunOpenPoseInference(model);
    }

    return 0;
}
