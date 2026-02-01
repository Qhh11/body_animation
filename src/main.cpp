#include <algorithm>
#include <fstream>
#include <iostream>
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
#include <filesystem>
namespace fs = std::filesystem;

using json = nlohmann::json;

// Forward declarations
void SaveOBJ(const std::string& filename,
             const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vertices,
             const Eigen::MatrixXi& faces);

// Global GMM pose prior
static GMMPosePrior g_gmm_prior;

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

    // depth (
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

    // arm-torso sep
    cfg.use_arm_torso_sep = true;

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
// MODE 2: Real OpenPose
// =========================================================
void RunOpenPoseInference(const SmplModel& model) {
    std::cout << "\n=== MODE 2: Running Real OpenPose Data ===\n";

    std::string json_path = "../../Data/openpose_output/courtyard_basketball_01/image_00047_keypoints.json";
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
    const std::string out_dir   = "./seq_out";
    fs::create_directories(out_dir);

    const int start = 0;
    const int end   = 200;

    const int img_w = 1080;
    const int img_h = 1920;
    CameraIntrinsics camera(1962.0f, 1969.0f, 540.0f, 960.0f, img_w, img_h);

    OptimizerConfig cfg = MakeCfgForOpenPose();

    cfg.use_temporal = true;
    cfg.w_temp_pose  = 12.0;
    cfg.w_temp_trans = 6.0;
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

    bool has_prev  = false;
    bool has_prev2 = false;

    double depth_affine[2]      = {1e-3, 3.0};
    double depth_affine_prev[2] = {1e-3, 3.0};

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
        // stable set (no wrists/feet)
        int ids[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14};
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
            EstimateGlobalTranslationXY(lms, camera, 3.0, trans);
            depth_affine[0] = 1e-3;
            depth_affine[1] = trans[2];
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

                depth_pts.push_back({smpl_idx, depth_val, c2d});
            }
            depth_file.close();
        } else {
            // it's ok: depth may be missing for some frames
        }

        if ((int)depth_pts.size() >= 6) {
            cfg.use_depth_affine = true;
            cfg.w_depth_affine = 30.0;
            cfg.depth_affine_sign = -1.0; // larger rel_depth => closer => smaller z
            cfg.depth_affine_data   = &depth_pts;
            cfg.depth_affine_params = depth_affine;

            cfg.use_temporal_depth_affine = has_prev;
            cfg.depth_affine_prev = has_prev ? depth_affine_prev : nullptr;
            cfg.w_temp_depth_scale = 80.0;
            cfg.w_temp_depth_bias  = 200.0;
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

        bool ok = false;
        if (!has_prev) {
            ok = OptimizeSMPL(lms, camera, model, beta, theta, trans, cfg);
        } else {
            ok = OptimizeSMPLTracked(lms, camera, model, beta, theta, trans, cfg,
                                     theta_prev, trans_prev,
                                     has_prev2 ? theta_prev2 : nullptr);
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
            } else {
                // allow downward update faster, upward update slower
                if (y_min < ground_y_est) {
                    ground_y_est = 0.7 * y_min + 0.3 * ground_y_est;
                } else {
                    ground_y_est = 0.98 * ground_y_est + 0.02 * y_min;
                }
            }

            cfg.ground_y = ground_y_est; // for next frame
        }
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V;
        SMPLForward<double>(beta, theta, trans, model, V);
        SaveOBJ(MakeOutObj(out_dir, f), V, model.faces);

        // shift prev buffers
        if (has_prev) {
            std::memcpy(theta_prev2, theta_prev, sizeof(theta_prev2));
            has_prev2 = true;
        }
        std::memcpy(theta_prev, theta, sizeof(theta_prev));
        std::memcpy(trans_prev, trans, sizeof(trans_prev));
        has_prev = true;

        // update affine prev
        depth_affine_prev[0] = depth_affine[0];
        depth_affine_prev[1] = depth_affine[1];

        std::cout << "[Frame " << f << "] done | trans=("
                  << trans[0] << "," << trans[1] << "," << trans[2] << ")"
                  << " | affine=(" << depth_affine[0] << "," << depth_affine[1] << ")"
                  << " | ground_y=" << cfg.ground_y
                  << " | contact(L,R)=(" << cfg.foot_contact_L << "," << cfg.foot_contact_R << ")"
                  << "\n";
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
        } else {
            RunAMASSValidation(model, arg1, 100, true);
        }
    } else {
        RunOpenPoseSequenceInference(model);
        //RunOpenPoseInference(model);
    }

    return 0;
}
