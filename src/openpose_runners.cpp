#include "openpose_runners.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <sstream>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>

#include "smpl_math.h"
#include "evaluation.h"
#include "logging.h"
#include "io_utils.h"
#include "openpose_io.h"
#include "init_utils.h"
#include "config_builders.h"
#include "threedpw_gt.h"
#include "camera_utils.h" 
#include "multi_hypothesis.h"
#include "init_utils.h"
#include "post_smoothing.h"

namespace fs = std::filesystem;

namespace app {

// =========================================================
// MODE 2: Real OpenPose (single frame)
// =========================================================
void RunOpenPoseInference(const SmplModel& model,
                          const GMMPosePrior* gmm_prior) {
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

    OptimizerConfig cfg = MakeCfgForOpenPose(gmm_prior);
    cfg.trans_z_prior_target = trans[2];

    std::vector<RelativeDepthPair> relative_depth_pairs;
    std::string depth_path = "../../Data/sapiens_depth/courtyard_basketball_01/image_00047_depth.txt";
    std::ifstream depth_file(depth_path);
    if (depth_file.is_open()) {
        std::vector<std::pair<int, double>> kp_depths;
        int op_idx; double depth_val;
        while (depth_file >> op_idx >> depth_val) {
            int smpl_idx = OPENPOSE_TO_SMPL[op_idx];
            if (smpl_idx >= 0) kp_depths.push_back({smpl_idx, depth_val});
        }
        depth_file.close();
        for (size_t i = 0; i < kp_depths.size(); ++i)
            for (size_t j = i + 1; j < kp_depths.size(); ++j)
                relative_depth_pairs.push_back({kp_depths[i].first, kp_depths[j].first,
                                                kp_depths[i].second, kp_depths[j].second});
        std::cout << "[Depth] Created " << relative_depth_pairs.size() << " relative depth pairs\n";
    }
    if (!relative_depth_pairs.empty()) {
        cfg.use_relative_depth = true;
        cfg.w_relative_depth = 20.0;
        cfg.relative_depth_pairs = &relative_depth_pairs;
    }

    const bool ok = OptimizeSMPL(lms, camera, model, beta, theta, trans, cfg);
    if (!ok) { std::cerr << "OptimizeSMPL failed.\n"; return; }

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_opt;
    SMPLForward<double>(beta, theta, trans, model, V_opt);
    SaveOBJ("mesh_after_opt.obj", V_opt, model.faces);
    std::cout << "[Done] saved mesh_after_opt.obj\n";
}


void RunOpenPoseSequenceInference(const SmplModel& model,
                                  const GMMPosePrior* gmm_prior,
                                  const std::string& seq_name,
                                  const std::string& out_dir,
                                  int start_frame,
                                  int end_frame) {
    std::cout << "\n=== MODE 3: OpenPose SEQUENCE (Camera-Space) + Two-Pass Smoothing ===\n";

    const std::string seq_dir   = "../../Data/openpose_output/" + seq_name;
    const std::string depth_dir = "../../Data/sapiens_depth/"    + seq_name;
    const std::string gt_dir    = "../../Data/3dpw_gt/"          + seq_name;
    fs::create_directories(out_dir);

    const int start = start_frame;
    const int end   = end_frame;

    if (end < start) {
        std::cerr << "[Args] invalid range: end < start (" << start << "," << end << ")\n";
        return;
    }

    // ---------------------------
    // Load ground truth
    // ---------------------------
    ThreeDPW_GT gt_data;
    bool has_gt = gt_data.LoadFromDir(gt_dir);
    if (has_gt) {
        std::cout << "[GT] Loaded " << gt_data.num_frames << " frames\n";
        if (gt_data.has_poses)      std::cout << "[GT] Has pose (theta)\n";
        if (gt_data.has_betas)      std::cout << "[GT] Has shape (beta)\n";
        if (gt_data.has_world_data) std::cout << "[GT] Has world-space joints\n";
    } else {
        std::cout << "[GT] No ground truth, metrics will be -1\n";
    }

    std::vector<CameraExtrinsics> cam_poses = LoadCameraPoses(gt_dir + "/cam_poses.txt");
    const bool has_cam_poses = !cam_poses.empty();
    if (has_cam_poses)
        std::cout << "[Camera] Loaded " << cam_poses.size() << " poses\n";

    // ---------------------------
    // Evaluation accumulators
    // ---------------------------
    std::vector<double> all_mpjpe, all_pa_mpjpe, all_pve, all_pose_error, all_reproj_2d;
    std::vector<double> all_mpjpe_rootrel;
    std::vector<double> all_cam_rootrel, all_cam_pa;

    const int img_w = 1080;
    const int img_h = 1920;
    CameraIntrinsics camera(1962.0f, 1969.0f, 540.0f, 960.0f, img_w, img_h);

    OptimizerConfig cfg = MakeCfgForOpenPoseSequence(gmm_prior);

    // ---------------------------
    // Logger
    // ---------------------------
    logutil::RunMeta meta;
    meta.mode = "OPENPOSE_SEQ_CAMSPACE_2PASS";
    meta.seq_dir = seq_dir;
    meta.depth_dir = depth_dir;
    meta.gt_dir = gt_dir;
    meta.out_dir = out_dir;
    meta.start_frame = start;
    meta.end_frame = end;
    meta.use_world_coords = false;
    meta.has_cam_poses = has_cam_poses;

    logutil::EvalLogger elog;
    bool logger_ok = false;
    if (!elog.Open(out_dir, "evaluation.txt", "config.txt")) {
        std::cerr << "[Logger] failed to open\n";
    } else {
        logger_ok = true;
        elog.WriteConfig(cfg, &camera, &meta);
        elog.WriteEvalHeader(meta);
    }

    // ---------------------------
    // Camera-space parameters
    // ---------------------------
    double beta[10]  = {0.0};
    double theta[72] = {0.0};
    double trans[3]  = {0.0, 0.0, 3.0};

    double theta_prev_raw[72]  = {0.0};
    double trans_prev_raw[3]   = {0.0, 0.0, 3.0};
    double theta_prev2_raw[72] = {0.0};
    double trans_prev2_raw[3]  = {0.0, 0.0, 3.0};

    double theta_prev_curr[72]  = {0.0};
    double trans_prev_curr[3]   = {0.0, 0.0, 3.0};
    double theta_prev2_curr[72] = {0.0};
    double trans_prev2_curr[3]  = {0.0, 0.0, 3.0};

    bool has_prev  = false;
    bool has_prev2 = false;
    int  prev_frame_idx  = -1;
    int  prev2_frame_idx = -1;

    double depth_affine[2]      = {3.0, 3.0};
    double depth_affine_prev[2] = {3.0, 3.0};

    // 2D keypoints EMA smoother
    bool kp_ema_init = false;
    double kp_x[25], kp_y[25], kp_seen[25];

    auto smooth_landmarks_inplace = [&](std::vector<Landmark2D>& lms) {
        if (!kp_ema_init) {
            for (int i=0;i<25;++i){ kp_x[i]=0; kp_y[i]=0; kp_seen[i]=0; }
            kp_ema_init = true;
        }
        static const std::unordered_set<int> arm_ids = {2,3,4,5,6,7};

        for (auto& lm : lms) {
            int id = lm.joint_id;
            if (id < 0 || id >= 25) continue;
            if (lm.confidence < 0.20f) continue;

            if (arm_ids.count(id)) {
                kp_x[id] = lm.x; kp_y[id] = lm.y; kp_seen[id] = 1.0;
                continue;
            }

            double a = 0.65 + 0.30 * std::min(1.0, std::max(0.0, (double)lm.confidence));
            if (kp_seen[id] < 0.5) {
                kp_x[id] = lm.x; kp_y[id] = lm.y; kp_seen[id] = 1.0;
            } else {
                kp_x[id] = a*lm.x + (1.0-a)*kp_x[id];
                kp_y[id] = a*lm.y + (1.0-a)*kp_y[id];
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
        for (auto& lm: lms) if (lm.joint_id == op_id) { x=lm.x; y=lm.y; return true; }
        return false;
    };

    // Stable joints for depth: all upper body
    auto is_stable_op = [&](int op_id) -> bool {
        for (int x : {1, 2, 3, 4, 5, 6, 7, 8}) if (x == op_id) return true;
        return false;
    };

    bool ground_init = false;
    double ground_y_est = 0.0;
    double z_anchor = 3.0;
    bool ankle_prev_valid = false;
    double prev_LA_x=0, prev_LA_y=0, prev_RA_x=0, prev_RA_y=0;

    auto compute_contact = [&](const std::vector<Landmark2D>& lms, bool is_left)->double{
        int op_ank = is_left ? 14 : 11;
        int op_kne = is_left ? 13 : 10;
        const double cA = get_conf(lms, op_ank);
        const double cK = get_conf(lms, op_kne);
        if (cA < 0.35 || cK < 0.35) return 0.0;
        double ax, ay;
        if (!get_xy(lms, op_ank, ax, ay)) return 0.0;
        double y_norm = ay / double(img_h);
        double ground_like = (y_norm > 0.80) ? 1.0 :
                             (y_norm > 0.72) ? (y_norm - 0.72) / 0.08 : 0.0;
        double speed_like = 1.0;
        if (ankle_prev_valid) {
            double px = is_left ? prev_LA_x : prev_RA_x;
            double py = is_left ? prev_LA_y : prev_RA_y;
            double sp = std::sqrt((ax-px)*(ax-px) + (ay-py)*(ay-py));
            if (sp > 18.0) speed_like = 0.0;
            else if (sp > 6.0) speed_like = (18.0 - sp) / 12.0;
        }
        return std::clamp(ground_like * speed_like * std::min(1.0, 0.5*(cA+cK)), 0.0, 1.0);
    };

    auto ComputeReproj2DScore = [&](const std::vector<Landmark2D>& lms,
                                const double beta_[10],
                                const double theta_[72],
                                const double trans_[3],
                                const OptimizerConfig& cfg_)->double
    {
        MatX3dRM J;
        SMPLJointsForward<double>(beta_, theta_, trans_, model, J);
        double err = 0.0; int cnt = 0;
        for (const auto& lm : lms) {
            if (lm.confidence < cfg_.min_confidence) continue;
            int op_idx = lm.joint_id;
            if (op_idx < 0 || op_idx >= 25) continue;
            int smpl_idx = OPENPOSE_TO_SMPL[op_idx];
            if (smpl_idx < 0 || smpl_idx >= J.rows()) continue;
            double X = J(smpl_idx, 0), Y = J(smpl_idx, 1), Z = J(smpl_idx, 2);
            if (Z <= 1e-6) continue;
            double u = camera.fx * X / Z + camera.cx;
            double v = camera.fy * (cfg_.flip_y_in_projection ? -Y : Y) / Z + camera.cy;
            double du = u - lm.x, dv = v - lm.y;
            err += std::sqrt(du*du + dv*dv);
            cnt++;
        }
        return (cnt > 0) ? (err / cnt) : 1e18;
    };

    // Helper: compute mean 2D confidence of landmarks
    auto ComputeMeanConfidence = [&](const std::vector<Landmark2D>& lms,
                                     int& num_visible) -> double
    {
        double sum_conf = 0.0;
        num_visible = 0;
        for (const auto& lm : lms) {
            if (lm.confidence > 0.1f) {
                sum_conf += lm.confidence;
                num_visible++;
            }
        }
        return (num_visible > 0) ? sum_conf / num_visible : 0.0;
    };

    // =====================================================
    // FIRST-FRAME YAW MULTI-START
    // =====================================================
    auto InitFirstFrameYawByMultiStart = [&](const std::vector<Landmark2D>& lms,
                                            double beta_[10],
                                            double theta_[72],
                                            double trans_[3],
                                            OptimizerConfig cfg_base)
    {
        cfg_base.use_temporal = false;
        cfg_base.use_temporal_acc = false;
        cfg_base.max_iter_stage1 = 60;
        cfg_base.max_iter_stage2 = 0;
        cfg_base.max_iter_stage3 = 0;
        cfg_base.print_progress = false;
        cfg_base.lock_hands = false;

        auto try_one = [&](double yaw_deg, double& out_score,
                        double out_theta[72], double out_trans[3])
        {
            double theta_try[72]; std::memcpy(theta_try, theta_, sizeof(theta_try));
            double trans_try[3];  std::memcpy(trans_try, trans_, sizeof(trans_try));
            double beta_try[10];  std::memcpy(beta_try,  beta_,  sizeof(beta_try));
            theta_try[1] = yaw_deg * M_PI / 180.0;
            bool ok = OptimizeSMPL(lms, camera, model, beta_try, theta_try, trans_try, cfg_base);
            if (!ok) { out_score = 1e18; return; }
            out_score = ComputeReproj2DScore(lms, beta_try, theta_try, trans_try, cfg_base);
            std::memcpy(out_theta, theta_try, sizeof(theta_try));
            std::memcpy(out_trans, trans_try, sizeof(trans_try));
        };

        double best_score = 1e18, best_yaw = 0.0;
        double best_theta[72], best_trans[3];

        for (double yaw = -90; yaw <= 90; yaw += 30) {
            double s; double th[72]; double tr[3];
            try_one(yaw, s, th, tr);
            if (s < best_score) {
                best_score = s; best_yaw = yaw;
                std::memcpy(best_theta, th, sizeof(th));
                std::memcpy(best_trans, tr, sizeof(tr));
            }
        }
        double fine_best_score = best_score;
        double fine_best_theta[72]; std::memcpy(fine_best_theta, best_theta, sizeof(best_theta));
        double fine_best_trans[3];  std::memcpy(fine_best_trans, best_trans, sizeof(best_trans));
        double fine_best_yaw = best_yaw;

        for (double yaw = best_yaw - 20; yaw <= best_yaw + 20; yaw += 5) {
            double s; double th[72]; double tr[3];
            try_one(yaw, s, th, tr);
            if (s < fine_best_score) {
                fine_best_score = s; fine_best_yaw = yaw;
                std::memcpy(fine_best_theta, th, sizeof(th));
                std::memcpy(fine_best_trans, tr, sizeof(tr));
            }
        }

        if (fine_best_score < 1e17) {
            double best_yaw_rad = fine_best_theta[1];
            std::memset(theta_, 0, 72 * sizeof(double));
            theta_[1] = best_yaw_rad;
            std::memcpy(trans_, fine_best_trans, sizeof(fine_best_trans));
            cfg.trans_z_prior_target = trans_[2];

            std::cout << "[InitYaw] best reproj=" << fine_best_score
                    << " px, yaw_deg=" << fine_best_yaw
                    << " (rad=" << theta_[1] << ")\n";
        }
    };


    // =====================================================
    // PASS 1: FORWARD CAUSAL OPTIMIZATION
    // Store all results for post-processing
    // =====================================================
    std::vector<postsmooth::FrameResult> all_results;
    all_results.reserve(end - start + 1);

    // Also store per-frame landmarks for evaluation later
    std::vector<std::vector<Landmark2D>> all_landmarks;
    all_landmarks.reserve(end - start + 1);

    int success_count = 0, fail_count = 0;

    for (int f = start; f <= end; ++f) {
        std::string json_path = MakeOpenPoseJsonPath(seq_dir, f);
        std::vector<Landmark2D> lms;

        // Reset per-frame depth flags
        cfg.use_relative_depth   = false;
        cfg.relative_depth_pairs = nullptr;
        cfg.use_depth            = false;
        cfg.depth_data           = nullptr;
        cfg.use_depth_affine     = false;
        cfg.depth_affine_data    = nullptr;
        cfg.depth_affine_params  = nullptr;
        cfg.depth_affine_prev    = nullptr;
        cfg.use_cross_limb_depth = false;
        cfg.cross_limb_pairs     = nullptr;

        if (!LoadLandmarksFromJSON(json_path, lms)) {
            std::cerr << "[Frame " << f << "] failed to load: " << json_path << "\n";
            fail_count++;

            // Store failed frame result
            postsmooth::FrameResult fr;
            fr.frame_idx = f;
            fr.solver_ok = false;
            all_results.push_back(fr);
            all_landmarks.push_back({});
            continue;
        }

        smooth_landmarks_inplace(lms);

        // Compute 2D observation quality for this frame
        int num_visible = 0;
        double mean_conf = ComputeMeanConfidence(lms, num_visible);

        // =====================================================
        // ADAPTIVE TEMPORAL WEIGHTING
        // Low confidence → trust temporal more, trust 2D less
        // =====================================================
        OptimizerConfig cfg_frame = cfg;  // per-frame copy for adaptive weights
        if (has_prev) {
            double conf_factor = std::clamp(mean_conf, 0.3, 1.0);
            // Scale temporal up when confidence is low (factor 1.0 to 1.7)
            double temporal_boost = 2.0 - conf_factor;
            cfg_frame.w_temp_pose  = cfg.w_temp_pose * temporal_boost;
            cfg_frame.w_temp_trans = cfg.w_temp_trans * temporal_boost;
            cfg_frame.w_temp_root_orient = cfg.w_temp_root_orient * temporal_boost;
        }

        // =====================================================
        // WARM START
        // =====================================================
        if (has_prev) {
            if (has_cam_poses &&
                prev_frame_idx >= 0 && prev_frame_idx < (int)cam_poses.size() &&
                f < (int)cam_poses.size())
            {
                Eigen::Matrix3d R_rel; Eigen::Vector3d t_rel;
                ComputeRelativeCameraTransform(cam_poses[prev_frame_idx],
                                               cam_poses[f], R_rel, t_rel);
                TransformPrevParamsToCurrCam(theta_prev_raw, trans_prev_raw,
                                            R_rel, t_rel,
                                            theta_prev_curr, trans_prev_curr);
                if (has_prev2 && prev2_frame_idx >= 0 &&
                    prev2_frame_idx < (int)cam_poses.size()) {
                    Eigen::Matrix3d R_rel2; Eigen::Vector3d t_rel2;
                    ComputeRelativeCameraTransform(cam_poses[prev2_frame_idx],
                                                   cam_poses[f], R_rel2, t_rel2);
                    TransformPrevParamsToCurrCam(theta_prev2_raw, trans_prev2_raw,
                                                R_rel2, t_rel2,
                                                theta_prev2_curr, trans_prev2_curr);
                }
            } else {
                std::memcpy(theta_prev_curr, theta_prev_raw, sizeof(theta_prev_curr));
                std::memcpy(trans_prev_curr, trans_prev_raw, sizeof(trans_prev_curr));
                std::memcpy(theta_prev2_curr, theta_prev2_raw, sizeof(theta_prev2_curr));
                std::memcpy(trans_prev2_curr, trans_prev2_raw, sizeof(trans_prev2_curr));
            }

            std::memcpy(theta, theta_prev_curr, sizeof(theta));
            std::memcpy(trans, trans_prev_curr, sizeof(trans));
            depth_affine[0] = depth_affine_prev[0];
            depth_affine[1] = depth_affine_prev[1];
            cfg_frame.trans_z_prior_target = z_anchor;
        } else {
            EstimateGlobalTranslationXY(lms, camera, 3.0, trans);
            InitFirstFrameYawByMultiStart(lms, beta, theta, trans, cfg_frame); 
            depth_affine[0] = 3.0;
            depth_affine[1] = 3.0;
        }

        // =====================================================
        // LOAD DEPTH with confidence weighting & outlier rejection
        // =====================================================
        std::vector<JointRelDepthData> depth_pts;
        {
            std::string depth_path = MakeOpenPoseDepthPath(depth_dir, f);
            std::ifstream df(depth_path);
            if (df.is_open()) {
                int op_idx; double depth_val;
                std::vector<std::pair<int, double>> all_depths;

                while (df >> op_idx >> depth_val) {
                    if (op_idx < 0 || op_idx >= 25) continue;
                    if (!is_stable_op(op_idx)) continue;
                    double c2d = get_conf(lms, op_idx);
                    if (c2d < 0.35) continue;
                    all_depths.push_back({op_idx, depth_val});
                }

                // Compute median torso depth for outlier detection
                std::vector<double> torso_depths;
                for (auto& [oid, dv] : all_depths) {
                    if (oid == 1 || oid == 8 || oid == 2 || oid == 5)
                        torso_depths.push_back(dv);
                }
                double median_d = 0;
                if (!torso_depths.empty()) {
                    std::sort(torso_depths.begin(), torso_depths.end());
                    median_d = torso_depths[torso_depths.size()/2];
                }

                for (auto& [oid, dv] : all_depths) {
                    int smpl_idx = OPENPOSE_TO_SMPL[oid];
                    if (smpl_idx < 0) continue;

                    double c2d = get_conf(lms, oid);
                    double depth_conf = c2d;
                    // Per-joint depth confidence
                    bool is_wrist = (oid == 4 || oid == 7);
                    bool is_elbow = (oid == 3 || oid == 6);
                    if (is_wrist) depth_conf *= 0.5;     
                    else if (is_elbow) depth_conf *= 0.4;

                    // Outlier rejection
                    if (median_d > 0 && std::abs(dv - median_d) > 3.0 * std::abs(median_d) + 0.1) {
                        depth_conf *= 0.1;
                    }

                    depth_pts.push_back({smpl_idx, dv, depth_conf});
                }
            }
        }

        // Cross-limb depth pairs
        std::vector<RelativeDepthPair> cross_limb_pairs;
        {
            std::unordered_map<int, double> jdepth;
            for (auto& dp : depth_pts) jdepth[dp.smpl_joint_idx] = dp.rel_depth;

            int cross_pairs[][2] = {{20, 21}, {18, 19}, {16, 17}};
            for (auto& cp : cross_pairs) {
                if (jdepth.count(cp[0]) && jdepth.count(cp[1]))
                    cross_limb_pairs.push_back({cp[0], cp[1], jdepth[cp[0]], jdepth[cp[1]]});
            }

            int chain_pairs[][2] = {{17, 19}, {19, 21}, {16, 18}, {18, 20}};
            for (auto& cp : chain_pairs) {
                if (jdepth.count(cp[0]) && jdepth.count(cp[1]))
                    cross_limb_pairs.push_back({cp[0], cp[1], jdepth[cp[0]], jdepth[cp[1]]});
            }

            int torso_joints[] = {0, 3, 6, 9};  
            int arm_joints[]   = {16, 17, 18, 19, 20, 21};  
            for (int tj : torso_joints) {
                for (int aj : arm_joints) {
                    if (jdepth.count(tj) && jdepth.count(aj)) {
                        cross_limb_pairs.push_back({aj, tj, jdepth[aj], jdepth[tj]});
                    }
                }
            }
        }

        // Setup depth affine
        if ((int)depth_pts.size() >= 6) {
            cfg_frame.use_depth_affine    = true;
            if (f == start) cfg_frame.w_depth_affine = 400.0;
            cfg_frame.depth_affine_sign   = 1.0;
            cfg_frame.depth_affine_data   = &depth_pts;
            cfg_frame.depth_affine_params = depth_affine;
            cfg_frame.use_temporal_depth_affine = has_prev;
            cfg_frame.depth_affine_prev   = has_prev ? depth_affine_prev : nullptr;
            cfg_frame.w_temp_depth_scale  = 10.0;
            cfg_frame.w_temp_depth_bias   = 20.0;
            cfg_frame.fix_depth_affine_bias = has_prev;
        }

        if (!cross_limb_pairs.empty()) {
            cfg_frame.use_cross_limb_depth = true;
            cfg_frame.cross_limb_pairs = &cross_limb_pairs;
        }

        // Foot contact
        {
            double LAx=0, LAy=0, RAx=0, RAy=0;
            bool hasLA = get_xy(lms, 14, LAx, LAy);
            bool hasRA = get_xy(lms, 11, RAx, RAy);
            cfg_frame.foot_contact_L = compute_contact(lms, true);
            cfg_frame.foot_contact_R = compute_contact(lms, false);
            if (hasLA) { prev_LA_x=LAx; prev_LA_y=LAy; ankle_prev_valid=true; }
            if (hasRA) { prev_RA_x=RAx; prev_RA_y=RAy; ankle_prev_valid=true; }
        }
        
        // =====================================================
        // OPTIMIZE
        // =====================================================
        bool ok = false;
        if (!has_prev) {
            cfg_frame.use_foot_ground = false;
            initutil::EstimateTransZFromKeypoints(lms, camera, trans);
            cfg_frame.trans_z_prior_target = trans[2];
            depth_affine[0] = 3.0;
            depth_affine[1] = trans[2]; 

            ok = OptimizeSMPL(lms, camera, model, beta, theta, trans, cfg_frame);
            z_anchor = trans[2];
            cfg_frame.max_iter_stage3 = 200;
        } else {
            cfg_frame.max_iter_stage3 = 0;
            multihyp::DepthGuidedZCorrection(
                model, beta, theta, trans,
                cfg_frame.depth_affine_data ? *cfg_frame.depth_affine_data : std::vector<JointRelDepthData>{},
                cfg_frame.depth_affine_params,
                cfg_frame.multi_hyp_max_z_shift);
            ok = OptimizeSMPLTracked(lms, camera, model, beta, theta, trans, cfg_frame,
                                    theta_prev_curr, trans_prev_curr,
                                    has_prev2 ? theta_prev2_curr : nullptr,
                                    has_prev2 ? trans_prev2_curr : nullptr);
        }

        // =====================================================
        // STORE FRAME RESULT (always, even if failed)
        // =====================================================
        {
            postsmooth::FrameResult fr;
            fr.frame_idx = f;
            fr.solver_ok = ok;
            std::memcpy(fr.theta, theta, sizeof(theta));
            std::memcpy(fr.trans, trans, sizeof(trans));
            std::memcpy(fr.beta,  beta,  sizeof(beta));
            std::memcpy(fr.depth_affine, depth_affine, sizeof(depth_affine));

            fr.mean_confidence = mean_conf;
            fr.num_visible_kps = num_visible;

            if (ok) {
                fr.reproj_px = ComputeReproj2DScore(lms, beta, theta, trans, cfg_frame);
            }

            all_results.push_back(fr);
            all_landmarks.push_back(lms);
        }

        if (!ok) {
            std::cerr << "[Frame " << f << "] Optimize failed\n";
            fail_count++;
            continue;
        }
        success_count++;

        // [Pass1] Save raw mesh immediately (will be overwritten by Pass2)
        {
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V;
            SMPLForward<double>(beta, theta, trans, model, V);
            SaveOBJ(MakeOutObj(out_dir, f), V, model.faces);
        }
        // Ground plane estimate
        {
            MatX3dRM Jcam;
            SMPLJointsForward<double>(beta, theta, trans, model, Jcam);
            double y_min = 1e9;
            for (int id : {7, 8, 10, 11})
                if (id < Jcam.rows()) y_min = std::min(y_min, Jcam(id, 1));
            if (!ground_init) {
                ground_y_est = y_min;
                ground_init = true;
            } 
            cfg.ground_y = ground_y_est;
        }

        // =====================================================
        // STORE PREV (for temporal warm start)
        // =====================================================
        if (has_prev) {
            std::memcpy(theta_prev2_raw, theta_prev_raw, sizeof(theta_prev2_raw));
            std::memcpy(trans_prev2_raw, trans_prev_raw, sizeof(trans_prev2_raw));
            prev2_frame_idx = prev_frame_idx;
            has_prev2 = true;
        }
        std::memcpy(theta_prev_raw, theta, sizeof(theta_prev_raw));
        std::memcpy(trans_prev_raw, trans, sizeof(trans_prev_raw));
        prev_frame_idx = f;
        has_prev = true;

        depth_affine_prev[0] = depth_affine[0];
        depth_affine_prev[1] = depth_affine[1];

        // Z drift monitor
        static double Z_cam_first = 0.0;
        if (f == start) Z_cam_first = trans[2];
        std::cout << "[F" << f << "] Z_cam=" << std::fixed << std::setprecision(3) << trans[2]
                  << " | drift=" << std::showpos << (trans[2] - Z_cam_first) << std::noshowpos
                  << " | scale=" << depth_affine[0]
                  << " | conf=" << std::fixed << std::setprecision(2) << mean_conf
                  << " | reproj=" << all_results.back().reproj_px << "px\n";

        std::cout << "[Frame " << f << "] pass1 done\n";
    }

    std::cout << "\n========== PASS 1 COMPLETE ==========\n"
              << "  Success: " << success_count << "\n"
              << "  Failed:  " << fail_count << "\n"
              << "=====================================\n\n";

    // =====================================================
    // PASS 2: POST-PROCESSING
    // Quality filter → interpolation → Gaussian smoothing
    // =====================================================
    {
        postsmooth::QualityThresholds qthres;
        qthres.max_pose_delta  = 2.0;   
        qthres.max_root_delta  = 1.2;   
        qthres.max_trans_delta = 0.6;   
        qthres.max_reproj_px   = 80.0;
        qthres.min_visible_kps = 6;

        postsmooth::SmoothConfig scfg;
        scfg.sigma_body  = 1.8;   
        scfg.sigma_arms  = 0.8;   
        scfg.sigma_root  = 1.5;   
        scfg.sigma_trans  = 1.5;
        scfg.smooth_root_special = true;

        auto pp_stats = postsmooth::RunPostProcessing(all_results, qthres, scfg);

        // Re-export all meshes from smoothed results
        std::cout << "[Pass2] Re-exporting smoothed meshes...\n";
        postsmooth::ReExportMeshes(all_results, model, out_dir);
    }

    // =====================================================
    // EVALUATION (on smoothed results)
    // =====================================================
    for (size_t ri = 0; ri < all_results.size(); ++ri) {
        const auto& fr = all_results[ri];
        const auto& lms = all_landmarks[ri];
        int f = fr.frame_idx;
        if (f < 0) continue;

        double mpjpe_frame    = -1.0;
        double pa_mpjpe_frame = -1.0;
        double pose_err_frame = -1.0;
        double reproj2d_frame = -1.0;

        if (has_gt && f < gt_data.num_frames) {
            MatX3dRM J_cam;
            SMPLJointsForward<double>(fr.beta, fr.theta, fr.trans, model, J_cam);

            bool eval_in_world = has_cam_poses && f < (int)cam_poses.size()
                                 && gt_data.has_world_data;

            MatX3dRM J_eval;
            if (eval_in_world)
                TransformJointsCamToWorld(J_cam, cam_poses[f], J_eval, cfg.flip_y_in_projection);
            else
                J_eval = J_cam;

            MatX3dRM J_gt;
            bool got_gt = eval_in_world ? gt_data.GetJointsWorld(f, J_gt)
                                        : gt_data.GetJoints(f, J_gt);
            if (got_gt) {
                mpjpe_frame    = ComputeMPJPE(J_eval, J_gt);
                pa_mpjpe_frame = ComputePAMPJPE(J_eval, J_gt);
                all_mpjpe.push_back(mpjpe_frame);
                all_pa_mpjpe.push_back(pa_mpjpe_frame);

                MatX3dRM J_eval_rr = J_eval, J_gt_rr = J_gt;
                for (int i = 0; i < J_eval_rr.rows(); ++i) J_eval_rr.row(i) -= J_eval.row(0);
                for (int i = 0; i < J_gt_rr.rows(); ++i)   J_gt_rr.row(i) -= J_gt.row(0);
                all_mpjpe_rootrel.push_back(ComputeMPJPE(J_eval_rr, J_gt_rr));

                if (ri == 0) {
                    std::cout << "\n===== FIRST FRAME DIAGNOSTIC (SMOOTHED) =====\n"
                              << "eval_in_world=" << eval_in_world << "\n"
                              << "Pred pelvis: " << J_eval.row(0) << "\n"
                              << "GT   pelvis: " << J_gt.row(0) << "\n"
                              << "Offset norm: " << (J_eval.row(0)-J_gt.row(0)).norm()*1000.0 << " mm\n"
                              << "Interpolated: " << fr.interpolated << "\n"
                              << "===================================\n\n";
                }
            }

            {
                MatX3dRM J_gt_cam;
                bool got_cam_gt = gt_data.GetJoints(f, J_gt_cam);
                if (got_cam_gt && J_gt_cam.rows() == J_cam.rows()) {
                    MatX3dRM Jc_rr = J_cam, Jg_rr = J_gt_cam;
                    for (int i = 0; i < Jc_rr.rows(); ++i) Jc_rr.row(i) -= J_cam.row(0);
                    for (int i = 0; i < Jg_rr.rows(); ++i) Jg_rr.row(i) -= J_gt_cam.row(0);
                    all_cam_rootrel.push_back(ComputeMPJPE(Jc_rr, Jg_rr));
                    all_cam_pa.push_back(ComputePAMPJPE(J_cam, J_gt_cam));
                }
            }

            double theta_gt[72];
            bool got_pose = (eval_in_world && !gt_data.poses_world.empty())
                            ? gt_data.GetPoseWorld(f, theta_gt)
                            : gt_data.GetPose(f, theta_gt);
            if (got_pose) {
                pose_err_frame = ComputePoseError(fr.theta, theta_gt, false);
                all_pose_error.push_back(pose_err_frame);
            }

            // 2D reprojection on smoothed params
            if (!lms.empty()) {
                double reproj_err = 0.0; int reproj_count = 0;
                for (const auto& lm : lms) {
                    if (lm.confidence < 0.3f) continue;
                    int op_idx = lm.joint_id;
                    if (op_idx < 0 || op_idx >= 25) continue;
                    int smpl_idx = OPENPOSE_TO_SMPL[op_idx];
                    if (smpl_idx < 0 || smpl_idx >= J_cam.rows()) continue;
                    double px = J_cam(smpl_idx, 0), py = J_cam(smpl_idx, 1), pz = J_cam(smpl_idx, 2);
                    if (pz <= 0) continue;
                    double u = camera.fx * px / pz + camera.cx;
                    double v = camera.fy * (cfg.flip_y_in_projection ? -py : py) / pz + camera.cy;
                    double err = std::sqrt((u-lm.x)*(u-lm.x) + (v-lm.y)*(v-lm.y));
                    reproj_err += err;
                    reproj_count++;
                }
                if (reproj_count > 0) {
                    reproj2d_frame = reproj_err / reproj_count;
                    all_reproj_2d.push_back(reproj2d_frame);
                }
            }
        }

        if (logger_ok) {
            logutil::FrameMetrics fm;
            fm.frame      = f;
            fm.mpjpe      = mpjpe_frame;
            fm.pa_mpjpe   = pa_mpjpe_frame;
            fm.pose_error = pose_err_frame;
            fm.reproj_2d  = reproj2d_frame;
            fm.pve        = -1.0;
            fm.shape_error= -1.0;
            elog.LogFrame(fm);
            elog.Push(fm);
        }
    }

    // ---------------------------
    // Summary
    // ---------------------------
    if (!all_mpjpe.empty()) {
        auto mean = [](const std::vector<double>& v)->double{
            if (v.empty()) return -1.0;
            double s=0; for(double x:v) s+=x; return s/(double)v.size();
        };
        std::cout << "\n========== EVALUATION SUMMARY (SMOOTHED) ==========\n"
                  << "  Frames: "      << all_mpjpe.size() << "\n"
                  << "--- World-space metrics ---\n"
                  << "  MPJPE:          " << mean(all_mpjpe)*1000.0 << " mm\n"
                  << "  MPJPE(root-rel):" << mean(all_mpjpe_rootrel)*1000.0 << " mm\n"
                  << "  PA-MPJPE:       " << mean(all_pa_mpjpe)*1000.0 << " mm\n"
                  << "--- Camera-space diagnostics ---\n"
                  << "  Cam root-rel:   " << mean(all_cam_rootrel)*1000.0 << " mm\n"
                  << "  Cam PA-MPJPE:   " << mean(all_cam_pa)*1000.0 << " mm\n";
        if (mean(all_pose_error)>=0) std::cout << "  Pose Err: " << mean(all_pose_error) << " rad\n";
        if (mean(all_reproj_2d)>=0)  std::cout << "  2D Reproj: " << mean(all_reproj_2d) << " px\n";
        std::cout << "====================================================\n";
    }

    if (logger_ok) {
        elog.SetSuccessFail(success_count, fail_count);
        elog.WriteSummary();
        std::cout << "[Saved] " << elog.EvalPath() << "\n";
        std::cout << "[Saved] " << elog.ConfigPath() << "\n";
    }

    std::cout << "[Done] sequence under " << out_dir << "\n";
}

} // namespace app