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

#include "smpl_math.h"
#include "evaluation.h"
#include "logging.h"
#include "io_utils.h"
#include "openpose_io.h"
#include "init_utils.h"
#include "config_builders.h"
#include "threedpw_gt.h"

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
    cfg.trans_z_prior_target = trans[2];  // anchor to initial estimate

    // -------- Load Sapiens relative depth and create pairs --------
    std::vector<RelativeDepthPair> relative_depth_pairs;
    std::string depth_path = "../../Data/sapiens_depth/courtyard_basketball_01/image_00047_depth.txt";
    std::ifstream depth_file(depth_path);
    if (depth_file.is_open()) {
        std::vector<std::pair<int, double>> kp_depths;
        int op_idx;
        double depth_val;
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

    // All optimization in CAMERA SPACE — no extrinsics
    const bool ok = OptimizeSMPL(lms, camera, model, beta, theta, trans, cfg);
    if (!ok) { std::cerr << "OptimizeSMPL failed.\n"; return; }

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_opt;
    SMPLForward<double>(beta, theta, trans, model, V_opt);
    SaveOBJ("mesh_after_opt.obj", V_opt, model.faces);
    std::cout << "[Done] saved mesh_after_opt.obj\n";
}


void RunOpenPoseSequenceInference(const SmplModel& model,
                                  const GMMPosePrior* gmm_prior) {
    std::cout << "\n=== MODE 3: OpenPose SEQUENCE (Camera-Space) ===\n";

    const std::string seq_dir   = "../../Data/openpose_output/courtyard_bodyScannerMotions_00";
    const std::string depth_dir = "../../Data/sapiens_depth/courtyard_bodyScannerMotions_00";
    const std::string gt_dir    = "../../Data/3dpw_gt/courtyard_bodyScannerMotions_00";
    const std::string out_dir   = "./seq_out_zhengmian3";
    fs::create_directories(out_dir);

    // ---------------------------
    // Load ground truth for evaluation
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
        std::cout << "[Camera] Loaded " << cam_poses.size()
                  << " poses (warm-start transform & evaluation only)\n";
    else
        std::cout << "[Camera] No camera poses (fixed camera)\n";

    // ---------------------------
    // Evaluation accumulators
    // ---------------------------
    std::vector<double> all_mpjpe, all_pa_mpjpe, all_pve, all_pose_error, all_reproj_2d;
    std::vector<double> all_mpjpe_rootrel;
    std::vector<double> all_cam_rootrel, all_cam_pa;

    const int start = 0;
    const int end   = 140;
    const int img_w = 1080;
    const int img_h = 1920;
    CameraIntrinsics camera(1962.0f, 1969.0f, 540.0f, 960.0f, img_w, img_h);

    OptimizerConfig cfg = MakeCfgForOpenPoseSequence(gmm_prior);
    // ---------------------------
    // Logger
    // ---------------------------
    logutil::RunMeta meta;
    meta.mode = "OPENPOSE_SEQ_CAMSPACE";
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
    //theta[1] = M_PI * -0.5;   ENABLE ONLY IF SIDE INIT  

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
        for (auto& lm : lms) {
            int id = lm.joint_id;
            if (id < 0 || id >= 25) continue;
            if (lm.confidence < 0.20f) continue;
            double a = 0.45 + 0.45 * std::min(1.0, std::max(0.0, (double)lm.confidence));
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

    auto is_stable_op = [&](int op_id)->bool{
        // Torso + arm chain: include wrists (4,7) for depth signal on arms
        for (int x : {1, 2, 3, 4, 5, 6, 7, 8}) if (x == op_id) return true;
        return false;
    };

    bool ground_init = false;
    double ground_y_est = 0.0;

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

        double err = 0.0;
        int cnt = 0;
        for (const auto& lm : lms) {
            if (lm.confidence < cfg_.min_confidence) continue;
            int op_idx = lm.joint_id;
            if (op_idx < 0 || op_idx >= 25) continue;
            int smpl_idx = OPENPOSE_TO_SMPL[op_idx];
            if (smpl_idx < 0 || smpl_idx >= J.rows()) continue;

            double X = J(smpl_idx, 0);
            double Y = J(smpl_idx, 1);
            double Z = J(smpl_idx, 2);
            if (Z <= 1e-6) continue;

            double u = camera.fx * X / Z + camera.cx;
            double v = camera.fy * (cfg_.flip_y_in_projection ? -Y : Y) / Z + camera.cy;

            double du = u - lm.x;
            double dv = v - lm.y;
            err += std::sqrt(du*du + dv*dv);
            cnt++;
        }
        return (cnt > 0) ? (err / cnt) : 1e18;
    };

    auto InitFirstFrameYawByMultiStart = [&](const std::vector<Landmark2D>& lms,
                                         double beta_[10],
                                         double theta_[72],
                                         double trans_[3],
                                         OptimizerConfig cfg_base)
    {
        // 只跑很少迭代，快速选 yaw basin
        cfg_base.use_temporal = false;
        cfg_base.use_temporal_acc = false;
        cfg_base.max_iter_stage1 = 25;
        cfg_base.max_iter_stage2 = 0;
        cfg_base.max_iter_stage3 = 0;
        cfg_base.print_progress = false;

        // 第一帧：建议先弱化 arm-torso sep，避免把手推到身后
        if (cfg_base.use_arm_torso_sep) {
            cfg_base.arm_torso_w_elbow = 6.0;
            cfg_base.arm_torso_w_wrist = 10.0;
        }

        // 可选：第一帧先别锁死手（否则更容易走奇怪解）
         cfg_base.lock_hands = false; // 你现在 lock_hands=true，可以先注释试试

        const double degs[] = {-120,-90,-60,-30,0,30,60,90,120};
        double best_score = 1e18;
        double best_theta[72];
        double best_trans[3];

        for (double d : degs) {
            double theta_try[72]; std::memcpy(theta_try, theta_, sizeof(theta_try));
            double trans_try[3];  std::memcpy(trans_try, trans_, sizeof(trans_try));
            double beta_try[10];  std::memcpy(beta_try,  beta_,  sizeof(beta_try));

            theta_try[1] = d * M_PI / 180.0;

            bool ok = OptimizeSMPL(lms, camera, model, beta_try, theta_try, trans_try, cfg_base);
            if (!ok) continue;

            double s = ComputeReproj2DScore(lms, beta_try, theta_try, trans_try, cfg_base);
            if (s < best_score) {
                best_score = s;
                std::memcpy(best_theta, theta_try, sizeof(best_theta));
                std::memcpy(best_trans, trans_try, sizeof(best_trans));
            }
        }

        if (best_score < 1e17) {
            std::memcpy(theta_, best_theta, sizeof(best_theta));
            std::memcpy(trans_, best_trans, sizeof(best_trans));
            cfg.trans_z_prior_target = trans_[2];
            std::cout << "[InitYaw] best reproj=" << best_score
                    << " px, yaw(rad)=" << theta_[1] << "\n";
        } else {
            std::cout << "[InitYaw] failed, keep default yaw\n";
        }
    };

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

        if (!LoadLandmarksFromJSON(json_path, lms)) {
            std::cerr << "[Frame " << f << "] failed to load: " << json_path << "\n";
            fail_count++;
            continue;
        }

        double sep_scale = 0.0;
        if (f == start) sep_scale = 0.0;
        else if (f < start + 10) sep_scale = (f - start) / 10.0;   // 0→1
        else sep_scale = 1.0;

        cfg.use_arm_torso_sep = (sep_scale > 1e-6);
        cfg.arm_torso_w_elbow = 10.0 * sep_scale;   
        cfg.arm_torso_w_wrist = 15.0 * sep_scale;

        smooth_landmarks_inplace(lms);

        // =====================================================
        // WARM START — transform prev results to current cam space
        // =====================================================
        if (has_prev) {
            if (has_cam_poses &&
                prev_frame_idx >= 0 && prev_frame_idx < (int)cam_poses.size() &&
                f < (int)cam_poses.size())
            {
                // Moving camera: compute relative transform and apply
                Eigen::Matrix3d R_rel;
                Eigen::Vector3d t_rel;
                ComputeRelativeCameraTransform(cam_poses[prev_frame_idx],
                                               cam_poses[f], R_rel, t_rel);
                TransformPrevParamsToCurrCam(theta_prev_raw, trans_prev_raw,
                                            R_rel, t_rel,
                                            theta_prev_curr, trans_prev_curr);

                if (has_prev2 && prev2_frame_idx >= 0 &&
                    prev2_frame_idx < (int)cam_poses.size())
                {
                    Eigen::Matrix3d R_rel2;
                    Eigen::Vector3d t_rel2;
                    ComputeRelativeCameraTransform(cam_poses[prev2_frame_idx],
                                                   cam_poses[f], R_rel2, t_rel2);
                    TransformPrevParamsToCurrCam(theta_prev2_raw, trans_prev2_raw,
                                                R_rel2, t_rel2,
                                                theta_prev2_curr, trans_prev2_curr);
                }
            } else {
                // Fixed camera: straight copy
                std::memcpy(theta_prev_curr, theta_prev_raw, sizeof(theta_prev_curr));
                std::memcpy(trans_prev_curr, trans_prev_raw, sizeof(trans_prev_curr));
                std::memcpy(theta_prev2_curr, theta_prev2_raw, sizeof(theta_prev2_curr));
                std::memcpy(trans_prev2_curr, trans_prev2_raw, sizeof(trans_prev2_curr));
                 
                //init beta for first frame
                EstimateGlobalTranslationXY(lms, camera, 3.0, trans);
                cfg.trans_z_prior_target = trans[2];

                InitFirstFrameYawByMultiStart(lms, beta, theta, trans, cfg);

                OptimizerConfig cfg_first = cfg;
                cfg_first.use_temporal = false;
                cfg_first.use_temporal_acc = false;

                cfg_first.max_iter_stage1 = 80;
                cfg_first.max_iter_stage2 = 200;
                cfg_first.max_iter_stage3 = 120;     
                cfg_first.lock_hands  = false;
                cfg_first.lock_wrists = false;
                cfg_first.use_arm_torso_sep = false; 

                bool ok0 = OptimizeSMPL(lms, camera, model, beta, theta, trans, cfg_first);
                if (!ok0) { std::cout << "not ok init beta\n";}
            }

            // Initialize current frame from transformed previous
            std::memcpy(theta, theta_prev_curr, sizeof(theta));
            std::memcpy(trans, trans_prev_curr, sizeof(trans));
            depth_affine[0] = depth_affine_prev[0];
            depth_affine[1] = depth_affine_prev[1];

            // Z prior target = prev frame's Z in current cam space
            cfg.trans_z_prior_target = trans_prev_curr[2];
    } else {
        // First frame: 2D init
        EstimateGlobalTranslationXY(lms, camera, 3.0, trans);
        cfg.trans_z_prior_target = trans[2];

        InitFirstFrameYawByMultiStart(lms, beta, theta, trans, cfg);

        depth_affine[0] = 3.0;
        depth_affine[1] = trans[2];
    }


        // Damp hands
        {
            auto damp = [&](int j, double s) {
                theta[3*j] *= s; theta[3*j+1] *= s; theta[3*j+2] *= s;
            };
            damp(20, 0.10);
            damp(21, 0.10);
        }

        // Load Sapiens depth (stable torso keypoints only)
        std::vector<JointRelDepthData> depth_pts;
        {
            std::string depth_path = MakeOpenPoseDepthPath(depth_dir, f);
            std::ifstream df(depth_path);
            if (df.is_open()) {
                int op_idx; double depth_val;
                while (df >> op_idx >> depth_val) {
                    if (op_idx < 0 || op_idx >= 25) continue;
                    if (!is_stable_op(op_idx)) continue;
                    double conf_thr = (f == start) ? 0.20 : 0.35; 
                    double c2d = get_conf(lms, op_idx);
                    if (c2d < conf_thr) continue;
                    if (c2d < 0.35) continue;
                    int smpl_idx = OPENPOSE_TO_SMPL[op_idx];
                    if (smpl_idx < 0) continue;
                    depth_pts.push_back({smpl_idx, depth_val, c2d});
                }
            }
        }

        if ((int)depth_pts.size() >= 6) {
            cfg.use_depth_affine    = true;
            if (f == start) cfg.w_depth_affine = 90.0;  
            cfg.w_depth_affine      = 60.0;
            cfg.depth_affine_sign   = 1.0;
            cfg.depth_affine_data   = &depth_pts;
            cfg.depth_affine_params = depth_affine;

            cfg.use_temporal_depth_affine = has_prev;
            cfg.depth_affine_prev   = has_prev ? depth_affine_prev : nullptr;
            cfg.w_temp_depth_scale  = 20.0;
            cfg.w_temp_depth_bias   = 50.0;
            cfg.fix_depth_affine_bias = has_prev;
        }

        // Foot contact
        {
            double LAx=0, LAy=0, RAx=0, RAy=0;
            bool hasLA = get_xy(lms, 14, LAx, LAy);
            bool hasRA = get_xy(lms, 11, RAx, RAy);
            cfg.foot_contact_L = compute_contact(lms, true);
            cfg.foot_contact_R = compute_contact(lms, false);
            if (hasLA) { prev_LA_x=LAx; prev_LA_y=LAy; ankle_prev_valid=true; }
            if (hasRA) { prev_RA_x=RAx; prev_RA_y=RAy; ankle_prev_valid=true; }
        }

        // Iteration counts
        if (has_prev) {
            cfg.max_iter_stage1 = 30;
            cfg.max_iter_stage2 = 100;   // was 40 — arms need more iters to move
            cfg.max_iter_stage3 = 0;
        } else {
            cfg.max_iter_stage1 = 60;
            cfg.max_iter_stage2 = 120;
            cfg.max_iter_stage3 = 80;
        }

        // =====================================================
        // OPTIMIZE — pure camera space, no extrinsics
        // =====================================================
        bool ok = false;
        if (!has_prev) {
            ok = OptimizeSMPL(lms, camera, model, beta, theta, trans, cfg);
        } else {
            // theta_prev_curr / trans_prev_curr are already in current cam space
            ok = OptimizeSMPLTracked(lms, camera, model, beta, theta, trans, cfg,
                                     theta_prev_curr, trans_prev_curr,
                                     has_prev2 ? theta_prev2_curr : nullptr,
                                     has_prev2 ? trans_prev2_curr : nullptr);
        }

        if (!ok) {
            std::cerr << "[Frame " << f << "] Optimize failed\n";
            fail_count++;
            continue;
        }
        success_count++;

        double mpjpe_frame    = -1.0;
        double pa_mpjpe_frame = -1.0;
        double pose_err_frame = -1.0;
        double reproj2d_frame = -1.0;

        // Update ground plane estimate (camera space)
        {
            MatX3dRM Jcam;
            SMPLJointsForward<double>(beta, theta, trans, model, Jcam);
            double y_min = 1e9;
            for (int id : {7, 8, 10, 11})
                if (id < Jcam.rows()) y_min = std::min(y_min, Jcam(id, 1));

            if (!ground_init) {
                ground_y_est = y_min;
                ground_init = true;
            } else {
                // Camera-space ground can shift with camera motion; use conservative EMA
                double alpha = (y_min < ground_y_est) ? 0.5 : 0.02;
                ground_y_est = alpha * y_min + (1.0 - alpha) * ground_y_est;
            }
            cfg.ground_y = ground_y_est;
        }

        // Save mesh (camera space)
        {
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V;
            SMPLForward<double>(beta, theta, trans, model, V);
            SaveOBJ(MakeOutObj(out_dir, f), V, model.faces);
        }

        // =====================================================
        // STORE PREV — keep in this frame's own camera space
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

        // Z drift monitor (camera-space Z = trans[2], trivially observable)
        static double Z_cam_first = 0.0;
        if (f == start) Z_cam_first = trans[2];
        std::cout << "[F" << f << "] Z_cam=" << std::fixed << std::setprecision(3) << trans[2]
                  << " | drift=" << std::showpos << (trans[2] - Z_cam_first) << std::noshowpos
                  << " | scale=" << depth_affine[0] << "\n";

        // =====================================================
        // EVALUATION — transform cam-space → world for GT comparison
        // =====================================================
        if (has_gt && f < gt_data.num_frames) {
            MatX3dRM J_cam;
            SMPLJointsForward<double>(beta, theta, trans, model, J_cam);

            bool eval_in_world = has_cam_poses && f < (int)cam_poses.size()
                                 && gt_data.has_world_data;

            MatX3dRM J_eval;
            if (eval_in_world) {
                TransformJointsCamToWorld(J_cam, cam_poses[f], J_eval);
            } else {
                J_eval = J_cam;
            }

            // MPJPE / PA-MPJPE
            MatX3dRM J_gt;
            bool got_gt = eval_in_world ? gt_data.GetJointsWorld(f, J_gt)
                                        : gt_data.GetJoints(f, J_gt);
            if (got_gt) {
                mpjpe_frame    = ComputeMPJPE(J_eval, J_gt);
                pa_mpjpe_frame = ComputePAMPJPE(J_eval, J_gt);
                all_mpjpe.push_back(mpjpe_frame);
                all_pa_mpjpe.push_back(pa_mpjpe_frame);

                // Root-relative MPJPE in world space
                MatX3dRM J_eval_rr = J_eval, J_gt_rr = J_gt;
                for (int i = 0; i < J_eval_rr.rows(); ++i)
                    J_eval_rr.row(i) -= J_eval.row(0);
                for (int i = 0; i < J_gt_rr.rows(); ++i)
                    J_gt_rr.row(i) -= J_gt.row(0);
                double mpjpe_rr = ComputeMPJPE(J_eval_rr, J_gt_rr);
                all_mpjpe_rootrel.push_back(mpjpe_rr);

                // ---------- First-frame diagnostic ----------
                if (f == start) {
                    std::cout << "\n===== FIRST FRAME DIAGNOSTIC =====\n";
                    std::cout << "eval_in_world=" << eval_in_world << "\n";
                    std::cout << "Pred pelvis: " << J_eval.row(0) << "\n";
                    std::cout << "GT   pelvis: " << J_gt.row(0) << "\n";
                    std::cout << "Offset:      " << (J_eval.row(0)-J_gt.row(0)) << "\n";
                    std::cout << "Offset norm: " << (J_eval.row(0)-J_gt.row(0)).norm()*1000.0 << " mm\n";
                    // Bone length check: left hip (1) to left knee (4)
                    if (J_eval.rows() > 5 && J_gt.rows() > 5) {
                        double bone_pred = (J_eval.row(1) - J_eval.row(4)).norm();
                        double bone_gt   = (J_gt.row(1) - J_gt.row(4)).norm();
                        std::cout << "Pred L-thigh: " << bone_pred*1000.0 << " mm\n";
                        std::cout << "GT   L-thigh: " << bone_gt*1000.0 << " mm\n";
                        std::cout << "Scale ratio:  " << bone_pred / std::max(bone_gt, 1e-6) << "\n";
                    }
                    // Cam-space pred pelvis
                    std::cout << "Cam  pelvis: " << J_cam.row(0) << "\n";
                    std::cout << "trans[2] (depth): " << trans[2] << "\n";
                    std::cout << "J_eval rows=" << J_eval.rows() << " J_gt rows=" << J_gt.rows() << "\n";
                    std::cout << "===================================\n\n";
                }
            }

            // Camera-space diagnostic: compare J_cam directly with cam-space GT
            // This bypasses the world transform entirely — if these are much better,
            // the world transform is wrong.
            {
                MatX3dRM J_gt_cam;
                bool got_cam_gt = gt_data.GetJoints(f, J_gt_cam);
                if (got_cam_gt && J_gt_cam.rows() == J_cam.rows()) {
                    // Root-relative in camera space
                    MatX3dRM Jc_rr = J_cam, Jg_rr = J_gt_cam;
                    for (int i = 0; i < Jc_rr.rows(); ++i) Jc_rr.row(i) -= J_cam.row(0);
                    for (int i = 0; i < Jg_rr.rows(); ++i) Jg_rr.row(i) -= J_gt_cam.row(0);
                    all_cam_rootrel.push_back(ComputeMPJPE(Jc_rr, Jg_rr));
                    all_cam_pa.push_back(ComputePAMPJPE(J_cam, J_gt_cam));
                }
            }

            // Pose error
            double theta_gt[72];
            bool got_pose = (eval_in_world && !gt_data.poses_world.empty())
                            ? gt_data.GetPoseWorld(f, theta_gt)
                            : gt_data.GetPose(f, theta_gt);
            if (got_pose) {
                pose_err_frame = ComputePoseError(theta, theta_gt, false);
                all_pose_error.push_back(pose_err_frame);
            }

            // 2D reprojection (camera-space, must match optimizer's flip_y)
            double reproj_err = 0.0;
            int reproj_count = 0;
            for (const auto& lm : lms) {
                if (lm.confidence < 0.3f) continue;
                int op_idx = lm.joint_id;
                if (op_idx < 0 || op_idx >= 25) continue;
                int smpl_idx = OPENPOSE_TO_SMPL[op_idx];
                if (smpl_idx < 0 || smpl_idx >= J_cam.rows()) continue;

                double px = J_cam(smpl_idx, 0);
                double py = J_cam(smpl_idx, 1);
                double pz = J_cam(smpl_idx, 2);
                if (pz <= 0) continue;

                double u = camera.fx * px / pz + camera.cx;
                // flip_y: optimizer uses -Y for projection, must match here
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

        // Log
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

        std::cout << "[Frame " << f << "] done\n";
    }

    // ---------------------------
    // Summary
    // ---------------------------
    if (!all_mpjpe.empty()) {
        auto mean = [](const std::vector<double>& v)->double{
            if (v.empty()) return -1.0;
            double s=0; for(double x:v) s+=x; return s/(double)v.size();
        };
        std::cout << "\n========== EVALUATION SUMMARY ==========\n"
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
        std::cout << "========================================\n";
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