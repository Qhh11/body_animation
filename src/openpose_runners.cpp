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

    // Camera-space Z prior for first frame
    cfg.use_trans_z_prior = true;
    cfg.trans_z_prior_target = trans[2];
    cfg.w_trans_z_prior = 80.0;

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

// =========================================================
// MODE 3: OpenPose SEQUENCE — CAMERA-SPACE OPTIMIZATION
// =========================================================
void RunOpenPoseSequenceInference(const SmplModel& model,
                                  const GMMPosePrior* gmm_prior) {
    std::cout << "\n=== MODE 3: OpenPose SEQUENCE (Camera-Space) ===\n";

    const std::string seq_dir   = "../../Data/openpose_output/courtyard_bodyScannerMotions_00";
    const std::string depth_dir = "../../Data/sapiens_depth/courtyard_bodyScannerMotions_00";
    const std::string gt_dir    = "../../Data/3dpw_gt/courtyard_bodyScannerMotions_00";
    const std::string out_dir   = "./seq_out_test3";
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

    // ---------------------------
    // Load camera extrinsics
    // Used ONLY for:
    //   (a) Computing relative transform for warm-start
    //   (b) Transforming results to world space for evaluation
    // ---------------------------
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

    const int start = 70;
    const int end   = 200;
    const int img_w = 1080;
    const int img_h = 1920;
    CameraIntrinsics camera(1962.0f, 1969.0f, 540.0f, 960.0f, img_w, img_h);

    OptimizerConfig cfg = MakeCfgForOpenPose(gmm_prior);

    cfg.use_temporal = true;
    cfg.w_temp_pose  = 4.0;
    cfg.w_temp_trans = 6.0;
    cfg.use_temporal_acc = true;
    cfg.w_temp_acc_pose  = 2.0;
    cfg.temporal_body_only = true;

    cfg.use_foot_ground = true;
    cfg.w_foot_ground   = 140.0;
    cfg.w_foot_flat     = 80.0;
    cfg.ground_y        = 0.0;

    // Camera-space Z prior — THE key constraint preventing Z drift
    cfg.use_trans_z_prior = true;
    cfg.w_trans_z_prior = 120.0;

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
    // theta/trans are always in the CURRENT frame's camera space
    // ---------------------------
    double beta[10]  = {0.0};
    double theta[72] = {0.0};
    double trans[3]  = {0.0, 0.0, 3.0};

    // Previous frame params stored in THEIR OWN camera space
    double theta_prev_raw[72]  = {0.0};
    double trans_prev_raw[3]   = {0.0, 0.0, 3.0};
    double theta_prev2_raw[72] = {0.0};
    double trans_prev2_raw[3]  = {0.0, 0.0, 3.0};

    // Transformed to current cam space (temporal prior targets + warm start)
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
        for (int x : {1, 2, 3, 5, 6, 8}) if (x == op_id) return true;
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
            }

            // Initialize current frame from transformed previous
            std::memcpy(theta, theta_prev_curr, sizeof(theta));
            std::memcpy(trans, trans_prev_curr, sizeof(trans));
            depth_affine[0] = depth_affine_prev[0];
            depth_affine[1] = depth_affine_prev[1];

            // Z prior target = prev frame's Z in current cam space
            cfg.trans_z_prior_target = trans_prev_curr[2];
        } else {
            // First frame: 2D-based initialization (already camera space)
            EstimateGlobalTranslationXY(lms, camera, 3.0, trans);
            depth_affine[0] = 3.0;
            depth_affine[1] = trans[2];
            cfg.trans_z_prior_target = trans[2];
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
                    double c2d = get_conf(lms, op_idx);
                    if (c2d < 0.35) continue;
                    int smpl_idx = OPENPOSE_TO_SMPL[op_idx];
                    if (smpl_idx < 0) continue;
                    depth_pts.push_back({smpl_idx, depth_val, c2d});
                }
            }
        }

        if ((int)depth_pts.size() >= 6) {
            cfg.use_depth_affine    = true;
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
            cfg.max_iter_stage1 = 20;
            cfg.max_iter_stage2 = 40;
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

            // 2D reprojection (always camera-space, direct)
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
                double v = camera.fy * py / pz + camera.cy;
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
                  << "  MPJPE:    "    << mean(all_mpjpe)*1000.0 << " mm\n"
                  << "  PA-MPJPE: "    << mean(all_pa_mpjpe)*1000.0 << " mm\n";
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