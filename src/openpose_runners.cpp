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
#include "logging.h"        // your EvalLogger / RunMeta
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

// =========================================================
// MODE 3: OpenPose SEQUENCE (your full version)
// =========================================================
void RunOpenPoseSequenceInference(const SmplModel& model,
                                  const GMMPosePrior* gmm_prior) {
    std::cout << "\n=== MODE 3: OpenPose SEQUENCE ===\n";

    const std::string seq_dir   = "../../Data/openpose_output/courtyard_bodyScannerMotions_00";
    const std::string depth_dir = "../../Data/sapiens_depth/courtyard_bodyScannerMotions_00";
    const std::string gt_dir    = "../../Data/3dpw_gt/courtyard_bodyScannerMotions_00";
    const std::string out_dir   = "./seq_out_test3";
    fs::create_directories(out_dir);

    // ---------------------------
    // Load 3DPW ground truth for evaluation
    // ---------------------------
    ThreeDPW_GT gt_data;
    bool has_gt = gt_data.LoadFromDir(gt_dir);
    if (has_gt) {
        std::cout << "[GT] Loaded " << gt_data.num_frames << " frames of ground truth\n";
        if (gt_data.has_poses) std::cout << "[GT] Has pose parameters (theta)\n";
        if (gt_data.has_betas) std::cout << "[GT] Has shape parameters (beta)\n";
        if (gt_data.has_world_data) std::cout << "[GT] Has world-space joints\n";
    } else {
        std::cout << "[GT] No ground truth available, evaluation metrics will be -1\n";
    }

    // ---------------------------
    // Load camera extrinsics (per-frame world-to-camera transforms)
    // ---------------------------
    std::vector<CameraExtrinsics> cam_poses = LoadCameraPoses(gt_dir + "/cam_poses.txt");
    bool has_cam_poses = !cam_poses.empty();
    if (has_cam_poses) {
        std::cout << "[Camera] Loaded " << cam_poses.size() << " camera poses (moving camera)\n";
    } else {
        std::cout << "[Camera] No camera poses found, assuming fixed camera at origin\n";
    }

    const bool use_world_coords = has_cam_poses;
    if (use_world_coords) std::cout << "[World] World-coordinate optimization ENABLED\n";
    else                  std::cout << "[World] Camera-coordinate optimization (no extrinsics)\n";

    auto cam_to_world = [](const Eigen::Vector3d& p_cam, const CameraExtrinsics& ext) -> Eigen::Vector3d {
        return ext.R.transpose() * (p_cam - ext.t);
    };

    // ---------------------------
    // Evaluation accumulators
    // ---------------------------
    std::vector<double> all_mpjpe;
    std::vector<double> all_pa_mpjpe;
    std::vector<double> all_pve;
    std::vector<double> all_pose_error;
    std::vector<double> all_reproj_2d;

    const int start = 70;
    const int end   = 200;

    const int img_w = 1080;
    const int img_h = 1920;
    CameraIntrinsics camera(1962.0f, 1969.0f, 540.0f, 960.0f, img_w, img_h);

    OptimizerConfig cfg = MakeCfgForOpenPose(gmm_prior);

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

    // ---------------------------
    // Logger (outside loop)
    // ---------------------------
    logutil::RunMeta meta;
    meta.mode = "OPENPOSE_SEQ";
    meta.seq_dir = seq_dir;
    meta.depth_dir = depth_dir;
    meta.gt_dir = gt_dir;
    meta.out_dir = out_dir;
    meta.start_frame = start;
    meta.end_frame = end;
    meta.use_world_coords = use_world_coords;
    meta.has_cam_poses = has_cam_poses;

    logutil::EvalLogger elog;
    bool logger_ok = false;
    if (!elog.Open(out_dir, "evaluation.txt", "config.txt")) {
        std::cerr << "[Logger] failed to open evaluation/config files under " << out_dir << "\n";
    } else {
        logger_ok = true;
        elog.WriteConfig(cfg, &camera, &meta);
        elog.WriteEvalHeader(meta);
    }

    // ---------------------------
    // Parameters + temporal buffers
    // ---------------------------
    double beta[10]  = {0.0};
    double theta[72] = {0.0};
    double trans[3]  = {0.0, 0.0, 3.0};

    double theta_prev[72]  = {0.0};
    double theta_prev2[72] = {0.0};
    double trans_prev[3]   = {0.0, 0.0, 3.0};
    double trans_prev2[3]  = {0.0, 0.0, 3.0};

    bool has_prev  = false;
    bool has_prev2 = false;

    double depth_affine[2]      = {3.0, 3.0};  // [scale, bias]
    double depth_affine_prev[2] = {3.0, 3.0};

    double Z_cam_prev  = 3.0;
    double Z_cam_prev2 = 3.0;

    // 2D keypoints EMA smoother
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
        int ids[] = {1, 2, 3, 5, 6, 8}; // neck, shoulders, elbows, midhip
        for (int x: ids) if (x == op_id) return true;
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
        double ground_like = 0.0;
        if (y_norm > 0.80) ground_like = 1.0;
        else if (y_norm > 0.72) ground_like = (y_norm - 0.72) / (0.80 - 0.72);

        double speed_like = 1.0;
        if (ankle_prev_valid) {
            double px = is_left ? prev_LA_x : prev_RA_x;
            double py = is_left ? prev_LA_y : prev_RA_y;
            double sp = std::sqrt((ax-px)*(ax-px) + (ay-py)*(ay-py));
            if (sp > 18.0) speed_like = 0.0;
            else if (sp > 6.0) speed_like = (18.0 - sp) / (18.0 - 6.0);
        }

        double conf_like = std::min(1.0, 0.5*(cA+cK));
        double contact = ground_like * speed_like * conf_like;
        return std::min(1.0, std::max(0.0, contact));
    };

    int success_count = 0;
    int fail_count = 0;

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

        cfg.use_cam_depth_prior = false;

        if (!LoadLandmarksFromJSON(json_path, lms)) {
            std::cerr << "[Frame " << f << "] failed to load: " << json_path << "\n";
            fail_count++;
            continue;
        }

        smooth_landmarks_inplace(lms);

        // init / warm start
        if (has_prev) {
            std::memcpy(theta, theta_prev, sizeof(theta));
            std::memcpy(trans, trans_prev, sizeof(trans));
            depth_affine[0] = depth_affine_prev[0];
            depth_affine[1] = depth_affine_prev[1];
        } else {
            double trans_cam[3];
            EstimateGlobalTranslationXY(lms, camera, 3.0, trans_cam);

            if (use_world_coords && f < (int)cam_poses.size()) {
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

            depth_affine[0] = 3.0;
            depth_affine[1] = trans_cam[2];
        }

        // damp hands (keep your original)
        {
            auto damp_joint = [&](int j, double s){
                theta[3*j+0] *= s;
                theta[3*j+1] *= s;
                theta[3*j+2] *= s;
            };
            damp_joint(20, 0.10);
            damp_joint(21, 0.10);
        }

        // load sapiens depth points (stable torso only)
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
        }

        if ((int)depth_pts.size() >= 6) {
            cfg.use_depth_affine = true;
            cfg.w_depth_affine = 60.0;
            cfg.depth_affine_sign = 1.0;
            cfg.depth_affine_data   = &depth_pts;
            cfg.depth_affine_params = depth_affine;

            cfg.use_temporal_depth_affine = has_prev;
            cfg.depth_affine_prev = has_prev ? depth_affine_prev : nullptr;
            cfg.w_temp_depth_scale = 20.0;
            cfg.w_temp_depth_bias  = 50.0;

            cfg.fix_depth_affine_bias = has_prev;
        }

        // world-coord: camera depth prior
        if (use_world_coords && has_prev) {
            cfg.use_cam_depth_prior = true;
            cfg.w_cam_depth_prior = 200.0;
            cfg.Z_cam_prev = Z_cam_prev;
            cfg.Z_cam_prev2 = Z_cam_prev2;
            cfg.has_Z_cam_prev2 = has_prev2;
        }

        // foot contact
        double LAx=0, LAy=0, RAx=0, RAy=0;
        bool hasLA = get_xy(lms, 14, LAx, LAy);
        bool hasRA = get_xy(lms, 11, RAx, RAy);

        cfg.foot_contact_L = compute_contact(lms, true);
        cfg.foot_contact_R = compute_contact(lms, false);

        if (hasLA) { prev_LA_x = LAx; prev_LA_y = LAy; ankle_prev_valid = true; }
        if (hasRA) { prev_RA_x = RAx; prev_RA_y = RAy; ankle_prev_valid = true; }

        // iterations
        if (has_prev) {
            cfg.max_iter_stage1 = 20;
            cfg.max_iter_stage2 = 40;
            cfg.max_iter_stage3 = 0;
        } else {
            cfg.max_iter_stage1 = 60;
            cfg.max_iter_stage2 = 120;
            cfg.max_iter_stage3 = 80;
        }

        // extrinsics pointer
        const CameraExtrinsics* frame_extrinsics = nullptr;
        if (use_world_coords && f < (int)cam_poses.size()) frame_extrinsics = &cam_poses[f];

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

        // update ground estimate
        {
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Jw;
            SMPLJointsForward<double>(beta, theta, trans, model, Jw);

            double y_min = 1e9;
            int ids[] = {7, 8, 10, 11};
            for (int id : ids) {
                if (id >= 0 && id < Jw.rows()) y_min = std::min(y_min, Jw(id, 1));
            }

            if (!ground_init) {
                ground_y_est = y_min;
                ground_init = true;
            } else if (!use_world_coords) {
                if (y_min < ground_y_est) ground_y_est = 0.7 * y_min + 0.3 * ground_y_est;
                else                      ground_y_est = 0.98 * ground_y_est + 0.02 * y_min;
            }
            cfg.ground_y = ground_y_est;
        }

        // save mesh
        {
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V;
            SMPLForward<double>(beta, theta, trans, model, V);
            SaveOBJ(MakeOutObj(out_dir, f), V, model.faces);
        }

        // shift prev buffers
        if (has_prev) {
            std::memcpy(theta_prev2, theta_prev, sizeof(theta_prev2));
            std::memcpy(trans_prev2, trans_prev, sizeof(trans_prev2));
            has_prev2 = true;
        }
        std::memcpy(theta_prev, theta, sizeof(theta_prev));
        std::memcpy(trans_prev, trans, sizeof(trans_prev));
        has_prev = true;

        depth_affine_prev[0] = depth_affine[0];
        depth_affine_prev[1] = depth_affine[1];

        if (use_world_coords && f < (int)cam_poses.size()) {
            const auto& ext = cam_poses[f];
            double Z_cam_cur = ext.R(2,0)*trans[0] + ext.R(2,1)*trans[1] + ext.R(2,2)*trans[2] + ext.t(2);
            Z_cam_prev2 = Z_cam_prev;
            Z_cam_prev  = Z_cam_cur;
        }

        // drift monitor print
        double Z_cam_cur = 0.0;
        if (use_world_coords && f < (int)cam_poses.size()) {
            const auto& ext = cam_poses[f];
            Z_cam_cur = ext.R(2,0)*trans[0] + ext.R(2,1)*trans[1] + ext.R(2,2)*trans[2] + ext.t(2);
        } else {
            Z_cam_cur = trans[2];
        }
        static double Z_cam_first = 0.0;
        if (f == start) Z_cam_first = Z_cam_cur;
        double Z_cam_drift = Z_cam_cur - Z_cam_first;

        std::cout << "[F" << f << "] Z_cam=" << std::fixed << std::setprecision(3) << Z_cam_cur
                  << " | drift=" << std::showpos << Z_cam_drift << std::noshowpos
                  << " | scale=" << depth_affine[0] << "\n";

        // Evaluation against GT
        if (has_gt && f < gt_data.num_frames) {
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_pred;
            SMPLJointsForward<double>(beta, theta, trans, model, J_pred);

            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_gt;
            bool got_gt = false;
            if (use_world_coords && gt_data.has_world_data) got_gt = gt_data.GetJointsWorld(f, J_gt);
            else                                            got_gt = gt_data.GetJoints(f, J_gt);

            if (got_gt) {
                mpjpe_frame    = ComputeMPJPE(J_pred, J_gt);
                pa_mpjpe_frame = ComputePAMPJPE(J_pred, J_gt);
                all_mpjpe.push_back(mpjpe_frame);
                all_pa_mpjpe.push_back(pa_mpjpe_frame);
            }

            double theta_gt[72];
            bool got_pose = false;
            if (use_world_coords && !gt_data.poses_world.empty()) got_pose = gt_data.GetPoseWorld(f, theta_gt);
            else                                                  got_pose = gt_data.GetPose(f, theta_gt);

            if (got_pose) {
                pose_err_frame = ComputePoseError(theta, theta_gt, false);
                all_pose_error.push_back(pose_err_frame);
            }

            double reproj_err = 0.0;
            int reproj_count = 0;

            for (const auto& lm : lms) {
                if (lm.confidence < 0.3f) continue;
                int op_idx = lm.joint_id;
                if (op_idx < 0 || op_idx >= 25) continue;
                int smpl_idx = OPENPOSE_TO_SMPL[op_idx];
                if (smpl_idx < 0 || smpl_idx >= J_pred.rows()) continue;

                double px, py, pz;
                if (use_world_coords && f < (int)cam_poses.size()) {
                    Eigen::Vector3d p_world(J_pred(smpl_idx,0), J_pred(smpl_idx,1), J_pred(smpl_idx,2));
                    Eigen::Vector3d p_cam = cam_poses[f].R * p_world + cam_poses[f].t;
                    px = p_cam(0); py = p_cam(1); pz = p_cam(2);
                } else {
                    px = J_pred(smpl_idx,0);
                    py = J_pred(smpl_idx,1);
                    pz = J_pred(smpl_idx,2);
                }
                if (pz <= 0) continue;

                double u_pred = camera.fx * px / pz + camera.cx;
                double v_pred = camera.fy * py / pz + camera.cy;

                double err = std::sqrt((u_pred - lm.x) * (u_pred - lm.x) +
                                       (v_pred - lm.y) * (v_pred - lm.y));
                reproj_err += err;
                reproj_count++;
            }

            if (reproj_count > 0) {
                reproj2d_frame = reproj_err / reproj_count;
                all_reproj_2d.push_back(reproj2d_frame);
            }
        }

        // Log one line per frame
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

    // Final console summary + logger summary
    if (!all_mpjpe.empty()) {
        auto compute_mean = [](const std::vector<double>& v)->double{
            if (v.empty()) return -1.0;
            double s=0; for(double x: v) s+=x;
            return s / (double)v.size();
        };

        double mean_mpjpe   = compute_mean(all_mpjpe);
        double mean_pa      = compute_mean(all_pa_mpjpe);
        double mean_pose    = compute_mean(all_pose_error);
        double mean_reproj  = compute_mean(all_reproj_2d);

        std::cout << "\n========== EVALUATION SUMMARY (console) ==========\n";
        std::cout << "  Frames with GT logged: " << all_mpjpe.size() << "\n";
        std::cout << "  Mean MPJPE:            " << mean_mpjpe * 1000.0 << " mm\n";
        std::cout << "  Mean PA-MPJPE:         " << mean_pa * 1000.0 << " mm\n";
        if (mean_pose >= 0)   std::cout << "  Mean Pose Error:       " << mean_pose << " rad\n";
        if (mean_reproj >= 0) std::cout << "  Mean 2D Reproj:        " << mean_reproj << " px\n";
        std::cout << "=========================================\n";
    } else {
        std::cout << "\n[Warn] No GT frames evaluated (all_mpjpe empty)\n";
    }

    if (logger_ok) {
        elog.SetSuccessFail(success_count, fail_count);
        elog.WriteSummary();
        std::cout << "[Saved] " << elog.EvalPath() << "\n";
        std::cout << "[Saved] " << elog.ConfigPath() << "\n";
    }

    std::cout << "[Done] sequence saved under " << out_dir << "\n";
}

} // namespace app
