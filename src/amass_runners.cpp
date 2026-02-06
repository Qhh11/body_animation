#include "amass_runners.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <chrono>
#include <filesystem>

#include "smpl_math.h"
#include "amass_loader.h"
#include "evaluation.h"
#include "io_utils.h"
#include "config_builders.h"

namespace fs = std::filesystem;

namespace app {

// Synthetic landmarks for AMASS validation
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

double RunAMASSValidation(const SmplModel& model,
                          const GMMPosePrior* gmm_prior,
                          const std::string& amass_path,
                          int frame_idx,
                          bool save_mesh) {
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

    std::string base_name = StemNoExt(amass_path);

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

    OptimizerConfig cfg = MakeCfgForAMASS(&depth_gt, gmm_prior);

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

void RunAmassValidationSequence(const SmplModel& model,
                                const GMMPosePrior* gmm_prior,
                                const std::string& pose_file_path) {
    std::cout << "\n========== AMASS SEQUENCE VALIDATION (Frames 0-50) ==========\n";
    std::cout << "File: " << pose_file_path << "\n";

    std::string base_name = fs::path(pose_file_path).stem().string();

    const std::string mesh_dir = "./meshes/" + base_name;
    const std::string gt_mesh_dir = mesh_dir + "/gt";
    const std::string opt_mesh_dir = mesh_dir + "/optimized";

    fs::create_directories(gt_mesh_dir);
    fs::create_directories(opt_mesh_dir);

    std::vector<AmassFrame> amass_frames;
    if (!AmassLoader::Load(pose_file_path, amass_frames)) {
        std::cerr << "Failed to load file: " << pose_file_path << "\n";
        return;
    }
    if (amass_frames.empty()) {
        std::cerr << "No frames in file\n";
        return;
    }

    const int start_frame = 0;
    const int end_frame = std::min(50, (int)amass_frames.size() - 1);
    const int num_frames = end_frame - start_frame + 1;

    CameraIntrinsics camera = CreateDefaultCamera(800, 800);

    std::vector<double> all_mpjpe, all_pa_mpjpe, all_pve, all_pose_error, all_shape_error, all_reproj_2d;
    std::vector<std::vector<double>> per_joint_errors(24);

    all_mpjpe.reserve(num_frames);
    all_pa_mpjpe.reserve(num_frames);

    int success_count = 0;
    int fail_count = 0;

    for (int frame_idx = start_frame; frame_idx <= end_frame; ++frame_idx) {
        const AmassFrame& frame_gt = amass_frames[frame_idx];

        double beta_gt[10];
        double theta_gt[72];
        double trans_gt[3];

        for (int i = 0; i < 10; ++i) beta_gt[i] = frame_gt.beta[i];
        for (int i = 0; i < 72; ++i) theta_gt[i] = frame_gt.theta[i];
        trans_gt[0] = 0.0; trans_gt[1] = 0.0; trans_gt[2] = 3.0;

        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_gt;
        SMPLForward<double>(beta_gt, theta_gt, trans_gt, model, V_gt);

        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_gt;
        SMPLJointsForward<double>(beta_gt, theta_gt, trans_gt, model, J_gt);

        char gt_filename[512];
        std::snprintf(gt_filename, sizeof(gt_filename), "%s/frame_%05d_gt.obj", gt_mesh_dir.c_str(), frame_idx);
        SaveOBJ(gt_filename, V_gt, model.faces);

        auto observed_landmarks = GenerateSyntheticLandmarks(model, camera, beta_gt, theta_gt, trans_gt);
        if (observed_landmarks.empty()) { fail_count++; continue; }

        std::vector<JointDepthData> depth_gt;
        depth_gt.reserve(24);
        for (int i = 0; i < 24; i++) depth_gt.push_back({i, J_gt(i, 2)});

        double beta_opt[10]  = {0};
        double theta_opt[72] = {0};
        double trans_opt[3]  = {0.0, 0.0, 2.5};

        OptimizerConfig cfg = MakeCfgForAMASS(&depth_gt, gmm_prior);
        cfg.verbose = false;

        bool ok = OptimizeSMPL(observed_landmarks, camera, model, beta_opt, theta_opt, trans_opt, cfg);
        if (!ok) { fail_count++; continue; }

        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_opt;
        SMPLForward<double>(beta_opt, theta_opt, trans_opt, model, V_opt);

        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_opt;
        SMPLJointsForward<double>(beta_opt, theta_opt, trans_opt, model, J_opt);

        char opt_filename[512];
        std::snprintf(opt_filename, sizeof(opt_filename), "%s/frame_%05d_opt.obj", opt_mesh_dir.c_str(), frame_idx);
        SaveOBJ(opt_filename, V_opt, model.faces);

        double mpjpe = ComputeMPJPE(J_opt, J_gt);
        double pa_mpjpe = ComputePAMPJPE(J_opt, J_gt);
        all_mpjpe.push_back(mpjpe);
        all_pa_mpjpe.push_back(pa_mpjpe);

        double pve = ComputePVE(V_opt, V_gt);
        all_pve.push_back(pve);

        double pose_error = ComputePoseError(theta_opt, theta_gt, false);
        all_pose_error.push_back(pose_error);

        double shape_error = ComputeShapeError(beta_opt, beta_gt);
        all_shape_error.push_back(shape_error);

        double reproj_err = 0.0;
        int reproj_count = 0;
        for (const auto& lm : observed_landmarks) {
            if (lm.confidence < 0.3f) continue;
            int op_idx = lm.joint_id;
            if (op_idx < 0 || op_idx >= 25) continue;
            int smpl_idx = OPENPOSE_TO_SMPL[op_idx];
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
        all_reproj_2d.push_back(reproj_count > 0 ? reproj_err / reproj_count : -1.0);

        for (int j = 0; j < 24; ++j) {
            double joint_error = (J_opt.row(j) - J_gt.row(j)).norm();
            per_joint_errors[j].push_back(joint_error);
        }

        success_count++;
        std::cout << "[Frame " << frame_idx << "] "
                  << "MPJPE=" << mpjpe * 1000.0 << "mm, "
                  << "PA-MPJPE=" << pa_mpjpe * 1000.0 << "mm, "
                  << "PVE=" << pve * 1000.0 << "mm\n";
    }

    if (all_mpjpe.empty()) {
        std::cout << "\n[Error] No successful frames to evaluate\n";
        return;
    }

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

    std::string eval_file = mesh_dir + "/evaluation.txt";
    std::ofstream outfile(eval_file);
    if (outfile.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);

        outfile << "AMASS Sequence Validation Results\n";
        outfile << "==================================\n";
        outfile << "File: " << pose_file_path << "\n";
        outfile << "Date: " << std::ctime(&t);
        outfile << "Frames evaluated: " << all_mpjpe.size() << " (" << start_frame << "-" << end_frame << ")\n";
        outfile << "Successful: " << success_count << "\n";
        outfile << "Failed: " << fail_count << "\n\n";

        outfile << "Mean MPJPE (mm): " << mean_mpjpe * 1000.0 << "\n";
        outfile << "Mean PA-MPJPE (mm): " << mean_pa_mpjpe * 1000.0 << "\n";
        outfile << "Mean PVE (mm): " << mean_pve * 1000.0 << "\n";
        outfile << "Mean Pose Error (rad): " << mean_pose_error << "\n";
        outfile << "Mean Shape Error: " << mean_shape_error << "\n";
        outfile << "Mean 2D Reproj (px): " << mean_reproj_2d << "\n\n";

        outfile << "Per-Frame Results:\n";
        outfile << "Frame\tMPJPE(mm)\tPA-MPJPE(mm)\tPVE(mm)\tPoseErr(rad)\tShapeErr\t2DReproj(px)\n";
        for (size_t i = 0; i < all_mpjpe.size(); ++i) {
            outfile << (start_frame + (int)i) << "\t"
                    << all_mpjpe[i] * 1000.0 << "\t"
                    << all_pa_mpjpe[i] * 1000.0 << "\t"
                    << (i < all_pve.size() ? all_pve[i] * 1000.0 : -1.0) << "\t"
                    << (i < all_pose_error.size() ? all_pose_error[i] : -1.0) << "\t"
                    << (i < all_shape_error.size() ? all_shape_error[i] : -1.0) << "\t"
                    << (i < all_reproj_2d.size() ? all_reproj_2d[i] : -1.0) << "\n";
        }
        outfile.close();
        std::cout << "\n[Saved] Evaluation results: " << eval_file << "\n";
    }

    std::cout << "\n[Done] Meshes saved to: " << mesh_dir << "\n";
}

void RunBatchValidation(const SmplModel& model,
                        const GMMPosePrior* gmm_prior) {
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
        double err = RunAMASSValidation(model, gmm_prior, item.first, item.second, true);
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

} // namespace app
