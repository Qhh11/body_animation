#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>

#include "optimizer.h"    
#include "evaluation.h"  

namespace logutil {

// ------------------------------
// small time helpers
// ------------------------------
inline std::string NowString() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto tt  = system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

inline void EnsureDir(const std::string& /*dir*/) {
}

// ------------------------------
// run meta info
// ------------------------------
struct RunMeta {
    std::string mode;
    std::string seq_dir;
    std::string depth_dir;
    std::string gt_dir;
    std::string out_dir;
    int start_frame = -1;
    int end_frame   = -1;
    bool use_world_coords = false;
    bool has_cam_poses    = false;

    std::map<std::string, std::string> extra;
};

// ------------------------------
// one frame metrics record
// ------------------------------
struct FrameMetrics {
    int frame = -1;
    double mpjpe      = -1.0;
    double pa_mpjpe   = -1.0;
    double pve        = -1.0;
    double pose_error = -1.0;
    double shape_error = -1.0;
    double reproj_2d  = -1.0;
};

// ------------------------------
// stats helper
// ------------------------------
struct Stats {
    double mean = -1.0;
    double std  = -1.0;
    double min  = -1.0;
    double max  = -1.0;
};

inline Stats ComputeStats(const std::vector<double>& v) {
    Stats s;
    if (v.empty()) return s;
    double sum = 0.0;
    s.min = v[0]; s.max = v[0];
    for (double x : v) { sum += x; s.min = std::min(s.min, x); s.max = std::max(s.max, x); }
    s.mean = sum / (double)v.size();
    double var = 0.0;
    for (double x : v) var += (x - s.mean) * (x - s.mean);
    s.std = std::sqrt(var / (double)v.size());
    return s;
}

// ------------------------------
// dump OptimizerConfig
// ------------------------------
inline void DumpOptimizerConfig(std::ostream& os,
                                const OptimizerConfig& cfg,
                                const CameraIntrinsics* cam = nullptr,
                                const RunMeta* meta = nullptr) {
    os << "====================================================\n";
    os << "CONFIG DUMP\n";
    os << "Time: " << NowString() << "\n";
    if (meta) {
        os << "Mode: " << meta->mode << "\n";
        os << "OutDir: " << meta->out_dir << "\n";
        if (!meta->seq_dir.empty())   os << "SeqDir: " << meta->seq_dir << "\n";
        if (!meta->depth_dir.empty()) os << "DepthDir: " << meta->depth_dir << "\n";
        if (!meta->gt_dir.empty())    os << "GTDir: " << meta->gt_dir << "\n";
        if (meta->start_frame >= 0)   os << "StartFrame: " << meta->start_frame << "\n";
        if (meta->end_frame >= 0)     os << "EndFrame: " << meta->end_frame << "\n";
        os << "UseWorldCoords: " << (meta->use_world_coords ? "true" : "false") << "\n";
        os << "HasCamPoses:    " << (meta->has_cam_poses ? "true" : "false") << "\n";
        if (!meta->extra.empty()) {
            os << "Extra:\n";
            for (const auto& kv : meta->extra)
                os << "  - " << kv.first << ": " << kv.second << "\n";
        }
    }

    if (cam) {
        os << "\n[CameraIntrinsics]\n";
        os << "  fx=" << cam->fx << " fy=" << cam->fy
           << " cx=" << cam->cx << " cy=" << cam->cy;
    }

    os << "\n[General]\n";
    os << "  verbose=" << cfg.verbose << "\n";
    os << "  threads=" << cfg.threads << "\n";
    os << "  print_progress=" << cfg.print_progress << "\n";
    os << "  min_confidence=" << cfg.min_confidence << "\n";
    os << "  use_stable_joints_only=" << cfg.use_stable_joints_only << "\n";
    os << "  flip_y_in_projection=" << cfg.flip_y_in_projection << "\n";
    os << "  use_joint_importance=" << cfg.use_joint_importance << "\n";

    os << "\n[GMM]\n";
    os << "  use_gmm_prior=" << cfg.use_gmm_prior << "\n";
    os << "  w_gmm=" << cfg.w_gmm << "\n";

    os << "\n[Hinge/Bend]\n";
    os << "  use_hinge_priors=" << cfg.use_hinge_priors << "\n";
    os << "  use_bend_limits=" << cfg.use_bend_limits << "\n";
    os << "  w_bend=" << cfg.w_bend << "\n";
    os << "  hinge_min=" << cfg.hinge_min << " hinge_max=" << cfg.hinge_max << "\n";
    os << "  w_hinge_perp_elbow=" << cfg.w_hinge_perp_elbow << "\n";
    os << "  w_hinge_range_elbow=" << cfg.w_hinge_range_elbow << "\n";
    os << "  w_hinge_perp_knee=" << cfg.w_hinge_perp_knee << "\n";
    os << "  w_hinge_range_knee=" << cfg.w_hinge_range_knee << "\n";

    os << "\n[Root Upright]\n";
    os << "  use_root_upright=" << cfg.use_root_upright << "\n";
    os << "  root_upright_w_roll=" << cfg.root_upright_w_roll << "\n";
    os << "  root_upright_w_pitch=" << cfg.root_upright_w_pitch << "\n";

    os << "\n[Soft Limits]\n";
    os << "  use_shoulder_collar_limits=" << cfg.use_shoulder_collar_limits << "\n";
    os << "  w_collar_limit=" << cfg.w_collar_limit << " collar_amax=" << cfg.collar_amax << "\n";
    os << "  w_shoulder_limit=" << cfg.w_shoulder_limit << " shoulder_amax=" << cfg.shoulder_amax << "\n";
    os << "  use_wrist_limits=" << cfg.use_wrist_limits << "\n";
    os << "  w_wrist_limit=" << cfg.w_wrist_limit << " wrist_amax=" << cfg.wrist_amax << "\n";
    os << "  use_spine_limits=" << cfg.use_spine_limits << "\n";
    os << "  w_spine=" << cfg.w_spine << " spine_amax=" << cfg.spine_amax << "\n";
    os << "  use_neck_head_limits=" << cfg.use_neck_head_limits << "\n";
    os << "  w_neck_limit=" << cfg.w_neck_limit << " neck_amax=" << cfg.neck_amax << "\n";
    os << "  w_head_limit=" << cfg.w_head_limit << " head_amax=" << cfg.head_amax << "\n";


    os << "\n[Extra Priors]\n";
    os << "  use_pelvis_over_ankles=" << cfg.use_pelvis_over_ankles << "\n";
    os << "  use_arm_torso_sep=" << cfg.use_arm_torso_sep << "\n";
    os << "  arm_torso_seg_a=" << cfg.arm_torso_seg_a << " arm_torso_seg_b=" << cfg.arm_torso_seg_b << "\n";
    os << "  arm_torso_d_elbow=" << cfg.arm_torso_d_elbow << " arm_torso_d_wrist=" << cfg.arm_torso_d_wrist << "\n";
    os << "  arm_torso_w_elbow=" << cfg.arm_torso_w_elbow << " arm_torso_w_wrist=" << cfg.arm_torso_w_wrist << "\n";

    os << "\n[Locks]\n";
    os << "  lock_wrists=" << cfg.lock_wrists << "\n";
    os << "  lock_hands=" << cfg.lock_hands << "\n";
    os << "  lock_feet=" << cfg.lock_feet << "\n";

    os << "\n[Depth]\n";
    os << "  use_depth=" << cfg.use_depth << " w_depth=" << cfg.w_depth << "\n";
    os << "  use_relative_depth=" << cfg.use_relative_depth << " w_relative_depth=" << cfg.w_relative_depth << "\n";
    os << "  use_depth_affine=" << cfg.use_depth_affine << " w_depth_affine=" << cfg.w_depth_affine << "\n";
    os << "  depth_affine_sign=" << cfg.depth_affine_sign << "\n";
    os << "  fix_depth_affine_bias=" << cfg.fix_depth_affine_bias << "\n";
    os << "  use_temporal_depth_affine=" << cfg.use_temporal_depth_affine << "\n";
    os << "  w_temp_depth_scale=" << cfg.w_temp_depth_scale << " w_temp_depth_bias=" << cfg.w_temp_depth_bias << "\n";

    os << "\n[Camera-Space Z Prior]\n";
    os << "  use_trans_z_prior=" << cfg.use_trans_z_prior << "\n";
    os << "  trans_z_prior_target=" << cfg.trans_z_prior_target << "\n";
    os << "  w_trans_z_prior=" << cfg.w_trans_z_prior << "\n";

    os << "\n[Temporal]\n";
    os << "  use_temporal=" << cfg.use_temporal << "\n";
    os << "  w_temp_pose=" << cfg.w_temp_pose << " w_temp_trans=" << cfg.w_temp_trans << "\n";
    os << "  w_temp_root_orient=" << cfg.w_temp_root_orient << "\n";
    os << "  use_temporal_acc=" << cfg.use_temporal_acc << "\n";
    os << "  w_temp_acc_pose=" << cfg.w_temp_acc_pose << "\n";
    os << "  temporal_body_only=" << cfg.temporal_body_only << "\n";
    os << "  temporal_arm_scale=" << cfg.temporal_arm_scale << "\n";

    os << "\n[Foot Ground]\n";
    os << "  use_foot_ground=" << cfg.use_foot_ground << "\n";
    os << "  w_foot_ground=" << cfg.w_foot_ground << " w_foot_flat=" << cfg.w_foot_flat << "\n";
    os << "  ground_y=" << cfg.ground_y << "\n";
    os << "  foot_contact_L=" << cfg.foot_contact_L << " foot_contact_R=" << cfg.foot_contact_R << "\n";

    os << "\n[Capsule/CrossLimb]\n";
    os << "  use_capsule_collision=" << cfg.use_capsule_collision << " w=" << cfg.w_capsule_collision << "\n";
    os << "  use_cross_limb_depth=" << cfg.use_cross_limb_depth << " w=" << cfg.w_cross_limb_depth << "\n";
    os << "  capsule_unit_scale=" <<     cfg.capsule_unit_scale << "\n";


    os << "\n[Solver Iters]\n";
    os << "  max_iter_stage1=" << cfg.max_iter_stage1 << "\n";
    os << "  max_iter_stage2=" << cfg.max_iter_stage2 << "\n";
    os << "  max_iter_stage3=" << cfg.max_iter_stage3 << "\n";

    os << "\n[Graduated Energy]\n";
    os << "  use_graduated_energy=" << cfg.use_graduated_energy << "\n";
    os << "  grad_stage1_depth_scale=" << cfg.grad_stage1_depth_scale << "\n";
    os << "  grad_stage1_prior_scale=" << cfg.grad_stage1_prior_scale << "\n";
    os << "  grad_stage2_depth_scale=" << cfg.grad_stage2_depth_scale << "\n";
    os << "  grad_stage2_prior_scale=" << cfg.grad_stage2_prior_scale << "\n";

    os << "\n[Multi-Hypothesis]\n";
    os << "  multi_hyp_max_z_shift=" << cfg.multi_hyp_max_z_shift << "\n";

    os << "====================================================\n";
}

// ------------------------------
// EvalLogger
// ------------------------------
class EvalLogger {
public:
    EvalLogger() = default;

    bool Open(const std::string& out_dir,
              const std::string& eval_filename = "evaluation.txt",
              const std::string& config_filename = "config.txt") {
        out_dir_ = out_dir;
        eval_path_ = out_dir_ + "/" + eval_filename;
        cfg_path_  = out_dir_ + "/" + config_filename;
        eval_.open(eval_path_, std::ios::out);
        cfg_.open(cfg_path_, std::ios::out);
        return eval_.is_open() && cfg_.is_open();
    }

    const std::string& EvalPath()   const { return eval_path_; }
    const std::string& ConfigPath() const { return cfg_path_; }

    void WriteConfig(const OptimizerConfig& cfg,
                     const CameraIntrinsics* cam = nullptr,
                     const RunMeta* meta = nullptr) {
        if (!cfg_.is_open()) return;
        DumpOptimizerConfig(cfg_, cfg, cam, meta);
        cfg_.flush();
    }

    void WriteEvalHeader(const RunMeta& meta) {
        if (!eval_.is_open()) return;
        eval_ << "EVALUATION LOG\n";
        eval_ << "Time: " << NowString() << "\n";
        eval_ << "Mode: " << meta.mode << "\n";
        eval_ << "OutDir: " << meta.out_dir << "\n";
        if (!meta.seq_dir.empty())   eval_ << "SeqDir: " << meta.seq_dir << "\n";
        if (!meta.depth_dir.empty()) eval_ << "DepthDir: " << meta.depth_dir << "\n";
        if (!meta.gt_dir.empty())    eval_ << "GTDir: " << meta.gt_dir << "\n";
        if (meta.start_frame >= 0)   eval_ << "StartFrame: " << meta.start_frame << "\n";
        if (meta.end_frame >= 0)     eval_ << "EndFrame: " << meta.end_frame << "\n";
        eval_ << "UseWorldCoords: " << (meta.use_world_coords ? "true" : "false") << "\n";
        eval_ << "HasCamPoses: " << (meta.has_cam_poses ? "true" : "false") << "\n";
        if (!meta.extra.empty()) {
            eval_ << "Extra:\n";
            for (const auto& kv : meta.extra) eval_ << "  - " << kv.first << ": " << kv.second << "\n";
        }
        eval_ << "\n# Per-frame metrics\n";
        eval_ << "# frame  MPJPE(mm)  PA-MPJPE(mm)  PVE(mm)  PoseErr(rad)  ShapeErr  2DReproj(px)\n";
        eval_.flush();
    }

    void LogFrame(const FrameMetrics& m) {
        if (!eval_.is_open()) return;
        eval_ << m.frame << " "
              << (m.mpjpe >= 0 ? m.mpjpe * 1000.0 : -1.0) << " "
              << (m.pa_mpjpe >= 0 ? m.pa_mpjpe * 1000.0 : -1.0) << " "
              << (m.pve >= 0 ? m.pve * 1000.0 : -1.0) << " "
              << (m.pose_error >= 0 ? m.pose_error : -1.0) << " "
              << (m.shape_error >= 0 ? m.shape_error : -1.0) << " "
              << (m.reproj_2d >= 0 ? m.reproj_2d : -1.0)
              << "\n";
        eval_.flush();
    }

    void Push(const FrameMetrics& m) {
        frames_.push_back(m);
        if (m.mpjpe >= 0) mpjpe_.push_back(m.mpjpe);
        if (m.pa_mpjpe >= 0) pa_.push_back(m.pa_mpjpe);
        if (m.pve >= 0) pve_.push_back(m.pve);
        if (m.pose_error >= 0) pose_.push_back(m.pose_error);
        if (m.reproj_2d >= 0) reproj_.push_back(m.reproj_2d);
        if (m.shape_error >= 0) shape_.push_back(m.shape_error);
    }

    void SetSuccessFail(int success, int fail) { success_ = success; fail_ = fail; }

    void WriteSummary() {
        if (!eval_.is_open()) return;
        auto s_mpjpe = ComputeStats(mpjpe_);
        auto s_pa    = ComputeStats(pa_);
        auto s_pve   = ComputeStats(pve_);
        auto s_pose  = ComputeStats(pose_);
        auto s_repr  = ComputeStats(reproj_);
        auto s_shape = ComputeStats(shape_);

        eval_ << "\n========================================\n";
        eval_ << "SUMMARY\n";
        eval_ << "Frames logged: " << frames_.size() << "\n";
        eval_ << "Success: " << success_ << "\n";
        eval_ << "Fail:    " << fail_ << "\n\n";

        auto print_stats = [&](const char* name, const Stats& s, bool mm=false) {
            if (s.mean < 0) return;
            eval_ << name << ":\n";
            if (mm)
                eval_ << "  mean=" << s.mean*1000.0 << "  std=" << s.std*1000.0
                      << "  min=" << s.min*1000.0   << "  max=" << s.max*1000.0 << " (mm)\n";
            else
                eval_ << "  mean=" << s.mean << "  std=" << s.std
                      << "  min=" << s.min  << "  max=" << s.max << "\n";
        };

        print_stats("MPJPE", s_mpjpe, true);
        print_stats("PA-MPJPE", s_pa, true);
        print_stats("PVE", s_pve, true);
        print_stats("PoseErr", s_pose, false);
        print_stats("ShapeErr", s_shape, false);
        print_stats("2DReproj(px)", s_repr, false);

        eval_ << "========================================\n";
        eval_.flush();
    }

private:
    std::string out_dir_, eval_path_, cfg_path_;
    std::ofstream eval_, cfg_;
    int success_ = 0, fail_ = 0;
    std::vector<FrameMetrics> frames_;
    std::vector<double> mpjpe_, pa_, pve_, pose_, reproj_, shape_;
};

} // namespace logutil