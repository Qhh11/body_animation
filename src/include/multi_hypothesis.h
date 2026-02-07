// multi_hypothesis.h
// Solves the warm-start local-minimum problem in monocular 2D→3D fitting.
//
// Three mechanisms:
//   1. DepthGuidedZCorrection — cheap pre-optimization Z shift using depth cues
//   2. ComputeCombinedScore   — multi-term scoring (reproj + depth + temporal)
//   3. RunMultiHypothesisOptimization — run 2-3 warm-start hypotheses,
//      quick-evaluate, pick best, then full optimize
//
// Usage in openpose_runners.cpp:
//   #include "multi_hypothesis.h"
//   ...
//   // For tracked frames, replace direct OptimizeSMPLTracked call with:
//   multihyp::RunMultiHypothesisOptimization(lms, camera, model, beta, theta, trans,
//       cfg, theta_prev_curr, trans_prev_curr, ...);
//
#pragma once

#include "optimizer.h"
#include "smpl_model.h"
#include "smpl_math.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

namespace multihyp {

// ---------------------------------------------------------
// 1. Depth-Guided Z Correction
//    After warm-start, shift trans[2] so predicted joint Zs
//    better match the affine-converted depth observations.
//    Cost: one SMPL forward pass + arithmetic. ~0.1ms.
// ---------------------------------------------------------
inline void DepthGuidedZCorrection(
    const SmplModel& model,
    const double* beta,
    const double* theta,
    double* trans,                               // modified in-place
    const std::vector<JointRelDepthData>& depth_pts,
    const double* depth_affine,                  // [scale, bias]
    double max_z_shift = 0.5,
    double min_confidence = 0.3)
{
    if (depth_pts.empty() || !depth_affine) return;

    MatX3dRM J;
    SMPLJointsForward<double>(beta, theta, trans, model, J);

    double sum_delta = 0.0;
    int count = 0;

    for (const auto& dp : depth_pts) {
        if (dp.weight < min_confidence) continue;
        const int j = dp.smpl_joint_idx;
        if (j < 0 || j >= J.rows()) continue;

        const double z_pred = J(j, 2);
        const double z_tgt  = depth_affine[1] + depth_affine[0] * dp.rel_depth;

        // sanity: skip if target Z is nonsensical
        if (z_tgt < 0.3 || z_tgt > 30.0) continue;

        sum_delta += (z_tgt - z_pred);
        ++count;
    }

    if (count < 3) return;

    double z_shift = sum_delta / count;
    z_shift = std::clamp(z_shift, -max_z_shift, max_z_shift);

    if (std::abs(z_shift) > 0.02) {
        trans[2] += z_shift;
        std::cout << "[DepthZCorr] shifted Z by " << z_shift
                  << " (from " << count << " joints)\n";
    }
}

// ---------------------------------------------------------
// 2. Combined Score
//    Evaluates a hypothesis across reproj, depth consistency,
//    and temporal smoothness. Lower = better.
// ---------------------------------------------------------
inline double ComputeCombinedScore(
    const std::vector<Landmark2D>& lms,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    const double* beta,
    const double* theta,
    const double* trans,
    const OptimizerConfig& cfg,
    const std::vector<JointRelDepthData>& depth_pts,
    const double* depth_affine,
    const double* theta_prev,
    const double* trans_prev)
{
    MatX3dRM J;
    SMPLJointsForward<double>(beta, theta, trans, model, J);

    // --- (a) Average 2D reprojection error (pixels) ---
    double reproj = 0.0;
    int rc = 0;
    for (const auto& lm : lms) {
        if (lm.confidence < cfg.min_confidence) continue;
        if (lm.joint_id < 0 || lm.joint_id >= 25) continue;
        const int sid = OPENPOSE_TO_SMPL[lm.joint_id];
        if (sid < 0 || sid >= J.rows()) continue;

        const double Z = J(sid, 2);
        if (Z <= 1e-6) continue;

        const double u = camera.fx * J(sid, 0) / Z + camera.cx;
        const double v = camera.fy * (cfg.flip_y_in_projection ? -J(sid, 1) : J(sid, 1)) / Z + camera.cy;

        reproj += std::sqrt((u - lm.x) * (u - lm.x) + (v - lm.y) * (v - lm.y));
        ++rc;
    }
    if (rc > 0) reproj /= rc;

    // --- (b) Depth consistency (meters) ---
    double depth_err = 0.0;
    if (!depth_pts.empty() && depth_affine) {
        int dc = 0;
        for (const auto& dp : depth_pts) {
            if (dp.weight < 0.3) continue;
            const int j = dp.smpl_joint_idx;
            if (j < 0 || j >= J.rows()) continue;

            const double z_pred = J(j, 2);
            const double z_tgt  = depth_affine[1] + depth_affine[0] * dp.rel_depth;
            depth_err += std::abs(z_pred - z_tgt);
            ++dc;
        }
        if (dc > 0) depth_err /= dc;
    }

    // --- (c) Temporal smoothness ---
    double temp_err = 0.0;
    if (theta_prev && trans_prev) {
        // Pose change (body only, skip root)
        double pose_diff = 0.0;
        for (int i = 3; i < 72; ++i) {
            const double d = theta[i] - theta_prev[i];
            pose_diff += d * d;
        }
        temp_err += std::sqrt(pose_diff / 69.0);

        // Translation Z jump (weighted more)
        const double dz = trans[2] - trans_prev[2];
        temp_err += std::abs(dz) * 2.0;
    }

    // --- Weighted combination ---
    return cfg.multi_hyp_w_reproj  * reproj
         + cfg.multi_hyp_w_depth   * depth_err
         + cfg.multi_hyp_w_temporal * temp_err;
}

// ---------------------------------------------------------
// 3. Multi-Hypothesis Optimization
//    Generates 2-3 warm-start candidates, quick-evaluates each,
//    then runs full optimization from the best starting point.
// ---------------------------------------------------------
struct WarmStartHypothesis {
    double theta[72];
    double trans[3];
    const char* label;
};

inline bool RunMultiHypothesisOptimization(
    const std::vector<Landmark2D>& lms,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    double* beta,
    double* theta,       // in/out: warm-started, returns best result
    double* trans,        // in/out
    OptimizerConfig& cfg, // non-const: we modify iter counts for quick pass
    const double* theta_prev,
    const double* trans_prev,
    const double* theta_prev2,
    const double* trans_prev2,
    bool has_prev2,
    const std::vector<JointRelDepthData>& depth_pts,
    const double* depth_affine)
{
    // ------- Build hypotheses -------
    std::vector<WarmStartHypothesis> hyps;

    // Hyp 0: standard warm-start (as-is)
    {
        WarmStartHypothesis h;
        std::memcpy(h.theta, theta, sizeof(h.theta));
        std::memcpy(h.trans, trans, sizeof(h.trans));
        h.label = "warm-start";
        hyps.push_back(h);
    }

    // Hyp 1: depth-corrected Z
    if (!depth_pts.empty() && depth_affine) {
        WarmStartHypothesis h;
        std::memcpy(h.theta, theta, sizeof(h.theta));
        std::memcpy(h.trans, trans, sizeof(h.trans));
        DepthGuidedZCorrection(model, beta, h.theta, h.trans,
                               depth_pts, depth_affine,
                               cfg.multi_hyp_max_z_shift);
        // Only add if it's actually different from hyp 0
        if (std::abs(h.trans[2] - hyps[0].trans[2]) > 0.02) {
            h.label = "depth-z-corrected";
            hyps.push_back(h);
        }
    }

    // Hyp 2: Z perturbation (push further away)
    {
        WarmStartHypothesis h;
        std::memcpy(h.theta, theta, sizeof(h.theta));
        std::memcpy(h.trans, trans, sizeof(h.trans));

        // Determine perturbation direction from depth hint
        double dir = cfg.multi_hyp_z_perturb;
        if (!depth_pts.empty() && depth_affine && !hyps.empty()) {
            // If depth-correction wanted to push Z forward, perturb backward (explore)
            double delta = hyps.back().trans[2] - hyps[0].trans[2];
            if (delta > 0.05) dir = -cfg.multi_hyp_z_perturb;
        }
        h.trans[2] += dir;
        // Clamp to valid range
        h.trans[2] = std::clamp(h.trans[2], 0.5, 20.0);

        if (std::abs(h.trans[2] - hyps[0].trans[2]) > 0.05) {
            h.label = "z-perturbed";
            hyps.push_back(h);
        }
    }

    // If only 1 hypothesis, just run normal optimization
    if (hyps.size() <= 1) {
        return OptimizeSMPLTracked(lms, camera, model, beta, theta, trans, cfg,
                                    theta_prev, trans_prev,
                                    has_prev2 ? theta_prev2 : nullptr,
                                    has_prev2 ? trans_prev2 : nullptr);
    }

    // ------- Quick evaluation pass -------
    const int save_s1 = cfg.max_iter_stage1;
    const int save_s2 = cfg.max_iter_stage2;
    const int save_s3 = cfg.max_iter_stage3;
    const bool save_verbose = cfg.verbose;
    const bool save_print = cfg.print_progress;

    cfg.max_iter_stage1 = cfg.multi_hyp_quick_iter_s1;
    cfg.max_iter_stage2 = cfg.multi_hyp_quick_iter_s2;
    cfg.max_iter_stage3 = 0;  // skip stage 3 in quick pass
    cfg.verbose = false;
    cfg.print_progress = false;

    double best_score = 1e18;
    int best_idx = 0;

    std::cout << "[MultiHyp] Evaluating " << hyps.size() << " hypotheses...\n";

    for (int i = 0; i < (int)hyps.size(); ++i) {
        auto& h = hyps[i];

        // Make working copies
        double bt[10];
        std::memcpy(bt, beta, sizeof(double) * 10);

        bool ok = OptimizeSMPLTracked(
            lms, camera, model,
            bt, h.theta, h.trans, cfg,
            theta_prev, trans_prev,
            has_prev2 ? theta_prev2 : nullptr,
            has_prev2 ? trans_prev2 : nullptr);

        if (!ok) {
            std::cout << "  [Hyp " << i << ": " << h.label << "] FAILED\n";
            continue;
        }

        double score = ComputeCombinedScore(
            lms, camera, model, bt, h.theta, h.trans,
            cfg, depth_pts, depth_affine,
            theta_prev, trans_prev);

        std::cout << "  [Hyp " << i << ": " << h.label
                  << "] score=" << score
                  << " (Z=" << h.trans[2] << ")\n";

        if (score < best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    // ------- Restore config -------
    cfg.max_iter_stage1 = save_s1;
    cfg.max_iter_stage2 = save_s2;
    cfg.max_iter_stage3 = save_s3;
    cfg.verbose = save_verbose;
    cfg.print_progress = save_print;

    // ------- Full optimization from best hypothesis -------
    std::cout << "[MultiHyp] Selected hyp " << best_idx
              << " (" << hyps[best_idx].label
              << ") score=" << best_score << "\n";

    std::memcpy(theta, hyps[best_idx].theta, sizeof(double) * 72);
    std::memcpy(trans, hyps[best_idx].trans, sizeof(double) * 3);

    return OptimizeSMPLTracked(
        lms, camera, model, beta, theta, trans, cfg,
        theta_prev, trans_prev,
        has_prev2 ? theta_prev2 : nullptr,
        has_prev2 ? trans_prev2 : nullptr);
}

} // namespace multihyp