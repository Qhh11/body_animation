#pragma once

#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <numeric>

#include <Eigen/Dense>

#include "smpl_model.h"
#include "smpl_math.h"
#include "optimizer.h"
#include "io_utils.h"

namespace postsmooth {

// ============================================================
// Per-frame result storage
// ============================================================
struct FrameResult {
    int frame_idx = -1;

    double theta[72]  = {0.0};
    double trans[3]   = {0.0};
    double beta[10]   = {0.0};

    double depth_affine[2] = {3.0, 3.0};

    // Quality metrics
    double reproj_px      = 1e18;   // mean 2D reprojection error (pixels)
    double pose_delta_rad = 0.0;    // total angular change from previous frame
    double trans_delta_m  = 0.0;    // translation change from previous (meters)
    double root_delta_rad = 0.0;    // root orientation change from previous

    bool solver_ok   = false;
    bool rejected    = false;       // set by quality filter
    bool interpolated = false;      // set by gap-fill pass

    // 2D observation quality
    double mean_confidence = 0.0;
    int    num_visible_kps = 0;
};

// ============================================================
// Compute per-frame quality deltas (call after forward pass)
// ============================================================
inline void ComputeFrameDeltas(std::vector<FrameResult>& results) {
    for (size_t i = 1; i < results.size(); ++i) {
        if (!results[i].solver_ok || !results[i-1].solver_ok) continue;

        // Body pose delta (theta[3:72])
        double pd = 0.0;
        for (int k = 3; k < 72; ++k) {
            double d = results[i].theta[k] - results[i-1].theta[k];
            pd += d * d;
        }
        results[i].pose_delta_rad = std::sqrt(pd);

        // Root orientation delta (theta[0:3])
        double rd = 0.0;
        for (int k = 0; k < 3; ++k) {
            double d = results[i].theta[k] - results[i-1].theta[k];
            rd += d * d;
        }
        results[i].root_delta_rad = std::sqrt(rd);

        // Translation delta
        double td = 0.0;
        for (int k = 0; k < 3; ++k) {
            double d = results[i].trans[k] - results[i-1].trans[k];
            td += d * d;
        }
        results[i].trans_delta_m = std::sqrt(td);
    }
}

// ============================================================
// Mark bad frames based on physically implausible motion
// ============================================================
struct QualityThresholds {
    double max_pose_delta   = 1.0;   // rad total body change per frame
    double max_root_delta   = 0.8;   // rad root orientation change per frame
    double max_trans_delta  = 0.4;   // meters translation per frame
    double max_reproj_px    = 32.0;  // pixels mean reprojection error
    int    min_visible_kps  = 8;     // minimum visible keypoints
};

inline int MarkBadFrames(std::vector<FrameResult>& results,
                         const QualityThresholds& thres = QualityThresholds())
{
    int bad_count = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        auto& fr = results[i];

        // Solver failure
        if (!fr.solver_ok) {
            fr.rejected = true;
            bad_count++;
            continue;
        }

        // Too few keypoints visible
        if (fr.num_visible_kps < thres.min_visible_kps) {
            fr.rejected = true;
            bad_count++;
            continue;
        }

        // Skip delta checks for first frame
        if (i == 0) continue;

        // Physics-based rejection: jumps too large for one frame
        bool jumped = false;
        if (fr.pose_delta_rad > thres.max_pose_delta) jumped = true;
        if (fr.root_delta_rad > thres.max_root_delta) jumped = true;
        if (fr.trans_delta_m  > thres.max_trans_delta) jumped = true;

        // Additional heuristic: if reprojection is bad AND there's a jump,
        // almost certainly a bad local minimum
        if (jumped && fr.reproj_px > thres.max_reproj_px * 0.5) {
            fr.rejected = true;
            bad_count++;
            continue;
        }

        // Pure jump rejection (even if reproj is ok, physics says no)
        if (fr.pose_delta_rad > thres.max_pose_delta * 1.5 ||
            fr.root_delta_rad > thres.max_root_delta * 1.5 ||
            fr.trans_delta_m  > thres.max_trans_delta * 1.5) {
            fr.rejected = true;
            bad_count++;
        }
    }
    return bad_count;
}

// ============================================================
// SLERP for axis-angle root orientation
// ============================================================
inline void SlerpAxisAngle(const double aa_a[3], const double aa_b[3],
                           double t, double out[3])
{
    // Convert to rotation matrices
    Eigen::Vector3d va(aa_a[0], aa_a[1], aa_a[2]);
    Eigen::Vector3d vb(aa_b[0], aa_b[1], aa_b[2]);

    double ang_a = va.norm();
    double ang_b = vb.norm();

    Eigen::Matrix3d R_a = (ang_a > 1e-8)
        ? Eigen::AngleAxisd(ang_a, va / ang_a).toRotationMatrix()
        : Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_b = (ang_b > 1e-8)
        ? Eigen::AngleAxisd(ang_b, vb / ang_b).toRotationMatrix()
        : Eigen::Matrix3d::Identity();

    // R_interp = R_a * exp(t * log(R_a^T * R_b))
    Eigen::Matrix3d R_diff = R_a.transpose() * R_b;
    Eigen::AngleAxisd aa_diff(R_diff);

    double interp_angle = t * aa_diff.angle();
    Eigen::Matrix3d R_interp;
    if (std::abs(interp_angle) < 1e-10) {
        R_interp = R_a;
    } else {
        R_interp = R_a * Eigen::AngleAxisd(interp_angle, aa_diff.axis()).toRotationMatrix();
    }

    Eigen::AngleAxisd aa_result(R_interp);
    Eigen::Vector3d result = aa_result.axis() * aa_result.angle();
    out[0] = result(0);
    out[1] = result(1);
    out[2] = result(2);
}

// ============================================================
// Bidirectional gap-fill: interpolate rejected frames
// from nearest good neighbors
// ============================================================
inline int InterpolateBadFrames(std::vector<FrameResult>& results) {
    int filled = 0;

    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].rejected) continue;

        // Find nearest good frame before
        int prev_good = -1;
        for (int j = (int)i - 1; j >= 0; --j) {
            if (!results[j].rejected) { prev_good = j; break; }
        }

        // Find nearest good frame after
        int next_good = -1;
        for (int j = (int)i + 1; j < (int)results.size(); ++j) {
            if (!results[j].rejected) { next_good = j; break; }
        }

        if (prev_good >= 0 && next_good >= 0) {
            // Bidirectional interpolation
            double t = (double)(i - prev_good) / (double)(next_good - prev_good);

            // SLERP for root orientation (theta[0:3])
            SlerpAxisAngle(results[prev_good].theta,
                           results[next_good].theta,
                           t, results[i].theta);

            // LERP for body pose (theta[3:72])
            for (int k = 3; k < 72; ++k) {
                results[i].theta[k] = (1.0 - t) * results[prev_good].theta[k]
                                     + t * results[next_good].theta[k];
            }

            // LERP for translation
            for (int k = 0; k < 3; ++k) {
                results[i].trans[k] = (1.0 - t) * results[prev_good].trans[k]
                                     + t * results[next_good].trans[k];
            }

            // Copy beta from previous good frame
            std::memcpy(results[i].beta, results[prev_good].beta, sizeof(double) * 10);

            // Interpolate depth affine
            for (int k = 0; k < 2; ++k) {
                results[i].depth_affine[k] = (1.0 - t) * results[prev_good].depth_affine[k]
                                            + t * results[next_good].depth_affine[k];
            }

            results[i].interpolated = true;
            filled++;

        } else if (prev_good >= 0) {
            // Only have past — copy forward
            std::memcpy(results[i].theta, results[prev_good].theta, 72 * sizeof(double));
            std::memcpy(results[i].trans, results[prev_good].trans, 3 * sizeof(double));
            std::memcpy(results[i].beta,  results[prev_good].beta,  10 * sizeof(double));
            std::memcpy(results[i].depth_affine, results[prev_good].depth_affine, 2 * sizeof(double));
            results[i].interpolated = true;
            filled++;

        } else if (next_good >= 0) {
            // Only have future — copy backward
            std::memcpy(results[i].theta, results[next_good].theta, 72 * sizeof(double));
            std::memcpy(results[i].trans, results[next_good].trans, 3 * sizeof(double));
            std::memcpy(results[i].beta,  results[next_good].beta,  10 * sizeof(double));
            std::memcpy(results[i].depth_affine, results[next_good].depth_affine, 2 * sizeof(double));
            results[i].interpolated = true;
            filled++;
        }
        // else: no good neighbors at all, leave as-is
    }
    return filled;
}

// ============================================================
// 1D Gaussian smoothing kernel
// ============================================================
inline void GaussianSmooth1D(std::vector<double>& signal, double sigma,
                             const std::vector<bool>* mask = nullptr)
{
    if (signal.size() < 3 || sigma < 0.3) return;

    int k = (int)std::ceil(3.0 * sigma);
    std::vector<double> kernel(2 * k + 1);
    double sum = 0.0;
    for (int i = -k; i <= k; ++i) {
        kernel[i + k] = std::exp(-0.5 * i * i / (sigma * sigma));
        sum += kernel[i + k];
    }
    for (auto& v : kernel) v /= sum;

    std::vector<double> out(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        double val = 0.0;
        double wsum = 0.0;
        for (int j = -k; j <= k; ++j) {
            int idx = std::clamp((int)i + j, 0, (int)signal.size() - 1);
            // Skip masked-out frames in the kernel
            if (mask && !(*mask)[idx]) continue;
            val  += kernel[j + k] * signal[idx];
            wsum += kernel[j + k];
        }
        out[i] = (wsum > 1e-12) ? val / wsum : signal[i];
    }
    signal = out;
}

// ============================================================
// Smooth root orientation in rotation-matrix space
// (averaging axis-angle directly is wrong for large angles)
// ============================================================
inline void SmoothRootOrientation(std::vector<FrameResult>& results, double sigma) {
    if (results.size() < 3 || sigma < 0.3) return;

    int k = (int)std::ceil(3.0 * sigma);
    std::vector<double> kernel(2 * k + 1);
    double sum = 0.0;
    for (int i = -k; i <= k; ++i) {
        kernel[i + k] = std::exp(-0.5 * i * i / (sigma * sigma));
        sum += kernel[i + k];
    }
    for (auto& v : kernel) v /= sum;

    // Store smoothed roots
    std::vector<std::array<double,3>> smoothed(results.size());

    for (size_t i = 0; i < results.size(); ++i) {
        // Weighted average in tangent space around frame i's rotation
        Eigen::Vector3d aa_i(results[i].theta[0], results[i].theta[1], results[i].theta[2]);
        double ang_i = aa_i.norm();
        Eigen::Matrix3d R_i = (ang_i > 1e-8)
            ? Eigen::AngleAxisd(ang_i, aa_i / ang_i).toRotationMatrix()
            : Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R_i_inv = R_i.transpose();

        Eigen::Vector3d log_sum = Eigen::Vector3d::Zero();
        double wsum = 0.0;

        for (int j = -k; j <= k; ++j) {
            int idx = std::clamp((int)i + j, 0, (int)results.size() - 1);
            double w = kernel[j + k];

            Eigen::Vector3d aa_j(results[idx].theta[0], results[idx].theta[1], results[idx].theta[2]);
            double ang_j = aa_j.norm();
            Eigen::Matrix3d R_j = (ang_j > 1e-8)
                ? Eigen::AngleAxisd(ang_j, aa_j / ang_j).toRotationMatrix()
                : Eigen::Matrix3d::Identity();

            // log(R_i^T * R_j) — tangent vector at R_i
            Eigen::Matrix3d R_diff = R_i_inv * R_j;
            Eigen::AngleAxisd aa_diff(R_diff);
            Eigen::Vector3d log_diff = aa_diff.axis() * aa_diff.angle();

            log_sum += w * log_diff;
            wsum += w;
        }

        if (wsum > 1e-12) log_sum /= wsum;

        // Reconstruct: R_smooth = R_i * exp(log_mean)
        double log_ang = log_sum.norm();
        Eigen::Matrix3d R_smooth;
        if (log_ang < 1e-10) {
            R_smooth = R_i;
        } else {
            R_smooth = R_i * Eigen::AngleAxisd(log_ang, log_sum / log_ang).toRotationMatrix();
        }

        Eigen::AngleAxisd aa_smooth(R_smooth);
        Eigen::Vector3d v = aa_smooth.axis() * aa_smooth.angle();
        smoothed[i] = {v(0), v(1), v(2)};
    }

    // Write back
    for (size_t i = 0; i < results.size(); ++i) {
        results[i].theta[0] = smoothed[i][0];
        results[i].theta[1] = smoothed[i][1];
        results[i].theta[2] = smoothed[i][2];
    }
}

// ============================================================
// Per-DOF Gaussian smoothing with arm-aware sigma
// ============================================================
struct SmoothConfig {
    double sigma_body = 2.0;       // body joints
    double sigma_arms = 1.0;       // arm joints (faster motion)
    double sigma_root = 1.5;       // root orientation (special SLERP)
    double sigma_trans = 1.5;      // translation
    bool   smooth_root_special = true;  // use rotation-space smoothing for root
};

inline void SmoothAllParameters(std::vector<FrameResult>& results,
                                const SmoothConfig& scfg = SmoothConfig())
{
    if (results.size() < 3) return;

    const int N = (int)results.size();

    // Build mask: only smooth through non-rejected OR interpolated frames
    // (all frames at this point should be valid after interpolation)
    std::vector<bool> valid(N, true);

    auto is_arm_joint = [](int j) -> bool {
        return j == 13 || j == 14 || j == 16 || j == 17 ||
               j == 18 || j == 19 || j == 20 || j == 21 ||
               j == 22 || j == 23;
    };

    // 1. Smooth root orientation in SO(3) space
    if (scfg.smooth_root_special) {
        SmoothRootOrientation(results, scfg.sigma_root);
    }

    // 2. Smooth body pose per-DOF
    for (int dof = 3; dof < 72; ++dof) {
        int joint = dof / 3;
        double sigma = is_arm_joint(joint) ? scfg.sigma_arms : scfg.sigma_body;

        std::vector<double> traj(N);
        for (int i = 0; i < N; ++i) traj[i] = results[i].theta[dof];

        GaussianSmooth1D(traj, sigma, &valid);

        for (int i = 0; i < N; ++i) results[i].theta[dof] = traj[i];
    }

    // 3. Smooth translation
    for (int dim = 0; dim < 3; ++dim) {
        std::vector<double> traj(N);
        for (int i = 0; i < N; ++i) traj[i] = results[i].trans[dim];

        GaussianSmooth1D(traj, scfg.sigma_trans, &valid);

        for (int i = 0; i < N; ++i) results[i].trans[dim] = traj[i];
    }
}

// ============================================================
// Full post-processing pipeline
// ============================================================
struct PostProcessStats {
    int total_frames    = 0;
    int solver_failed   = 0;
    int quality_rejected = 0;
    int interpolated    = 0;
};

inline PostProcessStats RunPostProcessing(
    std::vector<FrameResult>& results,
    const QualityThresholds& qthres = QualityThresholds(),
    const SmoothConfig& scfg = SmoothConfig())
{
    PostProcessStats stats;
    stats.total_frames = (int)results.size();

    // Count solver failures
    for (auto& fr : results) {
        if (!fr.solver_ok) stats.solver_failed++;
    }

    // Step 1: Compute inter-frame deltas
    ComputeFrameDeltas(results);

    // Step 2: Mark bad frames
    stats.quality_rejected = MarkBadFrames(results, qthres);

    std::cout << "\n[PostProcess] === Quality Filter ===\n"
              << "  Total frames:     " << stats.total_frames << "\n"
              << "  Solver failures:  " << stats.solver_failed << "\n"
              << "  Quality rejected: " << stats.quality_rejected << "\n";

    // Step 3: Bidirectional interpolation of bad frames
    stats.interpolated = InterpolateBadFrames(results);

    std::cout << "  Interpolated:     " << stats.interpolated << "\n";

    // Step 4: Gaussian smoothing
    SmoothAllParameters(results, scfg);

    std::cout << "  Smoothing applied (body_sigma=" << scfg.sigma_body
              << ", arm_sigma=" << scfg.sigma_arms
              << ", root_sigma=" << scfg.sigma_root << ")\n"
              << "[PostProcess] Done.\n\n";

    return stats;
}

// ============================================================
// Re-export meshes from smoothed results
// ============================================================
inline void ReExportMeshes(const std::vector<FrameResult>& results,
                           const SmplModel& model,
                           const std::string& out_dir)
{
    for (const auto& fr : results) {
        if (fr.frame_idx < 0) continue;

        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V;
        SMPLForward<double>(fr.beta, fr.theta, fr.trans, model, V);

        std::string obj_path = app::MakeOutObj(out_dir, fr.frame_idx);
        app::SaveOBJ(obj_path, V, model.faces);
    }
    std::cout << "[PostProcess] Re-exported " << results.size()
              << " meshes to " << out_dir << "\n";
}

} // namespace postsmooth