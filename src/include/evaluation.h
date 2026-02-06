#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

// ============================================================
// Evaluation Metrics for SMPL Body Pose
// ============================================================

// MPJPE: Mean Per Joint Position Error
// Direct comparison of 3D joint positions (sensitive to global translation)
inline double ComputeMPJPE(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_pred,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt
) {
    assert(J_pred.rows() == J_gt.rows());
    return (J_pred - J_gt).rowwise().norm().mean();
}

// Procrustes alignment: find optimal rotation, translation, and scale
// to align J_pred to J_gt
// Returns aligned J_pred
inline Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> ProcrustesAlign(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_pred,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt
) {
    const int n = J_pred.rows();

    // 1. Center both point sets
    Eigen::Vector3d mu_pred = J_pred.colwise().mean();
    Eigen::Vector3d mu_gt = J_gt.colwise().mean();

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> X = J_pred.rowwise() - mu_pred.transpose();
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Y = J_gt.rowwise() - mu_gt.transpose();

    // 2. Compute scale
    double sx = X.norm();
    double sy = Y.norm();

    // Normalize
    X /= sx;
    Y /= sy;

    // 3. Compute optimal rotation using SVD
    // H = X^T * Y
    Eigen::Matrix3d H = X.transpose() * Y;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // R = V * U^T
    Eigen::Matrix3d R = V * U.transpose();

    // Handle reflection case
    if (R.determinant() < 0) {
        V.col(2) *= -1;
        R = V * U.transpose();
    }

    // 4. Compute scale factor
    double scale = sy / sx;

    // 5. Apply transformation: aligned = scale * R * (J_pred - mu_pred) + mu_gt
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_aligned(n, 3);
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d p = J_pred.row(i).transpose() - mu_pred;
        Eigen::Vector3d p_rot = R * p;
        J_aligned.row(i) = (scale * p_rot + mu_gt).transpose();
    }

    return J_aligned;
}

// PA-MPJPE: Procrustes-Aligned Mean Per Joint Position Error
// First align with Procrustes (removes global rotation, translation, scale), then compute MPJPE
inline double ComputePAMPJPE(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_pred,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt
) {
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_aligned = ProcrustesAlign(J_pred, J_gt);
    return ComputeMPJPE(J_aligned, J_gt);
}

// Per-joint errors (for detailed analysis)
inline std::vector<double> ComputePerJointError(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_pred,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt
) {
    std::vector<double> errors(J_pred.rows());
    for (int i = 0; i < J_pred.rows(); ++i) {
        errors[i] = (J_pred.row(i) - J_gt.row(i)).norm();
    }
    return errors;
}

// ============================================================
// Additional Metrics: PVE, Pose Error, Shape Error, 2D Reprojection
// ============================================================

// PVE: Per Vertex Error (mean L2 distance between predicted and GT vertices)
inline double ComputePVE(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V_pred,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V_gt
) {
    assert(V_pred.rows() == V_gt.rows());
    return (V_pred - V_gt).rowwise().norm().mean();
}

// PA-PVE: Procrustes-Aligned Per Vertex Error
inline double ComputePAPVE(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V_pred,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V_gt
) {
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_aligned = ProcrustesAlign(V_pred, V_gt);
    return ComputePVE(V_aligned, V_gt);
}

// Pose Error: L2 distance between pose parameters (theta)
// Computes error for body joints only (excluding root rotation for fairness)
inline double ComputePoseError(
    const double* theta_pred,
    const double* theta_gt,
    bool include_root = false
) {
    double error = 0.0;
    int start = include_root ? 0 : 3;  // skip root if not included
    int count = 0;
    for (int i = start; i < 72; ++i) {
        double diff = theta_pred[i] - theta_gt[i];
        error += diff * diff;
        count++;
    }
    return std::sqrt(error / count);  // RMSE per parameter
}

// Pose Error per joint (returns 24 joint rotation errors)
inline std::vector<double> ComputePoseErrorPerJoint(
    const double* theta_pred,
    const double* theta_gt
) {
    std::vector<double> errors(24);
    for (int j = 0; j < 24; ++j) {
        double err = 0.0;
        for (int k = 0; k < 3; ++k) {
            double diff = theta_pred[j*3+k] - theta_gt[j*3+k];
            err += diff * diff;
        }
        errors[j] = std::sqrt(err);  // L2 norm of axis-angle
    }
    return errors;
}

// Shape Error: L2 distance between beta parameters
inline double ComputeShapeError(
    const double* beta_pred,
    const double* beta_gt
) {
    double error = 0.0;
    for (int i = 0; i < 10; ++i) {
        double diff = beta_pred[i] - beta_gt[i];
        error += diff * diff;
    }
    return std::sqrt(error);
}

// 2D Reprojection Error: pixel distance between projected joints and GT 2D keypoints
inline double Compute2DReprojectionError(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_3d,
    const std::vector<std::pair<int, Eigen::Vector2d>>& gt_2d,  // (joint_idx, (u,v))
    double fx, double fy, double cx, double cy
) {
    if (gt_2d.empty()) return -1.0;

    double total_error = 0.0;
    int count = 0;
    for (const auto& [joint_idx, uv_gt] : gt_2d) {
        if (joint_idx < 0 || joint_idx >= J_3d.rows()) continue;

        Eigen::Vector3d p = J_3d.row(joint_idx).transpose();
        if (p(2) <= 0) continue;  // behind camera

        // Project to 2D
        double u_pred = fx * p(0) / p(2) + cx;
        double v_pred = fy * (-p(1)) / p(2) + cy;  // flip Y for image coord

        double err = std::sqrt((u_pred - uv_gt(0)) * (u_pred - uv_gt(0)) +
                               (v_pred - uv_gt(1)) * (v_pred - uv_gt(1)));
        total_error += err;
        count++;
    }

    return count > 0 ? total_error / count : -1.0;
}

// Evaluation result struct
struct EvaluationResult {
    double mpjpe = -1;       // Mean Per Joint Position Error (meters)
    double pa_mpjpe = -1;    // Procrustes-Aligned MPJPE (meters)
    double pve = -1;         // Per Vertex Error (meters)
    double pose_error = -1;  // Pose parameter error (radians RMSE)
    double shape_error = -1; // Shape parameter error
    double reproj_2d = -1;   // 2D reprojection error (pixels)
    std::vector<double> per_joint_errors;  // Per-joint errors

    void Print() const {
        std::cout << "  MPJPE:       " << (mpjpe >= 0 ? std::to_string(mpjpe * 1000.0) + " mm" : "N/A") << "\n";
        std::cout << "  PA-MPJPE:    " << (pa_mpjpe >= 0 ? std::to_string(pa_mpjpe * 1000.0) + " mm" : "N/A") << "\n";
        std::cout << "  PVE:         " << (pve >= 0 ? std::to_string(pve * 1000.0) + " mm" : "N/A") << "\n";
        std::cout << "  Pose Error:  " << (pose_error >= 0 ? std::to_string(pose_error) + " rad" : "N/A") << "\n";
        std::cout << "  Shape Error: " << (shape_error >= 0 ? std::to_string(shape_error) : "N/A") << "\n";
        std::cout << "  2D Reproj:   " << (reproj_2d >= 0 ? std::to_string(reproj_2d) + " px" : "N/A") << "\n";
    }
};

// Compute all metrics at once
inline EvaluationResult EvaluatePose(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_pred,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt
) {
    EvaluationResult result;
    result.mpjpe = ComputeMPJPE(J_pred, J_gt);
    result.pa_mpjpe = ComputePAMPJPE(J_pred, J_gt);
    result.per_joint_errors = ComputePerJointError(J_pred, J_gt);
    return result;
}
