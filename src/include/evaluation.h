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

// Evaluation result struct
struct EvaluationResult {
    double mpjpe;       // Mean Per Joint Position Error (meters)
    double pa_mpjpe;    // Procrustes-Aligned MPJPE (meters)
    std::vector<double> per_joint_errors;  // Per-joint errors

    void Print() const {
        std::cout << "  MPJPE:    " << mpjpe * 1000.0 << " mm\n";
        std::cout << "  PA-MPJPE: " << pa_mpjpe * 1000.0 << " mm\n";
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
