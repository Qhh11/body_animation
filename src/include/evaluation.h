#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>

// ============================================================
// Evaluation Metrics for SMPL Body Pose
// ============================================================

// MPJPE: Mean Per Joint Position Error
inline double ComputeMPJPE(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_pred,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt
) {
    if (J_pred.rows() != J_gt.rows()) return -1.0;
    return (J_pred - J_gt).rowwise().norm().mean();
}

// Procrustes alignment
inline Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> ProcrustesAlign(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_pred,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt
) {
    const int n = J_pred.rows();
    Eigen::Vector3d mu_pred = J_pred.colwise().mean();
    Eigen::Vector3d mu_gt = J_gt.colwise().mean();

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> X = J_pred.rowwise() - mu_pred.transpose();
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Y = J_gt.rowwise() - mu_gt.transpose();

    double sx = X.norm();
    double sy = Y.norm();
    X /= sx;
    Y /= sy;

    Eigen::Matrix3d H = X.transpose() * Y;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();

    if (R.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1;
        R = V * svd.matrixU().transpose();
    }

    double scale = sy / sx;

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_aligned(n, 3);
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d p = J_pred.row(i).transpose() - mu_pred;
        J_aligned.row(i) = (scale * R * p + mu_gt).transpose();
    }
    return J_aligned;
}

// PA-MPJPE
inline double ComputePAMPJPE(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_pred,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt
) {
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_aligned = ProcrustesAlign(J_pred, J_gt);
    return ComputeMPJPE(J_aligned, J_gt);
}

// PVE: Per Vertex Error
inline double ComputePVE(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V_pred,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V_gt
) {
    if (V_pred.rows() != V_gt.rows()) return -1.0;
    return (V_pred - V_gt).rowwise().norm().mean();
}

// Pose Parameter Error (excluding root by default)
inline double ComputePoseError(
    const double* theta_pred,
    const double* theta_gt,
    bool include_root = false
) {
    double error = 0.0;
    int start = include_root ? 0 : 3;
    int count = 0;
    for (int i = start; i < 72; ++i) {
        double diff = theta_pred[i] - theta_gt[i];
        error += diff * diff;
        count++;
    }
    return (count > 0) ? std::sqrt(error / count) : 0.0;
}

// Shape Parameter Error
inline double ComputeShapeError(
    const double* beta_pred,
    const double* beta_gt
) {
    double error = 0.0;
    for (int i = 0; i < 10; ++i) {
        double diff = beta_pred[i] - beta_gt[i];
        error += diff * diff;
    }
    return std::sqrt(error / 10.0);
}