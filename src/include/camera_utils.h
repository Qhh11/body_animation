#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cmath>

#include <Eigen/Dense>

#include "optimizer.h"   // MatX3dRM

// ============================================================
// Camera extrinsics: R (3x3) + t (3x1)
// Convention: world-to-camera  =>  X_cam = R * X_world + t
// ============================================================
struct CameraExtrinsics {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

// ============================================================
// Load per-frame camera poses from a text file.
//
// Supported format: flat list of numbers, 16 per frame (4x4)
// or 12 per frame (3x4).  No blank-line separator needed.
// ============================================================
inline std::vector<CameraExtrinsics> LoadCameraPoses(const std::string& path) {
    std::vector<CameraExtrinsics> poses;
    std::ifstream f(path);
    if (!f.is_open()) {
        // Silently return empty — caller checks .empty()
        return poses;
    }

    std::vector<double> nums;
    double v;
    while (f >> v) nums.push_back(v);

    // 4x4 matrices (16 numbers each)
    if (nums.size() >= 16 && nums.size() % 16 == 0) {
        int n = (int)nums.size() / 16;
        for (int i = 0; i < n; ++i) {
            CameraExtrinsics ce;
            const double* d = &nums[i * 16];
            ce.R(0,0) = d[0];  ce.R(0,1) = d[1];  ce.R(0,2) = d[2];  ce.t(0) = d[3];
            ce.R(1,0) = d[4];  ce.R(1,1) = d[5];  ce.R(1,2) = d[6];  ce.t(1) = d[7];
            ce.R(2,0) = d[8];  ce.R(2,1) = d[9];  ce.R(2,2) = d[10]; ce.t(2) = d[11];
            poses.push_back(ce);
        }
    }
    // 3x4 matrices (12 numbers each)
    else if (nums.size() >= 12 && nums.size() % 12 == 0) {
        int n = (int)nums.size() / 12;
        for (int i = 0; i < n; ++i) {
            CameraExtrinsics ce;
            const double* d = &nums[i * 12];
            ce.R(0,0) = d[0];  ce.R(0,1) = d[1];  ce.R(0,2) = d[2];  ce.t(0) = d[3];
            ce.R(1,0) = d[4];  ce.R(1,1) = d[5];  ce.R(1,2) = d[6];  ce.t(1) = d[7];
            ce.R(2,0) = d[8];  ce.R(2,1) = d[9];  ce.R(2,2) = d[10]; ce.t(2) = d[11];
            poses.push_back(ce);
        }
    }

    if (!poses.empty())
        std::cout << "[CamPoses] Loaded " << poses.size()
                  << " camera poses from " << path << "\n";
    return poses;
}

// ============================================================
// Relative camera transform:  X_camB = R_rel * X_camA + t_rel
//
//   R_rel = R_b * R_a^T
//   t_rel = t_b - R_rel * t_a
// ============================================================
inline void ComputeRelativeCameraTransform(
    const CameraExtrinsics& cam_a,
    const CameraExtrinsics& cam_b,
    Eigen::Matrix3d& R_rel,
    Eigen::Vector3d& t_rel)
{
    R_rel = cam_b.R * cam_a.R.transpose();
    t_rel = cam_b.t - R_rel * cam_a.t;
}

// ============================================================
// Transform previous-frame SMPL params into the current camera
// frame using relative camera motion.
//
//   trans_curr = R_rel * trans_prev + t_rel
//   root_curr  = R_rel * root_prev   (axis-angle composed)
//   body pose is copied unchanged.
// ============================================================
inline void TransformPrevParamsToCurrCam(
    const double* theta_prev,   // [72]
    const double* trans_prev,   // [3]
    const Eigen::Matrix3d& R_rel,
    const Eigen::Vector3d& t_rel,
    double* theta_curr,         // [72] output
    double* trans_curr)         // [3]  output
{
    // Copy full pose first
    std::memcpy(theta_curr, theta_prev, 72 * sizeof(double));

    // --- Transform translation ---
    Eigen::Vector3d tp(trans_prev[0], trans_prev[1], trans_prev[2]);
    Eigen::Vector3d tc = R_rel * tp + t_rel;
    trans_curr[0] = tc(0);
    trans_curr[1] = tc(1);
    trans_curr[2] = tc(2);

    // --- Transform root orientation (axis-angle, first 3 params) ---
    Eigen::Vector3d aa_prev(theta_prev[0], theta_prev[1], theta_prev[2]);
    double angle = aa_prev.norm();
    Eigen::Matrix3d R_root;
    if (angle < 1e-12) {
        R_root = Eigen::Matrix3d::Identity();
    } else {
        R_root = Eigen::AngleAxisd(angle, aa_prev / angle).toRotationMatrix();
    }

    Eigen::Matrix3d R_root_new = R_rel * R_root;
    Eigen::AngleAxisd aa_new(R_root_new);
    Eigen::Vector3d aa_vec = aa_new.axis() * aa_new.angle();

    theta_curr[0] = aa_vec(0);
    theta_curr[1] = aa_vec(1);
    theta_curr[2] = aa_vec(2);
}

// ============================================================
// Transform joints from camera space to world space.
//   X_world = R^T * (X_cam - t)
//
// If flip_y: negate Y before transforming (undo SMPL Y-flip).
// ============================================================
inline void TransformJointsCamToWorld(
    const MatX3dRM& J_cam,
    const CameraExtrinsics& cam,
    MatX3dRM& J_world,
    bool flip_y)
{
    J_world.resize(J_cam.rows(), 3);
    Eigen::Matrix3d R_inv = cam.R.transpose();

    for (int i = 0; i < J_cam.rows(); ++i) {
        Eigen::Vector3d p = J_cam.row(i).transpose();
        if (flip_y) p(1) = -p(1);
        Eigen::Vector3d pw = R_inv * (p - cam.t);
        J_world.row(i) = pw.transpose();
    }
}