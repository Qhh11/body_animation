#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace smpl {

// SMPL model parameters loaded from NPZ
struct SmplParams {
    Eigen::MatrixXf v_template;      // (N, 3) - template vertices
    Eigen::MatrixXf shapedirs;       // (N*3, n_beta) - shape blend shapes
    Eigen::MatrixXf posedirs;        // (N*3, D_pose) - pose blend shapes
    Eigen::MatrixXf J_regressor;     // (J, N) - joint regressor
    Eigen::MatrixXf weights;         // (N, J) - skinning weights
    Eigen::MatrixXi kintree_table;   // (2, J) - kinematic tree

    int N;        // number of vertices
    int J;        // number of joints
    int n_beta;   // number of shape parameters
    int D_pose;   // pose blend shape dimension
};

// Pose-corrective data in CSR-packed format
struct PoseCorrectiveData {
    int J;                              // number of joints
    int K;                              // number of basis components per region
    std::vector<int> region_ptr;        // (J+1) - CSR pointers
    std::vector<int> region_idx;        // (sum_len) - vertex indices
    Eigen::MatrixXf basis;              // (sum_len*3, K) - concatenated basis

    int sum_len;  // total number of vertices across all regions
};

// SMPL model containing parameters and optional corrective data
struct SmplModel {
    SmplParams params;
    PoseCorrectiveData corrective;
    bool has_corrective = false;
};

// Load SMPL model from NPZ file
// Returns true on success, false on error
bool LoadSmplNPZ(const std::string& npz_path, SmplParams* params);

// Load pose-corrective data from NPZ file
// Returns true on success, false on error
bool LoadCorrectiveNPZ(const std::string& npz_path, PoseCorrectiveData* corrective);

}  // namespace smpl
