#include "smpl_model.h"
#include <iostream>

// Helper function: Convert cnpy data to Eigen Matrix
template <typename T>
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> 
NpArrayToEigen(cnpy::NpyArray& arr, int rows, int cols) {
    // Get data pointer
    T* data_ptr = arr.data<T>();
    
    // Map to Eigen Matrix
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data_ptr, rows, cols).template cast<double>();
}

bool SmplModel::LoadFromNPZ(const std::string& path) {
    try {
        std::cout << "[SMPL] Loading model from: " << path << std::endl;
        cnpy::npz_t npz = cnpy::npz_load(path);

        // 1. 读取 v_template (N x 3)
        cnpy::NpyArray arr_v = npz["v_template"];
        this->N = arr_v.shape[0];
        this->v_template = NpArrayToEigen<double>(arr_v, N, 3);
        
        // 2. 读取 J_regressor (J x N)
        // 注意：npz 里可能是 sparse matrix 的 dense 格式，或者是普通 array
        cnpy::NpyArray arr_J = npz["J_regressor"];
        this->J = arr_J.shape[0];
        this->J_regressor = NpArrayToEigen<double>(arr_J, J, N);

        // 3. 读取 weights (N x J)
        cnpy::NpyArray arr_w = npz["weights"];
        this->weights = NpArrayToEigen<double>(arr_w, N, J);

        // 4. 读取 kintree_table (2 x J)
        cnpy::NpyArray arr_k = npz["kintree_table"];
        long* k_data = arr_k.data<long>(); 
        this->kintree_table.resize(2, J);
        for(int i=0; i<2*J; ++i) this->kintree_table.data()[i] = (int)k_data[i];
        
        // 5. 读取 shapedirs
        // 原始形状: (N, 3, 10) -> 我们把它读成 (N*3, 10)
        cnpy::NpyArray arr_s = npz["shapedirs"];
        int n_betas = arr_s.shape[2]; 
        this->shapedirs = NpArrayToEigen<double>(arr_s, N * 3, n_betas);

        // 6. 读取 posedirs
        // 原始形状: (N, 3, 207) -> 我们把它读成 (N*3, 207)
        cnpy::NpyArray arr_p = npz["posedirs"];
        int n_pose_basis = arr_p.shape[2]; 
        this->posedirs = NpArrayToEigen<double>(arr_p, N * 3, n_pose_basis);

        std::cout << "[SMPL] Load Success!" << std::endl;
        std::cout << "  - Vertices: " << N << std::endl;
        std::cout << "  - Joints: " << J << std::endl;
        std::cout << "  - Shape Basis: " << n_betas << std::endl;
        
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[SMPL] Error loading NPZ: " << e.what() << std::endl;
        return false;
    }
}