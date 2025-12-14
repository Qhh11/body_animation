#include "smpl_model.h"
#include <iostream>

// Helper function: Convert cnpy data to Eigen Matrix
template <typename T>
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> 
NpArrayToEigen(cnpy::NpyArray& arr, int rows, int cols) {
    T* data_ptr = arr.data<T>();
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data_ptr, rows, cols).template cast<double>();
}

// [NEW] Helper function for Integer matrices (for Faces)
template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> 
NpArrayToEigenInt(cnpy::NpyArray& arr, int rows, int cols) {
    T* data_ptr = arr.data<T>();
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data_ptr, rows, cols).template cast<int>();
}

bool SmplModel::LoadFromNPZ(const std::string& path) {
    try {
        std::cout << "[SMPL] Loading model from: " << path << std::endl;
        cnpy::npz_t npz = cnpy::npz_load(path);

        // ... [Keep previous loading code for v_template, J_regressor, weights, etc.] ...
        // (For brevity, I assume you kept the previous code here)
        // Re-paste your previous loading logic here
        
        cnpy::NpyArray arr_v = npz["v_template"];
        this->N = arr_v.shape[0];
        this->v_template = NpArrayToEigen<double>(arr_v, N, 3);

        cnpy::NpyArray arr_J = npz["J_regressor"];
        this->J = arr_J.shape[0];
        this->J_regressor = NpArrayToEigen<double>(arr_J, J, N);

        cnpy::NpyArray arr_w = npz["weights"];
        this->weights = NpArrayToEigen<double>(arr_w, N, J);

        cnpy::NpyArray arr_k = npz["kintree_table"];
        long* k_data = arr_k.data<long>(); 
        this->kintree_table.resize(2, J);
        for(int i=0; i<2*J; ++i) this->kintree_table.data()[i] = (int)k_data[i];
        
        cnpy::NpyArray arr_s = npz["shapedirs"];
        this->shapedirs = NpArrayToEigen<double>(arr_s, N * 3, arr_s.shape[2]);

        cnpy::NpyArray arr_p = npz["posedirs"];
        this->posedirs = NpArrayToEigen<double>(arr_p, N * 3, arr_p.shape[2]);

        // Load Faces
        if (npz.count("f")) {
            cnpy::NpyArray arr_f = npz["f"];
            // Faces are usually stored as int32 or int64 (N_faces x 3)
            if (arr_f.word_size == 4) {
                 this->faces = NpArrayToEigenInt<int>(arr_f, arr_f.shape[0], 3);
            } else if (arr_f.word_size == 8) {
                 this->faces = NpArrayToEigenInt<long>(arr_f, arr_f.shape[0], 3); // cnpy usually reads long for int64
            }
            std::cout << "  - Faces: " << this->faces.rows() << std::endl;
        } else {
            std::cerr << "[Warning] No 'f' or 'faces' found in NPZ. Output will be point cloud only." << std::endl;
        }

        std::cout << "[SMPL] Load Success!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[SMPL] Error loading NPZ: " << e.what() << std::endl;
        return false;
    }
}