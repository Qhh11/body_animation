#include "smpl_model.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <limits>
#include <cnpy.h>

// -------- helpers --------
static void Require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

template <typename T>
static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
LoadMatRowMajor(const cnpy::NpyArray& arr, int rows, int cols) {
    Require(arr.word_size == sizeof(T), "word_size mismatch");
    Require((int)arr.shape.size() == 2, "expected 2D array");
    Require((int)arr.shape[0] == rows && (int)arr.shape[1] == cols, "shape mismatch");

    const T* p = arr.data<T>();

    if (arr.fortran_order) {
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> M(p, rows, cols);
        return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(M);
    } else {
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> M(p, rows, cols);
        return M;
    }
}


template <typename T>
static Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>
LoadMatNx3RowMajor(const cnpy::NpyArray& arr, int rows) {
    Require(arr.word_size == sizeof(T), "word_size mismatch");
    Require((int)arr.shape.size() == 2, "expected 2D array");
    Require((int)arr.shape[0] == rows && (int)arr.shape[1] == 3, "shape mismatch Nx3");

    const T* p = arr.data<T>();

    if (arr.fortran_order) {
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::ColMajor>> M(p, rows, 3);
        return Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>(M);
    } else {
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>> M(p, rows, 3);
        return M;
    }
}


static Eigen::MatrixXi LoadFacesInt(const cnpy::NpyArray& arr) {
    Require((int)arr.shape.size() == 2, "faces must be 2D");
    Require((int)arr.shape[1] == 3, "faces must be Fx3");
    const int F = (int)arr.shape[0];

    Eigen::MatrixXi faces(F, 3);

    if (arr.word_size == 8) {
        const long long* p = arr.data<long long>();

        auto idx = [&](int r, int c)->size_t {
            // arr shape: (F, 3)
            if (arr.fortran_order) return (size_t)c * (size_t)F + (size_t)r; // col-major
            else                   return (size_t)r * (size_t)3 + (size_t)c; // row-major
        };

        for (int i = 0; i < F; ++i) {
            faces(i,0) = (int)p[idx(i,0)];
            faces(i,1) = (int)p[idx(i,1)];
            faces(i,2) = (int)p[idx(i,2)];
        }

    } else if (arr.word_size == 4) {
        const int* p = arr.data<int>();

        auto idx = [&](int r, int c)->size_t {
            // arr shape: (F, 3)
            if (arr.fortran_order) return (size_t)c * (size_t)F + (size_t)r; // col-major
            else                   return (size_t)r * (size_t)3 + (size_t)c; // row-major
        };

        for (int i = 0; i < F; ++i) {
            faces(i,0) = p[idx(i,0)];
            faces(i,1) = p[idx(i,1)];
            faces(i,2) = p[idx(i,2)];
        }

    } else {
        throw std::runtime_error("Unsupported faces dtype");
    }

    return faces;
}


static Eigen::Matrix<int, 2, Eigen::Dynamic, Eigen::RowMajor>
LoadKintreeInt(const cnpy::NpyArray& arr) {
    Require((int)arr.shape.size() == 2, "kintree_table must be 2D");
    Require((int)arr.shape[0] == 2, "kintree_table must be 2xJ");
    const int J = (int)arr.shape[1];

    Eigen::Matrix<int, 2, Eigen::Dynamic, Eigen::RowMajor> kt(2, J);

    if (arr.word_size == 8) {
        const long long* p = arr.data<long long>();

        auto idx = [&](int r, int c)->size_t {
            // arr shape: (2, J)
            if (arr.fortran_order) return (size_t)c * (size_t)2 + (size_t)r; // col-major
            else                   return (size_t)r * (size_t)J + (size_t)c; // row-major
        };

        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < J; ++c) {
                kt(r,c) = (int)p[idx(r,c)];
            }
        }

    } else if (arr.word_size == 4) {
        const int* p = arr.data<int>();

        auto idx = [&](int r, int c)->size_t {
            // arr shape: (2, J)
            if (arr.fortran_order) return (size_t)c * (size_t)2 + (size_t)r; // col-major
            else                   return (size_t)r * (size_t)J + (size_t)c; // row-major
        };

        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < J; ++c) {
                kt(r,c) = p[idx(r,c)];
            }
        }

    } else {
        throw std::runtime_error("Unsupported kintree dtype");
    }

    return kt;
}


static int DetectParentRow(const Eigen::Matrix<int,2,Eigen::Dynamic,Eigen::RowMajor>& kt) {
    int J = (int)kt.cols();

    auto score_row = [&](int r)->int {
        int ok = 0;
        for (int i = 1; i < J; ++i) {
            int p = kt(r,i);
            if (p >= -1 && p < J) ok++;
        }
        return ok;
    };

    int s0 = score_row(0);
    int s1 = score_row(1);
    return (s1 > s0) ? 1 : 0;
}

void SmplModel::BuildRegressorCache() {
    J_regressor_nz.clear();
    J_regressor_nz.resize(J);

    for (int j = 0; j < J; ++j) {
        J_regressor_nz[j].reserve(256);
        for (int v = 0; v < N; ++v) {
            double w = J_regressor(j, v);
            if (std::abs(w) > 1e-12) {
                J_regressor_nz[j].emplace_back(v, w);
            }
        }
        if (J_regressor_nz[j].empty()) {
            std::cerr << "[Warn] J_regressor row " << j << " has no nonzeros.\n";
        }
    }
    std::cout << "[SMPL] Built J_regressor nonzero cache.\n";
}

void SmplModel::BuildJointShapeCache(int num_beta) {
    J_template_3d.resize(J, 3);
    J_template_3d.setZero();

    J_shapedirs_3d.resize(J * 3, num_beta);
    J_shapedirs_3d.setZero();

    for (int j = 0; j < J; ++j) {
        const auto& nz = J_regressor_nz[j];
        for (const auto& vw : nz) {
            const int v = vw.first;
            const double w = vw.second;

            J_template_3d(j,0) += w * v_template(v,0);
            J_template_3d(j,1) += w * v_template(v,1);
            J_template_3d(j,2) += w * v_template(v,2);

            for (int k = 0; k < num_beta; ++k) {
                J_shapedirs_3d(j*3 + 0, k) += w * shapedirs(v*3 + 0, k);
                J_shapedirs_3d(j*3 + 1, k) += w * shapedirs(v*3 + 1, k);
                J_shapedirs_3d(j*3 + 2, k) += w * shapedirs(v*3 + 2, k);
            }
        }
    }

    std::cout << "[SMPL] Built joints-only cache: J_template_3d "
              << J_template_3d.rows() << "x" << J_template_3d.cols()
              << ", J_shapedirs_3d " << J_shapedirs_3d.rows() << "x" << J_shapedirs_3d.cols()
              << "\n";
}

bool SmplModel::LoadFromNPZ(const std::string& npz_path) {
    try {
        std::cout << "[SMPL] Loading model from: " << npz_path << "\n";

        auto npz = cnpy::npz_load(npz_path);

        // v_template: N x 3
        {
            auto& a = npz.at("v_template");
            Require(a.shape.size() == 2 && a.shape[1] == 3, "v_template shape must be Nx3");
            N = (int)a.shape[0];
            v_template = LoadMatNx3RowMajor<double>(a, N);
        }

        // faces: F x 3
        {
            auto& a = npz.at("f");
            faces = LoadFacesInt(a);
        }

        // J_regressor: J x N  (
        {
            auto& a = npz.at("J_regressor");
            Require(a.shape.size() == 2, "J_regressor must be 2D");
            J = (int)a.shape[0];
            Require((int)a.shape[1] == N, "J_regressor shape mismatch");
            J_regressor = LoadMatRowMajor<double>(a, J, N);
        }

        // weights: N x J
        {
            auto& a = npz.at("weights");
            Require(a.shape.size() == 2, "weights must be 2D");
            Require((int)a.shape[0] == N, "weights rows mismatch");
            Require((int)a.shape[1] == J, "weights cols mismatch");
            weights = LoadMatRowMajor<double>(a, N, J);
        }

        // shapedirs: (N*3) x num_betas  (yours shows 20670x300)
        {
            auto& a = npz.at("shapedirs");
            Require(a.shape.size() == 2, "shapedirs must be 2D");
            Require((int)a.shape[0] == N*3, "shapedirs rows must be N*3");
            int B = (int)a.shape[1];
            shapedirs = LoadMatRowMajor<double>(a, N*3, B);
        }

        // posedirs: (N*3) x ((J-1)*9)  
        {
            auto& a = npz.at("posedirs");
            Require(a.shape.size() == 2, "posedirs must be 2D");
            Require((int)a.shape[0] == N*3, "posedirs rows must be N*3");
            int P = (int)a.shape[1];
            posedirs = LoadMatRowMajor<double>(a, N*3, P);
        }

        // kintree_table: 2 x J (int)
        {
            auto& a = npz.at("kintree_table");
            kintree_table = LoadKintreeInt(a);
            int prow = DetectParentRow(kintree_table);
            if (prow == 1) {
                auto tmp = kintree_table;
                kintree_table.row(0) = tmp.row(1);
                kintree_table.row(1) = tmp.row(0);
            }

            // validation
            for (int i = 1; i < J; ++i) {
                int p = kintree_table(0,i);
                if (!(p >= -1 && p < J)) {
                    std::cerr << "[FATAL] kintree_table still invalid after int-load. joint "
                              << i << " parent=" << p << "\n";
                    return false;
                }
            }
        }

        std::cout << "  - Faces: " << faces.rows() << "\n";
        std::cout << "[SMPL] Load Success!\n";
        std::cout << "[SMPL] N=" << N << "  J=" << J << "\n";
        std::cout << "  v_template: " << v_template.rows() << "x" << v_template.cols() << "\n";
        std::cout << "  J_regressor: " << J_regressor.rows() << "x" << J_regressor.cols() << "\n";
        std::cout << "  weights: " << weights.rows() << "x" << weights.cols() << "\n";
        std::cout << "  kintree_table: " << kintree_table.rows() << "x" << kintree_table.cols() << "\n";
        std::cout << "  shapedirs: " << shapedirs.rows() << "x" << shapedirs.cols() << "\n";
        std::cout << "  posedirs: " << posedirs.rows() << "x" << posedirs.cols() << "\n";
        BuildRegressorCache();
        BuildJointShapeCache(10);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[SMPL] LoadFromNPZ failed: " << e.what() << "\n";
        return false;
    }
}
