#include "smpl.h"
#include "logging.h"
#include "cnpy.h"

namespace smpl {

bool LoadSmplNPZ(const std::string& npz_path, SmplParams* params) {
    LOG_INFO << "Loading SMPL model from: " << npz_path;

    try {
        cnpy::npz_t npz = cnpy::npz_load(npz_path);

        // Load v_template (N, 3)
        auto v_template_arr = npz["v_template"];
        if (v_template_arr.shape.size() != 2 || v_template_arr.shape[1] != 3) {
            LOG_ERROR << "v_template: expected shape (N, 3)";
            return false;
        }
        params->N = v_template_arr.shape[0];
        params->v_template.resize(params->N, 3);
        const float* v_data = v_template_arr.data<float>();
        for (int i = 0; i < params->N; ++i) {
            for (int j = 0; j < 3; ++j) {
                params->v_template(i, j) = v_data[i * 3 + j];
            }
        }

        // Load shapedirs (N, 3, n_beta) and reshape to (N*3, n_beta)
        auto shapedirs_arr = npz["shapedirs"];
        if (shapedirs_arr.shape.size() != 3) {
            LOG_ERROR << "shapedirs: expected 3D array (N, 3, n_beta)";
            return false;
        }
        int N = shapedirs_arr.shape[0];
        int n_beta = shapedirs_arr.shape[2];
        params->n_beta = n_beta;
        params->shapedirs.resize(N * 3, n_beta);
        const float* shape_data = shapedirs_arr.data<float>();
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < n_beta; ++k) {
                for (int j = 0; j < 3; ++j) {
                    params->shapedirs(i * 3 + j, k) = shape_data[i * 3 * n_beta + j * n_beta + k];
                }
            }
        }

        // Load posedirs (N*3, D_pose) or (N, 3, D_pose)
        auto posedirs_arr = npz["posedirs"];
        if (posedirs_arr.shape.size() == 3) {
            // Shape is (N, 3, D_pose), need to reshape
            int N = posedirs_arr.shape[0];
            int D_pose = posedirs_arr.shape[2];
            params->D_pose = D_pose;
            params->posedirs.resize(N * 3, D_pose);
            const float* pose_data = posedirs_arr.data<float>();
            for (int i = 0; i < N; ++i) {
                for (int k = 0; k < D_pose; ++k) {
                    for (int j = 0; j < 3; ++j) {
                        params->posedirs(i * 3 + j, k) = pose_data[i * 3 * D_pose + j * D_pose + k];
                    }
                }
            }
        } else if (posedirs_arr.shape.size() == 2) {
            // Already in (N*3, D_pose) format
            params->D_pose = posedirs_arr.shape[1];
            params->posedirs.resize(posedirs_arr.shape[0], params->D_pose);
            const float* pose_data = posedirs_arr.data<float>();
            for (size_t i = 0; i < posedirs_arr.shape[0]; ++i) {
                for (int j = 0; j < params->D_pose; ++j) {
                    params->posedirs(i, j) = pose_data[i * params->D_pose + j];
                }
            }
        } else {
            LOG_ERROR << "posedirs: unexpected shape";
            return false;
        }

        // Load J_regressor (J, N)
        auto J_reg_arr = npz["J_regressor"];
        params->J = J_reg_arr.shape[0];
        params->J_regressor.resize(params->J, params->N);
        const float* J_data = J_reg_arr.data<float>();
        for (int i = 0; i < params->J; ++i) {
            for (int j = 0; j < params->N; ++j) {
                params->J_regressor(i, j) = J_data[i * params->N + j];
            }
        }

        // Load weights (N, J)
        auto weights_arr = npz["weights"];
        params->weights.resize(params->N, params->J);
        const float* w_data = weights_arr.data<float>();
        for (int i = 0; i < params->N; ++i) {
            for (int j = 0; j < params->J; ++j) {
                params->weights(i, j) = w_data[i * params->J + j];
            }
        }

        // Load kintree_table (2, J)
        auto kintree_arr = npz["kintree_table"];
        params->kintree_table.resize(2, params->J);
        const int64_t* k_data = kintree_arr.data<int64_t>();
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < params->J; ++j) {
                params->kintree_table(i, j) = static_cast<int>(k_data[i * params->J + j]);
            }
        }

        LOG_INFO << "SMPL model loaded successfully:";
        LOG_INFO << "  N (vertices) = " << params->N;
        LOG_INFO << "  J (joints) = " << params->J;
        LOG_INFO << "  n_beta (shape params) = " << params->n_beta;
        LOG_INFO << "  D_pose (pose blendshape dim) = " << params->D_pose;

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR << "Failed to load SMPL NPZ: " << e.what();
        return false;
    }
}

}  // namespace smpl
