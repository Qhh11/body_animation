#include "smpl.h"
#include "logging.h"
#include "cnpy.h"

namespace smpl {

bool LoadCorrectiveNPZ(const std::string& npz_path, PoseCorrectiveData* corrective) {
    LOG_INFO << "Loading corrective data from: " << npz_path;

    try {
        cnpy::npz_t npz = cnpy::npz_load(npz_path);

        // Load J (number of joints)
        auto J_arr = npz["J"];
        const int64_t* J_data = J_arr.data<int64_t>();
        corrective->J = static_cast<int>(J_data[0]);

        // Load K (number of basis components)
        auto K_arr = npz["K"];
        const int64_t* K_data = K_arr.data<int64_t>();
        corrective->K = static_cast<int>(K_data[0]);

        // Load region_ptr (J+1,)
        auto region_ptr_arr = npz["region_ptr"];
        if (region_ptr_arr.shape[0] != static_cast<size_t>(corrective->J + 1)) {
            LOG_ERROR << "region_ptr: expected size " << (corrective->J + 1)
                      << ", got " << region_ptr_arr.shape[0];
            return false;
        }
        corrective->region_ptr.resize(corrective->J + 1);
        const int64_t* ptr_data = region_ptr_arr.data<int64_t>();
        for (int i = 0; i < corrective->J + 1; ++i) {
            corrective->region_ptr[i] = static_cast<int>(ptr_data[i]);
        }

        // Calculate sum_len from region_ptr
        corrective->sum_len = corrective->region_ptr[corrective->J];

        // Load region_idx (sum_len,)
        auto region_idx_arr = npz["region_idx"];
        if (region_idx_arr.shape[0] != static_cast<size_t>(corrective->sum_len)) {
            LOG_ERROR << "region_idx: expected size " << corrective->sum_len
                      << ", got " << region_idx_arr.shape[0];
            return false;
        }
        corrective->region_idx.resize(corrective->sum_len);
        const int64_t* idx_data = region_idx_arr.data<int64_t>();
        for (int i = 0; i < corrective->sum_len; ++i) {
            corrective->region_idx[i] = static_cast<int>(idx_data[i]);
        }

        // Load basis (sum_len*3, K)
        auto basis_arr = npz["basis"];
        if (basis_arr.shape[0] != static_cast<size_t>(corrective->sum_len * 3) ||
            basis_arr.shape[1] != static_cast<size_t>(corrective->K)) {
            LOG_ERROR << "basis: expected shape (" << (corrective->sum_len * 3)
                      << ", " << corrective->K << "), got ("
                      << basis_arr.shape[0] << ", " << basis_arr.shape[1] << ")";
            return false;
        }
        corrective->basis.resize(corrective->sum_len * 3, corrective->K);
        const float* basis_data = basis_arr.data<float>();
        for (int i = 0; i < corrective->sum_len * 3; ++i) {
            for (int j = 0; j < corrective->K; ++j) {
                corrective->basis(i, j) = basis_data[i * corrective->K + j];
            }
        }

        LOG_INFO << "Corrective data loaded successfully:";
        LOG_INFO << "  J (joints) = " << corrective->J;
        LOG_INFO << "  K (basis components) = " << corrective->K;
        LOG_INFO << "  sum_len (total vertices) = " << corrective->sum_len;

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR << "Failed to load corrective NPZ: " << e.what();
        return false;
    }
}

}  // namespace smpl
