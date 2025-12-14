#include "smpl.h"
#include "optimizer.h"
#include "logging.h"
#include <gflags/gflags.h>

DEFINE_string(smpl_npz, "", "Path to SMPL model NPZ file");
DEFINE_string(corrective_npz, "", "Path to corrective data NPZ file");

int main(int argc, char** argv) {
    // Initialize gflags and glog
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    LOG_INFO << "=== SMPL + Ceres Pipeline (Week 1) ===";

    // Validate arguments
    if (FLAGS_smpl_npz.empty()) {
        LOG_ERROR << "Missing required argument: --smpl_npz";
        return 1;
    }

    // Load SMPL model
    smpl::SmplParams smpl_params;
    if (!smpl::LoadSmplNPZ(FLAGS_smpl_npz, &smpl_params)) {
        LOG_ERROR << "Failed to load SMPL model";
        return 1;
    }

    LOG_INFO << "";
    LOG_INFO << "SMPL Model Dimensions:";
    LOG_INFO << "  N (vertices):     " << smpl_params.N;
    LOG_INFO << "  J (joints):       " << smpl_params.J;
    LOG_INFO << "  n_beta (shape):   " << smpl_params.n_beta;
    LOG_INFO << "  D_pose (pose BS): " << smpl_params.D_pose;

    // Load corrective data if provided
    bool has_corrective = false;
    smpl::PoseCorrectiveData corrective_data;

    if (!FLAGS_corrective_npz.empty()) {
        if (!smpl::LoadCorrectiveNPZ(FLAGS_corrective_npz, &corrective_data)) {
            LOG_ERROR << "Failed to load corrective data";
            return 1;
        }
        has_corrective = true;

        LOG_INFO << "";
        LOG_INFO << "Corrective Data Dimensions:";
        LOG_INFO << "  J (joints):        " << corrective_data.J;
        LOG_INFO << "  K (basis comps):   " << corrective_data.K;
        LOG_INFO << "  sum_len (verts):   " << corrective_data.sum_len;
    }

    // Initialize pose and shape parameters
    // theta: 3*J rotation parameters (axis-angle)
    // beta: n_beta shape parameters
    Eigen::VectorXf theta = Eigen::VectorXf::Random(smpl_params.J * 3) * 0.1f;
    Eigen::VectorXf beta = Eigen::VectorXf::Random(smpl_params.n_beta) * 0.1f;

    LOG_INFO << "";
    LOG_INFO << "Initial Parameters:";
    LOG_INFO << "  theta size: " << theta.size();
    LOG_INFO << "  beta size:  " << beta.size();

    // Run dummy Ceres solve
    LOG_INFO << "";
    if (!smpl::SolveDummyProblem(&theta, &beta)) {
        LOG_ERROR << "Optimization failed";
        return 1;
    }

    LOG_INFO << "";
    LOG_INFO << "Optimized Parameters:";
    LOG_INFO << "  theta norm: " << theta.norm();
    LOG_INFO << "  beta norm:  " << beta.norm();

    LOG_INFO << "";
    LOG_INFO << "=== Week 1 Pipeline Complete ===";
    LOG_INFO << "All systems operational!";

    if (has_corrective) {
        LOG_INFO << "Corrective data loaded and ready for Week 2+";
    }

    return 0;
}
