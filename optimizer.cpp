#include "optimizer.h"
#include "logging.h"
#include <ceres/ceres.h>

namespace smpl {

// ============================================================================
// Week 1: Simple L2 prior cost function (DUMMY SOLVE - NO GROUND TRUTH)
// ============================================================================
// This is a placeholder optimization that only tests Ceres integration.
// It encourages parameters to stay close to zero (L2 regularization).
//
// TODO(Week2+): Replace this with real cost functions that use GT data:
//   1. MeshFittingCost: fit to target 3D mesh vertices
//   2. KeypointProjectionCost: fit to 2D keypoints from images
//   3. ScanFittingCost: fit to 3D scan point clouds
// ============================================================================
struct L2PriorCost {
    L2PriorCost(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T* const param, T* residual) const {
        residual[0] = T(weight_) * param[0];
        return true;
    }

    static ceres::CostFunction* Create(double weight) {
        return new ceres::AutoDiffCostFunction<L2PriorCost, 1, 1>(
            new L2PriorCost(weight));
    }

private:
    double weight_;
};

// ============================================================================
// TODO(Week2+): Implement real optimization function to replace this dummy solve
// ============================================================================
// Current function: SolveDummyProblem() - minimizes ||theta||^2 + ||beta||^2
// Future function: SolveSMPLFitting() - minimizes data term + regularization
//
// Implementation roadmap:
// 1. Add GT data parameter (e.g., const Eigen::MatrixXf& target_vertices)
// 2. Add basis coefficient parameters w_j (J * K dimensional, J=24, K=8)
// 3. Implement MeshFittingCost functor:
//    - Call SMPL forward pass: vertices = SMPL(theta, beta, w_j)
//    - Compute residual: predicted_vertices - target_vertices
// 4. Add parameter blocks for theta, beta, and w_j to Ceres problem
// 5. Add data fitting residuals (main objective)
// 6. Add regularization residuals (optional: L2 priors, pose priors)
// 7. Configure solver options (may need sparse solver for large problems)
//
// Example signature:
//   bool SolveSMPLFitting(
//       const Eigen::MatrixXf& target_vertices,     // GT: (N, 3)
//       const SmplParams& smpl_params,              // Fixed SMPL model data
//       const PoseCorrectiveData& corrective_data,  // Fixed basis data
//       Eigen::VectorXf* theta,                     // Optimized: (J*3,)
//       Eigen::VectorXf* beta,                      // Optimized: (n_beta,)
//       Eigen::VectorXf* basis_weights);            // Optimized: (J*K,)
// ============================================================================
bool SolveDummyProblem(Eigen::VectorXf* theta, Eigen::VectorXf* beta) {
    LOG_INFO << "Running dummy Ceres optimization...";
    LOG_WARNING << "This is a DUMMY solve with NO ground truth data!";
    LOG_WARNING << "See TODO comments in optimizer.cpp for Week 2+ implementation.";

    // Convert Eigen vectors to raw arrays for Ceres
    std::vector<double> theta_data(theta->size());
    std::vector<double> beta_data(beta->size());

    for (int i = 0; i < theta->size(); ++i) {
        theta_data[i] = static_cast<double>((*theta)(i));
    }
    for (int i = 0; i < beta->size(); ++i) {
        beta_data[i] = static_cast<double>((*beta)(i));
    }

    // Build Ceres problem
    ceres::Problem problem;

    // TODO(Week2+): Replace these L2 priors with data fitting residuals
    // Currently: min ||theta||^2 + ||beta||^2 (just pulls params to zero)
    // Future: min ||SMPL(theta, beta, w_j) - target_vertices||^2 + regularization

    // Add L2 prior residuals for theta
    for (size_t i = 0; i < theta_data.size(); ++i) {
        ceres::CostFunction* cost_function = L2PriorCost::Create(1.0);
        problem.AddResidualBlock(cost_function, nullptr, &theta_data[i]);
    }

    // Add L2 prior residuals for beta
    for (size_t i = 0; i < beta_data.size(); ++i) {
        ceres::CostFunction* cost_function = L2PriorCost::Create(1.0);
        problem.AddResidualBlock(cost_function, nullptr, &beta_data[i]);
    }

    // TODO(Week2+): Add basis coefficient parameters w_j here
    // std::vector<double> basis_weights_data(J * K);  // J=24 joints, K=8 basis dims
    // for (int j = 0; j < J; ++j) {
    //     for (int v = 0; v < num_vertices_in_region[j]; ++v) {
    //         problem.AddResidualBlock(
    //             MeshFittingCost::Create(target_vertex, smpl_params, corrective_data, j, v),
    //             nullptr,  // or use Huber loss for robustness
    //             &theta_data[0], &beta_data[0], &basis_weights_data[j * K]
    //         );
    //     }
    // }

    // Configure solver
    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;  // TODO(Week2+): Use SPARSE_NORMAL_CHOLESKY for large problems
    options.minimizer_progress_to_stdout = false;

    // TODO(Week2+): Consider adding more solver options:
    // options.use_nonmonotonic_steps = true;
    // options.max_num_iterations = 100;
    // options.function_tolerance = 1e-6;
    // options.gradient_tolerance = 1e-10;
    // options.parameter_tolerance = 1e-8;

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    LOG_INFO << summary.BriefReport();

    // Copy optimized values back to Eigen vectors
    for (int i = 0; i < theta->size(); ++i) {
        (*theta)(i) = static_cast<float>(theta_data[i]);
    }
    for (int i = 0; i < beta->size(); ++i) {
        (*beta)(i) = static_cast<float>(beta_data[i]);
    }

    return summary.IsSolutionUsable();
}

// ============================================================================
// TODO(Week2+): Implement MeshFittingCost functor template
// ============================================================================
// This cost function will compute the residual between predicted SMPL vertices
// and ground truth target vertices.
//
// Uncomment and implement when ready:
// ============================================================================
/*
struct MeshFittingCost {
    // Constructor: store GT target vertex and fixed SMPL data
    MeshFittingCost(
        const Eigen::Vector3f& target_vertex,
        const SmplParams& smpl_params,
        const PoseCorrectiveData& corrective_data,
        int vertex_idx)
        : target_(target_vertex),
          smpl_params_(smpl_params),
          corrective_data_(corrective_data),
          vertex_idx_(vertex_idx) {}

    // Residual computation: called by Ceres during optimization
    // Parameters:
    //   theta: (J*3) pose parameters (axis-angle for each joint)
    //   beta: (n_beta) shape parameters
    //   basis_weights: (J*K) coefficients for pose-corrective basis
    // Outputs:
    //   residual: (3) difference between predicted and target vertex position
    template <typename T>
    bool operator()(
        const T* const theta,
        const T* const beta,
        const T* const basis_weights,
        T* residual) const {

        // Step 1: Shape blend shapes
        // v_shaped = v_template + shapedirs @ beta

        // Step 2: Pose-corrective blend shapes (STAR-style)
        // For each joint j affecting this vertex:
        //   w_j = basis_weights[j*K : (j+1)*K]
        //   offset_j = basis_j @ w_j
        //   v_posed += offset_j

        // Step 3: Forward Kinematics
        // Compute joint transformations from theta

        // Step 4: Linear Blend Skinning
        // v_final = LBS(v_posed, joint_transforms, weights)

        // Step 5: Compute residual
        // residual[0] = v_final.x() - T(target_.x());
        // residual[1] = v_final.y() - T(target_.y());
        // residual[2] = v_final.z() - T(target_.z());

        return true;
    }

    static ceres::CostFunction* Create(
        const Eigen::Vector3f& target_vertex,
        const SmplParams& smpl_params,
        const PoseCorrectiveData& corrective_data,
        int vertex_idx) {

        // AutoDiffCostFunction template parameters:
        // <Functor, ResidualDim, Param1Dim, Param2Dim, Param3Dim>
        return new ceres::AutoDiffCostFunction<
            MeshFittingCost,
            3,              // Residual dimension: (x, y, z)
            72,             // theta dimension: J*3 = 24*3
            300,            // beta dimension: n_beta
            192             // basis_weights dimension: J*K = 24*8
        >(new MeshFittingCost(target_vertex, smpl_params, corrective_data, vertex_idx));
    }

private:
    Eigen::Vector3f target_;
    const SmplParams& smpl_params_;
    const PoseCorrectiveData& corrective_data_;
    int vertex_idx_;
};
*/

// ============================================================================
// TODO(Week2+): Implement SolveSMPLFitting function
// ============================================================================
// This will replace SolveDummyProblem() with real data fitting.
//
// Uncomment and implement when ready:
// ============================================================================
/*
bool SolveSMPLFitting(
    const Eigen::MatrixXf& target_vertices,     // GT: (N, 3)
    const SmplParams& smpl_params,              // Fixed SMPL model data
    const PoseCorrectiveData& corrective_data,  // Fixed basis data
    Eigen::VectorXf* theta,                     // Optimized: (J*3,)
    Eigen::VectorXf* beta,                      // Optimized: (n_beta,)
    Eigen::VectorXf* basis_weights) {           // Optimized: (J*K,)

    LOG_INFO << "Running SMPL fitting with GT mesh data...";

    // Convert to double for Ceres
    std::vector<double> theta_data(theta->size());
    std::vector<double> beta_data(beta->size());
    std::vector<double> basis_weights_data(basis_weights->size());

    // ... (initialization code)

    ceres::Problem problem;

    // Add data fitting residuals for each vertex
    for (int v = 0; v < target_vertices.rows(); ++v) {
        Eigen::Vector3f target_v = target_vertices.row(v);
        ceres::CostFunction* cost_function =
            MeshFittingCost::Create(target_v, smpl_params, corrective_data, v);

        problem.AddResidualBlock(
            cost_function,
            new ceres::HuberLoss(1.0),  // Robust loss for outliers
            theta_data.data(),
            beta_data.data(),
            basis_weights_data.data()
        );
    }

    // Optional: Add L2 regularization on basis weights
    for (size_t i = 0; i < basis_weights_data.size(); ++i) {
        ceres::CostFunction* reg_cost = L2PriorCost::Create(0.01);  // Small weight
        problem.AddResidualBlock(reg_cost, nullptr, &basis_weights_data[i]);
    }

    // Solver configuration
    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    LOG_INFO << summary.FullReport();

    // Copy back results
    // ... (copy optimized values back to theta, beta, basis_weights)

    return summary.IsSolutionUsable();
}
*/

}  // namespace smpl
