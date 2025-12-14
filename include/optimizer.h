#pragma once

#include <Eigen/Dense>

namespace smpl {

// Run a dummy Ceres optimization problem with L2 priors only
// This is a Week 1 placeholder - only tests that Ceres is working
//
// Parameters:
//   theta - pose parameters (will be optimized)
//   beta  - shape parameters (will be optimized)
//
// Returns true on successful solve, false on error
bool SolveDummyProblem(Eigen::VectorXf* theta, Eigen::VectorXf* beta);

}  // namespace smpl
