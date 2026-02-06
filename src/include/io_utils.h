#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "optimizer.h" // Landmark2D

namespace app {

// Save OBJ
void SaveOBJ(const std::string& filename,
             const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vertices,
             const Eigen::MatrixXi& faces);

// Save 2D landmarks to text
void SaveLandmarksToFile(const std::string& filename,
                         const std::vector<Landmark2D>& landmarks);

// get stem without extension (best-effort)
std::string StemNoExt(const std::string& path);

} // namespace app
