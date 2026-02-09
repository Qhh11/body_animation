#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "optimizer.h" 

namespace app {

void SaveOBJ(const std::string& filename,
             const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vertices,
             const Eigen::MatrixXi& faces);

void SaveLandmarksToFile(const std::string& filename,
                         const std::vector<Landmark2D>& landmarks);

std::string StemNoExt(const std::string& path);

} // namespace app
