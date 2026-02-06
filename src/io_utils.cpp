#include "io_utils.h"

#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace app {

void SaveOBJ(const std::string& filename,
             const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vertices,
             const Eigen::MatrixXi& faces) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) return;

    for (int i = 0; i < vertices.rows(); ++i) {
        outfile << "v " << vertices(i, 0) << " " << vertices(i, 1) << " " << vertices(i, 2) << "\n";
    }
    for (int i = 0; i < faces.rows(); ++i) {
        outfile << "f " << faces(i, 0) + 1 << " " << faces(i, 1) + 1 << " " << faces(i, 2) + 1 << "\n";
    }
}

void SaveLandmarksToFile(const std::string& filename,
                         const std::vector<Landmark2D>& landmarks) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    for (const auto& lm : landmarks) {
        outfile << lm.joint_id << " " << lm.x << " " << lm.y << " " << lm.confidence << "\n";
    }
    outfile.close();
    std::cout << "Saved landmarks to " << filename << "\n";
}

std::string StemNoExt(const std::string& path) {
    try {
        return fs::path(path).stem().string();
    } catch (...) {
        // fallback: manual
        std::string s = path;
        auto slash = s.find_last_of("/\\");
        if (slash != std::string::npos) s = s.substr(slash + 1);
        auto dot = s.find_last_of('.');
        if (dot != std::string::npos) s = s.substr(0, dot);
        return s;
    }
}

} // namespace app
