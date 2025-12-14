#include <iostream>
#include <fstream> // Required for file writing
#include "smpl_model.h"
#include "smpl_math.h" 
#include "render_obj.h"
#include <ceres/ceres.h>


void SaveOBJ(const std::string& filename, 
             const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& vertices, 
             const Eigen::MatrixXi& faces) {
    
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    std::cout << "Saving OBJ to " << filename << "..." << std::endl;

    // 1. Write Vertices
    // OBJ format: "v x y z"
    for (int i = 0; i < vertices.rows(); ++i) {
        outfile << "v " << vertices(i, 0) << " " << vertices(i, 1) << " " << vertices(i, 2) << "\n";
    }

    // 2. Write Faces
    // OBJ format: "f v1 v2 v3"
    // Note: OBJ indices start at 1, but C++ arrays start at 0. So we add +1.
    for (int i = 0; i < faces.rows(); ++i) {
        outfile << "f " 
                << faces(i, 0) + 1 << " " 
                << faces(i, 1) + 1 << " " 
                << faces(i, 2) + 1 << "\n";
    }

    outfile.close();
    std::cout << "Saved successfully!" << std::endl;
}

int main() {
    google::InitGoogleLogging("SMPL_Optimizer");
    std::cout << ">>> Start SMPL Project..." << std::endl;

    // 1. Load Model
    SmplModel model;
    std::string npz_path = "../../Data/star_1_1/neutral/model.npz"; 
    
    if (!model.LoadFromNPZ(npz_path)) {
        return -1;
    }

    // 2. Prepare Inputs (T-Pose)
    double beta[10] = {0}; 
    double theta[72] = {0};
    double trans[3] = {0, 0, 0};

    // 3. Run Forward Pass
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out_verts;
    SMPLForward<double>(beta, theta, trans, model, out_verts);

    // 4. Save Result for Visualization
    // Save to the build folder (or any accessible path)
    SaveOBJ("output_tpose.obj", out_verts, model.faces);

    return 0;
}