#include <iostream>
#include "smpl_model.h"
#include <ceres/ceres.h>

int main() {
    // 1. Initialize Google Logging (Required by Ceres)
    google::InitGoogleLogging("SMPL_Optimizer");

    std::cout << ">>> Start SMPL Project..." << std::endl;

    // 2. Load Model
    SmplModel model;
    
    // Updated path based on your environment
    std::string npz_path = "../../Data/star_1_1/neutral/model.npz"; 
    
    if (!model.LoadFromNPZ(npz_path)) {
        std::cerr << "Failed to load model! Please check the path: " << npz_path << std::endl;
        return -1;
    }

    // 3. Simple test for Eigen (Verify if Eigen is working)
    std::cout << "Test Eigen: v_template mean = " << model.v_template.mean() << std::endl;

    // 4. Verify Ceres linking
    ceres::Problem problem;
    std::cout << "Ceres Solver initialized successfully." << std::endl;

    return 0;
}