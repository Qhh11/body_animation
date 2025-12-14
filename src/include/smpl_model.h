#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "cnpy.h" 

struct SmplModel {
    int N; // Number of vertices
    int J; // Number of joints

    Eigen::MatrixXd v_template; 
    Eigen::MatrixXd shapedirs; 
    Eigen::MatrixXd posedirs; 
    Eigen::MatrixXd J_regressor; 
    Eigen::MatrixXd weights; 
    Eigen::MatrixXi kintree_table; 

    // [NEW] Added faces matrix to store triangle indices (N_faces x 3)
    Eigen::MatrixXi faces; 

    SmplModel() : N(0), J(0) {}

    bool LoadFromNPZ(const std::string& path);
};