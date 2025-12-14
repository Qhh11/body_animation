#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "cnpy.h" 


struct SmplModel {
    int N;
    int J; 

    Eigen::MatrixXd v_template; 
    
    Eigen::MatrixXd shapedirs; 

    Eigen::MatrixXd posedirs; 

    Eigen::MatrixXd J_regressor; 
    
    Eigen::MatrixXd weights; 
    
    Eigen::MatrixXi kintree_table; 

    SmplModel() : N(0), J(0) {}

    bool LoadFromNPZ(const std::string& path);
};