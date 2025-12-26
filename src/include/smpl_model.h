// smpl_model.h
#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <utility>

struct SmplModel {
    int N = 0;
    int J = 0;

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> v_template;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> shapedirs;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> posedirs;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> J_regressor;
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kintree_table;
    Eigen::MatrixXi faces;

    std::vector<std::vector<std::pair<int,double>>> J_regressor_nz;

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> J_template_3d;   // J x 3
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> J_shapedirs_3d; // (J*3) x 10

    bool LoadFromNPZ(const std::string& npz_path);

private:
    void BuildRegressorCache();
    void BuildJointShapeCache(int num_beta = 10); 
};
