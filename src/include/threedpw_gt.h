#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace app {

struct ThreeDPW_GT {
    std::vector<Eigen::Matrix<double, 24, 3, Eigen::RowMajor>> joint_positions;
    std::vector<Eigen::Matrix<double, 24, 3, Eigen::RowMajor>> joint_positions_world;

    std::vector<Eigen::Matrix<double, 72, 1>> poses;
    std::vector<Eigen::Matrix<double, 72, 1>> poses_world;
    std::vector<Eigen::Vector3d> trans_world;

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> vertices;
    Eigen::Matrix<double, 10, 1> betas;

    bool has_poses = false;
    bool has_betas = false;
    bool has_vertices = false;
    bool has_world_data = false;
    bool loaded = false;
    int num_frames = 0;

    bool LoadFromTxt(const std::string& txt_path);
    bool LoadFromDir(const std::string& gt_dir);

    bool GetJoints(int frame_idx, Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt) const;
    bool GetPose(int frame_idx, double* theta_gt) const;
    bool GetBetas(double* beta_gt) const;
    bool GetVertices(int frame_idx, Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V_gt) const;

    bool GetJointsWorld(int frame_idx, Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt) const;
    bool GetPoseWorld(int frame_idx, double* theta_gt) const;
    bool GetTransWorld(int frame_idx, double* trans_gt) const;
};

} // namespace app
