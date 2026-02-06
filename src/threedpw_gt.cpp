#include "threedpw_gt.h"

#include <fstream>
#include <sstream>
#include <iostream>

namespace app {

bool ThreeDPW_GT::LoadFromTxt(const std::string& txt_path) {
    std::ifstream f(txt_path);
    if (!f.is_open()) return false;

    std::string line;
    joint_positions.clear();

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        Eigen::Matrix<double, 24, 3, Eigen::RowMajor> joints;
        for (int j = 0; j < 24; ++j) {
            iss >> joints(j, 0) >> joints(j, 1) >> joints(j, 2);
        }
        joint_positions.push_back(joints);
    }
    num_frames = (int)joint_positions.size();
    loaded = num_frames > 0;
    return loaded;
}

bool ThreeDPW_GT::LoadFromDir(const std::string& gt_dir) {
    // Load joints (required)
    std::string joints_path = gt_dir + "/gt_joints.txt";
    if (!LoadFromTxt(joints_path)) {
        // Fallback to single file path
        if (!LoadFromTxt(gt_dir)) return false;
    }

    // Load world-space joints (optional)
    {
        std::string joints_world_path = gt_dir + "/gt_joints_world.txt";
        std::ifstream jwf(joints_world_path);
        if (jwf.is_open()) {
            std::string line;
            while (std::getline(jwf, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                Eigen::Matrix<double, 24, 3, Eigen::RowMajor> joints;
                for (int j = 0; j < 24; ++j) {
                    iss >> joints(j, 0) >> joints(j, 1) >> joints(j, 2);
                }
                joint_positions_world.push_back(joints);
            }
            if (joint_positions_world.size() == (size_t)num_frames) {
                has_world_data = true;
                std::cout << "[GT] Loaded world-space joints\n";
            }
        }
    }

    // Load poses (optional)
    {
        std::string poses_path = gt_dir + "/gt_poses.txt";
        std::ifstream pf(poses_path);
        if (pf.is_open()) {
            std::string line;
            while (std::getline(pf, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                Eigen::Matrix<double, 72, 1> pose;
                for (int i = 0; i < 72; ++i) iss >> pose(i);
                poses.push_back(pose);
            }
            has_poses = (poses.size() == (size_t)num_frames);
            if (has_poses) std::cout << "[GT] Loaded poses\n";
        }
    }

    // Load world-space poses (optional)
    {
        std::string poses_world_path = gt_dir + "/gt_poses_world.txt";
        std::ifstream pwf(poses_world_path);
        if (pwf.is_open()) {
            std::string line;
            while (std::getline(pwf, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                Eigen::Matrix<double, 72, 1> pose;
                for (int i = 0; i < 72; ++i) iss >> pose(i);
                poses_world.push_back(pose);
            }
            if (poses_world.size() == (size_t)num_frames) {
                std::cout << "[GT] Loaded world-space poses\n";
            }
        }
    }

    // Load world-space translation (optional)
    {
        std::string trans_world_path = gt_dir + "/gt_trans_world.txt";
        std::ifstream twf(trans_world_path);
        if (twf.is_open()) {
            std::string line;
            while (std::getline(twf, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                Eigen::Vector3d t;
                iss >> t(0) >> t(1) >> t(2);
                trans_world.push_back(t);
            }
            if (trans_world.size() == (size_t)num_frames) {
                std::cout << "[GT] Loaded world-space translation\n";
            }
        }
    }

    // Load betas (optional)
    {
        std::string betas_path = gt_dir + "/gt_betas.txt";
        std::ifstream bf(betas_path);
        if (bf.is_open()) {
            for (int i = 0; i < 10; ++i) bf >> betas(i);
            has_betas = true;
            std::cout << "[GT] Loaded betas\n";
        }
    }

    // Load vertices (optional)
    {
        std::string verts_path = gt_dir + "/gt_vertices.txt";
        std::ifstream vf(verts_path);
        if (vf.is_open()) {
            std::string line;
            while (std::getline(vf, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> verts(6890, 3);
                for (int i = 0; i < 6890; ++i) {
                    iss >> verts(i, 0) >> verts(i, 1) >> verts(i, 2);
                }
                vertices.push_back(std::move(verts));
            }
            has_vertices = (vertices.size() == (size_t)num_frames);
            if (has_vertices) std::cout << "[GT] Loaded vertices\n";
        }
    }

    return loaded;
}

bool ThreeDPW_GT::GetJoints(int frame_idx,
                            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt) const {
    if (!loaded || frame_idx < 0 || frame_idx >= num_frames) return false;
    J_gt = joint_positions[frame_idx];
    return true;
}

bool ThreeDPW_GT::GetPose(int frame_idx, double* theta_gt) const {
    if (!has_poses || frame_idx < 0 || frame_idx >= (int)poses.size()) return false;
    for (int i = 0; i < 72; ++i) theta_gt[i] = poses[frame_idx](i);
    return true;
}

bool ThreeDPW_GT::GetBetas(double* beta_gt) const {
    if (!has_betas) return false;
    for (int i = 0; i < 10; ++i) beta_gt[i] = betas(i);
    return true;
}

bool ThreeDPW_GT::GetVertices(int frame_idx,
                              Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V_gt) const {
    if (!has_vertices || frame_idx < 0 || frame_idx >= (int)vertices.size()) return false;
    V_gt = vertices[frame_idx];
    return true;
}

bool ThreeDPW_GT::GetJointsWorld(int frame_idx,
                                 Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& J_gt) const {
    if (!has_world_data || frame_idx < 0 || frame_idx >= (int)joint_positions_world.size()) return false;
    J_gt = joint_positions_world[frame_idx];
    return true;
}

bool ThreeDPW_GT::GetPoseWorld(int frame_idx, double* theta_gt) const {
    if (frame_idx < 0 || frame_idx >= (int)poses_world.size()) return false;
    for (int i = 0; i < 72; ++i) theta_gt[i] = poses_world[frame_idx](i);
    return true;
}

bool ThreeDPW_GT::GetTransWorld(int frame_idx, double* trans_gt) const {
    if (frame_idx < 0 || frame_idx >= (int)trans_world.size()) return false;
    trans_gt;
    trans_gt;
    trans_gt;
    return true;
}

} // namespace app
