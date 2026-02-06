#include "openpose_io.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace app {

bool LoadLandmarksFromJSON(const std::string& json_path,
                           std::vector<Landmark2D>& landmarks) {
    landmarks.clear();

    std::ifstream infile(json_path);
    if (!infile.is_open()) return false;

    json j;
    try { infile >> j; } catch (...) { return false; }

    if (!j.contains("people") || j["people"].empty()) return false;

    const auto& people = j["people"];

    int best_idx = -1;
    double best_score = -1.0;

    for (int i = 0; i < (int)people.size(); ++i) {
        if (!people[i].contains("pose_keypoints_2d")) continue;
        const auto& kp = people[i]["pose_keypoints_2d"];
        if (kp.size() < 75) continue;

        float min_x = 1e9f, max_x = -1e9f;
        float min_y = 1e9f, max_y = -1e9f;
        double sum_c = 0.0;
        int cnt = 0;

        const float conf_thr = 0.20f;

        for (int jnt = 0; jnt < 25; ++jnt) {
            float x = kp[3 * jnt + 0].get<float>();
            float y = kp[3 * jnt + 1].get<float>();
            float c = kp[3 * jnt + 2].get<float>();
            if (c < conf_thr) continue;

            min_x = std::min(min_x, x); max_x = std::max(max_x, x);
            min_y = std::min(min_y, y); max_y = std::max(max_y, y);

            sum_c += c;
            cnt++;
        }

        if (cnt < 6) continue;

        const float w = std::max(1.0f, max_x - min_x);
        const float h = std::max(1.0f, max_y - min_y);
        const double area = (double)w * (double)h;
        const double mean_c = sum_c / (double)cnt;

        const double score = area * mean_c * (double)cnt;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx < 0) {
        best_idx = 0;
        if (!people[best_idx].contains("pose_keypoints_2d")) return false;
        if (people[best_idx]["pose_keypoints_2d"].size() < 75) return false;
    }

    const auto& keypoints = people[best_idx]["pose_keypoints_2d"];

    for (int joint_id = 0; joint_id < 25; ++joint_id) {
        float x = keypoints[3 * joint_id + 0].get<float>();
        float y = keypoints[3 * joint_id + 1].get<float>();
        float c = keypoints[3 * joint_id + 2].get<float>();
        if (c > 0.05f) landmarks.emplace_back(x, y, c, joint_id);
    }

    return !landmarks.empty();
}

std::string MakeOpenPoseJsonPath(const std::string& dir, int frame) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/image_%05d_keypoints.json", dir.c_str(), frame);
    return std::string(buf);
}

std::string MakeOpenPoseDepthPath(const std::string& dir, int frame) {
    char buf1[512], buf2[512];
    std::snprintf(buf1, sizeof(buf1), "%s/image_%05d_depth.txt", dir.c_str(), frame);
    std::snprintf(buf2, sizeof(buf2), "%s/image_%05d.txt", dir.c_str(), frame);

    if (fs::exists(buf1)) return std::string(buf1);
    if (fs::exists(buf2)) return std::string(buf2);
    return std::string(buf1);
}

std::string MakeOutObj(const std::string& out_dir, int frame) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/mesh_%05d.obj", out_dir.c_str(), frame);
    return std::string(buf);
}

} // namespace app
