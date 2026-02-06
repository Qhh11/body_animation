#pragma once
#include <string>
#include <vector>

#include "optimizer.h" // Landmark2D

namespace app {

// OpenPose json loader (pick best person)
bool LoadLandmarksFromJSON(const std::string& json_path,
                           std::vector<Landmark2D>& landmarks);

std::string MakeOpenPoseJsonPath(const std::string& dir, int frame);

// supports both image_%05d_depth.txt and image_%05d.txt
std::string MakeOpenPoseDepthPath(const std::string& dir, int frame);

std::string MakeOutObj(const std::string& out_dir, int frame);

} // namespace app
