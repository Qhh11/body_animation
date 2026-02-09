#pragma once
#include <string>
#include <vector>

#include "optimizer.h" 

namespace app {

bool LoadLandmarksFromJSON(const std::string& json_path,
                           std::vector<Landmark2D>& landmarks);

std::string MakeOpenPoseJsonPath(const std::string& dir, int frame);

std::string MakeOpenPoseDepthPath(const std::string& dir, int frame);

std::string MakeOutObj(const std::string& out_dir, int frame);

} // namespace app
