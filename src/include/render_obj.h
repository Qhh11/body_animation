#pragma once
#include <string>

bool RenderObjToPng(
    const std::string& obj_path,
    const std::string& png_path,
    int width,
    int height,
    float cam_distance
);