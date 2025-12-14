#pragma once
#include <string>

// Render an OBJ file to a PNG image using Open3D offscreen rendering.
// Returns true on success, false on failure.
bool RenderObjToPng(
    const std::string& obj_path,
    const std::string& png_path,
    int width,
    int height,
    float cam_distance
);