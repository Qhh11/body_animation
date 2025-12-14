#include "render.h"
#include <open3d/Open3D.h>
#include <iostream>

using namespace open3d;

bool RenderObjToPng(
    const std::string& obj_path,
    const std::string& png_path,
    int width,
    int height,
    float cam_distance
) {
    // 1. Load mesh from OBJ file
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    if (!io::ReadTriangleMesh(obj_path, *mesh)) {
        std::cerr << "Failed to read mesh: " << obj_path << "\n";
        return false;
    }

    // Check if the mesh is valid
    if (mesh->vertices_.empty() || mesh->triangles_.empty()) {
        std::cerr << "Mesh is empty.\n";
        return false;
    }

    // Compute vertex normals for lighting
    mesh->ComputeVertexNormals();

    // Center the mesh at the origin to ensure it is visible by the camera
    mesh->Translate(-mesh->GetCenter());

    // 2. Create offscreen renderer and scene
    visualization::rendering::OffscreenRenderer renderer(width, height);
    auto& scene = renderer.GetScene();

    // Set white background color (RGBA in range [0, 1])
    scene.SetBackground({1.0f, 1.0f, 1.0f, 1.0f});

    // Use a default lit material (includes basic lighting)
    visualization::rendering::MaterialRecord mat;
    mat.shader = "defaultLit";
    scene.AddGeometry("mesh", mesh, mat);

    // 3. Configure camera using LookAt
    scene.GetCamera()->LookAt(
        {0.0f, 0.0f, 0.0f},           // Camera target (center)
        {0.0f, 0.0f, cam_distance},   // Camera position (eye)
        {0.0f, 1.0f, 0.0f}            // Up direction
    );

    // 4. Render the scene to an image
    auto img = renderer.RenderToImage();
    if (!img) {
        std::cerr << "RenderToImage failed.\n";
        return false;
    }

    // Write the rendered image to a PNG file
    if (!io::WriteImage(png_path, *img)) {
        std::cerr << "Failed to write image: " << png_path << "\n";
        return false;
    }

    return true;
}
