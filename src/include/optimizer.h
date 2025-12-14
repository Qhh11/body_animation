#pragma once

#include <vector>
#include <string>
#include "cost_functions.h"
#include "smpl_model.h"

// Optimizer Configuration
struct OptimizerConfig {
    // Regularization weights
    double lambda_shape = 0.01;      // Shape prior weight (prevent extreme body shapes)
    double lambda_pose = 0.001;      // Pose prior weight (prefer natural poses)
    double lambda_depth = 0.1;       // Depth prior weight (optional)
    double expected_depth = 2.5;     // Expected camera-to-person distance (meters)

    // Ceres solver options
    int max_iterations = 100;        // Maximum optimization iterations
    bool verbose = true;             // Print optimization progress
    double function_tolerance = 1e-6;
    double gradient_tolerance = 1e-10;
    double parameter_tolerance = 1e-8;

    // Confidence threshold
    float min_confidence = 0.3;      // Ignore landmarks with confidence below this
};

// Main Optimization Function
/**
 * Optimize SMPL parameters to fit 2D observed landmarks.
 *
 * @param observed_keypoints: Vector of 2D landmarks from OpenPose (BODY_25 format)
 * @param camera: Camera intrinsic parameters
 * @param model: Loaded SMPL/STAR model data
 * @param beta: [IN/OUT] Shape parameters [10], initialized to zero, optimized output
 * @param theta: [IN/OUT] Pose parameters [72], initialized to zero, optimized output
 * @param trans: [IN/OUT] Global translation [3], initialized to zero, optimized output
 * @param config: Optimizer configuration
 * @return: true if optimization succeeded, false otherwise
 */
bool OptimizeSMPL(
    const std::vector<Landmark2D>& observed_keypoints,
    const CameraIntrinsics& camera,
    const SmplModel& model,
    double* beta,   // [10]
    double* theta,  // [72]
    double* trans,  // [3]
    const OptimizerConfig& config = OptimizerConfig()
);

// Utility Functions
/**
 * Load 2D landmarks from JSON file (OpenPose output format).
 * @param json_path: Path to OpenPose JSON file
 * @param landmarks: Output vector of landmarks
 * @return: true if load succeeded
 */
bool LoadLandmarksFromJSON(
    const std::string& json_path,
    std::vector<Landmark2D>& landmarks
);

/**
 * Create a simple camera with typical parameters for debugging.
 * @param image_width: Image width in pixels
 * @param image_height: Image height in pixels
 * @return: Camera intrinsics with reasonable defaults
 */
CameraIntrinsics CreateDefaultCamera(int image_width, int image_height);
