#pragma once
#include "optimizer.h" 
#include <cmath>
#include <vector>
#include <iostream>

namespace app {

// estimate trans from bbox center with assumed z
void EstimateGlobalTranslationXY(const std::vector<Landmark2D>& landmarks,
                                 const CameraIntrinsics& camera,
                                 double z_assumed,
                                 double* trans_init);

} // namespace app

namespace initutil {

inline void EstimateRootFromKeypoints(
    const std::vector<Landmark2D>& lms,
    const CameraIntrinsics& camera,
    double* theta,       // modifies theta[0], theta[1], theta[2]
    float min_conf = 0.3f)
{
    // --- Extract key 2D positions ---
    // OpenPose IDs: 2=RShoulder, 5=LShoulder, 9=RHip, 12=LHip, 1=Neck, 8=MidHip
    struct KP { float x, y, c; bool valid; };
    auto find = [&](int op_id) -> KP {
        for (const auto& lm : lms) {
            if (lm.joint_id == op_id && lm.confidence >= min_conf)
                return {lm.x, lm.y, lm.confidence, true};
        }
        return {0, 0, 0, false};
    };

    KP rsh = find(2);   // Right shoulder
    KP lsh = find(5);   // Left shoulder
    KP rhip = find(9);  // Right hip
    KP lhip = find(12); // Left hip
    KP neck = find(1);  // Neck
    KP mhip = find(8);  // Mid hip

    // Need at least shoulders
    if (!rsh.valid || !lsh.valid) {
        std::cout << "[InitRoot] Insufficient keypoints, using default orientation\n";
        return;
    }

    // --- Estimate Y rotation (facing direction) ---
    float shoulder_dx = std::abs(rsh.x - lsh.x);
    float shoulder_dy = std::abs(rsh.y - lsh.y);
    float shoulder_width = std::sqrt(shoulder_dx * shoulder_dx + shoulder_dy * shoulder_dy);

    float torso_height = 0.0f;
    if (neck.valid && mhip.valid) {
        float dx = neck.x - mhip.x;
        float dy = neck.y - mhip.y;
        torso_height = std::sqrt(dx * dx + dy * dy);
    } else if (neck.valid && rhip.valid) {
        float dx = neck.x - rhip.x;
        float dy = neck.y - rhip.y;
        torso_height = std::sqrt(dx * dx + dy * dy);
    }

    // Ratio: shoulder_width / torso_height
    // Frontal: ~0.7-1.0, Profile: ~0.1-0.3
    float ratio = 0.5f;
    if (torso_height > 10.0f) {
        ratio = shoulder_width / torso_height;
    }

    // Map ratio to Y rotation angle
    // ratio ~0.8 → facing camera → angle ~0
    // ratio ~0.15 → side view → angle ~±π/2
    float y_angle = 0.0f;

    const float frontal_ratio = 0.65f;  
    const float profile_ratio = 0.25f;  

    if (ratio < profile_ratio) {
        y_angle = (float)(M_PI * 0.45);
    } else if (ratio < frontal_ratio) {
        // Linear interpolation
        float t = (ratio - profile_ratio) / (frontal_ratio - profile_ratio);
        y_angle = (float)(M_PI * 0.45) * (1.0f - t);
    } else {
        y_angle = 0.0f;  // frontal
    }

    float sh_mid_x = (rsh.x + lsh.x) * 0.5f;
    float hip_mid_x = sh_mid_x;
    if (rhip.valid && lhip.valid) {
        hip_mid_x = (rhip.x + lhip.x) * 0.5f;
    } else if (mhip.valid) {
        hip_mid_x = mhip.x;
    }

    float offset = sh_mid_x - hip_mid_x;
    bool shoulders_reversed = (rsh.x > lsh.x);

    if (ratio < frontal_ratio) {
        // Side-ish view: determine direction
        if (shoulders_reversed) {
            // Person's right shoulder is to the right in image → facing right
            y_angle = -y_angle;
        } else {
            // Check the offset
            if (offset > 5.0f) {
                y_angle = -y_angle;  // facing left in image → positive Y rotation
            }
            // else: default positive (facing right in image)
        }
    }

    theta[0] = 0.0;
    theta[1] = (double)y_angle;
    theta[2] = 0.0;


    std::cout << "[InitRoot] shoulder_width=" << shoulder_width
              << " torso_height=" << torso_height
              << " ratio=" << ratio
              << " y_angle=" << y_angle * 180.0f / M_PI << " deg"
              << " (reversed=" << shoulders_reversed
              << ",)";
}

inline void EstimateTransZFromKeypoints(
    const std::vector<Landmark2D>& lms,
    const CameraIntrinsics& camera,
    double* trans,
    float min_conf = 0.3f,
    double assumed_torso_height_m = 0.55)
{
    const Landmark2D* neck = nullptr;
    const Landmark2D* mhip = nullptr;
    for (const auto& lm : lms) {
        if (lm.confidence < min_conf) continue;
        if (lm.joint_id == 1) neck = &lm;
        if (lm.joint_id == 8) mhip = &lm;
    }
    if (!neck || !mhip) return;

    float dy = std::abs(neck->y - mhip->y);
    if (dy < 20.0f) return;

    double z_est = (double)camera.fy * assumed_torso_height_m / (double)dy;
    z_est = std::clamp(z_est, 1.0, 15.0);

    trans[2] = z_est;  

    std::cout << "[InitTrans] pixel_torso=" << dy
              << " Z_est=" << z_est << "\n";
}

} // namespace initutil
















