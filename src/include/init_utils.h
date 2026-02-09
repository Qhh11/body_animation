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
    double* theta,      
    float min_conf = 0.3f)
{
   
    struct KP { float x, y, c; bool valid; };
    auto find = [&](int op_id) -> KP {
        for (const auto& lm : lms) {
            if (lm.joint_id == op_id && lm.confidence >= min_conf)
                return {lm.x, lm.y, lm.confidence, true};
        }
        return {0, 0, 0, false};
    };

    KP rsh = find(2);   
    KP lsh = find(5);   
    KP rhip = find(9);  
    KP lhip = find(12); 
    KP neck = find(1);  
    KP mhip = find(8);  

    if (!rsh.valid || !lsh.valid) {
        std::cout << "[InitRoot] Insufficient keypoints, using default orientation\n";
        return;
    }

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

    float ratio = 0.5f;
    if (torso_height > 10.0f) {
        ratio = shoulder_width / torso_height;
    }


    float y_angle = 0.0f;

    const float frontal_ratio = 0.65f;  
    const float profile_ratio = 0.25f;  

    if (ratio < profile_ratio) {
        y_angle = (float)(M_PI * 0.45);
    } else if (ratio < frontal_ratio) {
        float t = (ratio - profile_ratio) / (frontal_ratio - profile_ratio);
        y_angle = (float)(M_PI * 0.45) * (1.0f - t);
    } else {
        y_angle = 0.0f;  
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
        if (shoulders_reversed) {
            y_angle = -y_angle;
        } else {
            if (offset > 5.0f) {
                y_angle = -y_angle; 
            }
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
















