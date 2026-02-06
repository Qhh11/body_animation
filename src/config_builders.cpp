#include "config_builders.h"

#include <iostream>

namespace app {

OptimizerConfig MakeCfgForOpenPose(const GMMPosePrior* gmm_prior) {
    OptimizerConfig cfg;
    cfg.verbose = true;

    cfg.min_confidence = 0.10f;
    cfg.use_stable_joints_only = true;
    cfg.flip_y_in_projection = true;

    // GMM
    if (gmm_prior && gmm_prior->loaded) {
        cfg.use_gmm_prior = true;
        cfg.w_gmm = 1.0;
        cfg.gmm_prior = gmm_prior;
        std::cout << "[Config] Using GMM pose prior\n";
    }

    // hinge/bend
    cfg.use_hinge_priors = true;
    cfg.use_bend_limits  = true;
    cfg.w_bend           = 0.05;

    cfg.w_hinge_perp_elbow  = 15.0;
    cfg.w_hinge_range_elbow = 8.0;
    cfg.w_hinge_perp_knee   = 120.0;
    cfg.w_hinge_range_knee  = 60.0;
    cfg.hinge_min = -2.6;
    cfg.hinge_max =  2.6;

    // root upright
    cfg.use_root_upright = true;
    cfg.root_upright_w_stage2 = 5.0;
    cfg.root_upright_w_stage3 = 1.0;

    // shoulder/collar/wrist/spine
    cfg.use_shoulder_collar_limits = true;
    cfg.w_collar_limit = 40.0;
    cfg.collar_amax    = 0.8;
    cfg.w_shoulder_limit = 60.0;
    cfg.shoulder_amax    = 1.7;

    cfg.use_wrist_limits = true;
    cfg.w_wrist_limit = 80.0;
    cfg.wrist_amax    = 0.6;

    cfg.use_spine_limits = true;
    cfg.w_spine     = 10.0;
    cfg.spine_amax  = 1.2;

    cfg.use_pelvis_over_ankles = true;

    // arm-torso sep
    cfg.use_arm_torso_sep = true;
    cfg.arm_torso_seg_a = 3;
    cfg.arm_torso_seg_b = 12;
    cfg.arm_torso_d_elbow = 0.035;
    cfg.arm_torso_d_wrist = 0.045;
    cfg.arm_torso_w_elbow = 20.0;
    cfg.arm_torso_w_wrist = 35.0;

    cfg.lock_wrists = true;
    cfg.lock_hands  = true;
    cfg.lock_feet   = true;

    // foot constraints: enable in sequence after ground_y known
    cfg.use_foot_ground = false;

    // solver
    cfg.threads = -1;
    cfg.print_progress = false;
    cfg.max_iter_stage1 = 60;
    cfg.max_iter_stage2 = 200;
    cfg.max_iter_stage3 = 200;

    return cfg;
}

OptimizerConfig MakeCfgForAMASS(const std::vector<JointDepthData>* depth_gt,
                                const GMMPosePrior* gmm_prior) {
    OptimizerConfig cfg;
    cfg.verbose = true;
    cfg.amass_unit_test = true;
    cfg.unit_test_exact = true;

    // synthetic landmarks accurate => use all joints
    cfg.use_stable_joints_only = false;

    // depth
    cfg.use_depth = true;
    cfg.w_depth = 50.0;
    cfg.depth_data = depth_gt;

    // GMM
    if (gmm_prior && gmm_prior->loaded) {
        cfg.use_gmm_prior = true;
        cfg.w_gmm = 1.0;
        cfg.gmm_prior = gmm_prior;
        std::cout << "[Config] Using GMM pose prior\n";
    } else {
        std::cout << "[Config] GMM not loaded\n";
    }

    // hinge/bend
    cfg.use_hinge_priors = true;
    cfg.use_bend_limits  = true;
    cfg.w_bend           = 0.05;

    cfg.w_hinge_perp_elbow  = 30.0;
    cfg.w_hinge_range_elbow = 10.0;

    // root upright
    cfg.use_root_upright = true;
    cfg.root_upright_w_stage2 = 0.5;
    cfg.root_upright_w_stage3 = 0.0;

    // soft limits
    cfg.use_shoulder_collar_limits = true;
    cfg.w_collar_limit = 20.0;
    cfg.collar_amax    = 1.0;
    cfg.w_shoulder_limit = 30.0;
    cfg.shoulder_amax    = 2.0;

    cfg.use_wrist_limits = true;
    cfg.w_wrist_limit = 60.0;
    cfg.wrist_amax    = 0.8;

    cfg.use_spine_limits = true;
    cfg.w_spine    = 5.0;
    cfg.spine_amax = 1.5;

    // neck/head limits
    cfg.use_neck_head_limits = true;
    cfg.w_neck_limit = 40.0;
    cfg.neck_amax = 0.5;
    cfg.w_head_limit = 50.0;
    cfg.head_amax = 0.4;

    // arm-torso sep
    cfg.use_arm_torso_sep = true;

    cfg.lock_wrists = true;
    cfg.lock_hands  = true;
    cfg.lock_feet   = true;

    return cfg;
}

} // namespace app
