#include "config_builders.h"

#include <iostream>

namespace app {

OptimizerConfig MakeCfgForOpenPose(const GMMPosePrior* gmm_prior) {
    OptimizerConfig cfg;
    cfg.verbose = true;

    cfg.min_confidence = 0.25f;
    cfg.use_stable_joints_only = false;
    cfg.flip_y_in_projection = true;

    // Per-joint importance (enabled by default for OpenPose)
    cfg.use_joint_importance = true;

    // GMM
    if (gmm_prior && gmm_prior->loaded) {
        cfg.use_gmm_prior = true;
        cfg.w_gmm = 1.0;
        cfg.gmm_prior = gmm_prior;
        std::cout << "[Config] Using GMM pose prior\n";
    }

    cfg.use_depth = true;
    cfg.use_depth_affine=true;
    // Hinge/bend
    cfg.use_hinge_priors = true;
    cfg.use_bend_limits  = true;
    cfg.w_bend           = 0.005;

    cfg.w_hinge_perp_elbow  = 60.0;
    cfg.w_hinge_range_elbow = 30.0;
    cfg.w_hinge_perp_knee   = 120.0;
    cfg.w_hinge_range_knee  = 60.0;
    cfg.hinge_min = -2.6;
    cfg.hinge_max =  2.6;

    // Root upright
    cfg.use_root_upright = true;
    cfg.root_upright_w_roll  = 30.0;   
    cfg.root_upright_w_pitch = 1.5;    

    // Shoulder/collar — wider range for arm movement
    cfg.use_shoulder_collar_limits = true;
    cfg.w_collar_limit = 30.0;
    cfg.collar_amax    = 1.3;
    cfg.w_shoulder_limit = 60.0;
    cfg.shoulder_amax    = 1.5;

    cfg.use_wrist_limits = true;
    cfg.w_wrist_limit = 90.0;
    cfg.wrist_amax    = 0.8;

    cfg.use_spine_limits = true;
    cfg.w_spine     = 20.0;
    cfg.spine_amax  = 1.2;

    cfg.use_arm_torso_sep = true;

    // Capsule collision: ON
    cfg.use_capsule_collision = true;
    cfg.w_capsule_collision = 80.0;

    // Extremity locks
    cfg.lock_wrists = false;
    cfg.lock_hands  = true;
    cfg.lock_feet   = true;

    // Camera-space Z prior
    cfg.use_trans_z_prior = true;
    cfg.w_trans_z_prior = 50.0;

    // Foot: disabled by default, enable per-frame in sequence
    cfg.use_foot_ground = false;

    // Solver
    cfg.threads = -1;
    cfg.print_progress = false;
    cfg.max_iter_stage1 = 60;
    cfg.max_iter_stage2 = 300;
    cfg.max_iter_stage3 = 200;
    return cfg;
}

OptimizerConfig MakeCfgForOpenPoseSequence(const GMMPosePrior* gmm_prior) {
    OptimizerConfig cfg = MakeCfgForOpenPose(gmm_prior);

    cfg.use_graduated_energy = true;
    // Temporal smoothness
    cfg.use_temporal = true;
    cfg.w_temp_pose  = 18.0;   
    cfg.w_temp_trans = 30.0;
    cfg.use_temporal_acc = true;
    cfg.w_temp_acc_pose = 2.0;
    cfg.temporal_body_only = true;
    cfg.w_temp_root_orient = 40.0;
    
    cfg.w_hinge_perp_elbow  = 30.0;  
    cfg.w_hinge_range_elbow = 10.0;   

    cfg.use_neck_head_limits = true;
    cfg.w_neck_limit = 100.0;
    cfg.neck_amax = 0.2;
    cfg.w_head_limit = 90.0;
    cfg.head_amax = 0.2;

    cfg.use_arm_in_front    = true;
    cfg.w_arm_in_front      = 100.0;
    cfg.arm_in_front_margin = 0.03;

    cfg.lock_wrists = false;
    cfg.use_wrist_limits = true;
    cfg.w_wrist_limit = 80.0;
    cfg.wrist_amax = 0.8;   

    cfg.temporal_arm_scale = 0.3;
    cfg.w_cross_limb_depth = 160.0;
    cfg.w_depth_affine = 380.0;

    cfg.use_pelvis_over_ankles = true;
    cfg.w_pelvis_over_ankles = 10.0;

    // Foot contact
    cfg.use_foot_ground = true;
    cfg.w_foot_ground   = 100.0;
    cfg.w_foot_flat     = 80.0;
    cfg.ground_y        = 0.0;
    cfg.foot_contact_L = 1.0;
    cfg.foot_contact_R = 1.0;

    return cfg;
}

OptimizerConfig MakeCfgForAMASS(const std::vector<JointDepthData>* depth_gt,
                                const GMMPosePrior* gmm_prior) {
    OptimizerConfig cfg;
    cfg.verbose = true;
    cfg.amass_unit_test = true;
    cfg.unit_test_exact = true;
    cfg.use_stable_joints_only = false;
    cfg.use_joint_importance = false;  // synthetic data, all joints equal

    cfg.use_depth = true;
    cfg.w_depth = 50.0;
    cfg.depth_data = depth_gt;

    if (gmm_prior && gmm_prior->loaded) {
        cfg.use_gmm_prior = true;
        cfg.w_gmm = 1.0;
        cfg.gmm_prior = gmm_prior;
        std::cout << "[Config] Using GMM pose prior\n";
    } else {
        std::cout << "[Config] GMM not loaded\n";
    }

    cfg.use_hinge_priors = true;
    cfg.use_bend_limits  = true;
    cfg.w_bend           = 0.05;
    cfg.w_hinge_perp_elbow  = 30.0;
    cfg.w_hinge_range_elbow = 10.0;

    cfg.use_root_upright = true;

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

    cfg.use_neck_head_limits = true;
    cfg.w_neck_limit = 80.0;
    cfg.neck_amax = 0.4;
    cfg.w_head_limit = 100.0;
    cfg.head_amax = 0.35;

    cfg.use_arm_torso_sep = true;
    cfg.arm_torso_seg_a = 0;   // Pelvis
    cfg.arm_torso_seg_b = 9;   // Spine3

    cfg.lock_wrists = true;
    cfg.lock_hands  = true;
    cfg.lock_feet   = true;

    return cfg;
}

} // namespace app