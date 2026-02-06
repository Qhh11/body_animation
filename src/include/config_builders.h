#pragma once

#include "optimizer.h"

namespace app {

// Single-frame OpenPose optimization config
OptimizerConfig MakeCfgForOpenPose(const GMMPosePrior* gmm_prior);

// Sequence OpenPose config: builds on single-frame, adds temporal + foot + arm freedom
OptimizerConfig MakeCfgForOpenPoseSequence(const GMMPosePrior* gmm_prior);

// AMASS unit-test config
OptimizerConfig MakeCfgForAMASS(const std::vector<JointDepthData>* depth_gt,
                                const GMMPosePrior* gmm_prior);

} // namespace app