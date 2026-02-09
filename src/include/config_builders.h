#pragma once

#include "optimizer.h"

namespace app {

OptimizerConfig MakeCfgForOpenPose(const GMMPosePrior* gmm_prior);

OptimizerConfig MakeCfgForOpenPoseSequence(const GMMPosePrior* gmm_prior);

OptimizerConfig MakeCfgForAMASS(const std::vector<JointDepthData>* depth_gt,
                                const GMMPosePrior* gmm_prior);

} // namespace app