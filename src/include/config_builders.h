#pragma once
#include <vector>

#include "optimizer.h"   // OptimizerConfig, JointDepthData, GMMPosePrior

namespace app {

OptimizerConfig MakeCfgForOpenPose(const GMMPosePrior* gmm_prior);

OptimizerConfig MakeCfgForAMASS(const std::vector<JointDepthData>* depth_gt,
                                const GMMPosePrior* gmm_prior);

} // namespace app
