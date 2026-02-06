#pragma once
#include "smpl_model.h"
#include "optimizer.h"

namespace app {

void RunOpenPoseInference(const SmplModel& model,
                          const GMMPosePrior* gmm_prior);

void RunOpenPoseSequenceInference(const SmplModel& model,
                                  const GMMPosePrior* gmm_prior);

} // namespace app
