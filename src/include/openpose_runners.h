#pragma once
#include <string>
#include "smpl_model.h"
#include "optimizer.h"

namespace app {

void RunOpenPoseInference(const SmplModel& model,
                          const GMMPosePrior* gmm_prior);

void RunOpenPoseSequenceInference(const SmplModel& model,
                                  const GMMPosePrior* gmm_prior,
                                  const std::string& seq_name,
                                  const std::string& out_dir,
                                  int start_frame,
                                  int end_frame);

} // namespace app
