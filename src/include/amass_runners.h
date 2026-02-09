#pragma once
#include <string>

#include "smpl_model.h"
#include "optimizer.h"

namespace app {

double RunAMASSValidation(const SmplModel& model,
                          const GMMPosePrior* gmm_prior,
                          const std::string& amass_path,
                          int frame_idx = -1,
                          bool save_mesh = true);

void RunAmassValidationSequence(const SmplModel& model,
                                const GMMPosePrior* gmm_prior,
                                const std::string& pose_file_path);

void RunBatchValidation(const SmplModel& model,
                        const GMMPosePrior* gmm_prior);

} // namespace app
