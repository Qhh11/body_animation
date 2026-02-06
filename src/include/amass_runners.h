#pragma once
#include <string>

#include "smpl_model.h"
#include "optimizer.h"

namespace app {

// MODE 1: single-frame AMASS validation
double RunAMASSValidation(const SmplModel& model,
                          const GMMPosePrior* gmm_prior,
                          const std::string& amass_path,
                          int frame_idx = -1,
                          bool save_mesh = true);

// AMASS sequence validation frames 0-50
void RunAmassValidationSequence(const SmplModel& model,
                                const GMMPosePrior* gmm_prior,
                                const std::string& pose_file_path);

// Batch validation (predefined list)
void RunBatchValidation(const SmplModel& model,
                        const GMMPosePrior* gmm_prior);

} // namespace app
