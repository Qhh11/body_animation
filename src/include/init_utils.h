#pragma once
#include <vector>
#include "optimizer.h" // Landmark2D, CameraIntrinsics

namespace app {

// estimate trans from bbox center with assumed z
void EstimateGlobalTranslationXY(const std::vector<Landmark2D>& landmarks,
                                 const CameraIntrinsics& camera,
                                 double z_assumed,
                                 double* trans_init);

} // namespace app
