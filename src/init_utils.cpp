#include "init_utils.h"

#include <iostream>
#include <algorithm>

namespace app {

void EstimateGlobalTranslationXY(const std::vector<Landmark2D>& landmarks,
                                 const CameraIntrinsics& camera,
                                 double z_assumed,
                                 double* trans_init) {
    float min_y = 1e9f, max_y = -1e9f;
    float min_x = 1e9f, max_x = -1e9f;

    for (const auto& lm : landmarks) {
        if (lm.confidence < 0.2f) continue;
        min_x = std::min(min_x, lm.x);
        max_x = std::max(max_x, lm.x);
        min_y = std::min(min_y, lm.y);
        max_y = std::max(max_y, lm.y);
    }

    const float center_x = 0.5f * (min_x + max_x);
    const float center_y = 0.5f * (min_y + max_y);

    trans_init[0] = (center_x - camera.cx) * z_assumed / camera.fx;
    trans_init[1] = -(center_y - camera.cy) * z_assumed / camera.fy;
    trans_init[2] = z_assumed;

    std::cout << "[Init] trans init = (" << trans_init[0] << ", " << trans_init[1] << ", " << trans_init[2] << ")\n";
}

} // namespace app
