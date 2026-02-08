// multi_hypothesis.h
#pragma once

#include "optimizer.h"
#include "smpl_model.h"
#include "smpl_math.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

namespace multihyp {

// ---------------------------------------------------------
// 1. Depth-Guided Z Correction
//    After warm-start, shift trans[2] so predicted joint Zs
//    better match the affine-converted depth observations.
//    Cost: one SMPL forward pass + arithmetic. ~0.1ms.
// ---------------------------------------------------------
inline void DepthGuidedZCorrection(
    const SmplModel& model,
    const double* beta,
    const double* theta,
    double* trans,                               // modified in-place
    const std::vector<JointRelDepthData>& depth_pts,
    const double* depth_affine,                  // [scale, bias]
    double max_z_shift = 0.5,
    double min_confidence = 0.3)
{
    if (depth_pts.empty() || !depth_affine) return;

    MatX3dRM J;
    SMPLJointsForward<double>(beta, theta, trans, model, J);

    double sum_delta = 0.0;
    int count = 0;

    for (const auto& dp : depth_pts) {
        if (dp.weight < min_confidence) continue;
        const int j = dp.smpl_joint_idx;
        if (j < 0 || j >= J.rows()) continue;

        const double z_pred = J(j, 2);
        const double z_tgt  = depth_affine[1] + depth_affine[0] * dp.rel_depth;

        // sanity: skip if target Z is nonsensical
        if (z_tgt < 0.3 || z_tgt > 30.0) continue;

        sum_delta += (z_tgt - z_pred);
        ++count;
    }

    if (count < 3) return;

    double z_shift = sum_delta / count;
    z_shift = std::clamp(z_shift, -max_z_shift, max_z_shift);

    if (std::abs(z_shift) > 0.02) {
        trans[2] += z_shift;
        std::cout << "[DepthZCorr] shifted Z by " << z_shift
                  << " (from " << count << " joints)\n";
    }
}


} // namespace multihyp