# SMPL Body Fitting - Monocular 2D-to-3D Optimization Pipeline



## Build

```bash
cd body_animation/src/build
cmake ..
make
```

Dependencies: Eigen3, Ceres Solver, zlib, C++17.

## Data Layout

```
Data/
  models/
    smpl_neutral.npz           # SMPL neutral body model
    gmm_08.npz                 # GMM pose prior (8 components)
  openpose_output/{seq}/       # OpenPose BODY_25 JSON detections per frame
  sapiens_depth/{seq}/         # Sapiens relative depth estimates per frame
  3dpw_gt/{seq}/               # 3DPW ground truth
    cam_intrinsics.txt         #   fx fy cx cy (auto-loaded per sequence)
    cam_poses.txt              #   per-frame 3x4 camera extrinsics
    gt_joints.txt              #   24 joints per frame (camera space)
    gt_poses.txt               #   72-dim SMPL pose per frame
    gt_betas.txt               #   10-dim shape parameters
```

## Running Sequences (OpenPose Pipeline)

All commands run from `body_animation/src/build/`.

### Command Format

```bash
./main --openpose-seq \
  --input_seq <sequence_name> \
  --output_dir <output_directory> \
  --start <start_frame> \
  --end <end_frame>
```

### Available Sequences

| Sequence | Frames | Notes |
|----------|--------|-------|
| `courtyard_basketball_01` | 0 - 957 | |
| `courtyard_bodyScannerMotions_00` | 0 - 1256 | |
| `outdoors_fencing_01` | 0 - 941 | **Must disable foot_ground** (see below) |

### Run Commands

```bash
# courtyard_basketball_01
./main --openpose-seq \
  --input_seq courtyard_basketball_01 \
  --output_dir ./seq_out_basketball \
  --start 0 --end 200

# courtyard_bodyScannerMotions_00
./main --openpose-seq \
  --input_seq courtyard_bodyScannerMotions_00 \
  --output_dir ./seq_out_bodyscanner \
  --start 0 --end 200

# outdoors_fencing_01 (foot_ground must be disabled first!)
./main --openpose-seq \
  --input_seq outdoors_fencing_01 \
  --output_dir ./seq_out_fencing \
  --start 0 --end 200

```

### Disabling foot_ground for fencing

The `outdoors_fencing_01` sequence contains jumping and airborne poses. The foot-ground contact constraint must be disabled, otherwise it will force feet to stay on the ground plane and produce incorrect results.

In `src/config_builders.cpp`, function `MakeCfgForOpenPoseSequence()`, change:

```cpp
cfg.use_foot_ground = false;   // disable for fencing (default: true)
```

Then rebuild:

```bash
make
```

Remember to set it back to `true` for other sequences that involve walking/standing.

## Output

Each run produces a directory containing:

```
{output_dir}/
  frame_00000.obj      # SMPL mesh per frame
  frame_00001.obj
  ...
  config.txt           # optimizer configuration used
  evaluation.txt       # per-frame metrics + summary (requires GT)
```


### Evaluation Metrics (printed in evaluation.txt)

| Metric | Unit | Description |
|--------|------|-------------|
| MPJPE | mm | Mean Per-Joint Position Error |
| PA-MPJPE | mm | Procrustes-Aligned MPJPE (scale/rotation invariant) |
| MPJPE (root-rel) | mm | MPJPE centered at pelvis |
| Pose Error | rad | RMS of pose parameter difference |
| 2D Reproj | px | Mean 2D reprojection error |

## AMASS Validation Pipeline

For controlled validation using synthetic 2D observations from motion capture data:

```bash
# Single-frame validation
./main ../../Data/sampled/CMU_01.txt

# Sequence validation (frames 0-50)
./main --seq-val ../../Data/sampled/CMU_01.txt

# Batch validation (predefined test set)
./main --batch
```