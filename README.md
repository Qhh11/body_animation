# Group2_Body Animation

The environment used in the project is the same docker container used in 3dScanning exercise.

## Build

```bash
cd body_animation/src/build
cmake ..
make
```


## Data Layout

```
Data/
  models/
    smpl_neutral.npz           # SMPL neutral body model
    gmm_08.npz                 # GMM pose prior (8 components)
  openpose_output/{seq}/       # OpenPose BODY_25 JSON detections per frame
  sapiens_depth/{seq}/         # Sapiens relative depth estimates per frame
  3dpw_gt/{seq}/               # 3DPW ground truth
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

| Sequence | Frames |
|----------|--------|
| `courtyard_basketball_01` | 0 - 957 | 
| `courtyard_bodyScannerMotions_00` | 0 - 1256 | 
| `outdoors_fencing_01` | 0 - 941 | 

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

# outdoors_fencing_01 
./main --openpose-seq \
  --input_seq outdoors_fencing_01 \
  --output_dir ./seq_out_fencing \
  --start 0 --end 200

```

### Note on Regularization Parameters

The SMPL body model has high-dimensional parameters (72-DoF pose + 10-DoF shape), and different motion types require different regularization settings for optimal results. The reproduction quality is highly sensitive to these configuration parameters (e.g., temporal smoothness weights, foot-ground contact, depth affine weights). Reference configuration parameters and example outputs for each sequence are available at [Dropbox](https://www.dropbox.com/scl/fo/n0kmyas3oq4d13muemlf0/AFIvtXufj9d1GaUFIwmvpm8?rlkey=ovyuznxd9jdb1a0jjm856v5eb&st=62fbx3vi&dl=0).
All parameters can be tuned in script config_builders.cpp.

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