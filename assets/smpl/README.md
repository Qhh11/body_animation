# SMPL Model Assets

This directory should contain the SMPL model file: `model.npz`

## Obtaining the SMPL Model

The SMPL model must be obtained legally from the official source:

**Official Website**: https://smpl.is.tue.mpg.de/

## Important Notes

1. **Do NOT commit the SMPL model file** to version control
   - The model file is copyrighted material
   - Redistribution is not permitted without authorization
   - The `.gitignore` file excludes `*.npz` files from this directory

2. **Required File**: `model.npz`
   - Download from the official SMPL website
   - Place it in this directory as `assets/smpl/model.npz`

3. **Expected Format**
   - The NPZ file should contain the following keys:
     - `v_template`: (N, 3) - template mesh vertices
     - `shapedirs`: (N, 3, n_beta) - shape blend shapes
     - `posedirs`: (N, 3, D_pose) or (N*3, D_pose) - pose blend shapes
     - `J_regressor`: (J, N) - joint regressor matrix
     - `weights`: (N, J) - skinning weights
     - `kintree_table`: (2, J) - kinematic tree

## Usage

Once you have `model.npz` in this directory, you can:

1. Run the pipeline directly:
   ```bash
   ./run_pipeline --smpl_npz=assets/smpl/model.npz
   ```

2. Generate corrective data:
   ```bash
   python3 tools/build_corrective_data.py \
       --smpl_npz assets/smpl/model.npz \
       --out_npz assets/corrective/corrective_data.npz
   ```
