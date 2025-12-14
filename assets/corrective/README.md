# Pose-Corrective Data

This directory contains generated pose-corrective data derived from the SMPL model.

## File: `corrective_data.npz`

This file is **generated** by the `tools/build_corrective_data.py` script and contains
region-based, dimensionality-reduced pose blend shape data.

## Generation

To generate this file:

```bash
python3 tools/build_corrective_data.py \
    --smpl_npz assets/smpl/model.npz \
    --out_npz assets/corrective/corrective_data.npz \
    --M 500 \
    --K 8
```

### Parameters

- `--M`: Number of top-weighted vertices to select per joint (default: 500)
- `--K`: Number of SVD basis components to keep per region (default: 8)

## Data Format

The generated NPZ file contains the following arrays in CSR (Compressed Sparse Row) format:

### Scalar Values

- `J`: (1,) - Number of joints
- `K`: (1,) - Number of basis components per region

### CSR-Packed Arrays

- `region_ptr`: (J+1,) - CSR pointer array
  - `region_ptr[j]` to `region_ptr[j+1]` gives the range of vertices for joint j
  - Total vertices: `sum_len = region_ptr[J]`

- `region_idx`: (sum_len,) - Vertex indices for all regions concatenated
  - For joint j: vertex indices are `region_idx[region_ptr[j]:region_ptr[j+1]]`

- `basis`: (sum_len*3, K) - Concatenated basis matrices
  - Each region has `len*3` rows (3 per vertex for x,y,z)
  - Each region has K columns (basis components)
  - Stacked vertically for all joints

## Usage

The C++ code loads this data using `LoadCorrectiveNPZ()` in `corrective_io.cpp`:

```cpp
smpl::PoseCorrectiveData corrective;
if (smpl::LoadCorrectiveNPZ("assets/corrective/corrective_data.npz", &corrective)) {
    // Use corrective.J, corrective.K, corrective.region_ptr, etc.
}
```

## Week 1 vs Week 2+

- **Week 1**: Data is generated and loaded, but not yet used in computation
- **Week 2+**: Will be used in `ComputePoseCorrectiveSTARStyle()` to compute
  pose-dependent vertex offsets during SMPL forward pass
