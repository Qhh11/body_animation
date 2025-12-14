# Week 2+ Implementation Roadmap

## Current Status (Week 1 Complete ✅)
- ✅ SMPL model loading from NPZ
- ✅ Corrective data generation (region-based SVD basis)
- ✅ Corrective data loading
- ✅ Ceres integration (dummy L2 solve only)
- ❌ Forward pass (stubs only)
- ❌ Real optimization with GT data

---

## What Needs to Be Done (Week 2+)

### 1. Implement SMPL Forward Pass (`smpl_forward.cpp`)

#### A. `ForwardKinematics()` - Lines 21-29
**Input**: 
- `theta` (72-dim): axis-angle rotations for 24 joints
- `J_regressor`, `kintree_table` from SMPL

**Output**: 
- `joint_transforms` (24 x 4x4): world-space transformation matrices

**Algorithm**:
```cpp
for each joint j:
    1. Convert theta[j*3:(j+1)*3] to rotation matrix R_j
    2. Get parent joint p from kintree_table
    3. Compute world transform: T_j = T_p @ [R_j | t_j]
```

#### B. `ComputePoseCorrectiveSTARStyle()` - Lines 9-17
**Input**: 
- `theta` (72-dim)
- `corrective_data`: region_ptr, region_idx, basis
- `basis_weights` (J*K = 192-dim): **optimized coefficients**

**Output**: 
- `pose_offsets` (N x 3): vertex-level displacement

**Algorithm**:
```cpp
for each joint j:
    1. Get region: indices = region_idx[region_ptr[j]:region_ptr[j+1]]
    2. Get basis: B_j = basis[region_ptr[j]*3 : region_ptr[j+1]*3, :]
    3. Get weights: w_j = basis_weights[j*K : (j+1)*K]
    4. Compute offset: delta_j = B_j @ w_j  (region_len*3 vector)
    5. Add to vertices: pose_offsets[indices] += reshape(delta_j, (region_len, 3))
```

#### C. `LinearBlendSkinning()` - Lines 33-42
**Input**: 
- `vertices` (N x 3): posed vertices
- `joint_transforms` (24 x 4x4)
- `weights` (N x 24): skinning weights

**Output**: 
- `deformed_vertices` (N x 3)

**Algorithm**:
```cpp
for each vertex v:
    v_out = Σ_j weights[v,j] * (T_j @ [v; 1])
```

---

### 2. Implement Real Optimization (`optimizer.cpp`)

#### Current State (Lines 62-137)
- Function: `SolveDummyProblem()`
- Objective: `min ||theta||² + ||beta||²`
- **NO ground truth data**

#### New Implementation (Lines 230-286, currently commented)

**Function**: `SolveSMPLFitting()`

**Inputs**:
- `target_vertices` (N x 3): **Ground Truth mesh**
- `smpl_params`, `corrective_data`: fixed model data
- `theta`, `beta`, `basis_weights`: **optimization variables**

**Objective**:
```
min_{theta, beta, w_j}  Σ_v ||SMPL(theta, beta, w_j)[v] - target[v]||²
                         + λ ||w_j||²  (regularization)
```

**Steps**:
1. Uncomment `MeshFittingCost` struct (Lines 148-220)
2. Implement residual computation in `operator()`:
   ```cpp
   v_shaped = v_template + shapedirs @ beta
   v_posed = v_shaped + ComputePoseCorrective(theta, w_j)
   transforms = ForwardKinematics(theta)
   v_final = LBS(v_posed, transforms)
   residual = v_final - target_vertex
   ```
3. Uncomment `SolveSMPLFitting()` function (Lines 230-286)
4. Add parameter blocks for `theta`, `beta`, `basis_weights`
5. Add residual blocks for each vertex
6. Optional: add regularization on `basis_weights`

---

### 3. Update Main Pipeline (`main.cpp`)

**Current** (Line 67):
```cpp
SolveDummyProblem(&theta, &beta);
```

**Future**:
```cpp
// Load GT mesh (from file or synthetic data)
Eigen::MatrixXf target_vertices = LoadTargetMesh("path/to/gt.obj");

// Initialize basis weights
Eigen::VectorXf basis_weights = Eigen::VectorXf::Zero(smpl_params.J * corrective_data.K);

// Run real fitting
SolveSMPLFitting(target_vertices, smpl_params, corrective_data,
                 &theta, &beta, &basis_weights);
```

---

## Implementation Order (Recommended)

### Phase 1: Forward Pass (No optimization yet)
1. ✅ Start with `ForwardKinematics()`
2. ✅ Then `LinearBlendSkinning()`
3. ✅ Test with fixed theta/beta, check output mesh
4. ✅ Add `ComputePoseCorrectiveSTARStyle()` last

### Phase 2: Add to Ceres
1. ✅ Implement `MeshFittingCost::operator()`
2. ✅ Test gradient with Ceres autodiff (check `summary.IsValid()`)
3. ✅ Start with small test case (10 vertices, not 6890)

### Phase 3: Full Optimization
1. ✅ Implement `SolveSMPLFitting()`
2. ✅ Create synthetic GT data (run forward pass once, save output)
3. ✅ Run fitting and verify convergence
4. ✅ Tune solver options and regularization weights

---

## Key Questions to Resolve

### Q1: Where to get GT data?
**Options**:
- A) Synthetic: run forward pass with known theta/beta, use output as GT
- B) Real scan: load 3D scan mesh from file (.obj, .ply)
- C) Multi-view: fit to 2D keypoints (different cost function)

### Q2: How to handle basis_weights initialization?
**Current**: Zero initialization is fine (pose correctives start as "off")
**Future**: Could initialize from SMPL's original posedirs via projection

### Q3: Should we optimize all vertices or subsample?
**Week 2**: Start with all 6890 vertices (full problem)
**Later**: Could subsample for speed (e.g., every 10th vertex)

---

## Debugging Tips

1. **Forward pass validation**:
   ```cpp
   // Set theta=0, beta=0 → should get v_template back
   ```

2. **Gradient checking**:
   ```cpp
   ceres::NumericDiffOptions diff_options;
   ceres::GradientChecker checker(...);
   ```

3. **Visualization**:
   - Export vertices to `.obj` file after each step
   - Use MeshLab/Blender to visualize

---

## Files Modified

| File | Status | Changes Needed |
|------|--------|----------------|
| `smpl_forward.cpp` | ❌ Stubs | Implement all 3 functions |
| `optimizer.cpp` | ⚠️ Partial | Uncomment + implement MeshFittingCost |
| `main.cpp` | ✅ Ready | Add GT loading + call SolveSMPLFitting |
| `optimizer.h` | ❌ Missing | Add declaration for SolveSMPLFitting |

---

## See Detailed TODOs in Code

- **optimizer.cpp**: Lines 7-17, 36-61, 81-107, 115-120, 139-286
- **smpl_forward.cpp**: Lines 7-16, 19-28, 31-42

