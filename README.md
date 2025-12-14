# SMPL + Ceres Optimization Pipeline

A C++17 implementation of SMPL (Skinned Multi-Person Linear model) with Ceres-based optimization and pose-corrective blendshapes.

## Week 1 Status

This is the **Week 1 scaffold** with:

✅ Docker environment setup (Ubuntu 22.04 + CMake + Eigen + Ceres)
✅ SMPL model loading from NPZ
✅ Pose-corrective data generation and loading
✅ Dummy Ceres optimization (L2 priors only)
✅ Project compiles and runs successfully

❌ Forward kinematics (stub only - Week 2+)
❌ Linear blend skinning (stub only - Week 2+)
❌ Pose-corrective computation (stub only - Week 2+)

## Project Structure

```
.
├── Dockerfile                      # Ubuntu 22.04 + dependencies
├── docker-compose.yml              # Mount repo to /workspace
├── CMakeLists.txt                  # C++17 build configuration
│
├── include/                        # Header files
│   ├── smpl.h                      # SMPL data structures & loaders
│   ├── optimizer.h                 # Ceres optimization wrapper
│   └── logging.h                   # Logging macros (glog)
│
├── smpl_loader.cpp                 # SMPL NPZ loader (FULL ✅)
├── corrective_loader.cpp           # Corrective data loader (FULL ✅)
├── optimizer.cpp                   # Dummy Ceres solve (FULL ✅)
├── smpl_forward.cpp                # FK/LBS/corrective stubs (TODO ❌)
├── main.cpp                        # Main application
│
├── tools/
│   └── build_corrective_data.py   # Generate corrective data (FULL ✅)
│
└── assets/
    ├── smpl/                       # SMPL model.npz (user must provide)
    └── corrective/                 # Generated corrective_data.npz
```

## Prerequisites

1. **Docker** installed on your system
2. **SMPL model** (`model.npz`) - must be obtained legally from https://smpl.is.tue.mpg.de/

## Quick Start

### 1. Build Docker Image

```bash
docker compose build
```

### 2. Start Container

```bash
docker compose up -d
docker exec -it smpl_ceres_dev bash
```

Or use VSCode "Attach to Running Container" feature.

### 3. Place SMPL Model

Download `model.npz` from the official SMPL website and place it in:

```
assets/smpl/model.npz
```

**Important**: Do not commit this file (it's in `.gitignore`)

### 4. Generate Corrective Data

```bash
python3 tools/build_corrective_data.py \
    --smpl_npz assets/smpl/model.npz \
    --out_npz assets/corrective/corrective_data.npz
```

### 5. Build and Run

```bash
# Inside container
mkdir -p build && cd build
cmake .. -GNinja
ninja

# Run pipeline
./smpl_pipeline \
    --smpl_npz=../assets/smpl/model.npz \
    --corrective_npz=../assets/corrective/corrective_data.npz
```

## Expected Output (Week 1)

```
=== SMPL + Ceres Pipeline (Week 1) ===
Loading SMPL model from: assets/smpl/model.npz
SMPL model loaded successfully:
  N (vertices) = 6890
  J (joints) = 24
  n_beta (shape params) = 10
  D_pose (pose blendshape dim) = 207

Loading corrective data from: assets/corrective/corrective_data.npz
Corrective data loaded successfully:
  J (joints) = 24
  K (basis components) = 8
  sum_len (total vertices) = 12000

SMPL Model Dimensions:
  N (vertices):     6890
  J (joints):       24
  n_beta (shape):   10
  D_pose (pose BS): 207

Corrective Data Dimensions:
  J (joints):        24
  K (basis comps):   8
  sum_len (verts):   12000

Initial Parameters:
  theta size: 72
  beta size:  10

Running dummy Ceres optimization...
[Ceres solver output]

Optimized Parameters:
  theta norm: 0.xxxx
  beta norm:  0.xxxx

=== Week 1 Pipeline Complete ===
All systems operational!
Corrective data loaded and ready for Week 2+
```

## Development

### File Descriptions

- **smpl_loader.cpp**: Loads SMPL NPZ with validation of all required keys
- **corrective_loader.cpp**: Loads CSR-packed corrective data
- **optimizer.cpp**: Sets up and solves Ceres problem (L2 priors only in Week 1)
- **smpl_forward.cpp**: Contains TODO stubs for:
  - `ComputePoseCorrectiveSTARStyle()` - Region-based pose correction
  - `ForwardKinematics()` - Joint transforms from pose
  - `LinearBlendSkinning()` - Vertex deformation

### Building with VSCode

1. Attach to the running container
2. Install C++ extensions in container
3. CMake Tools will auto-detect configuration
4. Build target: `smpl_pipeline`

## Next Steps (Week 2+)

1. Implement `ForwardKinematics()` - rodrigues rotation + kinematic chain
2. Implement `LinearBlendSkinning()` - weighted transformation blending
3. Implement `ComputePoseCorrectiveSTARStyle()` - region-based pose correction
4. Add Ceres residuals for actual fitting objectives
5. Add visualization output (OBJ files)

## Dependencies

- CMake 3.16+
- C++17 compiler (GCC/Clang)
- Eigen3 (linear algebra)
- Ceres Solver (optimization)
- glog, gflags (logging/flags)
- SuiteSparse (sparse linear algebra)
- cnpy (NPZ file I/O)
- Python 3 + NumPy (for tools)

## License

This is educational/research code. The SMPL model itself is subject to its own license terms.

## References

- SMPL: https://smpl.is.tue.mpg.de/
- STAR: https://star.is.tue.mpg.de/
- Ceres Solver: http://ceres-solver.org/
