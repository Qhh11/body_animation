# SMPL/STAR Body Model Optimization Pipeline

This project implements a C++ pipeline for 3D human body reconstruction using the SMPL/STAR model. It features a complete Forward Kinematics (FK) and Linear Blend Skinning (LBS) engine implemented from scratch, integrated with the Ceres Solver for optimization tasks.

## 📂 Project Structure

The project is organized as follows:

```text
/workspace/Exercises/
├── Libs/                   # External Libraries
│   ├── cnpy/               # Library for reading/writing .npz files
│   │   ├── cnpy.cpp
│   │   ├── cnpy.h
│   │   └── CMakeLists.txt 
│   ├── Eigen/              # Linear algebra library (Standard in environment)
│   └── ...
├── Data/                   # Model Data
│   └── star_1_1/           # STAR/SMPL model files
│       └── neutral/
│           └── model.npz   # The body model parameters
└── src/                    # Source Code (Working Directory)
    ├── build/              # Build artifacts
    ├── include/            # Header files
    │   ├── smpl_model.h    # Model data structure definition
    │   └── smpl_math.h     # (Optional) Math & FK/LBS implementations
    ├── CMakeLists.txt      # CMake build configuration
    ├── main.cpp            # Entry point and pipeline logic
    ├── smpl_model.cpp      # Model loading and logic
    ├── optimizer.cpp       # Ceres-based SMPL parameter optimization
    ├── include/
    │   ├── cost_functions.h  # Ceres cost functions (reprojection, regularization)
    │   └── optimizer.h       # Optimizer interface
    └── visualize_optimization.py  # Python visualization script
```

## 🚀 Quick Start

### Build and Run
```bash
cd src/build
cmake ..
make
./main
```

This will:
1. Load the SMPL/STAR model from `Data/star_1_1/neutral/model.npz`
2. Run a synthetic optimization demo (test with artificial data)
3. Generate output files in `src/build/` directory

### Visualize Results
After running the optimizer, visualize the results:

```bash
# Install Python dependencies (if needed)
pip3 install numpy matplotlib

# Run visualization script
cd src/build
python3 ../../visualize_optimization.py
```

This generates:
- `landmarks_2d.png` - 2D landmark comparison (GT vs optimized)
- `reprojection_errors.png` - Per-joint reprojection error chart
- `mesh_comparison_3d.png` - 3D mesh comparison (before/after/GT)
- `vertex_errors.png` - Per-vertex error analysis

You can also view the `.obj` files in MeshLab, Blender, or any 3D viewer.



## ✨ Features

- ✅ SMPL/STAR forward kinematics and linear blend skinning
- ✅ Ceres Solver integration for parameter optimization
- ✅ 2D landmark reprojection cost function
- ✅ Shape and pose regularization
- ✅ Synthetic test data generation for validation
- ✅ Python visualization scripts
- 🔄 Real dataset support (AMASS/OpenPose)