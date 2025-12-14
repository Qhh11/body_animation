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
    └── smpl_model.cpp      # Model loading and logic