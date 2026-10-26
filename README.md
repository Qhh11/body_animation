# Body Animation · SMPL/STAR Body Models

A C++ project exploring parametric 3D human body models, forward kinematics, and linear blend skinning.

## What is implemented

- Loading model parameters from NumPy `.npz` files using `cnpy`.
- A forward pass from shape, pose, and translation parameters to mesh vertices.
- OBJ mesh export for inspection in a 3D viewer.
- A CMake configuration using Eigen, Ceres Solver, and ZLIB.

The current `src/main.cpp` demonstrates a neutral T-pose forward pass and writes `output_tpose.obj`. It is not a complete video-to-motion application.

## Dependencies

- C++17 compiler and CMake 3.10 or newer.
- Eigen3, Ceres Solver, and ZLIB development packages.
- `cnpy`, included under `Libs/cnpy/` with its original license.
- Separately obtained STAR model parameters.

## Model data

STAR model parameters are excluded from the public migration because the bundled license prohibits redistribution. Obtain them from the model provider under its terms, then place the neutral model at:

```text
Data/star_1_1/neutral/model.npz
```

The existing `Data/star_1_1/LICENSE.txt` documents the model terms. A public code repository does not grant rights to the model data.

## Build configuration and current limitation

The upstream `main.cpp` still includes `render_obj.h`, which is absent from the current source tree. That include must be resolved before this revision will compile. No rendering function from that header is called by the current T-pose demo. The migration does not claim a successful build.

After resolving that upstream issue and installing the dependencies, the intended build layout is:

```sh
cmake -S src -B src/build
cmake --build src/build
cd src/build
./main
```

Run from `src/build` so the demo's relative model path resolves correctly. The executable location may differ for multi-configuration generators.

## Layout

```text
src/                  C++ implementation and CMake configuration
src/include/          Model and forward-pass headers
Libs/cnpy/            NumPy file loader
Data/star_1_1/        Model license; add licensed model files locally
```

## Credits

Originally developed in the [LRZ GitLab repository](https://gitlab.lrz.de/3dsmc_finalproject/body_animation). Original contributor attribution is retained in Git history.

STAR: Ahmed A. A. Osman, Timo Bolkart, and Michael J. Black, *STAR: Sparse Trained Articulated Human Body Regressor*, ECCV 2020.
