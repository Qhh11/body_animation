# Preprocessing: AMASS (.npz) → SMPL meshes (.obj) → renders (.png)

This folder contains scripts to convert AMASS-style motion files into per-frame SMPL meshes,
and optionally render the meshes to PNG images.

## 1) Environment (minimal: npz → obj)

```bash
conda create -n sav3d python=3.9 -y
conda activate sav3d

pip install -U pip setuptools wheel
pip install numpy scipy
pip install --no-build-isolation git+https://github.com/mattloper/chumpy.git
