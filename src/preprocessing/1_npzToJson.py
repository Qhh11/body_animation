import numpy as np
import json
from pathlib import Path


def convert_npz_to_json(npz_path: Path):
    """
    Convert one AMASS .npz file to .json
    and save it in the same directory.
    """
    data = np.load(npz_path, allow_pickle=True)

    # Required keys (adjust if needed)
    poses = data["poses"]    # (T, 72)
    trans = data["trans"]    # (T, 3)
    betas = data["betas"]    # (10,)

    json_data = {
        "num_frames": int(poses.shape[0]),
        "poses": poses.tolist(),
        "trans": trans.tolist(),
        "betas": betas.tolist(),
    }

    json_path = npz_path.with_suffix(".json")

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved: {json_path}")


def batch_convert(root_dir: Path):
    """
    Recursively traverse root_dir and
    convert all .npz files to .json.
    """
    npz_files = list(root_dir.rglob("*.npz"))

    print(f"Found {len(npz_files)} .npz files")

    for npz_path in npz_files:
        try:
            convert_npz_to_json(npz_path)
        except Exception as e:
            print(f"[ERROR] {npz_path}: {e}")


if __name__ == "__main__":
    # Change this to your AMASS root directory
    root_directory = Path("../../Data/amass/ACCAD/ACCAD_HG/Female1General_c3d/")

    batch_convert(root_directory)
