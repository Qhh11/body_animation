import subprocess
import numpy as np
from pathlib import Path
from SMPL_python_v_1_1.smpl.smpl_webuser.serialization import load_model

from pathlib import Path
import numpy as np

"""
!!!Please put all the .pkl in one folder. This script will convert all the .pkl in a folder to .obj and .png
"""

def obj_to_png(obj_path: Path):
    """
    Call obj2png.py to convert one OBJ file to PNG.
    The PNG will be saved next to the OBJ file.
    """
    obj_path = Path(obj_path).resolve()

    obj2png_script = Path(__file__).parent / "obj2png" / "src" / "obj2png.py"

    cmd = [
        "python",
        str(obj2png_script),
        "-i",
        str(obj_path),
        "-o",
        str(obj_path.with_suffix(".png")),
    ]

    subprocess.run(cmd, check=True)

def write_obj(path: Path, verts, faces):
    """
    Write a single mesh to an OBJ file.

    Parameters
    ----------
    path : Path
        Output OBJ file path.
    verts : ndarray (V, 3)
        Mesh vertices.
    faces : ndarray (F, 3)
        Mesh faces (0-based indexing).
    """
    # Ensure the output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as fp:
        # Write vertices
        for v in verts:
            fp.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces (OBJ format uses 1-based indexing)
        for f in (faces + 1):
            fp.write(f"f {int(f[0])} {int(f[1])} {int(f[2])}\n")


def convert_npz_to_obj_per_frame(
    npz_path: Path,
    m,
    pose_dim: int = 72,
    betas_dim: int = 10,
    poses_key: str = "poses",
    trans_key: str = "trans",
    betas_key: str = "betas",
):
    """
    Convert an AMASS-style .npz file into per-frame OBJ meshes.

    For an input file:
        /path/to/sequence.npz

    The output will be:
        /path/to/sequence/
            sequence_frame000000.obj
            sequence_frame000001.obj
            ...

    Parameters
    ----------
    npz_path : Path
        Path to the input .npz motion file.
    m : SMPL model instance
        Initialized SMPL model with attributes:
        - m.pose
        - m.trans
        - m.betas
        - m.r (vertices)
        - m.f (faces)
    pose_dim : int
        Number of pose parameters used by the SMPL model (default: 72).
    betas_dim : int
        Number of shape coefficients used by the SMPL model (default: 10).
    poses_key, trans_key, betas_key : str
        Keys used to access data inside the .npz file.
    """
    npz_path = Path(npz_path)

    # Create an output directory with the same name as the npz file
    # e.g. sequence.npz -> sequence/
    out_dir = npz_path.with_suffix("")
    out_dir.mkdir(exist_ok=True)

    # Load motion data
    data = np.load(npz_path, allow_pickle=True)
    poses = data[poses_key]   # (N, 156)
    trans = data[trans_key]   # (N, 3)
    betas = data[betas_key]   # (16,)
    print("1")
    # Trim pose and shape parameters to match SMPL input dimensions
    poses72 = poses[:, :pose_dim]     # (N, 72)
    betas10 = betas[:betas_dim]       # (10,)
    print("2")
    # Set shape parameters once (they are constant across frames)
    m.betas[:] = 0.0
    m.betas[:betas_dim] = betas10
    print("3")
    # Generate one OBJ file per frame
    num_frames = poses72.shape[0]
    for t in range(num_frames):
        # Set pose and global translation for the current frame
        m.pose[:] = poses72[t]
        m.trans[:] = trans[t]

        # Output file name: <npz_stem>_frameXXXXXX.obj
        out_obj = out_dir / f"{npz_path.stem}_frame{t:06d}.obj"

        # Write mesh to disk
        write_obj(out_obj, m.r, m.f)
        obj_to_png(out_obj)

    print(f"✔ Wrote {num_frames} OBJ files to: {out_dir}")


def batch_convert(root_dir: Path, m):
    """
    Recursively traverse root_dir and
    convert all .npz files to .json.
    """
    npz_files = list(root_dir.rglob("*.npz"))

    print(f"Found {len(npz_files)} .npz files")

    for npz_path in npz_files:
        try:
            convert_npz_to_obj_per_frame(npz_path, m)
        except Exception as e:
            print(f"[ERROR] {npz_path}: {e}")

if __name__ == "__main__":
    m = load_model( '../../Data/models/SMPL_python_v_1_1_4_0/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl' )
    root_directory = Path("../../Data/amass/ACCAD/ACCAD_HG/test/")
    batch_convert(root_directory, m)
    