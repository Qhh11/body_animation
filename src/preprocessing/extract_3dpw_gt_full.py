#!/usr/bin/env python3
"""
Extract full 3DPW ground truth data (joints, poses, betas, 2D keypoints) from pkl file.
Transforms to camera coordinate system.

Output format:
- gt_joints.txt: 24*3 values per line (joint positions in camera space)
- gt_joints_world.txt: 24*3 values per line (joint positions in world space)
- gt_poses.txt: 72 values per line (SMPL theta parameters, root in camera space)
- gt_poses_world.txt: 72 values per line (SMPL theta parameters, original world space)
- gt_betas.txt: 10 values (single line, shared across frames)
- gt_trans.txt: 3 values per line (translation in camera space)
- gt_trans_world.txt: 3 values per line (translation in world space)
- gt_2d.txt: 2D keypoints for reprojection evaluation
- cam_intrinsics.txt: camera intrinsics (fx, fy, cx, cy)
- cam_poses.txt: 12 values per line (3x4 camera extrinsic matrix, flattened row-major)

Usage:
    python extract_3dpw_gt_full.py --pkl /path/to/sequence.pkl --output_dir /path/to/output
"""
import numpy as np
import pickle
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def load_3dpw_pkl(pkl_path):
    """Load 3DPW sequence pkl file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def extract_full_gt(data, person_idx=0):
    """
    Extract full GT data and transform to camera coordinate system.

    Returns:
        joints_camera: (num_frames, 24, 3)
        joints_world: (num_frames, 24, 3)
        poses_camera: (num_frames, 72) - theta with root transformed to camera
        poses_world: (num_frames, 72) - original theta in world space
        trans_camera: (num_frames, 3) - translation in camera space
        trans_world: (num_frames, 3) - original translation in world space
        betas: (10,)
        poses2d: (num_frames, 18, 2) or None - 2D keypoints
        cam_intrinsics: (3, 3) or None
        cam_poses: (num_frames, 4, 4) - camera extrinsic matrices
    """
    # Joint positions: (num_frames, 72) -> (num_frames, 24, 3)
    joint_positions = data['jointPositions'][person_idx]
    num_frames = joint_positions.shape[0]
    joints_world = joint_positions.reshape(num_frames, 24, 3)

    # SMPL poses: (num_frames, 72)
    poses = data['poses'][person_idx]  # axis-angle

    # SMPL betas: (10,)
    betas = data['betas'][person_idx]

    # Translation: (num_frames, 3)
    trans = data['trans'][person_idx]

    # Camera poses: (num_frames, 4, 4) - world to camera transform
    cam_poses = data['cam_poses']

    # Camera intrinsics
    cam_intrinsics = data.get('cam_intrinsics', None)

    # 2D poses: (num_frames, 3, 18) -> (num_frames, 18, 3) where last dim is (x, y, conf)
    poses2d = None
    if 'poses2d' in data and data['poses2d'][person_idx] is not None:
        p2d = data['poses2d'][person_idx]  # (num_frames, 3, 18)
        poses2d = np.transpose(p2d, (0, 2, 1))  # (num_frames, 18, 3)

    # Transform joints, poses, trans from world to camera coordinate
    joints_camera = []
    poses_camera = []
    trans_camera = []

    for f in range(num_frames):
        # cam_pose is world-to-camera transform
        R_cam = cam_poses[f, :3, :3]  # 3x3 rotation
        t_cam = cam_poses[f, :3, 3]   # 3x1 translation

        # Transform joints: p_cam = R @ p_world + t
        joints_f = joints_world[f]  # (24, 3)
        joints_cam_f = (R_cam @ joints_f.T).T + t_cam  # (24, 3)

        # Flip Y axis to match our SMPL convention (Y+ = up)
        # 3DPW camera coord has Y+ = down
        joints_cam_f[:, 1] = -joints_cam_f[:, 1]
        joints_camera.append(joints_cam_f)

        # Transform translation
        trans_f = trans[f]
        trans_cam_f = R_cam @ trans_f + t_cam
        trans_cam_f[1] = -trans_cam_f[1]  # flip Y
        trans_camera.append(trans_cam_f)

        # Transform root orientation to camera space
        pose_f = poses[f].copy()
        root_aa = pose_f[:3]  # root axis-angle

        # Convert to rotation matrix
        R_root_world = R.from_rotvec(root_aa).as_matrix()

        # Transform root rotation to camera space
        R_root_cam = R_cam @ R_root_world

        # Flip Y axis (negate Y row)
        flip_y = np.diag([1, -1, 1])
        R_root_cam = flip_y @ R_root_cam

        # Convert back to axis-angle
        root_aa_cam = R.from_matrix(R_root_cam).as_rotvec()
        pose_f[:3] = root_aa_cam

        poses_camera.append(pose_f)

    joints_camera = np.array(joints_camera)
    poses_camera = np.array(poses_camera)
    trans_camera = np.array(trans_camera)

    # Also return world-space data and cam_poses for optimization with extrinsics
    return (joints_camera, joints_world,
            poses_camera, poses,  # poses is original world-space
            trans_camera, trans,  # trans is original world-space
            betas, poses2d, cam_intrinsics, cam_poses)


def save_txt(data, path, fmt='%.6f'):
    """Save numpy array to txt file."""
    if data is None:
        return
    np.savetxt(path, data.reshape(data.shape[0], -1) if len(data.shape) > 1 else data.reshape(1, -1), fmt=fmt)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description='Extract full 3DPW GT data')
    parser.add_argument('--pkl', required=True, help='Path to 3DPW sequence pkl')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--person', type=int, default=0, help='Person index (default: 0)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.pkl}...")
    data = load_3dpw_pkl(args.pkl)

    # Print available keys
    print(f"Available keys: {list(data.keys())}")

    print(f"Extracting full GT for person {args.person}...")
    (joints_cam, joints_world,
     poses_cam, poses_world,
     trans_cam, trans_world,
     betas, poses2d, cam_intrinsics, cam_poses) = extract_full_gt(data, args.person)

    print(f"  Joints shape: {joints_cam.shape}")
    print(f"  Poses shape: {poses_cam.shape}")
    print(f"  Trans shape: {trans_cam.shape}")
    print(f"  Betas shape: {betas.shape}")
    print(f"  Cam poses shape: {cam_poses.shape}")
    if poses2d is not None:
        print(f"  2D poses shape: {poses2d.shape}")

    # Save camera-space outputs (for evaluation with moving camera assumption)
    save_txt(joints_cam, output_dir / 'gt_joints.txt')
    save_txt(poses_cam, output_dir / 'gt_poses.txt')
    save_txt(trans_cam, output_dir / 'gt_trans.txt')

    # Save world-space outputs (for optimization with camera extrinsics)
    save_txt(joints_world, output_dir / 'gt_joints_world.txt')
    save_txt(poses_world, output_dir / 'gt_poses_world.txt')
    save_txt(trans_world, output_dir / 'gt_trans_world.txt')

    save_txt(betas.reshape(1, -1), output_dir / 'gt_betas.txt')

    # Save camera extrinsics (4x4 -> 3x4, flattened to 12 values per line)
    # Format: R[0,0] R[0,1] R[0,2] t[0] R[1,0] R[1,1] R[1,2] t[1] R[2,0] R[2,1] R[2,2] t[2]
    cam_poses_flat = cam_poses[:, :3, :].reshape(cam_poses.shape[0], 12)
    save_txt(cam_poses_flat, output_dir / 'cam_poses.txt', fmt='%.10f')

    # Save 2D poses (for reprojection error)
    if poses2d is not None:
        # Format: frame_idx joint_idx x y confidence
        with open(output_dir / 'gt_2d.txt', 'w') as f:
            f.write("# frame joint_idx x y confidence\n")
            for frame_idx in range(poses2d.shape[0]):
                for joint_idx in range(poses2d.shape[1]):
                    x, y, conf = poses2d[frame_idx, joint_idx]
                    if conf > 0:
                        f.write(f"{frame_idx} {joint_idx} {x:.2f} {y:.2f} {conf:.4f}\n")
        print(f"Saved {output_dir / 'gt_2d.txt'}")

    # Save camera intrinsics if available
    if cam_intrinsics is not None:
        fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
        cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]
        with open(output_dir / 'cam_intrinsics.txt', 'w') as f:
            f.write(f"{fx} {fy} {cx} {cy}\n")
        print(f"Saved camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    print(f"\nDone! Files saved to {output_dir}")


if __name__ == '__main__':
    main()
