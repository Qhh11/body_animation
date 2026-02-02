#!/usr/bin/env python3
"""
Extract 3DPW ground truth joint positions from pkl file.
Transforms joints from world coordinate to camera coordinate.

Usage:
    python extract_3dpw_gt.py --pkl /path/to/sequence.pkl --output /path/to/gt_joints.txt
"""
import numpy as np
import pickle
import argparse


def load_3dpw_pkl(pkl_path):
    """Load 3DPW sequence pkl file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def extract_gt_joints(data, person_idx=0):
    """
    Extract GT joint positions and transform to camera coordinate system.

    3DPW stores joints in world coordinate. We need to transform them to camera space
    using cam_poses (4x4 extrinsic matrices).
    """
    # Joint positions: (num_frames, 72) -> (num_frames, 24, 3)
    joint_positions = data['jointPositions'][person_idx]
    num_frames = joint_positions.shape[0]
    joints_world = joint_positions.reshape(num_frames, 24, 3)

    # Camera poses: (num_frames, 4, 4) - world to camera transform
    cam_poses = data['cam_poses']

    # Transform joints from world to camera coordinate
    joints_camera = []
    for f in range(num_frames):
        # cam_pose is world-to-camera transform
        R = cam_poses[f, :3, :3]  # 3x3 rotation
        t = cam_poses[f, :3, 3]   # 3x1 translation

        # Transform: p_cam = R @ p_world + t
        joints_f = joints_world[f]  # (24, 3)
        joints_cam_f = (R @ joints_f.T).T + t  # (24, 3)

        # Flip Y axis to match our SMPL convention (Y+ = up)
        # 3DPW camera coord has Y+ = down
        joints_cam_f[:, 1] = -joints_cam_f[:, 1]

        joints_camera.append(joints_cam_f)

    return np.array(joints_camera)  # (num_frames, 24, 3)


def save_joints_txt(joints, output_path):
    """
    Save joints to txt file.
    Format: one frame per line, 72 values (24 joints * 3 coords)
    """
    num_frames = joints.shape[0]
    with open(output_path, 'w') as f:
        for frame in range(num_frames):
            values = joints[frame].flatten()  # 72 values
            line = ' '.join(f'{v:.6f}' for v in values)
            f.write(line + '\n')
    print(f"Saved {num_frames} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract 3DPW GT joints')
    parser.add_argument('--pkl', required=True, help='Path to 3DPW sequence pkl')
    parser.add_argument('--output', required=True, help='Output txt file path')
    parser.add_argument('--person', type=int, default=0, help='Person index (default: 0)')
    args = parser.parse_args()

    print(f"Loading {args.pkl}...")
    data = load_3dpw_pkl(args.pkl)

    print(f"Extracting GT joints for person {args.person}...")
    joints_camera = extract_gt_joints(data, args.person)
    print(f"  Shape: {joints_camera.shape}")

    # Print some stats
    print(f"  Joint position range:")
    print(f"    X: {joints_camera[:,:,0].min():.3f} to {joints_camera[:,:,0].max():.3f}")
    print(f"    Y: {joints_camera[:,:,1].min():.3f} to {joints_camera[:,:,1].max():.3f}")
    print(f"    Z: {joints_camera[:,:,2].min():.3f} to {joints_camera[:,:,2].max():.3f}")

    save_joints_txt(joints_camera, args.output)


if __name__ == '__main__':
    main()
