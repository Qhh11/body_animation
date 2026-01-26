#!/usr/bin/env python3
"""
Generate depth values at OpenPose keypoint locations from Sapiens depth map.

Usage:
    python generate_depth_from_sapiens.py \
        --depth_npy /path/to/sapiens_depth.npy \
        --keypoints /path/to/openpose_keypoints.json \
        --output /path/to/output_depth.txt
"""
import numpy as np
import json
import argparse
from pathlib import Path


def load_openpose_keypoints(json_path):
    """Load OpenPose keypoints from JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    if not data['people']:
        raise ValueError("No people detected in OpenPose output")

    # Take first person
    kp = data['people'][0]['pose_keypoints_2d']

    # Reshape to (25, 3) - x, y, confidence
    keypoints = np.array(kp).reshape(-1, 3)
    return keypoints


def sample_depth_at_keypoints(depth_map, keypoints):
    """
    Sample depth values at keypoint locations.

    Args:
        depth_map: (H, W) numpy array of depth values
        keypoints: (N, 3) array where each row is [x, y, confidence]

    Returns:
        List of (keypoint_idx, depth_value) for valid keypoints
    """
    H, W = depth_map.shape
    results = []

    for idx, (x, y, conf) in enumerate(keypoints):
        if conf < 0.1:  # Skip low confidence keypoints
            continue

        # Convert to integer pixel coordinates
        px, py = int(round(x)), int(round(y))

        # Check bounds
        if 0 <= px < W and 0 <= py < H:
            depth = depth_map[py, px]
            if depth > 0:  # Valid depth
                results.append((idx, float(depth)))

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate depth at OpenPose keypoints from Sapiens depth')
    parser.add_argument('--depth_npy', required=True, help='Path to Sapiens depth map (.npy)')
    parser.add_argument('--keypoints', required=True, help='Path to OpenPose keypoints JSON')
    parser.add_argument('--output', required=True, help='Output path for depth text file')
    args = parser.parse_args()

    # Load depth map
    depth_map = np.load(args.depth_npy)
    print(f"Loaded depth map: shape={depth_map.shape}, dtype={depth_map.dtype}")
    print(f"  Depth range: {depth_map.min():.4f} to {depth_map.max():.4f}")

    # Load keypoints
    keypoints = load_openpose_keypoints(args.keypoints)
    print(f"Loaded {len(keypoints)} keypoints")

    # Sample depth at keypoints
    kp_depths = sample_depth_at_keypoints(depth_map, keypoints)
    print(f"Sampled depth at {len(kp_depths)} valid keypoints")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for idx, depth in kp_depths:
            f.write(f"{idx} {depth}\n")

    print(f"Saved to {output_path}")

    # Print summary
    print("\nKeypoint depths:")
    openpose_names = [
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
        "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
        "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
        "REye", "LEye", "REar", "LEar", "LBigToe",
        "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
    ]
    for idx, depth in kp_depths:
        name = openpose_names[idx] if idx < len(openpose_names) else f"Joint{idx}"
        print(f"  {idx:2d} ({name:12s}): {depth:.4f}")


if __name__ == '__main__':
    main()
