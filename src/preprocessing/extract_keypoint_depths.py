#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract depth values at OpenPose keypoint locations from Sapiens depth map.

Usage:
    python extract_keypoint_depths.py \
        --openpose_json path/to/keypoints.json \
        --depth_npz path/to/depth.npz \
        --output_json path/to/output_depth.json

Or batch processing:
    python extract_keypoint_depths.py \
        --openpose_dir path/to/openpose_jsons/ \
        --depth_dir path/to/depth_npzs/ \
        --output_dir path/to/output_depths/
"""

import argparse
import json
import numpy as np
from pathlib import Path


def load_openpose_keypoints(json_path: str) -> list:
    """
    Load OpenPose keypoints from JSON file.
    Returns list of (x, y, confidence, joint_id) for valid keypoints.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'people' not in data or len(data['people']) == 0:
        return []

    # Find best person (largest bounding box with high confidence)
    best_idx = 0
    best_score = -1.0

    for i, person in enumerate(data['people']):
        if 'pose_keypoints_2d' not in person:
            continue
        kp = person['pose_keypoints_2d']
        if len(kp) < 75:
            continue

        # Calculate score based on bbox size and confidence
        xs, ys, confs = [], [], []
        for j in range(25):
            x, y, c = kp[j*3], kp[j*3+1], kp[j*3+2]
            if c > 0.2:
                xs.append(x)
                ys.append(y)
                confs.append(c)

        if len(xs) < 6:
            continue

        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        area = max(1, w) * max(1, h)
        mean_conf = sum(confs) / len(confs)
        score = area * mean_conf * len(confs)

        if score > best_score:
            best_score = score
            best_idx = i

    # Extract keypoints from best person
    keypoints = []
    kp = data['people'][best_idx]['pose_keypoints_2d']

    for joint_id in range(25):
        x = kp[joint_id * 3]
        y = kp[joint_id * 3 + 1]
        conf = kp[joint_id * 3 + 2]

        if conf > 0.05:  # Keep low threshold, let optimizer filter
            keypoints.append({
                'joint_id': joint_id,
                'x': float(x),
                'y': float(y),
                'confidence': float(conf)
            })

    return keypoints


def load_depth_map(depth_path: str) -> np.ndarray:
    """
    Load depth map from NPY or NPZ file.
    """
    path = Path(depth_path)

    if path.suffix == '.npy':
        # Direct numpy array
        return np.load(depth_path)

    elif path.suffix == '.npz':
        # NPZ archive
        data = np.load(depth_path)

        # Try common key names
        for key in ['depth', 'depth_map', 'disparity', 'data']:
            if key in data:
                return data[key]

        # If only one array, use it
        keys = list(data.keys())
        if len(keys) == 1:
            return data[keys[0]]

        print(f"[Warning] Available keys in NPZ: {keys}")
        raise KeyError(f"Cannot find depth data in {depth_path}. Available keys: {keys}")

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def sample_depth_at_keypoints(
    keypoints: list,
    depth_map: np.ndarray,
    image_size: tuple = None,  # (width, height) of original image
    sample_radius: int = 2
) -> list:
    """
    Sample depth values at keypoint locations.
    Uses a small neighborhood average to be more robust.

    Args:
        keypoints: List of keypoint dicts with x, y, confidence, joint_id
        depth_map: 2D numpy array of depth values
        image_size: (width, height) of original image for scaling. If None, assumes same size as depth_map.
        sample_radius: Radius for neighborhood sampling (0 = single pixel)

    Returns:
        List of keypoint depth dicts
    """
    depth_h, depth_w = depth_map.shape[:2]
    result = []

    # Calculate scale factors if image size differs from depth map size
    if image_size is not None:
        img_w, img_h = image_size
        scale_x = depth_w / img_w
        scale_y = depth_h / img_h
        print(f"[Info] Scaling keypoints: image ({img_w}x{img_h}) -> depth ({depth_w}x{depth_h}), scale=({scale_x:.3f}, {scale_y:.3f})")
    else:
        scale_x, scale_y = 1.0, 1.0

    for kp in keypoints:
        # Scale keypoint coordinates to depth map resolution
        x = int(round(kp['x'] * scale_x))
        y = int(round(kp['y'] * scale_y))

        # Skip if out of bounds
        if x < 0 or x >= depth_w or y < 0 or y >= depth_h:
            print(f"[Warn] Keypoint {kp['joint_id']} out of bounds: ({x}, {y})")
            continue

        # Sample neighborhood
        x_min = max(0, x - sample_radius)
        x_max = min(depth_w, x + sample_radius + 1)
        y_min = max(0, y - sample_radius)
        y_max = min(depth_h, y + sample_radius + 1)

        patch = depth_map[y_min:y_max, x_min:x_max]

        # Use median to be robust to outliers
        if patch.size > 0:
            depth_value = float(np.median(patch))
        else:
            depth_value = float(depth_map[y, x])

        result.append({
            'joint_id': kp['joint_id'],
            'depth': depth_value,
            'confidence': kp['confidence']
        })

    return result


def save_depth_json(depths: list, output_path: str, frame_id: str = None):
    """Save depth data to JSON file."""
    output = {
        'keypoint_depths': depths
    }
    if frame_id:
        output['frame'] = frame_id

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[OK] Saved {len(depths)} keypoint depths to {output_path}")


def process_single(openpose_json: str, depth_file: str, output_json: str, image_size: tuple = None):
    """
    Process a single frame.

    Args:
        openpose_json: Path to OpenPose keypoints JSON
        depth_file: Path to depth map (NPY or NPZ)
        output_json: Path to output depth JSON
        image_size: (width, height) of original image. If None, will try to infer.
    """
    # Load OpenPose keypoints
    keypoints = load_openpose_keypoints(openpose_json)
    if not keypoints:
        print(f"[Error] No keypoints found in {openpose_json}")
        return False

    print(f"[Info] Loaded {len(keypoints)} keypoints from {openpose_json}")

    # Load depth map
    depth_map = load_depth_map(depth_file)
    print(f"[Info] Loaded depth map: shape={depth_map.shape}, range=[{depth_map.min():.3f}, {depth_map.max():.3f}]")

    # Try to infer image size from keypoint coordinates if not provided
    if image_size is None:
        max_x = max(kp['x'] for kp in keypoints)
        max_y = max(kp['y'] for kp in keypoints)
        # Check if keypoints are way larger than depth map (need scaling)
        depth_h, depth_w = depth_map.shape[:2]
        if max_x > depth_w * 1.5 or max_y > depth_h * 1.5:
            # Likely need scaling - assume common video resolutions
            if max_x > 1000:
                image_size = (1280, 720)  # 720p
            elif max_x > 600:
                image_size = (640, 480)   # VGA
            print(f"[Info] Auto-detected image size: {image_size}")

    # Sample depths at keypoint locations
    depths = sample_depth_at_keypoints(keypoints, depth_map, image_size)

    # Save output
    save_depth_json(depths, output_json)
    return True


def process_batch(openpose_dir: str, depth_dir: str, output_dir: str):
    """Process multiple frames."""
    openpose_path = Path(openpose_dir)
    depth_path = Path(depth_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all OpenPose JSON files
    json_files = sorted(openpose_path.glob("*_keypoints.json"))
    print(f"[Info] Found {len(json_files)} OpenPose JSON files")

    success_count = 0
    for json_file in json_files:
        # Extract frame identifier (e.g., "dance_000000000120" from "dance_000000000120_keypoints.json")
        frame_id = json_file.stem.replace("_keypoints", "")

        # Extract frame number (e.g., "000000000120" from "dance_000000000120")
        # Handle different naming conventions
        frame_num = frame_id.split("_")[-1] if "_" in frame_id else frame_id

        # Find corresponding depth file (NPY or NPZ)
        # Try common naming patterns
        depth_candidates = [
            depth_path / f"{frame_id}.npy",
            depth_path / f"{frame_id}_depth.npy",
            depth_path / f"frame_{frame_id}_depth.npy",
            depth_path / f"frame_{frame_num}_depth.npy",  # e.g., frame_000000000120_depth.npy
            depth_path / f"{frame_num}_depth.npy",
            depth_path / f"{frame_id}.npz",
            depth_path / f"{frame_id}_depth.npz",
            depth_path / f"depth_{frame_id}.npz",
            depth_path / f"frame_{frame_num}_depth.npz",
        ]

        depth_file = None
        for candidate in depth_candidates:
            if candidate.exists():
                depth_file = candidate
                break

        if depth_file is None:
            print(f"[Warn] No depth file found for {frame_id}")
            continue

        # Output path
        output_file = output_path / f"{frame_id}_depth.json"

        # Process
        if process_single(str(json_file), str(depth_file), str(output_file)):
            success_count += 1

    print(f"\n[Done] Processed {success_count}/{len(json_files)} frames")


def main():
    parser = argparse.ArgumentParser(description="Extract keypoint depths from Sapiens depth maps")

    # Single file mode
    parser.add_argument("--openpose_json", type=str, help="Path to OpenPose keypoints JSON")
    parser.add_argument("--depth_file", type=str, help="Path to Sapiens depth file (NPY or NPZ)")
    parser.add_argument("--output_json", type=str, help="Path to output depth JSON")

    # Batch mode
    parser.add_argument("--openpose_dir", type=str, help="Directory containing OpenPose JSONs")
    parser.add_argument("--depth_dir", type=str, help="Directory containing depth files")
    parser.add_argument("--output_dir", type=str, help="Directory for output depth JSONs")

    # Optional
    parser.add_argument("--image_width", type=int, default=496, help="Original image width (default: 496)")
    parser.add_argument("--image_height", type=int, default=640, help="Original image height (default: 640)")

    args = parser.parse_args()

    image_size = (args.image_width, args.image_height)

    # Single file mode
    if args.openpose_json and args.depth_file and args.output_json:
        process_single(args.openpose_json, args.depth_file, args.output_json, image_size)

    # Batch mode
    elif args.openpose_dir and args.depth_dir and args.output_dir:
        process_batch(args.openpose_dir, args.depth_dir, args.output_dir)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  Single file:")
        print("    python extract_keypoint_depths.py \\")
        print("        --openpose_json ../../Data/OpenPose/my_openpose/output_jsons/dance_000000000120_keypoints.json \\")
        print("        --depth_file ../../Data/OpenPose/my_openpose/depths/frame_000000000120_depth.npy \\")
        print("        --output_json ../../Data/OpenPose/my_openpose/depth_jsons/dance_000000000120_depth.json")
        print()
        print("  Batch processing:")
        print("    python extract_keypoint_depths.py \\")
        print("        --openpose_dir ../../Data/OpenPose/my_openpose/output_jsons/ \\")
        print("        --depth_dir ../../Data/OpenPose/my_openpose/depths/ \\")
        print("        --output_dir ../../Data/OpenPose/my_openpose/depth_jsons/")


if __name__ == "__main__":
    main()
