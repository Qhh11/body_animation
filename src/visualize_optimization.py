#!/usr/bin/env python3
"""
Visualization script for SMPL optimization results.
This script loads the output files from the C++ optimizer and creates visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def load_landmarks(filename):
    """Load landmarks from text file."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                joint_id, x, y, conf = map(float, parts)
                data.append({'joint_id': int(joint_id), 'x': x, 'y': y, 'conf': conf})
    return data


def load_camera_params(filename):
    """Load camera parameters from text file."""
    with open(filename, 'r') as f:
        fx, fy, cx, cy = map(float, f.readline().strip().split())
        width, height = map(int, f.readline().strip().split())
    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'width': width, 'height': height}


def load_obj(filename):
    """Load vertices and faces from OBJ file."""
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # OBJ indices start at 1, convert to 0-based
                faces.append([int(p.split('/')[0]) - 1 for p in parts[1:4]])
    return np.array(vertices), np.array(faces)


def visualize_2d_landmarks(landmarks_gt, landmarks_opt, camera, save_path='landmarks_2d.png'):
    """Visualize 2D landmark comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot ground truth landmarks
    gt_x = [lm['x'] for lm in landmarks_gt]
    gt_y = [lm['y'] for lm in landmarks_gt]
    ax.scatter(gt_x, gt_y, c='green', marker='o', s=100, label='Ground Truth', alpha=0.7)

    # Plot optimized landmarks
    opt_x = [lm['x'] for lm in landmarks_opt]
    opt_y = [lm['y'] for lm in landmarks_opt]
    ax.scatter(opt_x, opt_y, c='red', marker='x', s=100, label='Optimized', alpha=0.7)

    # Draw connections between corresponding points
    for gt_lm, opt_lm in zip(landmarks_gt, landmarks_opt):
        ax.plot([gt_lm['x'], opt_lm['x']], [gt_lm['y'], opt_lm['y']],
                'b--', alpha=0.3, linewidth=0.5)

    # Annotate joint IDs
    for lm in landmarks_gt:
        ax.annotate(str(int(lm['joint_id'])),
                   (lm['x'], lm['y']),
                   fontsize=8,
                   alpha=0.5)

    ax.set_xlim(0, camera['width'])
    ax.set_ylim(camera['height'], 0)  # Invert Y-axis for image coordinates
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('2D Landmark Reprojection Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved 2D landmark visualization to {save_path}")
    plt.close()


def compute_reprojection_errors(landmarks_gt, landmarks_opt):
    """Compute per-joint reprojection errors."""
    errors = []
    for gt_lm, opt_lm in zip(landmarks_gt, landmarks_opt):
        dx = gt_lm['x'] - opt_lm['x']
        dy = gt_lm['y'] - opt_lm['y']
        error = np.sqrt(dx**2 + dy**2)
        errors.append({
            'joint_id': int(gt_lm['joint_id']),
            'error': error
        })
    return errors


def visualize_reprojection_errors(errors, save_path='reprojection_errors.png'):
    """Visualize reprojection errors as a bar chart."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    joint_ids = [e['joint_id'] for e in errors]
    error_values = [e['error'] for e in errors]

    colors = ['red' if err > 2.0 else 'orange' if err > 1.0 else 'green' for err in error_values]

    ax.bar(joint_ids, error_values, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='orange', linestyle='--', label='1 pixel threshold')
    ax.axhline(y=2.0, color='red', linestyle='--', label='2 pixel threshold')

    ax.set_xlabel('Joint ID')
    ax.set_ylabel('Reprojection Error (pixels)')
    ax.set_title('Per-Joint Reprojection Error')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Print statistics
    mean_error = np.mean(error_values)
    max_error = np.max(error_values)
    print(f"\nReprojection Error Statistics:")
    print(f"  Mean error: {mean_error:.3f} pixels")
    print(f"  Max error:  {max_error:.3f} pixels")
    print(f"  Median error: {np.median(error_values):.3f} pixels")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved reprojection error visualization to {save_path}")
    plt.close()


def visualize_3d_meshes(vertices_before, vertices_after, vertices_gt,
                        faces, save_path='mesh_comparison_3d.png'):
    """Visualize 3D mesh comparison."""
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: Before optimization
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_trisurf(vertices_before[:, 0], vertices_before[:, 1], vertices_before[:, 2],
                     triangles=faces, alpha=0.3, color='blue', edgecolor='none')
    ax1.scatter(vertices_before[:, 0], vertices_before[:, 1], vertices_before[:, 2],
                c='blue', s=0.1, alpha=0.1)
    ax1.set_title('Before Optimization')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    set_equal_aspect_3d(ax1, vertices_before)

    # Plot 2: After optimization
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_trisurf(vertices_after[:, 0], vertices_after[:, 1], vertices_after[:, 2],
                     triangles=faces, alpha=0.3, color='red', edgecolor='none')
    ax2.scatter(vertices_after[:, 0], vertices_after[:, 1], vertices_after[:, 2],
                c='red', s=0.1, alpha=0.1)
    ax2.set_title('After Optimization')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    set_equal_aspect_3d(ax2, vertices_after)

    # Plot 3: Ground truth
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_trisurf(vertices_gt[:, 0], vertices_gt[:, 1], vertices_gt[:, 2],
                     triangles=faces, alpha=0.3, color='green', edgecolor='none')
    ax3.scatter(vertices_gt[:, 0], vertices_gt[:, 1], vertices_gt[:, 2],
                c='green', s=0.1, alpha=0.1)
    ax3.set_title('Ground Truth')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    set_equal_aspect_3d(ax3, vertices_gt)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved 3D mesh comparison to {save_path}")
    plt.close()


def set_equal_aspect_3d(ax, vertices):
    """Set equal aspect ratio for 3D plot."""
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                         vertices[:, 1].max() - vertices[:, 1].min(),
                         vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0

    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def visualize_vertex_errors(vertices_gt, vertices_opt, save_path='vertex_errors.png'):
    """Visualize per-vertex errors between ground truth and optimized mesh."""
    errors = np.linalg.norm(vertices_gt - vertices_opt, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1.hist(errors, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.4f}m')
    ax1.axvline(x=np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.4f}m')
    ax1.set_xlabel('Vertex Error (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Per-Vertex Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 3D scatter with error coloring
    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(vertices_opt[:, 0], vertices_opt[:, 1], vertices_opt[:, 2],
                         c=errors, cmap='hot', s=1, alpha=0.6)
    ax2.set_title('Optimized Mesh (colored by error)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.colorbar(scatter, ax=ax2, label='Error (m)', shrink=0.5)
    set_equal_aspect_3d(ax2, vertices_opt)

    print(f"\nVertex Error Statistics:")
    print(f"  Mean error: {np.mean(errors):.4f} m")
    print(f"  Max error:  {np.max(errors):.4f} m")
    print(f"  Median error: {np.median(errors):.4f} m")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved vertex error visualization to {save_path}")
    plt.close()


def main():
    print("="*60)
    print("SMPL Optimization Visualization")
    print("="*60)

    # Check if files exist
    required_files = [
        'landmarks_gt.txt',
        'landmarks_optimized.txt',
        'mesh_before_opt.obj',
        'mesh_after_opt.obj',
        'mesh_ground_truth.obj',
        'camera_params.txt'
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"\nError: Missing required files: {missing_files}")
        print("Please run the C++ optimizer first to generate these files.")
        return

    # Load data
    print("\nLoading data...")
    landmarks_gt = load_landmarks('landmarks_gt.txt')
    landmarks_opt = load_landmarks('landmarks_optimized.txt')
    camera = load_camera_params('camera_params.txt')

    vertices_before, faces = load_obj('mesh_before_opt.obj')
    vertices_after, _ = load_obj('mesh_after_opt.obj')
    vertices_gt, _ = load_obj('mesh_ground_truth.obj')

    print(f"  Loaded {len(landmarks_gt)} ground truth landmarks")
    print(f"  Loaded {len(landmarks_opt)} optimized landmarks")
    print(f"  Loaded mesh with {len(vertices_gt)} vertices and {len(faces)} faces")

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. 2D landmark comparison
    visualize_2d_landmarks(landmarks_gt, landmarks_opt, camera)

    # 2. Reprojection errors
    errors = compute_reprojection_errors(landmarks_gt, landmarks_opt)
    visualize_reprojection_errors(errors)

    # 3. 3D mesh comparison
    visualize_3d_meshes(vertices_before, vertices_after, vertices_gt, faces)

    # 4. Vertex errors
    visualize_vertex_errors(vertices_gt, vertices_after)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - landmarks_2d.png: 2D landmark comparison")
    print("  - reprojection_errors.png: Per-joint reprojection errors")
    print("  - mesh_comparison_3d.png: 3D mesh comparison")
    print("  - vertex_errors.png: Per-vertex error analysis")
    print("\nYou can view the OBJ files in MeshLab, Blender, or any 3D viewer.")


if __name__ == '__main__':
    main()
