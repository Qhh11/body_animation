#!/usr/bin/env python3
"""
Build pose-corrective data for SMPL model.

This tool extracts region-based pose blend shape data from SMPL
and performs dimensionality reduction using SVD.

Week 1 Implementation:
- Reads SMPL weights and posedirs from model.npz
- For each joint: selects top-M vertices by skinning weight
- Extracts posedir data for each region
- Applies SVD to keep top-K components
- Saves CSR-packed output to corrective_data.npz
"""

import argparse
import numpy as np
import sys


def build_corrective_data(smpl_npz_path, output_npz_path, M=500, K=8):
    """
    Build corrective data from SMPL model.

    Args:
        smpl_npz_path: Path to SMPL model.npz
        output_npz_path: Path to output corrective_data.npz
        M: Number of top vertices to select per joint (default: 500)
        K: Number of SVD components to keep (default: 8)
    """
    print(f"Loading SMPL model from: {smpl_npz_path}")

    try:
        smpl_data = np.load(smpl_npz_path)
    except Exception as e:
        print(f"Error loading SMPL NPZ: {e}", file=sys.stderr)
        return False

    # Extract required arrays
    if 'weights' not in smpl_data or 'posedirs' not in smpl_data:
        print("Error: SMPL NPZ missing 'weights' or 'posedirs'", file=sys.stderr)
        return False

    weights = smpl_data['weights']  # (N, J)
    posedirs = smpl_data['posedirs']  # (N, 3, D_pose) or (N*3, D_pose)

    # Handle different posedirs formats
    if posedirs.ndim == 3:
        # Reshape (N, 3, D_pose) to (N*3, D_pose)
        N, _, D_pose = posedirs.shape
        posedirs_flat = np.zeros((N * 3, D_pose), dtype=posedirs.dtype)
        for i in range(N):
            for j in range(3):
                posedirs_flat[i * 3 + j, :] = posedirs[i, j, :]
        posedirs = posedirs_flat

    N = weights.shape[0]
    J = weights.shape[1]
    D_pose = posedirs.shape[1]

    print(f"SMPL model info:")
    print(f"  N (vertices): {N}")
    print(f"  J (joints): {J}")
    print(f"  D_pose: {D_pose}")
    print(f"  M (top vertices per joint): {M}")
    print(f"  K (SVD components): {K}")

    # CSR-packed arrays
    region_ptr = [0]  # Will have J+1 entries
    region_idx = []   # Will have sum_len entries
    basis_blocks = [] # Will concatenate to (sum_len*3, K)

    # Process each joint
    for j in range(J):
        # Select top M vertices by weight for this joint
        joint_weights = weights[:, j]
        top_m_indices = np.argsort(-joint_weights)[:M]  # Descending order
        top_m_indices = np.sort(top_m_indices)  # Sort for consistent ordering

        # Extract posedirs for this region
        # For each vertex, we have 3 rows (x, y, z)
        region_rows = []
        for v_idx in top_m_indices:
            for dim in range(3):
                region_rows.append(v_idx * 3 + dim)

        region_posedirs = posedirs[region_rows, :]  # (M*3, D_pose)

        # Apply SVD and keep top K components
        # U: (M*3, min(M*3, D_pose))
        # S: (min(M*3, D_pose),)
        # Vt: (min(M*3, D_pose), D_pose)
        U, S, Vt = np.linalg.svd(region_posedirs, full_matrices=False)

        # Keep only top K components from U
        K_actual = min(K, U.shape[1])
        basis = U[:, :K_actual]  # (M*3, K_actual)

        # Pad with zeros if K_actual < K
        if K_actual < K:
            basis_padded = np.zeros((basis.shape[0], K), dtype=basis.dtype)
            basis_padded[:, :K_actual] = basis
            basis = basis_padded

        # Store in CSR format
        region_idx.extend(top_m_indices.tolist())
        region_ptr.append(region_ptr[-1] + len(top_m_indices))
        basis_blocks.append(basis)

        if (j + 1) % 5 == 0 or j == J - 1:
            print(f"  Processed joint {j+1}/{J}")

    # Concatenate all basis blocks
    basis_concat = np.vstack(basis_blocks)  # (sum_len*3, K)
    sum_len = region_ptr[-1]

    print(f"\nOutput dimensions:")
    print(f"  sum_len (total vertices): {sum_len}")
    print(f"  region_ptr shape: ({J+1},)")
    print(f"  region_idx shape: ({sum_len},)")
    print(f"  basis shape: ({sum_len*3}, {K})")

    # Save to NPZ
    np.savez(
        output_npz_path,
        J=np.array([J], dtype=np.int64),
        K=np.array([K], dtype=np.int64),
        region_ptr=np.array(region_ptr, dtype=np.int64),
        region_idx=np.array(region_idx, dtype=np.int64),
        basis=basis_concat.astype(np.float32)
    )

    print(f"\nSaved corrective data to: {output_npz_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build pose-corrective data from SMPL model"
    )
    parser.add_argument(
        "--smpl_npz",
        type=str,
        required=True,
        help="Path to SMPL model.npz"
    )
    parser.add_argument(
        "--out_npz",
        type=str,
        required=True,
        help="Path to output corrective_data.npz"
    )
    parser.add_argument(
        "--M",
        type=int,
        default=500,
        help="Number of top vertices to select per joint (default: 500)"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=8,
        help="Number of SVD components to keep (default: 8)"
    )

    args = parser.parse_args()

    success = build_corrective_data(
        args.smpl_npz,
        args.out_npz,
        M=args.M,
        K=args.K
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
