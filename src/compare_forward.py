#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from pathlib import Path


# ----------------------------- utils -----------------------------

def stats_l2_per_vertex(A, B, name=""):
    """
    A,B: (N,3)
    """
    D = np.linalg.norm(A - B, axis=1)
    out = {
        "mean": float(np.mean(D)),
        "median": float(np.median(D)),
        "max": float(np.max(D)),
        "p95": float(np.percentile(D, 95.0)),
    }
    if name:
        print(f"{name}\nL2(m): mean={out['mean']:.6g}  median={out['median']:.6g}  max={out['max']:.6g}  p95={out['p95']:.6g}")
    return out


def vec_l2(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.linalg.norm(a - b))


def rodrigues_batch(aa):
    """
    aa: (J,3) axis-angle
    return: R (J,3,3)
    """
    aa = np.asarray(aa, dtype=np.float64)
    J = aa.shape[0]
    theta = np.linalg.norm(aa, axis=1, keepdims=True)  # (J,1)
    eps = 1e-12

    # unit axis
    axis = aa / np.maximum(theta, eps)

    x = axis[:, 0:1]
    y = axis[:, 1:2]
    z = axis[:, 2:3]

    zero = np.zeros((J, 1), dtype=np.float64)
    K = np.concatenate([
        np.concatenate([ zero, -z,   y], axis=1)[:, None, :],
        np.concatenate([   z, zero, -x], axis=1)[:, None, :],
        np.concatenate([  -y,   x, zero], axis=1)[:, None, :],
    ], axis=1)  # (J,3,3)

    I = np.eye(3, dtype=np.float64)[None, :, :]  # (1,3,3)
    ct = np.cos(theta)[:, None, :]  # (J,1,1)
    st = np.sin(theta)[:, None, :]  # (J,1,1)

    # Rodrigues: R = I + sinθ K + (1-cosθ) K^2
    K2 = K @ K
    R = I + st * K + (1.0 - ct) * K2

    # handle theta ~ 0: R ~ I
    small = (theta.reshape(-1) < 1e-10)
    if np.any(small):
        R[small] = np.eye(3, dtype=np.float64)

    return R


# ----------------------------- SMPL model load -----------------------------

def load_smpl_npz(npz_path):
    npz = np.load(npz_path, allow_pickle=False)

    def req(key):
        if key not in npz:
            raise KeyError(f"Missing key '{key}' in {npz_path}")
        return npz[key]

    v_template = np.array(req("v_template"), dtype=np.float64)         # (N,3)
    shapedirs  = np.array(req("shapedirs"), dtype=np.float64)          # (N*3,B)
    posedirs   = np.array(req("posedirs"), dtype=np.float64)           # (N*3,207)
    weights    = np.array(req("weights"), dtype=np.float64)            # (N,J)
    J_reg      = np.array(req("J_regressor"), dtype=np.float64)        # (J,N)
    kt         = np.array(req("kintree_table"), dtype=np.int64)        # (2,J)

    N = v_template.shape[0]
    J = J_reg.shape[0]

    assert v_template.shape == (N, 3)
    assert weights.shape == (N, J)
    assert J_reg.shape == (J, N)
    assert kt.shape[0] == 2 and kt.shape[1] == J

    parents = kt[0].copy()
    parents[0] = -1
    parents = parents.astype(np.int64)

    # only first 10 betas
    shapedirs10 = shapedirs[:, :10].copy()

    return {
        "N": N,
        "J": J,
        "v_template": v_template,
        "shapedirs10": shapedirs10,
        "posedirs": posedirs,
        "weights": weights,
        "J_regressor": J_reg,
        "parents": parents,
    }


# ----------------------------- Python forward (matches your C++ intent) -----------------------------

def smpl_forward_np(model, beta10, theta72, trans3, use_posedirs=True, return_debug=False):
    """
    beta10: (10,)
    theta72: (72,) axis-angle, 24*3
    trans3: (3,)
    """
    N = model["N"]
    J = model["J"]

    beta10 = np.asarray(beta10, dtype=np.float64).reshape(10)
    theta72 = np.asarray(theta72, dtype=np.float64).reshape(72)
    trans3 = np.asarray(trans3, dtype=np.float64).reshape(3)

    v_template = model["v_template"]
    shapedirs10 = model["shapedirs10"]
    posedirs = model["posedirs"]
    weights = model["weights"]
    J_reg = model["J_regressor"]
    parents = model["parents"]

    # ---- v_shaped ----
    # shapedirs is (N*3,10)
    v_shaped_flat = v_template.reshape(-1) + shapedirs10 @ beta10  # (N*3,)
    v_shaped = v_shaped_flat.reshape(N, 3)

    # ---- joints (J,3) from regressor ----
    # J_reg: (J,N), v_shaped: (N,3)
    joints = J_reg @ v_shaped  # (J,3)

    # ---- FK ----
    aa = theta72.reshape(J, 3)
    R = rodrigues_batch(aa)  # (J,3,3)

    # local transforms 4x4
    local = np.zeros((J, 4, 4), dtype=np.float64)
    local[:, 3, 3] = 1.0
    local[:, :3, :3] = R

    # translations: root = joints[0], others = joints[i]-joints[parent]
    local[:, :3, 3] = joints
    for i in range(1, J):
        p = int(parents[i])
        if p >= 0:
            local[i, :3, 3] = joints[i] - joints[p]

    # global transforms
    global_T = np.zeros((J, 4, 4), dtype=np.float64)
    for i in range(J):
        p = int(parents[i])
        if p < 0:
            global_T[i] = local[i]
        else:
            global_T[i] = global_T[p] @ local[i]

    # ---- pose blendshapes ----
    v_posed = v_shaped.copy()
    pose_feat = np.zeros((207,), dtype=np.float64)

    if use_posedirs:
        # IMPORTANT: flatten order must match your C++:
        # pose_feat((i-1)*9 + r*3 + c) = dR(r,c)
        idx = 0
        I3 = np.eye(3, dtype=np.float64)
        for i in range(1, 24):
            dR = R[i] - I3
            for r in range(3):
                for c in range(3):
                    pose_feat[idx] = dR[r, c]
                    idx += 1

        # posedirs: (N*3,207)
        v_posed_flat = v_shaped.reshape(-1) + posedirs @ pose_feat  # (N*3,)
        v_posed = v_posed_flat.reshape(N, 3)

    # ---- skinning transforms: Tskin = global_T * inv_bind ----
    inv_bind = np.zeros((J, 4, 4), dtype=np.float64)
    inv_bind[:, 3, 3] = 1.0
    inv_bind[:, 0, 0] = 1.0
    inv_bind[:, 1, 1] = 1.0
    inv_bind[:, 2, 2] = 1.0
    inv_bind[:, :3, 3] = -joints  # -joints(i)

    Tskin = global_T @ inv_bind  # (J,4,4)

    # ---- LBS ----
    V = np.zeros((N, 3), dtype=np.float64)
    # apply per-joint, accumulate weighted
    # p = T * [v;1]
    v_h = np.concatenate([v_posed, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N,4)
    for j in range(J):
        w = weights[:, j:j+1]  # (N,1)
        if np.max(w) <= 1e-12:
            continue
        p = (v_h @ Tskin[j].T)[:, :3]  # (N,3)
        V += w * p

    # global translation
    V = V + trans3[None, :]

    if not return_debug:
        return V

    return {
        "v_shaped": v_shaped,
        "pose_feat": pose_feat,
        "v_posed": v_posed,
        "V": V,
        "joints": joints,
    }


# ----------------------------- main: compare CaseB / CaseC -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="../../Data/models/smpl_neutral.npz")
    ap.add_argument("--mode", type=str, choices=["B", "C"], required=True)
    ap.add_argument("--dir", type=str, default=".", help="folder containing cpp_*.npy")
    args = ap.parse_args()

    workdir = Path(args.dir)
    model = load_smpl_npz(args.npz)

    if args.mode == "B":
        beta = np.load(workdir / "cpp_beta_gt.npy").astype(np.float64).reshape(10)
        theta0 = np.zeros((72,), dtype=np.float64)
        trans0 = np.zeros((3,), dtype=np.float64)

        dbg = smpl_forward_np(model, beta, theta0, trans0, use_posedirs=False, return_debug=True)
        V_py_shaped = dbg["v_shaped"]
        V_py_caseB  = dbg["V"]  # since theta=0 and posedirs off, this should just be shaped + skinning(identity-ish)

        V_cpp_shaped = np.load(workdir / "cpp_v_shaped_caseB.npy").astype(np.float64).reshape(-1, 3)
        V_cpp_caseB  = np.load(workdir / "cpp_V_caseB.npy").astype(np.float64).reshape(-1, 3)

        print("Shapes:")
        print("  beta:", beta.shape)
        print("  V_py_shaped:", V_py_shaped.shape)
        print("  V_cpp_shaped:", V_cpp_shaped.shape)
        print("  V_cpp_caseB:", V_cpp_caseB.shape)
        print()

        stats_l2_per_vertex(V_py_shaped, V_cpp_shaped, "[Raw] Python v_shaped vs C++ v_shaped_caseB")
        stats_l2_per_vertex(V_py_shaped, V_cpp_caseB, "[Raw] Python v_shaped vs C++ V_caseB")

        # optional: centroid align check
        t = V_cpp_caseB.mean(axis=0) - V_py_shaped.mean(axis=0)
        V_aligned = V_py_shaped + t[None, :]
        print("\n[Align centroid] Python v_shaped -> C++ V_caseB")
        print("  centroid offset t =", t)
        stats_l2_per_vertex(V_aligned, V_cpp_caseB, "Aligned L2")

    elif args.mode == "C":
        theta = np.load(workdir / "cpp_theta_gt.npy").astype(np.float64).reshape(72)
        beta0 = np.zeros((10,), dtype=np.float64)
        trans0 = np.zeros((3,), dtype=np.float64)

        dbg = smpl_forward_np(model, beta0, theta, trans0, use_posedirs=True, return_debug=True)
        V_py_shaped = dbg["v_shaped"]
        pose_feat   = dbg["pose_feat"]
        V_py_posed  = dbg["v_posed"]
        V_py_caseC  = dbg["V"]

        V_cpp_shaped = np.load(workdir / "cpp_v_shaped_caseC.npy").astype(np.float64).reshape(-1, 3)
        V_cpp_posed  = np.load(workdir / "cpp_v_posed_caseC.npy").astype(np.float64).reshape(-1, 3)
        V_cpp_caseC  = np.load(workdir / "cpp_V_caseC.npy").astype(np.float64).reshape(-1, 3)

        posefeat_path = workdir / "cpp_posefeat_caseC.npy"
        if posefeat_path.exists():
            posefeat_cpp = np.load(posefeat_path).astype(np.float64).reshape(-1)
        else:
            posefeat_cpp = None

        print("Shapes:")
        print("  theta:", theta.shape)
        print("  V_py_shaped:", V_py_shaped.shape)
        print("  V_py_posed:", V_py_posed.shape)
        print("  V_py_caseC:", V_py_caseC.shape)
        print()

        stats_l2_per_vertex(V_py_shaped, V_cpp_shaped, "[Diff] v_shaped: Python vs C++")

        print("\n[Diff] pose_feat: Python vs C++")
        if posefeat_cpp is None:
            print("  cpp_posefeat_caseC.npy not found (skip)")
        else:
            print("  posefeat L2 =", vec_l2(pose_feat, posefeat_cpp))
            print("  posefeat maxabs =", float(np.max(np.abs(pose_feat - posefeat_cpp))))

        stats_l2_per_vertex(V_py_posed, V_cpp_posed, "\n[Diff] v_posed: Python vs C++")
        stats_l2_per_vertex(V_py_caseC, V_cpp_caseC, "\n[Diff] V_caseC (final skinned verts): Python vs C++")

        # sanity: shaped should equal template when beta=0
        vt = model["v_template"]
        stats_l2_per_vertex(V_cpp_shaped, vt, "\n[Sanity] C++ v_shaped_caseC vs v_template")

    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
