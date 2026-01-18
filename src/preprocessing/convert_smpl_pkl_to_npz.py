#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import pickle
import numpy as np

PKL_PATH = "../../Data/models/SMPL_python_v.1.1-4.0/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
OUT_NPZ  = "../../Data/models/smpl_neutral.npz"


def normalize_protocol0_newlines(raw: bytes) -> bytes:
    # Protocol-0 ASCII pickle expects '\n'. Some files use CRLF which breaks parsing.
    raw = raw.replace(b"\r\n", b"\n")
    raw = raw.replace(b"\r", b"\n")
    return raw


# ------------------------------------------------------------
# Chumpy-free unpickler
# ------------------------------------------------------------
class _ChumpyDummy:
    """
    Stand-in for chumpy classes during unpickling.
    We only need to successfully unpickle the dict; we won't "compute" with these objects.
    """
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"<ChumpyDummy keys={list(self.__dict__.keys())[:5]}...>"


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map any chumpy.* reference to dummy so unpickling can continue
        if module.startswith("chumpy"):
            return _ChumpyDummy

        # Some very old pickles may refer to __builtin__ (py2)
        if module == "__builtin__":
            module = "builtins"

        return super().find_class(module, name)


def load_smpl_pkl_no_chumpy(pkl_path: str):
    raw = open(pkl_path, "rb").read()
    print("File size:", len(raw))
    print("Head:", raw[:32])

    raw = normalize_protocol0_newlines(raw)

    bio = io.BytesIO(raw)
    try:
        smpl = _CompatUnpickler(bio, fix_imports=True, encoding="latin1").load()
    except Exception as e:
        print("\n[ERROR] Unpickle failed.")
        print("Reason:", repr(e))
        print("\nIf you see scipy.sparse missing, install it via apt:")
        print("  apt-get update && apt-get install -y python3-scipy")
        raise

    if not isinstance(smpl, dict):
        raise TypeError(f"Expected dict at top-level, got {type(smpl)}")

    return smpl


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def to_dense_if_sparse(x):
    # scipy sparse matrix has .toarray()
    if hasattr(x, "toarray"):
        return x.toarray()
    return to_np(x)


def unwrap_possible_chumpy(x):
    """
    In case some fields are chumpy objects, try to extract underlying numeric arrays.
    Common internal attribute names in chumpy objects: 'r', 'x', 'value'
    """
    if isinstance(x, _ChumpyDummy):
        d = getattr(x, "__dict__", {})
        for k in ("r", "x", "value", "v", "data"):
            if k in d:
                return d[k]
    return x


def main():
    print("Using PKL:", PKL_PATH)
    print("Saving NPZ to:", OUT_NPZ)

    smpl = load_smpl_pkl_no_chumpy(PKL_PATH)

    # ---- Required fields (SMPL v1.1.0) ----
    def get_required(key):
        if key not in smpl:
            raise KeyError(f"Missing key '{key}' in pkl. Available keys: {list(smpl.keys())[:30]} ...")
        return unwrap_possible_chumpy(smpl[key])

    v_template   = to_np(get_required("v_template")).astype(np.float64)          # (6890,3)
    weights      = to_np(get_required("weights")).astype(np.float64)             # (6890,24)
    f            = to_np(get_required("f")).astype(np.int32)                     # (13776,3)
    shapedirs    = to_np(get_required("shapedirs")).astype(np.float64)            # (6890,3,300)
    posedirs     = to_np(get_required("posedirs")).astype(np.float64)             # (6890,3,207) or variants

    J_regressor  = to_dense_if_sparse(get_required("J_regressor")).astype(np.float64)  # (24,6890)
    kintree_table = to_np(get_required("kintree_table")).astype(np.int32)               # (2,24)

    N = v_template.shape[0]
    assert v_template.shape == (N, 3), f"v_template shape weird: {v_template.shape}"
    assert weights.shape[0] == N, "weights N mismatch"
    assert J_regressor.shape[1] == N, f"J_regressor should be (24,N), got {J_regressor.shape}"
    assert kintree_table.shape[1] == 24, f"kintree_table should be (2,24), got {kintree_table.shape}"

    # ---- Flatten dirs to match your C++ expectations ----
    # shapedirs: (N,3,B) -> (N*3,B)
    if shapedirs.ndim == 3:
        shapedirs_flat = shapedirs.reshape(N * 3, shapedirs.shape[2])
    elif shapedirs.ndim == 2 and shapedirs.shape[0] == N * 3:
        shapedirs_flat = shapedirs
    else:
        raise ValueError(f"Unexpected shapedirs shape: {shapedirs.shape}")

    # posedirs: handle (N,3,P), (N*3,P), (P,N*3)
    if posedirs.ndim == 3:
        posedirs_flat = posedirs.reshape(N * 3, posedirs.shape[2])
    elif posedirs.ndim == 2:
        if posedirs.shape[0] == N * 3:
            posedirs_flat = posedirs
        elif posedirs.shape[1] == N * 3:
            posedirs_flat = posedirs.T
        else:
            raise ValueError(f"Unexpected posedirs 2D shape: {posedirs.shape}")
    else:
        raise ValueError(f"Unexpected posedirs ndim: {posedirs.ndim}")

    print("\n[OK] Extracted fields:")
    print("  v_template:", v_template.shape)
    print("  f:", f.shape)
    print("  weights:", weights.shape)
    print("  J_regressor:", J_regressor.shape)
    print("  kintree_table:", kintree_table.shape)
    print("  shapedirs_flat:", shapedirs_flat.shape)
    print("  posedirs_flat:", posedirs_flat.shape)

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(
        OUT_NPZ,
        v_template=v_template,
        f=f,
        weights=weights,
        J_regressor=J_regressor,
        kintree_table=kintree_table,
        shapedirs=shapedirs_flat,
        posedirs=posedirs_flat,
    )

    print("\n[SUCCESS] Wrote:", OUT_NPZ)


if __name__ == "__main__":
    main()
