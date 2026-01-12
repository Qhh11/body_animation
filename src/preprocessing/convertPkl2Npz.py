# Convert the SMPL model from pkl to npz format, to fit the LoadFromNPZ loader.
import pickle
import numpy as np

with open("basicModel_f_lbs_10_207_0_v1.0.0.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

def to_np(x):
    if hasattr(x, "r"):
        x = x.r
    return np.asarray(x)

out = {}
out["v_template"] = to_np(data["v_template"])
out["shapedirs"]  = to_np(data["shapedirs"])
out["posedirs"]   = to_np(data["posedirs"])
out["weights"]    = to_np(data["weights"])
out["f"]          = np.asarray(data["f"], dtype=np.int32)
out["kintree_table"] = np.asarray(data["kintree_table"], dtype=np.int32)

J = data["J_regressor"]
if hasattr(J, "toarray"):
    J = J.toarray()
out["J_regressor"] = to_np(J)

np.savez("basicModel_f_lbs_10_207_0_v1.0.0_converted.npz", **out)
print("Done.")
