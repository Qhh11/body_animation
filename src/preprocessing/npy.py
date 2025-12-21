import numpy as np
import os

npz_path = "../../Data/amass/A1-Stand_poses.npz"
out_dir = "../../Data/amass/A1-Stand_npy"
os.makedirs(out_dir, exist_ok=True)

data = np.load(npz_path)

np.save(os.path.join(out_dir, "poses.npy"), data["poses"])
np.save(os.path.join(out_dir, "betas.npy"), data["betas"])
np.save(os.path.join(out_dir, "trans.npy"), data["trans"])

print("Saved poses.npy, betas.npy, trans.npy")