import numpy as np
import os

input_path = '../Data/CMU/06/06_09_poses.npz'
output_path = '../Data/CMU/06/06_09_poses_formatted.txt'

print(f"Loading {input_path}...")

if not os.path.exists(input_path):
    print("Error: File not found!")
    exit(1)

try:
    data = np.load(input_path)
    print("Keys found:", list(data.keys()))
    
    if 'poses' not in data:
        print("Error: 'poses' key is MISSING in this file!")
        for k in data.keys():
            print(f" - {k}: {data[k].shape}")
        exit(1)
        
    # AMASS: poses (N, 156), trans (N, 3), betas (16)
    poses = data['poses']
    trans = data['trans']
    
    if 'betas' in data:
        betas = data['betas']
        if len(betas.shape) == 1:
            betas = np.tile(betas[:10], (poses.shape[0], 1))
        else:
            betas = betas[:, :10]
    else:
        print("Warning: 'betas' not found, using zeros.")
        betas = np.zeros((poses.shape[0], 10))

    # Standard SMPL: 10 betas + 72 thetas + 3 trans
    # AMASS poses: 156 -> need first 66 (root + body), padding rest to 72
    
    num_frames = poses.shape[0]
    print(f"Processing {num_frames} frames...")
    
    with open(output_path, 'w') as f:
        f.write(f"{num_frames}\n")
        
        for i in range(num_frames):
            # Beta(0-9) Theta(0-71) Trans(0-2)
            
            # Write Betas [10]
            for b in betas[i]:
                f.write(f"{b:.6f} ")
            
            # Write Thetas [72]
            # AMASS 0-2 (Root)
            f.write(f"{poses[i, 0]:.6f} {poses[i, 1]:.6f} {poses[i, 2]:.6f} ")
            
            # AMASS 3-65 (Body 21 joints) -> SMPL 3-65
            for k in range(3, 66):
                f.write(f"{poses[i, k]:.6f} ")
                
            # SMPL 66-71 (Hands/Feet ends) -> Fill with 0.0 (6 zeros)
            for _ in range(6):
                f.write("0.000000 ")
                
            # Write Trans [3]
            f.write(f"{trans[i, 0]:.6f} {trans[i, 1]:.6f} {trans[i, 2]:.6f}")
            
            f.write("\n") # End of frame

    print(f"Success! Saved formatted data to {output_path}")

except Exception as e:
    print(f"Python processing failed: {e}")