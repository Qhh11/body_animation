#!/usr/bin/env python3
"""
从 CMU 80-94 文件夹中随机挑选 20 个 npz 文件并转换为 poses_formatted.txt
输出到 /config/workspace/body_animation/Data/sampled/ 目录
"""

import numpy as np
import os
import random
import glob

# 配置
CMU_ROOT = '/config/workspace/body_animation/Data/CMU'
OUTPUT_DIR = '/config/workspace/body_animation/Data/sampled'
FOLDERS = ['80', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '93', '94']
NUM_SAMPLES = 20

def convert_npz_to_txt(input_path, output_path):
    """转换单个 npz 文件"""
    data = np.load(input_path)

    if 'poses' not in data:
        print(f"  Error: 'poses' key not found in {input_path}")
        return False

    poses = data['poses']
    trans = data['trans']

    if 'betas' in data:
        betas = data['betas']
        if len(betas.shape) == 1:
            betas = np.tile(betas[:10], (poses.shape[0], 1))
        else:
            betas = betas[:, :10]
    else:
        betas = np.zeros((poses.shape[0], 10))

    num_frames = poses.shape[0]

    with open(output_path, 'w') as f:
        f.write(f"{num_frames}\n")

        for i in range(num_frames):
            # Beta [10]
            for b in betas[i]:
                f.write(f"{b:.6f} ")

            # Theta [72]
            # Root (0-2)
            f.write(f"{poses[i, 0]:.6f} {poses[i, 1]:.6f} {poses[i, 2]:.6f} ")
            # Body (3-65)
            for k in range(3, 66):
                f.write(f"{poses[i, k]:.6f} ")
            # Padding (66-71)
            for _ in range(6):
                f.write("0.000000 ")

            # Trans [3]
            f.write(f"{trans[i, 0]:.6f} {trans[i, 1]:.6f} {trans[i, 2]:.6f}")
            f.write("\n")

    return True

def main():
    # 收集所有 npz 文件
    all_npz = []
    for folder in FOLDERS:
        folder_path = os.path.join(CMU_ROOT, folder)
        if os.path.isdir(folder_path):
            npz_files = glob.glob(os.path.join(folder_path, '*.npz'))
            all_npz.extend(npz_files)

    print(f"Found {len(all_npz)} npz files in folders {FOLDERS}")

    if len(all_npz) < NUM_SAMPLES:
        print(f"Warning: Only {len(all_npz)} files available, using all of them")
        selected = all_npz
    else:
        selected = random.sample(all_npz, NUM_SAMPLES)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nConverting {len(selected)} files to {OUTPUT_DIR}:\n")

    success_count = 0
    for npz_path in selected:
        # 生成输出文件名: CMU_80_80_01_poses.txt
        folder_name = os.path.basename(os.path.dirname(npz_path))
        file_name = os.path.basename(npz_path).replace('.npz', '')
        output_name = f"CMU_{folder_name}_{file_name}.txt"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        print(f"  {npz_path}")
        print(f"    -> {output_name}")

        try:
            if convert_npz_to_txt(npz_path, output_path):
                success_count += 1
                # 读取帧数
                with open(output_path, 'r') as f:
                    num_frames = int(f.readline().strip())
                print(f"    OK ({num_frames} frames)")
        except Exception as e:
            print(f"    FAILED: {e}")
        print()

    print(f"\n=== Done: {success_count}/{len(selected)} files converted ===")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
