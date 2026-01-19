#!/usr/bin/env python3
"""
分析 sampled 目录下每个文件的姿态复杂度
找出每个文件中姿态最复杂的帧
"""
import os
import numpy as np

SAMPLED_DIR = '/config/workspace/body_animation/Data/sampled'

def load_amass_txt(filepath):
    """加载转换后的 AMASS txt 文件"""
    with open(filepath, 'r') as f:
        num_frames = int(f.readline().strip())
        frames = []
        for _ in range(num_frames):
            line = f.readline().strip().split()
            vals = [float(x) for x in line]
            # beta[10] + theta[72] + trans[3] = 85
            beta = np.array(vals[0:10])
            theta = np.array(vals[10:82])
            trans = np.array(vals[82:85])
            frames.append({'beta': beta, 'theta': theta, 'trans': trans})
    return frames

def pose_complexity(theta):
    """
    计算姿态复杂度：
    - 排除 root rotation (前3个值)
    - 计算其他关节角度的 L2 范数
    """
    body_pose = theta[3:]  # 去掉 root
    return np.linalg.norm(body_pose)

def analyze_file(filepath):
    """分析单个文件，返回最复杂帧的信息"""
    frames = load_amass_txt(filepath)

    complexities = []
    for i, f in enumerate(frames):
        c = pose_complexity(f['theta'])
        complexities.append((i, c))

    # 按复杂度排序
    complexities.sort(key=lambda x: x[1], reverse=True)

    # 统计
    all_c = [c for _, c in complexities]
    mean_c = np.mean(all_c)
    max_c = complexities[0][1]
    max_frame = complexities[0][0]

    # 找 top 5
    top5 = complexities[:5]

    return {
        'num_frames': len(frames),
        'mean_complexity': mean_c,
        'max_complexity': max_c,
        'max_frame': max_frame,
        'top5': top5
    }

def main():
    files = sorted([f for f in os.listdir(SAMPLED_DIR) if f.endswith('.txt')])

    print("=" * 80)
    print(f"{'File':<35} {'Frames':>7} {'Mean':>8} {'Max':>8} {'Best Frame':>12}")
    print("=" * 80)

    results = []
    for fname in files:
        fpath = os.path.join(SAMPLED_DIR, fname)
        info = analyze_file(fpath)
        results.append((fname, info))

        print(f"{fname:<35} {info['num_frames']:>7} {info['mean_complexity']:>8.2f} {info['max_complexity']:>8.2f} {info['max_frame']:>12}")

    print("=" * 80)
    print("\n详细信息（每个文件姿态最复杂的 Top 5 帧）:\n")

    for fname, info in results:
        print(f"\n{fname}:")
        print(f"  总帧数: {info['num_frames']}")
        print(f"  平均复杂度: {info['mean_complexity']:.3f}")
        print(f"  Top 5 复杂帧:")
        for rank, (frame_idx, complexity) in enumerate(info['top5'], 1):
            pct = frame_idx / info['num_frames'] * 100
            print(f"    {rank}. Frame {frame_idx:>5} ({pct:>5.1f}%) - complexity: {complexity:.3f}")

if __name__ == '__main__':
    main()
