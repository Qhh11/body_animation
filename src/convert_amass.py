#!/usr/bin/env python3
"""
Convert AMASS .npz files to SMPL formatted .txt files

Usage:
    # Single file mode
    python convert_amass.py --input ../Data/CMU/06/06_09_poses.npz --output output.txt
    
    # Batch mode
    python convert_amass.py --batch --input_dir ../Data/CMU/06 --output_dir ../Data/CMU/06_formatted
"""

import numpy as np
import os
import argparse
import re
from pathlib import Path
from tqdm import tqdm


def convert_amass_to_smpl(input_path, output_path, verbose=True):
    """
    Convert a single AMASS .npz file to SMPL formatted .txt file.
    
    Args:
        input_path: Path to input .npz file
        output_path: Path to output .txt file
        verbose: Whether to print detailed information
    
    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"Loading {input_path}...")

    if not os.path.exists(input_path):
        print(f"Error: File not found - {input_path}")
        return False

    try:
        data = np.load(input_path)
        
        if verbose:
            print("Keys found:", list(data.keys()))
        
        if 'poses' not in data:
            print(f"Error: 'poses' key is MISSING in {input_path}!")
            if verbose:
                for k in data.keys():
                    print(f" - {k}: {data[k].shape}")
            return False
            
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
            if verbose:
                print("Warning: 'betas' not found, using zeros.")
            betas = np.zeros((poses.shape[0], 10))

        # Standard SMPL: 10 betas + 72 thetas + 3 trans
        # AMASS poses: 156 -> need first 66 (root + body), padding rest to 72
        
        num_frames = poses.shape[0]
        if verbose:
            print(f"Processing {num_frames} frames...")
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
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
                
                f.write("\n")  # End of frame

        if verbose:
            print(f"Success! Saved formatted data to {output_path}")
        
        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_convert(input_dir, output_dir):
    """
    Batch convert all XX_XX_poses.npz files in a directory.
    Ignores files like XX_XX_stageii.npz
    
    Args:
        input_dir: Directory containing .npz files
        output_dir: Directory for output .txt files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found - {input_dir}")
        return
    
    # Find all .npz files
    all_npz_files = sorted(input_dir.glob('**/*.npz'))
    
    # Filter: only keep files matching XX_XX_poses.npz pattern
    pattern = re.compile(r'^\d+_\d+_poses\.npz$')
    npz_files = [f for f in all_npz_files if pattern.match(f.name)]
    
    if not npz_files:
        print(f"No files matching XX_XX_poses.npz pattern found in {input_dir}")
        if all_npz_files:
            print(f"Found {len(all_npz_files)} .npz files, but none match the pattern.")
            print("Sample files found:")
            for f in all_npz_files[:5]:
                print(f"  - {f.name}")
        return
    
    print(f"Found {len(npz_files)} files matching XX_XX_poses.npz pattern")
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    success_count = 0
    failed_files = []
    
    for npz_file in tqdm(npz_files, desc="Converting files"):
        # Generate output path (maintain subdirectory structure)
        relative_path = npz_file.relative_to(input_dir)
        output_path = output_dir / relative_path.with_suffix('.txt')
        
        # Convert
        if convert_amass_to_smpl(npz_file, output_path, verbose=False):
            success_count += 1
        else:
            failed_files.append(str(npz_file))
    
    # Summary
    print("\n" + "=" * 60)
    print("Batch conversion complete!")
    print(f"  Total files: {len(npz_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for fname in failed_files:
            print(f"  - {fname}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert AMASS .npz files to SMPL formatted .txt files'
    )
    
    parser.add_argument('--batch', action='store_true', 
                        help='Batch processing mode')
    
    # Single file mode arguments
    parser.add_argument('--input', help='Input .npz file path')
    parser.add_argument('--output', help='Output .txt file path')
    
    # Batch mode arguments
    parser.add_argument('--input_dir', help='Input directory containing .npz files')
    parser.add_argument('--output_dir', help='Output directory for .txt files')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode
        if not args.input_dir or not args.output_dir:
            parser.error("Batch mode requires --input_dir and --output_dir")
        
        batch_convert(args.input_dir, args.output_dir)
    
    else:
        # Single file mode
        if not args.input or not args.output:
            parser.error("Single file mode requires --input and --output")
        
        convert_amass_to_smpl(args.input, args.output, verbose=True)


if __name__ == '__main__':
    main()