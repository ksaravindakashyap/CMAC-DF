#!/usr/bin/env python3
"""
Debug script to diagnose AV-Deepfake1M dataset availability and setup.

This script checks:
1. If dataset exists at expected location
2. Where dataset might be on cluster
3. What directory structure exists
4. How to properly download/configure it
"""

import os
import sys
from pathlib import Path
import subprocess
import json

def check_expected_location():
    """Check if dataset exists at /mnt/av_deepfake1m"""
    print("=" * 70)
    print("1. CHECKING EXPECTED LOCATION: /mnt/av_deepfake1m")
    print("=" * 70)
    
    expected_path = Path("/mnt/av_deepfake1m")
    
    if expected_path.exists():
        print(f"✓ Directory exists: {expected_path}")
        
        # Check structure
        subdirs = []
        for item in expected_path.iterdir():
            if item.is_dir():
                subdirs.append(item.name)
        
        print(f"  Subdirectories: {subdirs}")
        
        # Check for expected splits
        for split in ['train', 'test', 'val']:
            split_dir = expected_path / split
            if split_dir.exists():
                print(f"  ✓ Found {split}/ directory")
                
                # Check for categories
                for cat in ['original', 'deepfake', 'audio', 'visual']:
                    cat_dir = split_dir / cat
                    if cat_dir.exists():
                        mp4_files = list(cat_dir.glob('*.mp4'))
                        print(f"    - {cat}/: {len(mp4_files)} MP4 files")
        
        return True
    else:
        print(f"✗ Directory NOT found: {expected_path}")
        print(f"  This is the root cause of job 5264866 failure")
        return False

def search_cluster_storage():
    """Search for dataset on cluster"""
    print("\n" + "=" * 70)
    print("2. SEARCHING CLUSTER STORAGE FOR AV-DEEPFAKE1M")
    print("=" * 70)
    
    search_paths = [
        "/mnt",
        "/localscratch",
        "/scratch",
        Path.home() / "datasets",
        Path.home() / "data",
    ]
    
    print("Searching common locations...")
    found = []
    
    for search_path in search_paths:
        search_path = Path(search_path)
        if not search_path.exists():
            continue
        
        print(f"\n  Checking {search_path}...")
        try:
            for item in search_path.iterdir():
                if 'av' in item.name.lower() and 'deepfake' in item.name.lower():
                    print(f"    ✓ Found: {item}")
                    found.append(item)
        except PermissionError:
            print(f"    (No permission to access)")
        except Exception as e:
            print(f"    (Error: {e})")
    
    if found:
        print(f"\n✓ Found {len(found)} potential matches")
        return found
    else:
        print("\n✗ No AV-Deepfake1M directories found on cluster")
        return []

def check_hf_dataset():
    """Check if dataset is available on HuggingFace"""
    print("\n" + "=" * 70)
    print("3. CHECKING HUGGINGFACE MIRROR")
    print("=" * 70)
    
    print("Looking for AV-Deepfake1M on HuggingFace...")
    print("  Known mirrors:")
    print("    - Reverb/av-deepfake1m (if available)")
    print("    - ControlNet/av-deepfake1m (if available)")
    
    # Try to list HF datasets
    try:
        print("\n  Attempting to check HuggingFace API...")
        import huggingface_hub
        print(f"  ✓ huggingface_hub is installed")
        
        # Try to search
        from huggingface_hub import list_datasets
        print("  Searching HuggingFace for deepfake datasets...")
        # Note: This might take a while
        
    except ImportError:
        print("  ✗ huggingface_hub not installed")
        print("    Install with: pip install huggingface-hub")
    except Exception as e:
        print(f"  ! Error checking HF: {e}")

def check_github_official():
    """Show info about official GitHub repo"""
    print("\n" + "=" * 70)
    print("4. OFFICIAL GITHUB REPOSITORY")
    print("=" * 70)
    
    print("Repository: https://github.com/ControlNet/AV-Deepfake1M")
    print("\nKey information:")
    print("  - Dataset is under EULA (requires agreement)")
    print("  - Official download links: https://github.com/ControlNet/AV-Deepfake1M")
    print("  - Official SDK: pip install avdeepfake1m")
    print("  - Challenge: https://deepfakes1m.github.io/2024")
    print("\nExpected directory structure:")
    print("""
    |- train_metadata.json
    |- train_metadata/
    |- train/
    |- val_metadata.json
    |- val_metadata/
    |- val/
    |- test_files.txt
    |- test/
    """)
    
    print("\nTo get the dataset:")
    print("  1. Visit: https://github.com/ControlNet/AV-Deepfake1M")
    print("  2. Agree to EULA at: https://github.com/ControlNet/AV-Deepfake1M/blob/master/eula.pdf")
    print("  3. Follow download instructions on GitHub")
    print("  4. Place at /mnt/av_deepfake1m or update script paths")

def check_sdk():
    """Check if official SDK is available"""
    print("\n" + "=" * 70)
    print("5. CHECKING OFFICIAL SDK")
    print("=" * 70)
    
    print("Attempting to import avdeepfake1m SDK...")
    try:
        import avdeepfake1m
        print(f"✓ avdeepfake1m is installed")
        print(f"  Version: {avdeepfake1m.__version__ if hasattr(avdeepfake1m, '__version__') else 'unknown'}")
        print(f"  Location: {avdeepfake1m.__file__}")
        
        # Try to import dataloader
        try:
            from avdeepfake1m.loader import AVDeepfake1mDataModule
            print(f"✓ AVDeepfake1mDataModule is available")
            print("  You can use: dm = AVDeepfake1mDataModule('/path/to/dataset')")
        except ImportError as e:
            print(f"✗ Cannot import AVDeepfake1mDataModule: {e}")
        
    except ImportError:
        print("✗ avdeepfake1m not installed")
        print("  Install with: pip install avdeepfake1m")
        print("  GitHub: https://github.com/ControlNet/AV-Deepfake1M")

def generate_recommendations():
    """Generate actionable recommendations"""
    print("\n" + "=" * 70)
    print("6. RECOMMENDATIONS")
    print("=" * 70)
    
    print("\nTo fix the AV-Deepfake1M setup, you need to:")
    print()
    print("STEP 1: Get the dataset")
    print("  Option A (Recommended): Download from official GitHub")
    print("    - Link: https://github.com/ControlNet/AV-Deepfake1M")
    print("    - Agree to EULA and follow download instructions")
    print()
    print("  Option B: Check if already downloaded elsewhere on cluster")
    print("    - Ask system administrator where it's located")
    print("    - Symlink to /mnt/av_deepfake1m")
    print()
    print("STEP 2: Validate dataset structure")
    print("  - Expected: /mnt/av_deepfake1m/train/{original,deepfake}/*.mp4")
    print("  - Expected: /mnt/av_deepfake1m/test/{original,deepfake}/*.mp4")
    print()
    print("STEP 3: Test with updated script")
    print("  - Run: python av_deepfake1m_rt60_parallel.py --dataset-dir /mnt/av_deepfake1m --split train --category original --limit 5")
    print()
    print("STEP 4: Submit SLURM job when ready")
    print("  - sbatch submit_rt60_job_av_deepfake_test.sh")

def main():
    print("\n" + "=" * 70)
    print("AV-DEEPFAKE1M DATASET DEBUG SCRIPT")
    print("=" * 70)
    print(f"Current date: {subprocess.check_output(['date']).decode().strip()}")
    print(f"Current user: {os.getenv('USER')}")
    print()
    
    # Run checks
    exists = check_expected_location()
    found_paths = search_cluster_storage()
    check_hf_dataset()
    check_github_official()
    check_sdk()
    generate_recommendations()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if exists:
        print("✓ Dataset found at /mnt/av_deepfake1m")
        print("✓ Run: sbatch submit_rt60_job_av_deepfake_test.sh")
    else:
        print("✗ Dataset NOT found at /mnt/av_deepfake1m")
        if found_paths:
            print(f"✓ But found alternatives at: {found_paths}")
            print("  Update script AV_DEEPFAKE1M_ROOT variable")
        else:
            print("✗ Dataset not found anywhere on cluster")
            print("! ACTION REQUIRED: Download dataset from GitHub")
            print("  Reference: https://github.com/ControlNet/AV-Deepfake1M")

if __name__ == '__main__':
    main()
