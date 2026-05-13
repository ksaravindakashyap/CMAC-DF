#!/usr/bin/env python3
"""
Debug script to diagnose and solve VoxCeleb2 HuggingFace issues.

The problem: Reverb/voxceleb2 mirror has mixed schemas - text files mixed with audio/video
Solution: Use huggingface-cli for direct download instead of datasets library

This script tests:
1. The current datasets library approach (to reproduce the error)
2. Alternative approaches using huggingface-cli
3. Proper chunked download setup
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

def show_current_problem():
    """Show why the current approach fails"""
    print("=" * 70)
    print("1. UNDERSTANDING THE CURRENT PROBLEM")
    print("=" * 70)
    
    print("\nCurrent script uses: from datasets import load_dataset('Reverb/voxceleb2')")
    print("\nError encountered in jobs 5264809 and 5264807:")
    print("""
    Error downloading from HuggingFace: An error occurred while generating the dataset
    
    All the data files must have the same columns, but at some point there are 
    1 new columns ({'text'}) and 3 missing columns 
    ({'video_id', 'audio', 'speaker_id'}).
    """)
    
    print("\nRoot cause:")
    print("  - HuggingFace Reverb/voxceleb2 has MIXED content")
    print("  - Contains BOTH text files (description) AND audio/video files")
    print("  - datasets library expects consistent schema across all files")
    print("  - It fails when encountering text files after audio files")

def show_hf_cli_approach():
    """Show the huggingface-cli approach"""
    print("\n" + "=" * 70)
    print("2. HUGGINGFACE-CLI SOLUTION (Recommended)")
    print("=" * 70)
    
    print("\nThe huggingface-cli tool downloads files directly without schema validation")
    print("\nStep 1: Check installation")
    print("  Command: which huggingface-cli")
    
    # Check if installed
    result = subprocess.run(['which', 'huggingface-cli'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Installed at: {result.stdout.strip()}")
    else:
        print("  ✗ Not installed. Install with:")
        print("    pip install huggingface-hub")
        print("    # This provides huggingface-cli command")
    
    print("\nStep 2: Enable high-speed chunked transfers")
    print("  Add to ~/.huggingface/hub_config.json or set env var:")
    print("    export HF_HUB_ENABLE_HF_TRANSFER=1")
    print("    export HF_HUB_CHUNK_SIZE=10485760  # 10MB chunks")
    
    print("\nStep 3: Download using huggingface-cli")
    print("  Command:")
    print("    huggingface-cli download Reverb/voxceleb2 --repo-type dataset \\")
    print("      --include 'vox2_dev_mp4.zip' \\")
    print("      --local-dir /localscratch/voxceleb2_hf")
    
    print("\nStep 4: Extract the downloaded file")
    print("  Command:")
    print("    unzip /localscratch/voxceleb2_hf/vox2_dev_mp4.zip -d /localscratch/voxceleb2_data")

def check_hf_cli_status():
    """Check huggingface-cli status"""
    print("\n" + "=" * 70)
    print("3. CURRENT HUGGINGFACE-CLI STATUS")
    print("=" * 70)
    
    try:
        result = subprocess.run(['huggingface-cli', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ huggingface-cli version: {result.stdout.strip()}")
        else:
            print("✗ huggingface-cli returned error")
    except FileNotFoundError:
        print("✗ huggingface-cli not found in PATH")
        print("  Install with: pip install huggingface-hub")
    except Exception as e:
        print(f"✗ Error checking huggingface-cli: {e}")
    
    # Check for hub_config
    hub_config = Path.home() / '.huggingface' / 'hub_config.json'
    print(f"\nHub config location: {hub_config}")
    if hub_config.exists():
        print("✓ Config file exists")
        with open(hub_config) as f:
            print(f"  Content:\n{f.read()}")
    else:
        print("✗ Config file not found (will use defaults)")

def show_voxceleb2_structure():
    """Show expected VoxCeleb2 structure"""
    print("\n" + "=" * 70)
    print("4. EXPECTED VOXCELEB2 STRUCTURE")
    print("=" * 70)
    
    print("\nVoxCeleb2 dataset hierarchy:")
    print("""
    /path/to/voxceleb2/
    ├── dev/                     # Development set
    │   └── mp4/
    │       ├── id00001/         # Speaker ID
    │       │   ├── 0000000/     # Video segment ID
    │       │   │   ├── 00001.mp4
    │       │   │   ├── 00002.mp4
    │       │   │   └── ...
    │       │   └── ...
    │       ├── id00002/
    │       │   └── ...
    │       └── ... (hundreds of speakers)
    └── test/                    # Test set (optional)
        └── mp4/
            └── ... (similar structure)
    """)
    
    print("Expected file count:")
    print("  - Development set: ~1,251 speakers, ~150K+ videos")
    print("  - Each speaker has 1-5 segments")
    print("  - Each segment has 1-5 videos")

def show_alternative_approaches():
    """Show alternative approaches if HF doesn't work"""
    print("\n" + "=" * 70)
    print("5. ALTERNATIVE APPROACHES")
    print("=" * 70)
    
    print("\nIf HuggingFace download is too slow or breaks:")
    
    print("\nOption A: Use official VoxCeleb2 source (when Oxford VGG is stable)")
    print("  - Official site: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/")
    print("  - Note: Currently unstable per your message")
    
    print("\nOption B: Synthetic dataset (already validated)")
    print("  - Already working: /localscratch/test_voxceleb2/")
    print("  - 50 videos validated with 100% success rate")
    print("  - Can be scaled to any size with create_test_voxceleb2_simple.py")
    
    print("\nOption C: Use existing local copy")
    print("  - Check if already downloaded elsewhere on cluster")
    print("  - Ask HPC administrator")

def generate_setup_script():
    """Generate a setup script for proper VoxCeleb2 download"""
    script_path = Path("/home/sengg/deep-learning-course-project/setup_voxceleb2_hf.sh")
    
    print("\n" + "=" * 70)
    print("6. GENERATING SETUP SCRIPT")
    print("=" * 70)
    
    script_content = """#!/bin/bash
# Setup script for VoxCeleb2 download using huggingface-cli
# This avoids the schema validation error by downloading files directly

set -e

echo "======================================================================"
echo "VoxCeleb2 Download Setup using huggingface-cli"
echo "======================================================================"

# Configuration
DOWNLOAD_DIR="/localscratch/voxceleb2_hf"
EXTRACT_DIR="/localscratch/voxceleb2"
DATASET_NAME="Reverb/voxceleb2"

echo ""
echo "1. Checking prerequisites..."

# Check huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "✗ huggingface-cli not found"
    echo "  Installing: pip install huggingface-hub"
    pip install huggingface-hub
fi
echo "✓ huggingface-cli found"

# Create directories
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$EXTRACT_DIR"

echo ""
echo "2. Configuring high-speed transfers..."
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CHUNK_SIZE=10485760  # 10MB chunks
echo "✓ Transfer optimization enabled"

echo ""
echo "3. Downloading VoxCeleb2 from HuggingFace..."
echo "   This may take several hours (dataset is ~100GB+)"
echo "   Repository: $DATASET_NAME"
echo "   Target: $DOWNLOAD_DIR"
echo ""

# Download without filtering to avoid schema issues
# Use --local-dir for incremental/resumable downloads
huggingface-cli download "$DATASET_NAME" \\
    --repo-type dataset \\
    --local-dir "$DOWNLOAD_DIR" \\
    --local-dir-use-symlinks False \\
    --resume-download

echo ""
echo "✓ Download complete"

echo ""
echo "4. Finding and extracting MP4 archives..."

# Look for zip files
for zipfile in "$DOWNLOAD_DIR"/vox2_*.zip; do
    if [ -f "$zipfile" ]; then
        filename=$(basename "$zipfile")
        echo "  Extracting: $filename"
        unzip -q "$zipfile" -d "$EXTRACT_DIR"
        echo "  ✓ Extracted"
    fi
done

echo ""
echo "5. Verifying extracted structure..."

if [ -d "$EXTRACT_DIR/dev/mp4" ]; then
    speaker_count=$(find "$EXTRACT_DIR/dev/mp4" -maxdepth 1 -type d | wc -l)
    video_count=$(find "$EXTRACT_DIR/dev/mp4" -name "*.mp4" | wc -l)
    echo "✓ VoxCeleb2 structure verified"
    echo "  Speakers: $(( speaker_count - 1 ))"
    echo "  Videos: $video_count"
else
    echo "✗ Expected structure not found at $EXTRACT_DIR/dev/mp4"
    echo "  Check the downloaded files"
fi

echo ""
echo "======================================================================"
echo "Setup complete! VoxCeleb2 is ready at: $EXTRACT_DIR"
echo ""
echo "Next step: Update script path"
echo "  voxceleb2_rt60_parallel.py --dataset-dir $EXTRACT_DIR/dev/mp4"
echo ""
echo "Or update submit scripts with:"
echo "  VOXCELEB2_ROOT=$EXTRACT_DIR/dev/mp4"
echo "======================================================================"
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    
    print(f"\n✓ Created setup script: {script_path}")
    print(f"  Usage: bash {script_path}")
    print(f"  This will:")
    print(f"    1. Check/install huggingface-cli")
    print(f"    2. Configure high-speed transfers")
    print(f"    3. Download VoxCeleb2 (~100GB)")
    print(f"    4. Extract to /localscratch/voxceleb2/dev/mp4")
    print(f"    5. Verify structure")

def show_recommendations():
    """Show final recommendations"""
    print("\n" + "=" * 70)
    print("7. FINAL RECOMMENDATIONS")
    print("=" * 70)
    
    print("\nFor VoxCeleb2 RT60 extraction:")
    print()
    print("SHORT TERM (Keep current tests working):")
    print("  ✓ Continue using synthetic test dataset")
    print("  ✓ Already validated: 100% success on 50 videos")
    print("  ✓ Jobs 5264874 and 5264881 are successful")
    print()
    print("MEDIUM TERM (Proper VoxCeleb2 setup):")
    print("  1. Run the generated setup script:")
    print("     bash setup_voxceleb2_hf.sh")
    print()
    print("  2. Wait for download to complete (~1-2 hours with good connection)")
    print()
    print("  3. Test with small subset:")
    print("     python voxceleb2_rt60_parallel.py \\")
    print("       --dataset-dir /localscratch/voxceleb2/dev/mp4 \\")
    print("       --limit 100 \\")
    print("       --output test_100.csv")
    print()
    print("  4. If successful, submit full job:")
    print("     sbatch submit_rt60_job.sh")
    print()
    print("KEY IMPROVEMENTS:")
    print("  - Direct file download (no schema validation)")
    print("  - High-speed chunked transfers (HF_HUB_ENABLE_HF_TRANSFER)")
    print("  - Resumable downloads (--local-dir)")
    print("  - No schema mismatch errors")

def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  VOXCELEB2 DEBUGGING & SETUP GUIDE".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    show_current_problem()
    show_hf_cli_approach()
    check_hf_cli_status()
    show_voxceleb2_structure()
    show_alternative_approaches()
    generate_setup_script()
    show_recommendations()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Current status:
  ✗ VoxCeleb2 via datasets library: BROKEN (schema mismatch)
  ✓ Synthetic test dataset: WORKING (100% success)

Next action:
  1. Review generated setup_voxceleb2_hf.sh script
  2. Run when ready to download full VoxCeleb2 (~24+ hours, 100GB+)
  3. OR continue with synthetic data for validation
  
Note: The setup script uses huggingface-cli which downloads files directly,
avoiding the schema validation errors from the datasets library.
    """)

if __name__ == '__main__':
    main()
