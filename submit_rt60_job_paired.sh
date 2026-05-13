#!/bin/bash
#SBATCH --job-name=rt60_paired
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-node=32G
#SBATCH --output=rt60_paired_%j.out
#SBATCH --error=rt60_paired_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@vt.edu

# RT60 Extraction Job for Paired AV-Deepfake1M Dataset
#
# This job extracts RT60 (reverberation time) from paired fake/real audio videos
# for comprehensive acoustic analysis and deepfake detection.
#
# Submit with:
#   sbatch submit_rt60_job_paired.sh --dataset-dir /path/to/paired_dataset
#
# Configuration:
#   - Nodes: 2 (32 cores total)
#   - Memory: 64 GB total
#   - Walltime: 24 hours
#   - Suitable for 1000 pairs (~2000 videos)

set -e

# Parse command line arguments
DATASET_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_ID=$SLURM_JOB_ID
NODES=$SLURM_JOB_NUM_NODES
CORES=$((SLURM_NTASKS_PER_NODE * NODES))

# Use provided dataset dir or default
if [ -z "$DATASET_DIR" ]; then
    DATASET_DIR="/tmp/paired_av_deepfake"
fi

FAKE_LIST="$DATASET_DIR/rt60_fake_videos.txt"
REAL_LIST="$DATASET_DIR/rt60_real_videos.txt"

echo "========================================"
echo "RT60 EXTRACTION - PAIRED DATASET"
echo "========================================"
echo "Job ID: $JOB_ID"
echo "Nodes: $NODES"
echo "Cores: $CORES"
echo "Dataset: $DATASET_DIR"
echo ""

# Check dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    echo "Download dataset first using: python download_paired_av_deepfake.py"
    exit 1
fi

# Check video lists exist
if [ ! -f "$FAKE_LIST" ]; then
    echo "Creating video lists from dataset..."
    python3 "$PROJECT_DIR/verify_paired_dataset.py" \
        --dataset-dir "$DATASET_DIR"
fi

if [ ! -f "$FAKE_LIST" ] || [ ! -f "$REAL_LIST" ]; then
    echo "ERROR: Video lists not found"
    echo "Run verify_paired_dataset.py first"
    exit 1
fi

NUM_PAIRS=$(wc -l < "$FAKE_LIST")
echo "Video pairs to process: $NUM_PAIRS"
echo ""

# Activate Python environment if needed
if [ -f "$HOME/venv/bin/activate" ]; then
    source "$HOME/venv/bin/activate"
    echo "Using virtual environment: $VIRTUAL_ENV"
fi

# Create output directory
mkdir -p "$DATASET_DIR/rt60_results"

echo "Starting RT60 extraction..."
echo "========================================"

# Phase 1: Extract RT60 from fake audio (synthetic/modified)
echo ""
echo "Phase 1: Extracting RT60 from FAKE AUDIO videos..."
echo "Output: $DATASET_DIR/rt60_results/fake_rt60_$JOB_ID.csv"

python3 "$PROJECT_DIR/rt60_extractor.py" \
    --video-list "$FAKE_LIST" \
    --output "$DATASET_DIR/rt60_results/fake_rt60_$JOB_ID.csv" \
    --workers "$CORES" \
    --batch-size 100

FAKE_EXIT=$?
if [ $FAKE_EXIT -ne 0 ]; then
    echo "ERROR: Fake audio extraction failed"
    exit 1
fi

# Phase 2: Extract RT60 from real audio (original)
echo ""
echo "Phase 2: Extracting RT60 from REAL AUDIO videos..."
echo "Output: $DATASET_DIR/rt60_results/real_rt60_$JOB_ID.csv"

python3 "$PROJECT_DIR/rt60_extractor.py" \
    --video-list "$REAL_LIST" \
    --output "$DATASET_DIR/rt60_results/real_rt60_$JOB_ID.csv" \
    --workers "$CORES" \
    --batch-size 100

REAL_EXIT=$?
if [ $REAL_EXIT -ne 0 ]; then
    echo "ERROR: Real audio extraction failed"
    exit 1
fi

# Phase 3: Merge and analyze results
echo ""
echo "Phase 3: Merging and analyzing results..."

python3 << 'PYTHON_ANALYSIS'
import pandas as pd
import sys
import os

dataset_dir = os.environ.get('DATASET_DIR')
job_id = os.environ.get('JOB_ID')

fake_csv = f"{dataset_dir}/rt60_results/fake_rt60_{job_id}.csv"
real_csv = f"{dataset_dir}/rt60_results/real_rt60_{job_id}.csv"
output_csv = f"{dataset_dir}/rt60_results/paired_rt60_analysis_{job_id}.csv"

print(f"Loading fake audio results: {fake_csv}")
fake_df = pd.read_csv(fake_csv)

print(f"Loading real audio results: {real_csv}")
real_df = pd.read_csv(real_csv)

# Merge results
print(f"Merging results...")
fake_df['audio_type'] = 'fake'
real_df['audio_type'] = 'real'

merged_df = pd.concat([fake_df, real_df], ignore_index=True)
merged_df.to_csv(output_csv, index=False)

# Compute statistics
print(f"\n{'='*60}")
print(f"PAIRED RT60 ANALYSIS")
print(f"{'='*60}")

fake_rt60 = fake_df['rt60'].dropna()
real_rt60 = real_df['rt60'].dropna()

print(f"\nFAKE AUDIO (Synthetic/Modified):")
print(f"  Count: {len(fake_rt60)}")
print(f"  Mean RT60: {fake_rt60.mean():.3f} s")
print(f"  Median RT60: {fake_rt60.median():.3f} s")
print(f"  Std Dev: {fake_rt60.std():.3f} s")
print(f"  Min: {fake_rt60.min():.3f} s")
print(f"  Max: {fake_rt60.max():.3f} s")

print(f"\nREAL AUDIO (Original):")
print(f"  Count: {len(real_rt60)}")
print(f"  Mean RT60: {real_rt60.mean():.3f} s")
print(f"  Median RT60: {real_rt60.median():.3f} s")
print(f"  Std Dev: {real_rt60.std():.3f} s")
print(f"  Min: {real_rt60.min():.3f} s")
print(f"  Max: {real_rt60.max():.3f} s")

print(f"\nCOMPARISON:")
if len(fake_rt60) > 0 and len(real_rt60) > 0:
    rt60_diff = (fake_rt60.mean() - real_rt60.mean())
    print(f"  Mean difference: {rt60_diff:.3f} s (Fake - Real)")
    print(f"  Percent difference: {(rt60_diff/real_rt60.mean())*100:.1f}%")
    
    # Simple t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(fake_rt60, real_rt60)
    print(f"  T-test p-value: {p_value:.2e}")
    if p_value < 0.05:
        print(f"  ✓ Significant difference between fake and real audio RT60")
    else:
        print(f"  No significant difference detected")

print(f"\n{'='*60}")
print(f"Results saved to: {output_csv}")
print(f"{'='*60}\n")

PYTHON_ANALYSIS

# Create final report
echo ""
echo "Creating comprehensive report..."

python3 << 'PYTHON_REPORT'
import json
import pandas as pd
import os
from datetime import datetime

dataset_dir = os.environ.get('DATASET_DIR')
job_id = os.environ.get('JOB_ID')

output_csv = f"{dataset_dir}/rt60_results/paired_rt60_analysis_{job_id}.csv"
report_file = f"{dataset_dir}/rt60_results/job_report_{job_id}.json"

report = {
    'job_id': job_id,
    'timestamp': datetime.now().isoformat(),
    'dataset_dir': dataset_dir,
    'output_csv': output_csv,
    'status': 'complete',
}

if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
    fake_data = df[df['audio_type'] == 'fake']['rt60'].dropna()
    real_data = df[df['audio_type'] == 'real']['rt60'].dropna()
    
    report['statistics'] = {
        'fake_audio': {
            'count': int(len(fake_data)),
            'mean': float(fake_data.mean()),
            'median': float(fake_data.median()),
            'std': float(fake_data.std()),
            'min': float(fake_data.min()),
            'max': float(fake_data.max()),
        },
        'real_audio': {
            'count': int(len(real_data)),
            'mean': float(real_data.mean()),
            'median': float(real_data.median()),
            'std': float(real_data.std()),
            'min': float(real_data.min()),
            'max': float(real_data.max()),
        }
    }

with open(report_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"Report saved to: {report_file}")
PYTHON_REPORT

echo ""
echo "========================================"
echo "✓ RT60 EXTRACTION COMPLETE"
echo "========================================"
echo ""
echo "Results Location:"
echo "  Dataset: $DATASET_DIR"
echo "  Fake audio RT60: $DATASET_DIR/rt60_results/fake_rt60_$JOB_ID.csv"
echo "  Real audio RT60: $DATASET_DIR/rt60_results/real_rt60_$JOB_ID.csv"
echo "  Paired analysis: $DATASET_DIR/rt60_results/paired_rt60_analysis_$JOB_ID.csv"
echo "  Report: $DATASET_DIR/rt60_results/job_report_$JOB_ID.json"
echo ""
echo "Download results:"
echo "  scp -r $DATASET_DIR/rt60_results YOUR_LOCAL_MACHINE:~/results/"
echo ""
