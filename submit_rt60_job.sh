#!/bin/bash
#SBATCH --job-name=rt60-voxceleb2
#SBATCH --account=cs5814

# ============================================================================
# Tinkercliff Job Configuration
# ============================================================================

# Partition: normal_q (CPU,7-day), can also use:
#   - tc_normal_short (1-day max, 2x billing, higher priority)
#   - tc_normal_long (14-day max, 0.5x priority, lower resource limits)
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_base

# Node/Core Configuration
# Using 4 nodes with 32 tasks each = 128 total parallel workers
# This fits within normal_q limits (10496 cores available per job)
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1

# Memory: ~2GB per core (256GB/128 cores)
#SBATCH --mem-per-cpu=2000

# Walltime: 24 hours (adjust based on dataset size)
# For 1M videos at 18 sec/video with 128 cores: ~1.3M seconds / 128 = ~10,400 seconds = ~3 hours
# For safety and larger batches, use 24 hours
#SBATCH --time=24:00:00

# Output and error logs (use %j for job ID)
#SBATCH --output=rt60_job_%j.out
#SBATCH --error=rt60_job_%j.err

# Email notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sengg@vt.edu

# ============================================================================
# Environment Setup
# ============================================================================

# Print job information
echo "============================================================"
echo "RT60 VoxCeleb2 Processing - Tinkercliff Cluster"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per node: $SLURM_CPUS_PER_NODE"
echo "Total CPUs: $SLURM_NTASKS"
echo "Memory per CPU: $SLURM_MEM_PER_CPU MB"
echo "Walltime: $SLURM_TIMELIMIT"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Load modules
module purge
module load GCC/13.2.0
module load OpenMPI/4.1.5
module load Python/3.11.5

# Activate virtual environment (if using conda/venv)
# source /path/to/venv/bin/activate

# OR create temporary virtual environment on compute node
cd /localscratch
python3 -m venv venv_rt60
source venv_rt60/bin/activate

# Install required packages
pip install --quiet librosa numpy scipy pandas blind-rt60 datasets huggingface-hub

echo "Environment loaded. Python version: $(python --version)"
echo ""

# ============================================================================
# Dataset Configuration
# ============================================================================

# Download VoxCeleb2 from HuggingFace (Reverb/voxceleb2 mirror)
echo "Downloading VoxCeleb2 dataset from HuggingFace..."
VOXCELEB2_ROOT="/localscratch/voxceleb2"
mkdir -p "$VOXCELEB2_ROOT"

python3 << 'PYTHONEOF'
from datasets import load_dataset
import os

cache_dir = "/localscratch/voxceleb2"
print(f"Downloading VoxCeleb2 to {cache_dir}...")

try:
    # Load the VoxCeleb2 dataset from HuggingFace
    ds = load_dataset("Reverb/voxceleb2", split="dev", streaming=False, cache_dir=cache_dir)
    print("✓ VoxCeleb2 dataset downloaded successfully")
except Exception as e:
    print(f"Error downloading from HuggingFace: {e}")
    print("Make sure you have HuggingFace credentials set up")
    exit(1)
PYTHONEOF

# Output CSV file (will be written to scratch)
OUTPUT_CSV="/localscratch/voxceleb2_rt60_${SLURM_JOB_ID}.csv"

# Split to process: 'dev' or 'test'
SPLIT="dev"

# Optional: limit number of videos for testing
# Leave empty to process all videos
VIDEO_LIMIT=""

# ============================================================================
# Error Handling
# ============================================================================

# Exit on first error
set -e

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Job cleanup at $(date)"
    if [ -d "/localscratch/venv_rt60" ]; then
        rm -rf /localscratch/venv_rt60
    fi
}

trap cleanup EXIT

# ============================================================================
# Processing
# ============================================================================

echo "Dataset Configuration:"
echo "  VoxCeleb2 Root: $VOXCELEB2_ROOT"
echo "  Split: $SPLIT"
echo "  Output CSV: $OUTPUT_CSV"
echo "  Workers: $SLURM_NTASKS"
echo ""

# Change to working directory
cd /home/sengg/deep-learning-course-project

# Run parallel RT60 extraction
echo "Starting RT60 extraction at $(date)"
echo ""

if [ -z "$VIDEO_LIMIT" ]; then
    python voxceleb2_rt60_parallel.py \
        --voxceleb2-root "$VOXCELEB2_ROOT" \
        --split "$SPLIT" \
        --output "$OUTPUT_CSV" \
        --workers "$SLURM_NTASKS"
else
    python voxceleb2_rt60_parallel.py \
        --voxceleb2-root "$VOXCELEB2_ROOT" \
        --split "$SPLIT" \
        --limit "$VIDEO_LIMIT" \
        --output "$OUTPUT_CSV" \
        --workers "$SLURM_NTASKS"
fi

EXTRACT_STATUS=$?

echo ""
echo "RT60 extraction completed at $(date)"
echo "Exit status: $EXTRACT_STATUS"
echo ""

# ============================================================================
# Post-Processing
# ============================================================================

if [ $EXTRACT_STATUS -eq 0 ]; then
    echo "Processing Results:"
    echo "  Output file: $OUTPUT_CSV"
    echo "  File size: $(ls -lh $OUTPUT_CSV | awk '{print $5}')"
    echo "  Row count: $(wc -l < $OUTPUT_CSV)"
    echo ""
    
    # Copy results to home directory (optional)
    if [ -f "$OUTPUT_CSV" ]; then
        cp "$OUTPUT_CSV" "/home/sengg/deep-learning-course-project/results_${SLURM_JOB_ID}.csv"
        echo "  Results copied to: /home/sengg/deep-learning-course-project/results_${SLURM_JOB_ID}.csv"
    fi
else
    echo "ERROR: Processing failed with exit code $EXTRACT_STATUS"
    exit $EXTRACT_STATUS
fi

echo ""
echo "============================================================"
echo "Job completed successfully at $(date)"
echo "============================================================"
