#!/bin/bash

# Quick-start script for downloading and processing paired AV-Deepfake1M dataset
# 
# Usage:
#   ./quick_start_paired_download.sh [num_pairs] [output_dir]
#
# Examples:
#   ./quick_start_paired_download.sh            # Download 1000 pairs to /tmp/paired_av_deepfake
#   ./quick_start_paired_download.sh 100        # Download 100 pairs (test)
#   ./quick_start_paired_download.sh 1000 /home/sengg/data  # Custom location

set -e

# Configuration
NUM_PAIRS=${1:-1000}
OUTPUT_DIR=${2:-/tmp/paired_av_deepfake}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Paired AV-Deepfake1M Dataset Download"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Pairs to download: $NUM_PAIRS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Status: Starting..."
echo ""

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
echo "  - Checking for Python3..."
if ! command -v python3 &> /dev/null; then
    echo "    ERROR: Python3 not found"
    exit 1
fi
echo "    ✓ Python3 found: $(python3 --version)"

echo "  - Checking for yt-dlp..."
if ! command -v yt-dlp &> /dev/null; then
    echo "    ERROR: yt-dlp not found"
    echo "    Install with: pip install yt-dlp"
    exit 1
fi
echo "    ✓ yt-dlp found: $(yt-dlp --version)"

echo "  - Checking for ffmpeg (for verification)..."
if ! command -v ffprobe &> /dev/null; then
    echo "    WARNING: ffmpeg not found (needed for verification)"
    echo "    Install with: sudo apt-get install ffmpeg"
fi

echo ""
echo "Step 2: Downloading $NUM_PAIRS paired videos..."
echo "  This may take several hours depending on bandwidth"
echo "  (Sequential download to respect YouTube rate limits)"
echo ""

# Run the main downloader
python3 "$SCRIPT_DIR/download_paired_av_deepfake.py" \
    --output "$OUTPUT_DIR" \
    --num-pairs "$NUM_PAIRS"

DOWNLOAD_EXIT=$?

if [ $DOWNLOAD_EXIT -ne 0 ]; then
    echo "ERROR: Download failed"
    exit 1
fi

echo ""
echo "Step 3: Verifying downloaded videos..."
if command -v ffprobe &> /dev/null; then
    python3 "$SCRIPT_DIR/verify_paired_dataset.py" \
        --dataset-dir "$OUTPUT_DIR"
else
    echo "  Skipping verification (ffmpeg not installed)"
fi

echo ""
echo "========================================"
echo "✓ Download Complete!"
echo "========================================"
echo ""
echo "Output location: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify all videos downloaded successfully:"
echo "     ls $OUTPUT_DIR/fake_audio/mp4/ -d */ | wc -l"
echo ""
echo "  2. Run RT60 extraction on Tinkercliff:"
echo "     sbatch submit_rt60_job_paired.sh --dataset-dir $OUTPUT_DIR"
echo ""
echo "  3. Or extract locally:"
echo "     python rt60_extractor.py --video-list $OUTPUT_DIR/rt60_paired_videos.txt"
echo ""
