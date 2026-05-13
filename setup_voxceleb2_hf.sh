#!/bin/bash
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
huggingface-cli download "$DATASET_NAME" \
    --repo-type dataset \
    --local-dir "$DOWNLOAD_DIR" \
    --local-dir-use-symlinks False \
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
