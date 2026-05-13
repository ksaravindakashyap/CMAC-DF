# Download Paired AV-Deepfake1M Dataset (RVFA-RVRA Pairs)

This guide explains how to download **1000 paired videos** from AV-Deepfake1M:
- **RVFA** (Real Video, Fake Audio): Videos with synthetic audio
- **RVRA** (Real Video, Real Audio): The original source videos from VoxCeleb2

Perfect for your RT60 acoustic analysis comparing real vs synthetic audio in the same visual environment.

## Overview

```
downloaded_pairs/
├── fake_audio/          # 1000 videos with synthetic/modified audio
│   └── mp4/
│       ├── pair_0000/video.mp4
│       ├── pair_0001/video.mp4
│       └── ...
├── real_audio/          # 1000 matching real videos
│   └── mp4/
│       ├── pair_0000/video.mp4
│       ├── pair_0001/video.mp4
│       └── ...
├── metadata/            # Downloaded metadata
│   └── av_deepfake1m_metadata.csv
└── manifest.json        # Pair mapping and download stats
```

## Prerequisites

### 1. Install yt-dlp (YouTube downloader)
```bash
pip install yt-dlp
```

### 2. Verify installation
```bash
yt-dlp --version
# Should output: version number (e.g., 2024.01.01)
```

### 3. Check internet bandwidth
- Each video: ~5-20 MB
- Total for 1000 pairs: ~10-40 GB
- Estimated download time: 6-24 hours (depends on bandwidth and YouTube rate limits)

## Step-by-Step Usage

### Option 1: Quick Start (1000 pairs)

```bash
cd /home/sengg/deep-learning-course-project

# Download to default location (/tmp/paired_av_deepfake)
python download_paired_av_deepfake.py

# Or specify custom output directory
python download_paired_av_deepfake.py \
    --output /home/sengg/paired_av_deepfake \
    --num-pairs 1000
```

### Option 2: Test First (small subset)

```bash
# Download just 10 pairs to test the pipeline
python download_paired_av_deepfake.py \
    --output /tmp/test_pairs \
    --num-pairs 10
```

### Option 3: Continue interrupted download

The script automatically skips existing files:

```bash
# If download was interrupted, just run again with same output path
python download_paired_av_deepfake.py \
    --output /home/sengg/paired_av_deepfake \
    --num-pairs 1000
# It will find existing videos and skip them
```

## What the Script Does

### Phase 1: Fetch Metadata
- Downloads official AV-Deepfake1M metadata CSV (~100 MB)
- Caches locally in `{output}/metadata/`
- Next run uses cached version

### Phase 2: Identify Pairs
- Filters for RVFA videos (fake audio)
- Matches with RVRA counterparts (real audio)
- Creates 1000 paired entries

### Phase 3: Extract Download Info
- Extracts YouTube URLs for each video
- Extracts timestamp (start/end seconds)
- Creates download queue

### Phase 4: Download Videos
- Uses yt-dlp to fetch from YouTube
- Trims to original timestamp duration
- Saves with MP4 codec
- Sequential download (respects YouTube rate limits)

### Phase 5: Generate Manifest
- Creates `manifest.json` with all pair mappings
- Includes download statistics
- Ready for RT60 extraction!

## Output Structure

### Directory Layout
```
fake_audio/mp4/
├── pair_0000/
│   └── video.mp4        # RVFA video 0 (synthetic audio)
├── pair_0001/
│   └── video.mp4        # RVFA video 1
└── ...

real_audio/mp4/
├── pair_0000/
│   └── video.mp4        # RVRA video 0 (original audio)
├── pair_0001/
│   └── video.mp4        # RVRA video 1 (original audio)
└── ...
```

### Manifest Structure (manifest.json)
```json
{
  "metadata": {
    "total_pairs": 1000,
    "successful_pairs": 950,
    "failed_pairs": 50,
    "fake_videos_downloaded": 950,
    "real_videos_downloaded": 950
  },
  "pairs": [
    {
      "pair_id": "pair_0000",
      "fake": {
        "video_id": "deepfake_12345",
        "url": "https://www.youtube.com/watch?v=...",
        "start": 10,
        "end": 25
      },
      "real": {
        "video_id": "voxceleb_12345",
        "url": "https://www.youtube.com/watch?v=...",
        "start": 10,
        "end": 25
      }
    },
    ...
  ]
}
```

## Running RT60 Extraction on Downloaded Pairs

Once downloads complete, extract RT60 values:

```bash
# Create extraction script for paired dataset
python -c "
import csv
import glob
from pathlib import Path

fake_videos = sorted(glob.glob('fake_audio/mp4/*/video.mp4'))
real_videos = sorted(glob.glob('real_audio/mp4/*/video.mp4'))

with open('paired_videos.txt', 'w') as f:
    for fake, real in zip(fake_videos, real_videos):
        f.write(f'{fake}\t{real}\n')

print(f'Created paired_videos.txt with {len(fake_videos)} pairs')
"

# Then run RT60 extraction
python rt60_extractor.py \
    --video-list paired_videos.txt \
    --output rt60_paired_results.csv
```

## Monitoring Progress

### Real-time Progress
```bash
# In another terminal, watch for downloaded files
watch -n 5 'ls fake_audio/mp4/ | wc -l'
```

### Check Download Logs
```bash
# View the last 50 lines of current session
tail -50 output_from_script.log
```

## Troubleshooting

### Issue: "yt-dlp not found"
```bash
pip install --upgrade yt-dlp
which yt-dlp
```

### Issue: "No matching RVRA for RVFA"
- Metadata may be incomplete
- Some deepfakes may not have direct VoxCeleb2 source
- Script handles this automatically (logs warnings)

### Issue: YouTube Rate Limiting
- Videos fail to download with timeout errors
- Solution: Run script with delays between requests
- The script is sequential to minimize rate limiting

### Issue: Disk Space
```bash
# Check available space
df -h /tmp/paired_av_deepfake

# Each pair occupies:
# - Fake video: 5-15 MB
# - Real video: 5-15 MB  
# - Total: 10-40 GB for 1000 pairs
```

### Issue: Corrupt Downloads
- If a video appears corrupted, delete it
- Rerun the script - it will redownload missing files

## Performance Expectations

| Metric | Value |
|--------|-------|
| Videos per pair | 2 (fake + real) |
| Total pairs | 1000 |
| Total videos | 2000 |
| Avg video size | 10 MB |
| Total storage | ~20 GB |
| Download time | 6-24 hours |
| Success rate | ~95% (some videos removed from YouTube) |

## Advanced Options

### Parallel Download (Experimental)

For faster downloads with multiple workers, use GNU Parallel:

```bash
# Install GNU Parallel
sudo apt-get install parallel

# Modify script to use parallel downloads
# (Not recommended due to YouTube rate limits)
```

### Custom Audio Modifications

To analyze specific types of audio modifications:

```bash
# Filter metadata for specific voice conversion types
# or verify known deepfake characteristics
grep "voice_conversion" av_deepfake1m_metadata.csv
```

## Next Steps After Download

### 1. Verify Downloaded Videos
```bash
# Check all videos are valid MP4s
for video in fake_audio/mp4/*/video.mp4; do
    ffprobe -v error -select_streams a:0 "$video" > /dev/null && echo "OK: $video" || echo "CORRUPT: $video"
done
```

### 2. Extract RT60 Values
```bash
# Run RT60 extraction on paired dataset
python rt60_extractor.py \
    --video-list paired_videos.txt \
    --output rt60_paired_analysis.csv
```

### 3. Analyze Results
```python
import pandas as pd

# Load RT60 results
df = pd.read_csv('rt60_paired_analysis.csv')

# Compare fake vs real RT60 values
print("Real audio RT60 - mean:", df[df['type'] == 'real']['rt60'].mean())
print("Fake audio RT60 - mean:", df[df['type'] == 'fake']['rt60'].mean())
```

## Metadata Details

The downloaded metadata contains:

| Column | Description |
|--------|-------------|
| `video_id` | Unique video identifier |
| `label` | 'RVFA' (fake audio) or 'RVRA' (real audio) |
| `voxceleb_id` | Source VoxCeleb2 speaker ID |
| `youtube_url` | YouTube video URL |
| `start_time` | Start timestamp (seconds) |
| `end_time` | End timestamp (seconds) |
| `modification_type` | Type of audio modification |

## References

- **AV-Deepfake1M Repository**: https://github.com/ControlNet/AV-Deepfake1M
- **VoxCeleb2 Dataset**: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/voxceleb2.html
- **yt-dlp Documentation**: https://github.com/yt-dlp/yt-dlp
- **RT60 Extraction**: Reverberation time analysis for acoustic characterization

## Computing Resources

### Local Machine (Standard)
- CPU: 4+ cores
- RAM: 8 GB minimum
- Storage: 25-50 GB available
- Network: Broadband (100+ Mbps recommended)
- Time: 6-24 hours

### Cluster/HPC (After Download)
```bash
# Submit RT60 extraction job on Tinkercliff
sbatch submit_rt60_job_paired.sh
```

## Support & Issues

If you encounter issues:

1. Check the output directory structure
2. Verify yt-dlp is working: `yt-dlp --version`
3. Check manifest.json for failed pairs
4. Review logs for specific error messages
5. Rerun the script to resume (it's resumable)

## Citation

If you use this dataset for research:

```bibtex
@article{deepfake1m,
  title={AV-Deepfake1M: A Large-Scale Audio-Visual Deepfake Dataset},
  author={...},
  year={2023}
}

@inproceedings{voxceleb2,
  title={VoxCeleb2: Deep Speaker Recognition},
  author={Chung et al},
  booktitle={INTERSPEECH},
  year={2018}
}
```
