# RT60 Extraction Pipeline for AV-Deepfake1M Dataset

This project extracts reverberation time (RT60) from the audio channels of videos in the AV-Deepfake1M dataset using the **blind_rt60** library.

## Overview

RT60 is the time it takes for sound to decay to -60dB. It's a key acoustic parameter that characterizes room acoustics and can be useful for:
- Audio quality assessment
- Deepfake detection (audio tampering)
- Room acoustic profiling

## Installation

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Verify installation (optional):
```bash
python test_rt60.py
```

## Usage

### Option A: Using Metadata JSON (for AV-Deepfake1M dataset)

If you have the AV-Deepfake1M dataset with metadata JSON:

```bash
python rt60_extractor.py \
    --metadata /path/to/train_metadata.json \
    --dataset-dir /path/to/dataset \
    --output results.csv \
    --limit 100
```

**Arguments:**
- `--metadata`: Path to metadata.json file from AV-Deepfake1M
- `--dataset-dir`: Base directory containing videos
- `--output`: Output CSV filename (default: rt60_results.csv)
- `--limit`: Process only first N videos (useful for testing)

### Option B: Using Video List

If you have a text file with one video path per line:

```bash
python rt60_extractor.py \
    --video-list videos.txt \
    --output results.csv
```

### Option C: Programmatic Usage

```python
from rt60_extractor import RT60Extractor
import numpy as np

# Initialize extractor
extractor = RT60Extractor()

# Extract audio and estimate RT60
video_path = "path/to/video.mp4"
audio = extractor.extract_audio_from_video(video_path)
rt60 = extractor.estimate_rt60(audio)

print(f"RT60: {rt60:.3f} seconds")

# Or process batch
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
extractor.process_batch(video_paths, "results.csv")
```

## Output

The script generates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `video_path` | Full path to video file |
| `filename` | Video filename |
| `rt60` | Estimated RT60 in seconds |
| `audio_duration` | Duration of audio in seconds |
| `status` | Processing status (success/failed) |

Example output:
```
video_path,filename,rt60,audio_duration,status
/data/videos/video1.mp4,video1.mp4,0.456,10.234,success
/data/videos/video2.mp4,video2.mp4,0.382,9.876,success
```

## Algorithm Details

The extraction pipeline uses **blind_rt60**, which implements the algorithm from:

> Ratnam, R., Jones, D., Wheeler, B., O'Brien, W., Lansing, C., & Feng, A. (2003). 
> "Blind estimation of reverberation time." 
> The Journal of the Acoustical Society of America, 114(6), 2877-2892.

**Key features:**
- Blind estimation: No need for impulse response or reference signals
- Works directly on speech/music audio
- Uses maximum likelihood estimation with exponential decay model
- Handles varying audio lengths and quality

## Performance Notes

- **Processing speed**: ~1-2 videos per minute (depending on audio length)
- **Accuracy**: Most reliable for audio >1 second with clear decay
- **Hardware**: CPU-based, no GPU required
- **Memory**: Minimal memory footprint (~100MB for typical audio)

## Troubleshooting

### Audio extraction fails
- Ensure ffmpeg is installed: `sudo apt-get install ffmpeg`
- Check video codec compatibility with librosa
- Verify file exists and is readable

### RT60 estimation returns None
- Audio too short (< 1 second)
- Very low signal-to-noise ratio
- No clear decay pattern

### Memory issues with large batches
- Reduce batch size or process videos sequentially
- Add `--limit` flag to test on subset first

## Next Steps

1. **Validation**: Compare results with traditional Schroeder method
2. **Analysis**: Correlate RT60 with deepfake type (audio_modified vs real)
3. **Visualization**: Create histograms/plots of RT60 distributions
4. **Feature extraction**: Use RT60 as feature for deepfake detection

## License

MIT License - See LICENSE file for details

## Reference

Original Paper: arXiv:2107.13832 - "Blind Room Parameter Estimation Using Multiple-Multichannel Speech Recordings"
