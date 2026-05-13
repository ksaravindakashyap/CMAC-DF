# Output File Generation & VoxCeleb2 Pipeline

## Part 1: How Output Files Are Generated

### Output Format: CSV (Comma-Separated Values)

The pipeline generates **CSV files** - a simple, human-readable, machine-readable text format.

#### Example Output Structure:

```
video_path,filename,rt60,audio_duration,status
/data/videos/video1.mp4,video1.mp4,0.456,10.234,success
/data/videos/video2.mp4,video2.mp4,0.382,9.876,success
/data/videos/video3.mp4,video3.mp4,,5.123,failed
```

### How CSV Files Are Generated (Python Code)

```python
import csv

# 1. Prepare results as list of dictionaries
results = [
    {
        'video_path': '/data/video1.mp4',
        'filename': 'video1.mp4',
        'rt60': 0.456,
        'audio_duration': 10.234,
        'status': 'success'
    },
    {
        'video_path': '/data/video2.mp4',
        'filename': 'video2.mp4',
        'rt60': 0.382,
        'audio_duration': 9.876,
        'status': 'success'
    }
]

# 2. Open file for writing
with open('results.csv', 'w', newline='') as csvfile:
    # 3. Create CSV writer with column names from dictionary keys
    fieldnames = results[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # 4. Write header row (column names)
    writer.writeheader()
    
    # 5. Write data rows (one dictionary per row)
    writer.writerows(results)
```

### Key Points About CSV Output:

| Aspect | Details |
|--------|---------|
| **File Format** | Plain text with comma separators |
| **Encoding** | UTF-8 (standard text) |
| **Line Endings** | `\n` (newline) |
| **Headers** | First row contains column names |
| **Data Types** | All stored as text (readable) |
| **Size** | Compact - ~100 bytes per video |
| **Compatibility** | Opens in Excel, Python, R, any tool |

### Opening CSV Files

**Python:**
```python
import pandas as pd
df = pd.read_csv('results.csv')
print(df.head())
```

**Command line:**
```bash
cat results.csv              # View all
head -20 results.csv         # First 20 rows
wc -l results.csv            # Count rows
```

**Excel/Sheets:**
- Right-click → Open With → Excel
- Or drag & drop into spreadsheet

### CSV Advantages

✓ **Human-readable** - Can view/edit in text editor
✓ **Tool-agnostic** - Works everywhere (Python, R, Excel, SQL, etc.)
✓ **Lightweight** - Small file size
✓ **Indexable** - Can query with pandas, SQL, etc.
✓ **Version-controllable** - Can track changes in git
✓ **Scriptable** - Easy to post-process

---

## Part 2: VoxCeleb2 RT60 Extraction

### What is VoxCeleb2?

- **1M+ utterances** from 6,112 speakers
- **Speaker identification** dataset
- **Audio-visual** (has both video and audio)
- **In-the-wild** videos from YouTube
- **3+ seconds** per clip
- **2,000+ hours** of speech

### Directory Structure

```
VoxCeleb2/
├── dev/                          # Development set (training)
│   ├── aac/                       # Audio files (.m4a)
│   │   ├── id10001/               # Speaker ID
│   │   │   ├── 1HDQeFPvL6c/       # Utterance ID
│   │   │   │   ├── 00001.m4a
│   │   │   │   └── 00002.m4a
│   │   └── id10002/
│   └── mp4/                       # Video files (.mp4)
│       ├── id10001/
│       │   └── 1HDQeFPvL6c/
│       │       ├── 00001.mp4
│       │       └── 00002.mp4
│       └── id10002/
└── test/                          # Test set
    ├── aac/
    └── mp4/
```

### Setup: Download & Extract RT60

#### Step 1: Download VoxCeleb2
```bash
# Register and download from:
# https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html

# Extract to a directory, e.g., /data/VoxCeleb2
```

#### Step 2: Extract RT60 from 100 videos (test run)
```bash
python voxceleb2_rt60_extractor.py \
    --voxceleb2-root /path/to/VoxCeleb2 \
    --split dev \
    --limit 100 \
    --output voxceleb2_rt60_sample.csv
```

#### Step 3: Extract RT60 from all dev videos
```bash
python voxceleb2_rt60_extractor.py \
    --voxceleb2-root /path/to/VoxCeleb2 \
    --split dev \
    --output voxceleb2_rt60_dev.csv
```

#### Step 4: Extract RT60 from test videos
```bash
python voxceleb2_rt60_extractor.py \
    --voxceleb2-root /path/to/VoxCeleb2 \
    --split test \
    --output voxceleb2_rt60_test.csv
```

### Output CSV Format (VoxCeleb2)

```
video_path,relative_path,video_filename,speaker_id,utterance_id,rt60,audio_duration,status
/data/VoxCeleb2/dev/mp4/id10001/1HDQeFPvL6c/00001.mp4,dev/mp4/id10001/1HDQeFPvL6c/00001.mp4,00001.mp4,id10001,1HDQeFPvL6c,0.456,10.234,success
/data/VoxCeleb2/dev/mp4/id10002/2yCdaNq7p3K/00001.mp4,dev/mp4/id10002/2yCdaNq7p3K/00001.mp4,00001.mp4,id10002,2yCdaNq7p3K,0.382,9.876,success
```

| Column | Meaning |
|--------|---------|
| `video_path` | Full absolute path |
| `relative_path` | Path relative to VoxCeleb2 root |
| `video_filename` | Just the filename (00001.mp4) |
| `speaker_id` | Speaker identifier (id10001) |
| `utterance_id` | Utterance identifier (1HDQeFPvL6c) |
| `rt60` | Reverberation time in seconds |
| `audio_duration` | Audio length in seconds |
| `status` | success/failed |

### Usage Examples

#### 1. Load and inspect results
```python
import pandas as pd

df = pd.read_csv('voxceleb2_rt60_sample.csv')
print(df.head())
print(f"Total videos: {len(df)}")
print(f"Mean RT60: {df['rt60'].mean():.3f}s")
```

#### 2. Find videos by speaker
```python
# Get all videos for speaker id10001
speaker_videos = df[df['speaker_id'] == 'id10001']
print(f"Found {len(speaker_videos)} videos for speaker id10001")
print(speaker_videos[['utterance_id', 'rt60', 'audio_duration']])
```

#### 3. Analyze RT60 distribution
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df['rt60'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('RT60 (seconds)')
plt.ylabel('Frequency')
plt.title('VoxCeleb2 RT60 Distribution')
plt.grid(True, alpha=0.3)
plt.savefig('voxceleb2_rt60_distribution.png')
print("Saved: voxceleb2_rt60_distribution.png")
```

#### 4. Filter by audio quality (duration)
```python
# Videos >5 seconds (longer, likely better for RT60 estimation)
long_videos = df[df['audio_duration'] > 5]
print(f"Videos >5 seconds: {len(long_videos)}")

# Videos with confident RT60 (those that succeeded)
successful = df[df['status'] == 'success']
print(f"Successful extractions: {len(successful)}")
```

#### 5. Export subset for other analysis
```python
# Get top 10 speakers with most videos
top_speakers = df['speaker_id'].value_counts().head(10)
print(top_speakers)

# Export RT60 data for specific speakers
for speaker in top_speakers.index:
    speaker_df = df[df['speaker_id'] == speaker]
    speaker_df.to_csv(f'rt60_{speaker}.csv', index=False)
```

---

## Part 3: Comparing Pipelines

### AV-Deepfake1M vs VoxCeleb2

| Aspect | AV-Deepfake1M | VoxCeleb2 |
|--------|---------------|-----------|
| **Underlying** | Based on VoxCeleb2 | Original source |
| **Videos** | 1M+ (modified versions) | 1M+ (original) |
| **Speakers** | 2K+ | 6,112 |
| **Purpose** | Deepfake detection | Speaker ID |
| **Audio** | Real + Fake versions | Original only |
| **Metadata** | Modification type | Speaker + utterance ID |
| **CSV Columns** | filename, rt60, ... | speaker_id, utterance_id, rt60, ... |

### When to Use Which

**Use AV-Deepfake1M when:**
- ✓ You want to compare real vs. fake audio RT60
- ✓ You need deepfake detection features
- ✓ Researching audio forgery

**Use VoxCeleb2 when:**
- ✓ You want original, unmodified audio
- ✓ You're studying speaker characteristics
- ✓ You need ground truth baseline
- ✓ You want larger dataset

---

## Part 4: Performance & Scaling

### Single Video Processing
```
Time per video: 15-20 seconds (including audio extraction + RT60 estimation)
Memory: ~100 MB constant
```

### Batch Processing Speed

| Dataset Size | Sequential | 8-core Parallel | Time |
|--------------|------------|-----------------|------|
| 100 videos | 25-35 min | 3-5 min | Fast test |
| 1,000 videos | 4-6 hours | 30-45 min | Slow but manageable |
| 10,000 videos | 40-60 hours | 5-8 hours | Use parallelization |
| 100,000 videos | 400+ hours | 50-80 hours | Distribute across machines |

### Parallelization Example

```python
from multiprocessing import Pool
import os

def extract_video_rt60(video_path):
    """Process one video"""
    extractor = VoxCeleb2RT60Extractor('/data/VoxCeleb2')
    return extractor.process_video(video_path)

# Get all mp4 files
import glob
videos = glob.glob('/data/VoxCeleb2/dev/mp4/**/*.mp4', recursive=True)

# Process with 8 workers
with Pool(os.cpu_count()) as p:
    results = p.map(extract_video_rt60, videos)

# Save results
results = [r for r in results if r]  # Filter None values
df = pd.DataFrame(results)
df.to_csv('voxceleb2_rt60_parallel.csv', index=False)
```

---

## Troubleshooting

### Issue: "Directory not found"
**Solution:** Ensure VoxCeleb2 is in correct location
```bash
ls /path/to/VoxCeleb2/dev/mp4  # Should show id10001, id10002, ...
```

### Issue: "No videos found"
**Solution:** Check split name
```bash
# Valid: "dev" or "test"
python voxceleb2_rt60_extractor.py --split dev  # Correct
python voxceleb2_rt60_extractor.py --split train  # Wrong
```

### Issue: Audio extraction slow
**Solution:** Use multiprocessing (see above) or reduce limit for testing
```bash
python voxceleb2_rt60_extractor.py --limit 10  # Test with 10 videos first
```

---

## Summary

**Output Generation:** CSV files via Python's csv module  
**VoxCeleb2 Pipeline:** `voxceleb2_rt60_extractor.py`  
**Basic Usage:** `python voxceleb2_rt60_extractor.py --voxceleb2-root /path --limit 100`  
**Analysis:** Use pandas to load and process CSV results
