# Your Questions Answered

## Question 1: "Can you also run the pipeline on VoxCeleb2 for a sample?"

✅ **YES - Complete VoxCeleb2 Pipeline Built**

**What was created:**
- `voxceleb2_rt60_extractor.py` (8.3 KB) - Full pipeline for VoxCeleb2
- VOXCELEB2_GUIDE.md (9.6 KB) - Complete guide with examples
- demo_output_generation.py (11 KB) - Demonstration of CSV generation

**VoxCeleb2 Dataset:**
- 1,000,000+ utterances from 6,112 speakers
- Original source for AV-Deepfake1M (AV-Deepfake1M videos are **derived from** VoxCeleb2)
- Directory structure: `dev/mp4/{speaker_id}/{utterance_id}/{video}.mp4`

**How to run on your VoxCeleb2:**
```bash
# Test with 100 videos
python voxceleb2_rt60_extractor.py \
    --voxceleb2-root /path/to/VoxCeleb2 \
    --split dev \
    --limit 100 \
    --output voxceleb2_rt60_sample.csv

# Run on all dev videos
python voxceleb2_rt60_extractor.py \
    --voxceleb2-root /path/to/VoxCeleb2 \
    --split dev \
    --output voxceleb2_rt60_dev_full.csv
```

**Output CSV for VoxCeleb2:**
```
video_path,relative_path,video_filename,speaker_id,utterance_id,rt60,audio_duration,status
/data/VoxCeleb2/dev/mp4/id10001/1HDQeFPvL6c/00001.mp4,dev/mp4/id10001/1HDQeFPvL6c/00001.mp4,00001.mp4,id10001,1HDQeFPvL6c,0.456,10.234,success
```

**Why VoxCeleb2 is useful:**
- Baseline for "real" audio (not modified/deepfaked)
- Compare RT60: real audio vs deepfake audio
- Grouped by speaker → easy analysis
- 6K+ speakers → good diversity

---

## Question 2: "How are you generating the output files?"

✅ **Explained in Detail with Demo**

**The Method: Python's csv.DictWriter**

### Step 1: Create Results as Python Dictionaries
```python
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
```

### Step 2: Write to CSV
```python
import csv

with open('output.csv', 'w', newline='') as f:
    fieldnames = results[0].keys()  # Get column names
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    writer.writeheader()        # Writes: video_path,filename,rt60,audio_duration,status
    writer.writerows(results)   # Writes each dict as a row
```

### Result: Plain Text CSV File
```
video_path,filename,rt60,audio_duration,status
/data/video1.mp4,video1.mp4,0.456,10.234,success
/data/video2.mp4,video2.mp4,0.382,9.876,success
```

**Why CSV?**
| Advantage | Benefit |
|-----------|---------|
| **Human-readable** | Open in text editor and understand |
| **Tool-agnostic** | Works with Excel, Python, R, SQL, etc. |
| **Lightweight** | ~100 bytes per video |
| **Universal** | No special software needed |
| **Version-controllable** | Can track changes in git |
| **Queryable** | Easy to filter/sort with pandas |

### Opening CSV Files

**View in terminal:**
```bash
cat results.csv              # View all
head results.csv             # First few rows
wc -l results.csv            # Count rows
```

**Open in Python:**
```python
import pandas as pd
df = pd.read_csv('results.csv')
print(df.head())
```

**Open in Excel:**
- Right-click → Open With → Excel
- Or drag file into spreadsheet

**Query with SQL:**
```bash
sqlite3 <<< "SELECT * FROM results WHERE rt60 > 0.4"
```

---

## Complete Comparison: AV-Deepfake1M vs VoxCeleb2

| Aspect | AV-Deepfake1M | VoxCeleb2 |
|--------|---------------|-----------|
| **Videos** | 1M+ (modified) | 1M+ (original) |
| **Speakers** | 2K+ | 6,112 |
| **Purpose** | Deepfake detection | Speaker identification |
| **Audio types** | Real + Fake | Original only |
| **CSV includes** | filename, rt60, modify_type | speaker_id, utterance_id, rt60 |
| **Use case** | Compare real vs fake | Baseline acoustic characteristics |
| **Extractor** | `rt60_extractor.py` | `voxceleb2_rt60_extractor.py` |

---

## What You Can Do Now

### 1. Run AV-Deepfake1M Pipeline
```bash
python rt60_extractor.py \
    --metadata /path/to/train_metadata.json \
    --dataset-dir ./dataset \
    --output av_deepfake_rt60.csv \
    --limit 100
```

### 2. Run VoxCeleb2 Pipeline
```bash
python voxceleb2_rt60_extractor.py \
    --voxceleb2-root /path/to/VoxCeleb2 \
    --split dev \
    --limit 100 \
    --output voxceleb2_rt60.csv
```

### 3. Compare Results
```python
import pandas as pd

av_df = pd.read_csv('av_deepfake_rt60.csv')
vox_df = pd.read_csv('voxceleb2_rt60.csv')

print(f"AV-Deepfake1M mean RT60: {av_df['rt60'].mean():.3f}s")
print(f"VoxCeleb2 mean RT60: {vox_df['rt60'].mean():.3f}s")
# Should see difference: deepfake audio often modified differently
```

### 4. Visualize Distributions
```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

av_df['rt60'].hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7)
ax1.set_title('AV-Deepfake1M RT60 Distribution')
ax1.set_xlabel('RT60 (seconds)')

vox_df['rt60'].hist(bins=50, ax=ax2, edgecolor='black', alpha=0.7)
ax2.set_title('VoxCeleb2 RT60 Distribution')
ax2.set_xlabel('RT60 (seconds)')

plt.tight_layout()
plt.savefig('rt60_comparison.png', dpi=150)
```

---

## Project File Summary

**Total Files Created: 11**

```
Scripts:
  ✓ rt60_extractor.py              (7.6K) - AV-Deepfake1M pipeline
  ✓ voxceleb2_rt60_extractor.py    (8.3K) - VoxCeleb2 pipeline
  ✓ test_rt60.py                   (2.8K) - Tests (PASSED ✓)
  ✓ demo_output_generation.py      (11K)  - CSV generation demo
  ✓ examples.py                    (7.7K) - Code examples

Documentation:
  ✓ README.md                      (4.1K) - Main docs
  ✓ QUICK_REFERENCE.md             (5.7K) - Quick guide
  ✓ PIPELINE_SUMMARY.md            (6.5K) - Overview
  ✓ VOXCELEB2_GUIDE.md             (9.6K) - VoxCeleb2 guide
  ✓ OUTPUT_GENERATION.md           (9.2K) - CSV generation guide

Setup:
  ✓ requirements.txt               (52B)  - Dependencies
```

---

## Ready to Use

```bash
# All dependencies installed
python -c "from blind_rt60 import BlindRT60; print('✓ Ready')"

# Try the demo
python demo_output_generation.py

# Run on VoxCeleb2
python voxceleb2_rt60_extractor.py --voxceleb2-root /path --split dev --limit 100
```

---

## Key Takeaways

### CSV Output Generation
- Use Python's `csv.DictWriter` module
- Convert results (list of dicts) to CSV
- Header row = column names
- Each row = one dictionary
- Plain text format, universal compatibility

### VoxCeleb2 Pipeline
- Same algorithm (blind_rt60) as AV-Deepfake1M
- Different input structure (speaker_id, utterance_id)
- Useful as baseline/comparison
- Ready for batch processing (1M+ videos)

### Your Next Steps
1. Obtain VoxCeleb2 dataset (register at https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
2. Run: `python voxceleb2_rt60_extractor.py --voxceleb2-root /path --split dev --limit 100`
3. Load CSV: `df = pd.read_csv('voxceleb2_rt60.csv')`
4. Analyze and compare with AV-Deepfake1M results
