# RT60 Extraction - Quick Reference

## Installation (Copy & Paste)
```bash
pip install blind-rt60 librosa scipy numpy matplotlib
```

## Verify Installation
```bash
python -c "from blind_rt60 import BlindRT60; print('✓ Ready')"
```

---

## Command Line Usage

### Process 100 videos from AV-Deepfake1M
```bash
python rt60_extractor.py \
    --metadata /path/to/train_metadata.json \
    --dataset-dir /path/to/dataset/train \
    --output rt60_results.csv \
    --limit 100
```

### Process all videos  
```bash
python rt60_extractor.py \
    --metadata /path/to/train_metadata.json \
    --dataset-dir /path/to/dataset/train \
    --output rt60_results_full.csv
```

### Process from video list file
```bash
# Create videos.txt with one path per line:
# /data/video1.mp4
# /data/video2.mp4
# ...

python rt60_extractor.py \
    --video-list videos.txt \
    --output rt60_results.csv
```

---

## Python One-Liners

### Extract RT60 from single video
```python
from rt60_extractor import RT60Extractor
e = RT60Extractor()
rt60 = e.estimate_rt60(e.extract_audio_from_video("video.mp4"))
print(f"RT60: {rt60:.3f}s")
```

### Extract from list of videos
```python
from rt60_extractor import RT60Extractor
e = RT60Extractor()
e.process_batch(["v1.mp4", "v2.mp4", "v3.mp4"], "results.csv")
```

### Direct blind_rt60 API
```python
from blind_rt60 import BlindRT60
import librosa
audio, sr = librosa.load("video.mp4", sr=16000, mono=True)
rt60 = BlindRT60()(audio, sr)
print(f"RT60: {rt60:.3f}s")
```

---

## Data Analysis

### Load and inspect results
```python
import pandas as pd

df = pd.read_csv('rt60_results.csv')
print(df.head())
print(df['rt60'].describe())
```

### Filter results
```python
# Only successful extractions
df_success = df[df['status'] == 'success']

# RT60 in specific range
df_filtered = df[(df['rt60'] > 0.2) & (df['rt60'] < 1.0)]

# Sort by RT60
df_sorted = df.sort_values('rt60', ascending=False)
```

### Visualize
```python
import matplotlib.pyplot as plt

# Histogram
df['rt60'].hist(bins=50)
plt.xlabel('RT60 (s)')
plt.ylabel('Count')
plt.savefig('rt60_hist.png')

# Box plot
df.boxplot(column='rt60')
plt.savefig('rt60_box.png')

# Scatter (RT60 vs duration)
plt.scatter(df['audio_duration'], df['rt60'], alpha=0.5)
plt.xlabel('Audio Duration (s)')
plt.ylabel('RT60 (s)')
plt.savefig('rt60_vs_duration.png')
```

---

## Integration with AV-Deepfake1M Metadata

```python
import pandas as pd
import json

# Load RT60 results
df_rt60 = pd.read_csv('rt60_results.csv')

# Load original metadata
with open('train_metadata.json') as f:
    metadata = json.load(f)

# Create lookup
metadata_dict = {item['file']: item for item in metadata}

# Add columns from metadata
df_rt60['modify_type'] = df_rt60['filename'].map(
    lambda x: metadata_dict.get(x, {}).get('modify_type', 'unknown')
)
df_rt60['audio_model'] = df_rt60['filename'].map(
    lambda x: metadata_dict.get(x, {}).get('audio_model', 'unknown')
)

# Analyze by type
print(df_rt60.groupby('modify_type')['rt60'].agg(['mean','std','count']))

# Save combined data
df_rt60.to_csv('rt60_with_metadata.csv', index=False)
```

---

## Performance Tips

### Process in parallel (8 cores)
```python
from rt60_extractor import RT60Extractor
from multiprocessing import Pool
import os

def process_video(path):
    e = RT60Extractor()
    return e.process_video(path)

videos = ["v1.mp4", "v2.mp4", ...]  # 1000+ videos

with Pool(os.cpu_count()) as p:
    results = p.map(process_video, videos)

# Save results
import csv
with open('results.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows([r for r in results if r])
```

### Monitor progress with tqdm
```python
from rt60_extractor import RT60Extractor
from tqdm import tqdm

e = RT60Extractor()
videos = ["v1.mp4", "v2.mp4", ...]

results = []
for video in tqdm(videos, desc="Processing"):
    result = e.process_video(video)
    if result:
        results.append(result)
```

---

## Troubleshooting

### Check for failed videos
```python
df = pd.read_csv('rt60_results.csv')
print(df[df['status'] != 'success'])
```

### Inspect audio before processing
```python
import librosa

audio, sr = librosa.load('video.mp4', sr=16000, mono=True)
print(f"Duration: {len(audio)/sr:.1f}s")
print(f"Peak amplitude: {abs(audio).max():.3f}")
print(f"RMS: {(audio**2).mean()**0.5:.3f}")
```

### Test specific video
```python
from rt60_extractor import RT60Extractor

e = RT60Extractor()
result = e.process_video('problematic_video.mp4')
if result:
    print(f"Success: RT60={result['rt60']:.3f}s")
else:
    print("Failed - check video format and audio quality")
```

---

## Help & Debugging

### Show all options
```bash
python rt60_extractor.py --help
```

### Run test
```bash
python test_rt60.py
```

### View examples
```bash
python examples.py
```

### Read full docs
```bash
cat README.md    # Algorithm & usage
cat PIPELINE_SUMMARY.md  # Complete overview
```

---

## Common Workflows

### Workflow 1: Quick Test (10 videos)
```bash
python rt60_extractor.py --metadata meta.json --dataset-dir ./data --limit 10
```

### Workflow 2: Full Dataset Processing
```bash
# Process in batches
python rt60_extractor.py --metadata meta.json --dataset-dir ./data --output rt60_full.csv

# Monitor progress
tail -f rt60_full.csv | wc -l  # Count lines
```

### Workflow 3: Correlate with Deepfake Type
```python
# (See Integration section above)
# Then analyze:
real = df[df['modify_type'] == 'real']['rt60']
fake = df[df['modify_type'] != 'real']['rt60']
print(f"Real audio RT60: {real.mean():.3f}s ± {real.std():.3f}s")
print(f"Fake audio RT60: {fake.mean():.3f}s ± {fake.std():.3f}s")
```

---

**Need Help?** Check the full README.md or PIPELINE_SUMMARY.md
