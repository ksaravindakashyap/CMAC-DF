# Complete Workflow: Download & Analyze Paired AV-Deepfake1M Dataset

This guide provides the complete end-to-end workflow for downloading 1000 paired real/fake videos and extracting RT60 (reverberation time) for acoustic analysis.

## 🎯 What You'll Get

**2000 videos (1000 pairs):**
- 1000 RVFA videos (Real Video, Fake Audio) - synthetic/TTS audio
- 1000 RVRA videos (Real Video, Real Audio) - matching original source videos
- Complete RT60 analysis comparing acoustic properties

## 📋 Prerequisites

### System Requirements
- **CPU:** 4+ cores (for RT60 extraction)
- **RAM:** 8 GB minimum
- **Storage:** 30-50 GB (for 1000 pairs)
- **Network:** Broadband internet (100+ Mbps recommended)
- **OS:** Linux/macOS (Windows with WSL2)

### Software Requirements

```bash
# Python dependencies
pip install -r requirements.txt

# Additional tool for downloading
pip install yt-dlp

# For verification (recommended)
sudo apt-get install ffmpeg

# For parallel processing on HPC
module load software/GCC-9.3.0
```

## 🚀 Quick Start (5 Steps)

### Step 1: Download Paired Videos (6-24 hours)

```bash
# Make script executable
chmod +x quick_start_paired_download.sh

# Download 1000 pairs (default)
./quick_start_paired_download.sh

# Or download custom amount to custom location
./quick_start_paired_download.sh 100 /home/sengg/test_pairs
```

**What happens:**
1. Downloads AV-Deepfake1M metadata
2. Identifies 1000 RVFA-RVRA pairs
3. Uses yt-dlp to fetch videos from YouTube
4. Organizes them into paired structure
5. Creates manifest and metadata files

**Progress tracking:**
```bash
# In another terminal, monitor downloads
watch -n 5 'ls fake_audio/mp4/ | wc -l'
```

### Step 2: Verify Downloaded Videos (5-10 minutes)

```bash
python verify_paired_dataset.py \
    --dataset-dir /home/sengg/paired_av_deepfake
```

**Verification checks:**
- ✓ All videos are valid MP4 files
- ✓ Videos have both video and audio streams
- ✓ Fake and real videos are properly paired
- ✓ No corruption detected
- ✓ Statistics on storage usage

**Output files created:**
- `rt60_paired_videos.txt` - Tab-separated fake/real pairs
- `rt60_fake_videos.txt` - Just fake videos
- `rt60_real_videos.txt` - Just real videos
- `verification_report.json` - Detailed report

### Step 3: Extract RT60 (Local or Cluster)

#### Option A: Local Extraction (CPU intensive, 10-30 hours)

```bash
python rt60_extractor.py \
    --video-list /home/sengg/paired_av_deepfake/rt60_paired_videos.txt \
    --output rt60_paired_results.csv \
    --workers 4 \
    --batch-size 50

# Monitor progress
tail -f rt60_paired_results.csv
```

#### Option B: Cluster Extraction (Recommended, 2-4 hours)

```bash
# Make script executable
chmod +x submit_rt60_job_paired.sh

# Submit to Tinkercliff
sbatch submit_rt60_job_paired.sh \
    --dataset-dir /home/sengg/paired_av_deepfake

# Check status
squeue -u $USER

# Monitor output
tail -f rt60_paired_*.out
```

### Step 4: Analyze RT60 Results (5 minutes)

```bash
python << 'EOF'
import pandas as pd

# Load results
df = pd.read_csv('rt60_paired_results.csv')

# Summary statistics
print("FAKE AUDIO (Synthetic) RT60:")
print(df[df['type'] == 'fake']['rt60'].describe())
print("\nREAL AUDIO (Original) RT60:")
print(df[df['type'] == 'real']['rt60'].describe())

# Compare means
fake_mean = df[df['type'] == 'fake']['rt60'].mean()
real_mean = df[df['type'] == 'real']['rt60'].mean()
print(f"\nDifference: {fake_mean - real_mean:.3f} seconds")
print(f"Percent change: {((fake_mean - real_mean) / real_mean * 100):.1f}%")

# Save detailed results
result_summary = df.groupby('type')['rt60'].agg(['count', 'mean', 'std', 'min', 'max'])
print("\n" + str(result_summary))
EOF
```

### Step 5: Review & Archive Results

```bash
# View all RT60 output files
ls -lh rt60_results/

# Summarize results
cat rt60_results/job_report_*.json | python -m json.tool

# Archive for analysis
tar -czf paired_av_deepfake_rt60_results.tar.gz \
    rt60_results/ \
    verification_report.json

# Transfer to local machine
scp -r rt60_results/ YOUR_LOCAL_MACHINE:~/analysis/
```

## 📁 Complete Directory Structure

```
paired_av_deepfake/
├── fake_audio/
│   └── mp4/
│       ├── pair_0000/video.mp4  (RVFA - synthetic audio)
│       ├── pair_0001/video.mp4
│       └── ... (1000 pairs)
├── real_audio/
│   └── mp4/
│       ├── pair_0000/video.mp4  (RVRA - original audio)
│       ├── pair_0001/video.mp4
│       └── ... (1000 pairs)
├── metadata/
│   └── av_deepfake1m_metadata.csv
├── rt60_results/
│   ├── fake_rt60_<JOBID>.csv
│   ├── real_rt60_<JOBID>.csv
│   ├── paired_rt60_analysis_<JOBID>.csv
│   └── job_report_<JOBID>.json
├── manifest.json
├── verification_report.json
├── rt60_paired_videos.txt      (for batch extraction)
├── rt60_fake_videos.txt
└── rt60_real_videos.txt
```

## 🔍 Understanding the Results

### RT60 CSV Output Format

```csv
video_path,rt60,confidence,audio_duration_s,date_processed
fake_audio/mp4/pair_0000/video.mp4,0.421,0.95,15.2,2024-05-07
real_audio/mp4/pair_0000/video.mp4,0.389,0.96,15.2,2024-05-07
...
```

### Interpretation

**RT60 = Reverberation Time**
- Measured in seconds
- Higher RT60 = more room reverb
- Typical values: 0.2 - 2.0 seconds
- Room acoustics characteristics

**Expected Analysis Findings:**
- Fake audio may have different RT60 signature
- Synthetic audio may lack natural room acoustics
- TTS/Voice conversion may create acoustic artifacts
- Useful for deepfake detection models

## 🎓 Analysis Examples

### 1. Simple Comparison

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('rt60_paired_results.csv')

# Box plot comparison
df.boxplot(column='rt60', by='type')
plt.title('RT60 Distribution: Fake vs Real Audio')
plt.show()
```

### 2. Pair-wise Differences

```python
# Load paired data
pairs = pd.read_csv('rt60_paired_videos.txt', sep='\t', header=None)

# Calculate difference for each pair
differences = []
for _, row in pairs.iterrows():
    fake_video = row[0]
    real_video = row[1]
    
    fake_rt60 = df[df['video_path'] == fake_video]['rt60'].values[0]
    real_rt60 = df[df['video_path'] == real_video]['rt60'].values[0]
    
    differences.append(fake_rt60 - real_rt60)

# Analyze distribution
import numpy as np
print(f"Mean difference: {np.mean(differences):.3f}")
print(f"Std deviation: {np.std(differences):.3f}")
```

### 3. Statistical Significance

```python
from scipy import stats

fake_rt60 = df[df['type'] == 'fake']['rt60'].dropna()
real_rt60 = df[df['type'] == 'real']['rt60'].dropna()

# T-test
t_stat, p_value = stats.ttest_ind(fake_rt60, real_rt60)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.2e}")

if p_value < 0.05:
    print("✓ Significant difference between fake and real audio")
else:
    print("No significant difference detected")
```

## ⚡ Performance Benchmarks

| Task | Scale | Time | Resources |
|------|-------|------|-----------|
| Download (1000 pairs) | 2000 videos | 6-24 hours | 100 Mbps connection |
| Verification | 2000 videos | 5-10 min | 4 CPU cores |
| RT60 Extraction (local) | 2000 videos | 10-30 hours | 4 CPU cores |
| RT60 Extraction (cluster) | 2000 videos | 2-4 hours | 32 cores |

## 🔧 Troubleshooting

### Download Issues

**Problem:** YouTube videos not downloading
```bash
# Solution: Check yt-dlp version
yt-dlp --update
```

**Problem:** Network timeouts
```bash
# Solution: Resume download from same directory
python download_paired_av_deepfake.py --output /path/to/existing --num-pairs 1000
```

### Verification Issues

**Problem:** Videos show as invalid
```bash
# Check with ffmpeg directly
ffprobe -v error fake_audio/mp4/pair_0000/video.mp4
```

### RT60 Extraction Issues

**Problem:** "rt60_extractor.py not found"
```bash
# Make sure you're in correct directory
cd /home/sengg/deep-learning-course-project
```

**Problem:** SLURM job fails
```bash
# Check error log
cat rt60_paired_*.err
```

## 📊 Advanced Analysis

### Acoustic Profile Analysis

```python
# Create acoustic fingerprints
fake_profiles = df[df['type'] == 'fake'].groupby('pair_id')['rt60'].agg(['mean', 'std', 'count'])
real_profiles = df[df['type'] == 'real'].groupby('pair_id')['rt60'].agg(['mean', 'std', 'count'])

# Identify outliers
import numpy as np
z_scores = np.abs(stats.zscore(fake_profiles['mean']))
outliers = z_scores > 3
```

### Machine Learning Features

```python
# Extract features for deepfake detection model
features = pd.DataFrame({
    'rt60_fake': fake_profiles['mean'],
    'rt60_real': real_profiles['mean'],
    'rt60_diff': fake_profiles['mean'] - real_profiles['mean'],
    'rt60_std_fake': fake_profiles['std'],
    'rt60_std_real': real_profiles['std'],
})
```

## 📚 Additional Resources

- **AV-Deepfake1M**: https://github.com/ControlNet/AV-Deepfake1M
- **VoxCeleb2**: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- **yt-dlp**: https://github.com/yt-dlp/yt-dlp
- **RT60 Theory**: https://en.wikipedia.org/wiki/Reverberation

## 🎯 Next Steps

1. **Data Collection** ✓ (using this guide)
2. **RT60 Analysis** ✓ (using this guide)
3. **Feature Engineering** → Create ML-ready features
4. **Model Training** → Train deepfake detection model
5. **Evaluation** → Test detection accuracy

## 📞 Support

For issues or questions:
1. Check the existing scripts for examples
2. Review error logs in detail
3. Verify prerequisites are installed
4. Re-run with verbose logging: add `--verbose` flag

---

**Last Updated:** May 2024  
**Status:** Production Ready  
**Version:** 1.0
