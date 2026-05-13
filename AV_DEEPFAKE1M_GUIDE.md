# AV-Deepfake1M RT60 Extraction - SLURM Job Guide

This guide explains how to use the RT60 extraction pipeline on Tinkercliff for the AV-Deepfake1M dataset.

## Files Overview

| File | Purpose | Scale |
|------|---------|-------|
| `av_deepfake1m_rt60_parallel.py` | Main extraction script | Any scale |
| `submit_rt60_job_av_deepfake_test.sh` | Debug/test job | 100 videos, 1 node, 1 hour |
| `submit_rt60_job_av_deepfake.sh` | Full production job | All videos, 4 nodes, 48 hours |

## Before You Start

### 1. Verify You Have the AV-Deepfake1M Dataset

The scripts expect the dataset at: `/mnt/av_deepfake1m`

**Directory structure should be:**
```
/mnt/av_deepfake1m/
├── train/
│   ├── original/
│   │   └── *.mp4
│   └── deepfake/
│       └── *.mp4
└── test/
    ├── original/
    │   └── *.mp4
    └── deepfake/
        └── *.mp4
```

If your dataset is elsewhere, update the `AV_DEEPFAKE1M_ROOT` variable in the job scripts.

### 2. Prepare the Job Scripts

Make the scripts executable:
```bash
chmod +x submit_rt60_job_av_deepfake_test.sh
chmod +x submit_rt60_job_av_deepfake.sh
```

## Quick Start: Test Job (Recommended)

Run the test job first to validate everything works:

```bash
# Submit test job (100 videos, ~20-30 min)
sbatch submit_rt60_job_av_deepfake_test.sh

# Check job status
squeue -u $USER

# Monitor output (in another terminal)
tail -f rt60_av_deepfake_test_<JOBID>.out

# View results once complete
cat results_av_deepfake_test_<JOBID>.csv
```

The test job will:
- Process 100 videos to validate the pipeline
- Generate statistics summary
- Create output CSV with RT60 values
- Provide guidance for full job submission

## Full Job: Production Run

Once the test job passes successfully:

```bash
# Submit full production job (all videos, ~40-50 hours)
sbatch submit_rt60_job_av_deepfake.sh

# Check job status
squeue -u $USER

# Monitor progress
tail -f rt60_av_deepfake_<JOBID>.out

# View results as they complete
tail -20 results_av_deepfake_<JOBID>.csv
```

### Expected Performance

| Configuration | Videos | Cores | Runtime | Walltime |
|---------------|--------|-------|---------|----------|
| Test | 100 | 8 | ~20-30 min | 1 hour |
| Production | 1M+ | 512 | ~40-50 hours | 48 hours |

### Output Files

Results are saved to your home directory:
- **Test results:** `results_av_deepfake_test_<JOBID>.csv`
- **Full results:** `results_av_deepfake_<JOBID>.csv` (separate files for each job)

## CSV Output Format

Each results file contains:
- `video_path` - Full path to video file
- `relative_path` - Path relative to dataset root
- `video_filename` - Just the filename
- `split` - train or test
- `category` - original or deepfake
- `rt60` - Estimated reverberation time (seconds)
- `audio_duration` - Duration of extracted audio (seconds)
- `status` - success or error message

## Advanced: Direct Script Usage (Without SLURM)

If you want to run the extraction script directly (not on cluster):

```bash
# Process 50 original videos from training set
python av_deepfake1m_rt60_parallel.py \
    --dataset-dir /mnt/av_deepfake1m \
    --split train \
    --category original \
    --limit 50 \
    --output my_results.csv \
    --workers 8

# Process all videos
python av_deepfake1m_rt60_parallel.py \
    --dataset-dir /mnt/av_deepfake1m \
    --split train \
    --output all_train_results.csv

# Process deepfake videos only
python av_deepfake1m_rt60_parallel.py \
    --dataset-dir /mnt/av_deepfake1m \
    --split train \
    --category deepfake \
    --output deepfake_results.csv
```

## Troubleshooting

### Dataset Not Found Error

**Error:** "Dataset directory not found: /mnt/av_deepfake1m"

**Solutions:**
1. Verify actual dataset location:
   ```bash
   find /mnt -name "*.mp4" -type f | head -5
   ```

2. Update path in submit scripts:
   - Edit `submit_rt60_job_av_deepfake_test.sh`
   - Edit `submit_rt60_job_av_deepfake.sh`
   - Change `AV_DEEPFAKE1M_ROOT="/mnt/av_deepfake1m"` to correct path

3. If using different partition, verify path is mounted there:
   ```bash
   srun --ntasks=1 ls /mnt/av_deepfake1m/
   ```

### Job Fails with Account Error

**Error:** "Invalid account or account/partition combination specified"

This job script uses account `cs5814`. If error occurs:
```bash
# Verify available accounts
sinfo -o "%a" | sort -u

# Update account in job script if needed
sed -i 's/--account=.*/--account=YOUR_ACCOUNT/' submit_rt60_job_av_deepfake.sh
```

### Memory Issues

If job runs out of memory, reduce workers or increase memory per core.

**In job script, change:**
```bash
#SBATCH --mem-per-cpu=2000
```

to:
```bash
#SBATCH --mem-per-cpu=4000
```

## Monitoring & Results

### Check Job Status
```bash
# View all your jobs
squeue -u $USER

# View specific job details
scontrol show job <JOBID>

# View job history
sacct -u $USER --format=JobID,JobName,State,ElapsedRaw
```

### View Results During Execution
```bash
# Watch output in real-time
tail -f rt60_av_deepfake_<JOBID>.out

# Count lines processed
wc -l results_av_deepfake_<JOBID>.csv

# Check first and last entries
head -5 results_av_deepfake_<JOBID>.csv
tail -5 results_av_deepfake_<JOBID>.csv
```

### Analyze Results

```python
import pandas as pd

# Load results
df = pd.read_csv('results_av_deepfake_<JOBID>.csv')

# Basic statistics
print(f"Total videos processed: {len(df)}")
print(f"Success rate: {100 * (df['status'] == 'success').sum() / len(df):.1f}%")

# RT60 by category
print("\nRT60 by category:")
for cat in ['original', 'deepfake']:
    subset = df[df['category'] == cat]
    print(f"  {cat}: mean={subset['rt60'].mean():.2f}s, std={subset['rt60'].std():.2f}s")

# Failed videos
failed = df[df['status'] != 'success']
print(f"\nFailed videos: {len(failed)}")
if len(failed) > 0:
    print(failed[['video_filename', 'status']].head())
```

## Comparison: VoxCeleb2 vs AV-Deepfake1M

| Aspect | VoxCeleb2 | AV-Deepfake1M |
|--------|-----------|---------------|
| Base dataset | Speaker verification videos | Voice deepfakes based on VoxCeleb2 |
| Structure | speaker_id/utterance_id | split/category/files |
| Categories | Speaker identity | original vs deepfake synthesis |
| Size | 1M+ videos | 1M+ videos (derived) |
| Use case | RT60 baseline | Deepfake detection feature |
| Script | `voxceleb2_rt60_parallel.py` | `av_deepfake1m_rt60_parallel.py` |
| SLURM job | `submit_rt60_job.sh` | `submit_rt60_job_av_deepfake.sh` |

## Next Steps

1. **Run test job** on AV-Deepfake1M to validate setup
2. **Compare RT60 distributions** between original and deepfake audio
3. **Analyze results** for deepfake detection patterns
4. **Extract VoxCeleb2 results** to baseline original audio
5. **Integrate RT60 features** into deepfake detection models

## Getting Help

If you encounter issues:

1. Check job error log:
   ```bash
   tail -100 rt60_av_deepfake_<JOBID>.err
   ```

2. Verify environment:
   ```bash
   bash test_cluster_setup.sh
   ```

3. Test script directly:
   ```bash
   python av_deepfake1m_rt60_parallel.py --help
   ```

---

**Quick Reference:**

```bash
# Test (100 videos, 1 node, 1 hour)
sbatch submit_rt60_job_av_deepfake_test.sh

# Production (all videos, 4 nodes, 48 hours)
sbatch submit_rt60_job_av_deepfake.sh

# Check status
squeue -u $USER

# View results
tail -20 results_av_deepfake_<JOBID>.csv
```
