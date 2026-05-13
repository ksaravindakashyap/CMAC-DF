# Running RT60 Extraction on Tinkercliff HPC Cluster

This guide explains how to process the entire VoxCeleb2 dataset (1M+ videos) on Virginia Tech's Tinkercliff cluster using parallel processing.

## Quick Start

### 1. Connect to Tinkercliff

```bash
ssh username@tinkercliffs1.arc.vt.edu
# or
ssh username@tinkercliffs2.arc.vt.edu
```

### 2. Prepare Your Job Script

Edit `submit_rt60_job.sh` and update these lines:

```bash
# Your VT email for job notifications
#SBATCH --mail-user=YOUR_EMAIL@vt.edu

# Path to your VoxCeleb2 dataset
VOXCELEB2_ROOT="/path/to/VoxCeleb2"

# Optional: limit for testing
VIDEO_LIMIT=""  # Leave empty for full dataset, or set to "100" for testing
```

### 3. Submit Your Job

```bash
# From the project directory
cd /home/sengg/deep-learning-course-project

# Make script executable
chmod +x submit_rt60_job.sh

# Submit to SLURM
sbatch submit_rt60_job.sh
```

### 4. Monitor Your Job

```bash
# Check job status
squeue -u $USER

# Check detailed job info
scontrol show job JOBID

# View job output (while running or after)
tail -f rt60_job_JOBID.out
tail -f rt60_job_JOBID.err

# Get job summary after completion
sacct -j JOBID --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,Start,End,Elapsed
```

## Processing Time Estimates

### Configuration: 4 nodes × 32 cores = 128 parallel workers

**For VoxCeleb2-dev split (1M videos):**

| Scenario | Videos | Time/Video | Total Time | Wall Clock |
|----------|--------|-----------|-----------|-----------|
| Baseline (1 core) | 1,000,000 | 18 sec | 208 days | - |
| Parallel (128 cores) | 1,000,000 | 18 sec | ~1.6 days | **~40 hours** |
| With I/O serialization | 1,000,000 | 18 sec | ~1.6 days | **~50 hours** |

**Scaling with different node counts:**

| Nodes | Cores | Estimated Time | Recommended For |
|-------|-------|-----------------|-----------------|
| 1 node | 128 | ~50 hours | Small test (limit 100k) |
| 2 nodes | 256 | ~25 hours | Medium dataset (250k videos) |
| 4 nodes | 512 | ~13 hours | Full dataset (1M videos) |
| 8 nodes | 1024 | ~6 hours | Fastest (full dataset) |

## Job Script Customization

### For Faster Processing (Higher Cost)

Use `tc_normal_short` QoS (1-day max, 2x billing multiplier):

```bash
#SBATCH --qos=tc_normal_short
#SBATCH --time=1-00:00:00    # Max 24 hours
```

With 8 nodes:
```bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
```

### For Longer Processing (Lower Priority)

Use `tc_normal_long` QoS (14-day max, 0.5x priority):

```bash
#SBATCH --qos=tc_normal_long
#SBATCH --time=14-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
```

### For Cheapest Processing (Preemptable)

Use preemptable queue (can be killed if higher-priority job arrives, but 10x cheaper):

```bash
#SBATCH --partition=preemptable_q
#SBATCH --qos=tc_preemptable_base
```

## Resource Limits on Tinkercliff

**normal_q (default):**
- Max CPU cores per job: 10,496
- Max memory: 22,845 GB
- Max wall time: 7 days
- Default priority: 1000

**tc_normal_short (higher priority):**
- Max CPU cores: 15,744
- Max memory: 34,268 GB
- Max wall time: 1 day
- Billing: 2x multiplier
- Priority: 2000

**tc_normal_long (lower priority):**
- Max CPU cores: 2,624
- Max memory: 5,712 GB
- Max wall time: 14 days
- Priority: 500

**preemptable_q:**
- Max wall time: 30 days
- Cores/memory reduced
- Can be preempted
- No billing cost

## Monitoring and Diagnostics

### Check cluster status
```bash
# See overall cluster usage
sinfo

# See available nodes by partition
sinfo -p normal_q

# Check your user's current usage
sstat -j JOBID  # While job is running

# Historical job stats
sacct --user=$USER --format=JobID,JobState,Elapsed,CPUTime,MaxRSS
```

### Check job progress
```bash
# Watch output in real-time
tail -f rt60_job_JOBID.out

# Count completed videos (from within script)
grep "Processed" rt60_job_JOBID.out | tail -1

# Estimate completion time
# If line shows "Processed 5000/1000000", then:
# (1000000 - 5000) / (5000 / elapsed_time) = remaining_time
```

### Troubleshoot issues
```bash
# Check if job timed out
grep "TIME_LIMIT" rt60_job_JOBID.out

# Check for memory issues
sacct -j JOBID --format=MaxRSS  # Max memory used

# Check I/O performance
grep "Processed" rt60_job_JOBID.out | tail -5  # Shows throughput

# Increase workers if underutilized
grep "#SBATCH --ntasks-per-node" submit_rt60_job.sh  # Current setting
```

## Handling Large Datasets

### Strategy 1: Process in chunks
```bash
# Process video file lists
for i in {1..10}; do
    echo "Processing chunk $i..."
    sbatch --array=0-9%3 submit_rt60_job.sh
done
```

### Strategy 2: Use SLURM arrays
Create `submit_rt60_array.sh`:
```bash
#!/bin/bash
#SBATCH --array=0-99%5   # 100 jobs, max 5 running
```

### Strategy 3: Incremental output
Results are written as processed (not buffered), so you can:
```bash
# Watch results accumulate in real-time
watch -n 5 "wc -l results_*.csv"

# Combine results from multiple runs
cat results_*.csv | head -1 > final_results.csv
cat results_*.csv | tail -n +2 >> final_results.csv
```

## Cost Estimation

**Based on core-hour pricing:**

Assuming $0.10 per core-hour:

| Job Config | Duration | Cores | Billing Units | Cost |
|-----------|----------|-------|---------------|------|
| 1 node, 24h | 24h | 128 | 3,072 | $307 |
| 4 nodes, 40h | 40h | 512 | 20,480 | $2,048 |
| 4 nodes (short), 40h | 40h | 512 | **40,960** | **$4,096** |
| 2 nodes (long), 50h | 50h | 256 | 12,800 | $1,280 |

*Note:* Budget may vary; check your allocation at https://coldfront.arc.vt.edu/

## After Processing

### Collect and analyze results
```bash
# Combine all result files
cat /home/sengg/deep-learning-course-project/results_*.csv > all_results.csv

# Quick stats
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('all_results.csv')
print(f"Total videos: {len(df)}")
print(f"Successful: {(df['status'] == 'success').sum()}")
print(f"Mean RT60: {df[df['status'] == 'success']['rt60'].mean():.2f}s")
print(f"Std RT60: {df[df['status'] == 'success']['rt60'].std():.2f}s")
EOF

# Copy large results to project storage
cp all_results.csv /projects/your_allocation/results/
```

### Archive results
```bash
# Compress results
gzip all_results.csv

# Archive to storage
tar czf rt60_results_$(date +%Y%m%d).tar.gz all_results.csv

# Check what's available
ls -lh /projects/your_allocation/
```

## Common Issues and Solutions

### Job timeout
- Increase `--time` parameter
- Use `tc_normal_long` QoS (14 days max)
- Reduce number of workers to focus on speed

### Memory issues
- Reduce `--ntasks-per-node` to reduce memory contention
- Increase `--mem-per-cpu`
- Check librosa memory usage with `top` command

### Slow I/O
- VoxCeleb2 is large; ensure it's on high-speed storage
- Use `/scratch` for output CSV (faster than `/projects`)
- Consider caching audio in local `/tmp` on compute nodes

### GPU queue preferred
- To use GPU nodes (A100/H200) instead of CPU:
```bash
#SBATCH --partition=a100_normal_q
#SBATCH --gres=gpu:1
#SBATCH --qos=tc_a100_normal_base
```

## Advanced: Multi-job Submission

Process multiple splits simultaneously:

```bash
#!/bin/bash
# submit_all_splits.sh

for split in dev test; do
    sbatch --job-name=rt60_${split} \
           --output=rt60_${split}_%j.out \
           --error=rt60_${split}_%j.err \
           submit_rt60_job.sh
done
```

## FAQ

**Q: How do I specify which nodes to use?**  
A: Add `--constraint=amd` for AMD nodes or `--constraint=avx512` for Intel nodes.

**Q: Can I run this locally first to test?**  
A: Yes! Run `python voxceleb2_rt60_parallel.py --voxceleb2-root /path --limit 100 --output test.csv`

**Q: How do I cancel a running job?**  
A: `scancel JOBID`

**Q: Can I modify a submitted but queued job?**  
A: No, but you can cancel and resubmit with different parameters.

**Q: What if processing fails mid-job?**  
A: Results are written incrementally to CSV, so you lose only the last batch.

## Best Practices

1. **Test first with small dataset:**
   ```bash
   # Modify script
   VIDEO_LIMIT="100"
   # Submit and verify output
   sbatch submit_rt60_job.sh
   ```

2. **Monitor initial progress:**
   ```bash
   tail -f rt60_job_JOBID.out to verify 128 cores are being used
   ```

3. **Use longest wall time you can afford:**
   - Reduces job urgency and may start faster
   - Better utilization of cluster scheduling

4. **Archive results immediately after completion:**
   - Don't rely on `/localscratch` (temporary)
   - Copy to `/projects` or download locally

5. **Check billing before large runs:**
   - Visit https://coldfront.arc.vt.edu/
   - Verify allocation balance
   - Understand QoS billing multipliers

## Support

For issues with:
- **VT ARC systems:** https://arc.vt.edu/help
- **SLURM job submission:** ARC office hours or https://docs.arc.vt.edu/
- **Your code/RT60 extraction:** Check project documentation

---

**Last updated:** May 2026  
**Contact:** Your ARC support team
