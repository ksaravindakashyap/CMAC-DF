#!/usr/bin/env python3
"""
Parse partial RT60 results from job log and save to CSV.
Also identify remaining videos that need processing.
"""

import re
import csv
from pathlib import Path
from collections import defaultdict

log_file = '/home/sengg/deep-learning-course-project/rt60_videos_subset_600_5292598.err'
dataset_dir = Path('/home/sengg/deep-learning-course-project/AVDeepfake1M_local/videos_subset_600')

# Parse results from log
results = []
processed_videos = set()

with open(log_file, 'r') as f:
    for line in f:
        # Look for successful extractions: ✓ real.mp4 (real): RT60=1.511s
        if '✓' in line:
            match = re.search(r'✓ (.+?) \((.+?)\): RT60=([\d.]+)s', line)
            if match:
                video_name, video_type, rt60 = match.groups()
                results.append({
                    'video_name': video_name,
                    'video_type': video_type,
                    'rt60': float(rt60),
                    'status': 'success'
                })
                processed_videos.add(video_name)

print(f"Extracted {len(results)} successful results from log")
print(f"Video types breakdown:")
type_counts = defaultdict(int)
for r in results:
    type_counts[r['video_type']] += 1
    
for vtype, count in sorted(type_counts.items()):
    print(f"  {vtype}: {count}")

# Save partial results
partial_csv = 'rt60_results_partial.csv'
with open(partial_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['video_name', 'video_type', 'rt60', 'status'])
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"\nPartial results saved to: {partial_csv}")
print(f"Total lines in CSV: {len(results) + 1}")

# Find remaining videos
print(f"\nFinding remaining videos to process...")
remaining_videos = defaultdict(list)

for video_file in dataset_dir.rglob('*.mp4'):
    video_name = video_file.name
    
    if video_name not in processed_videos:
        if video_name == 'real.mp4':
            remaining_videos['real'].append(str(video_file))
        elif video_name == 'fake_video_fake_audio.mp4':
            remaining_videos['fake_video_fake_audio'].append(str(video_file))
        elif video_name == 'real_video_fake_audio.mp4':
            remaining_videos['real_video_fake_audio'].append(str(video_file))

print(f"\nRemaining videos to process:")
total_remaining = 0
for vtype in sorted(remaining_videos.keys()):
    count = len(remaining_videos[vtype])
    total_remaining += count
    print(f"  {vtype}: {count}")

print(f"Total remaining: {total_remaining}")

# Save list of remaining videos for the next job
if total_remaining > 0:
    remaining_list_file = 'remaining_videos.txt'
    with open(remaining_list_file, 'w') as f:
        for vtype in sorted(remaining_videos.keys()):
            for video_path in remaining_videos[vtype]:
                f.write(f"{video_path}\n")
    
    print(f"\nList of remaining videos saved to: {remaining_list_file}")
