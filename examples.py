"""
Quick Start Guide: RT60 Extraction Pipeline

This script demonstrates how to use the RT60 extraction pipeline
once dependencies are installed.
"""

# ============================================================================
# INSTALLATION INSTRUCTIONS
# ============================================================================
"""
1. Using pip (recommended for Python 3.10-3.12):
   pip install blind-rt60 librosa scipy numpy

2. OR using conda with conda-forge:
   conda install -c conda-forge librosa scipy numpy
   pip install blind-rt60

3. If you encounter issues, try updating pip:
   pip install --upgrade pip setuptools
   pip install blind-rt60 librosa scipy numpy
"""

# ============================================================================
# IMPORT EXAMPLE (once installed)
# ============================================================================

try:
    from blind_rt60 import BlindRT60
    import librosa
    import numpy as np
    print("✓ All dependencies installed successfully!")
    HAS_DEPS = True
except ImportError as e:
    print(f"⚠ Dependencies not yet installed: {e}")
    print("\nRun: pip install blind-rt60 librosa scipy numpy")
    HAS_DEPS = False


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_1_simple_extraction():
    """Example 1: Extract RT60 from a single video"""
    if not HAS_DEPS:
        print("Dependencies required. Install with: pip install blind-rt60 librosa")
        return
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Extract RT60 from Single Video")
    print("="*60)
    
    from rt60_extractor import RT60Extractor
    
    # Path to video file
    video_path = "/path/to/video.mp4"
    
    # Initialize extractor
    extractor = RT60Extractor()
    
    # Extract RT60
    audio = extractor.extract_audio_from_video(video_path)
    if audio is not None:
        rt60 = extractor.estimate_rt60(audio)
        print(f"RT60: {rt60:.3f} seconds ({rt60*1000:.1f} ms)")


def example_2_batch_processing():
    """Example 2: Process multiple videos from AV-Deepfake1M"""
    if not HAS_DEPS:
        print("Dependencies required. Install with: pip install blind-rt60 librosa")
        return
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Processing (AV-Deepfake1M)")
    print("="*60)
    
    from rt60_extractor import RT60Extractor, load_dataset_metadata
    
    # Load metadata
    metadata_file = "/path/to/train_metadata.json"
    dataset_dir = "/path/to/dataset"
    
    video_paths = load_dataset_metadata(metadata_file)
    # Make paths absolute
    video_paths = [f"{dataset_dir}/{p}" for p in video_paths[:100]]  # First 100
    
    # Process batch
    extractor = RT60Extractor()
    extractor.process_batch(video_paths, "rt60_results.csv")
    
    print("Results saved to rt60_results.csv")


def example_3_direct_api():
    """Example 3: Direct blind_rt60 API usage"""
    if not HAS_DEPS:
        print("Dependencies required. Install with: pip install blind-rt60 librosa")
        return
    
    print("\n" + "="*60)
    print("EXAMPLE 3: Direct API Usage")
    print("="*60)
    
    from blind_rt60 import BlindRT60
    import librosa
    
    # Load audio
    audio_path = "path/to/audio.wav"
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Create estimator
    estimator = BlindRT60()
    
    # Estimate RT60
    rt60 = estimator(audio, sr)
    
    print(f"RT60: {rt60:.3f} seconds")
    
    # Visualize
    fig = estimator.visualize(audio, sr)
    # plt.show()


def example_4_analysis_and_visualization():
    """Example 4: Analyze results and create visualizations"""
    if not HAS_DEPS:
        print("Dependencies required. Install with: pip install blind-rt60 librosa")
        return
    
    print("\n" + "="*60)
    print("EXAMPLE 4: Analysis & Visualization")
    print("="*60)
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load results
    df = pd.read_csv("rt60_results.csv")
    
    print(f"Processed {len(df)} videos")
    print(f"Mean RT60: {df['rt60'].mean():.3f}s")
    print(f"Std RT60: {df['rt60'].std():.3f}s")
    print(f"Min RT60: {df['rt60'].min():.3f}s")
    print(f"Max RT60: {df['rt60'].max():.3f}s")
    
    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(df['rt60'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('RT60 (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of RT60 Values')
    plt.grid(True, alpha=0.3)
    plt.savefig('rt60_distribution.png')
    print("Saved: rt60_distribution.png")


# ============================================================================
# COMMAND LINE USAGE
# ============================================================================

"""
Once installed, you can use the pipeline from command line:

1. Process with metadata file:
   python rt60_extractor.py \
       --metadata /path/to/train_metadata.json \
       --dataset-dir /path/to/dataset \
       --output results.csv \
       --limit 100

2. Process with video list:
   python rt60_extractor.py \
       --video-list videos.txt \
       --output results.csv

3. Test the pipeline:
   python test_rt60.py

"""

# ============================================================================
# COMPLETE PIPELINE EXAMPLE (Template)
# ============================================================================

PIPELINE_TEMPLATE = """
# Complete workflow in Python

from rt60_extractor import RT60Extractor, load_dataset_metadata
import pandas as pd
import matplotlib.pyplot as plt
import json

# Step 1: Load dataset metadata
metadata_file = "/data/AV-Deepfake1M/train_metadata.json"
dataset_dir = "/data/AV-Deepfake1M/train"

video_paths = load_dataset_metadata(metadata_file)

# Step 2: Process batch (first 500 videos)
extractor = RT60Extractor()
extractor.process_batch(video_paths[:500], "rt60_results.csv")

# Step 3: Load and analyze results
df = pd.read_csv("rt60_results.csv")

# Step 4: Load original metadata for comparison
with open(metadata_file) as f:
    metadata = json.load(f)

# Merge with RT60 data
metadata_df = pd.DataFrame(metadata)
results_df = df.merge(metadata_df, left_on='filename', right_on='file')

# Step 5: Analyze by deepfake type
fake_types = results_df['modify_type'].unique()
for fake_type in fake_types:
    mask = results_df['modify_type'] == fake_type
    rt60_vals = results_df[mask]['rt60']
    print(f"{fake_type}: mean RT60 = {rt60_vals.mean():.3f}s")

# Step 6: Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, fake_type in enumerate(fake_types):
    ax = axes.flat[idx]
    mask = results_df['modify_type'] == fake_type
    ax.hist(results_df[mask]['rt60'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'RT60 Distribution: {fake_type}')
    ax.set_xlabel('RT60 (seconds)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rt60_analysis.png', dpi=150)
print("Saved: rt60_analysis.png")

# Step 7: Export statistics
stats = results_df.groupby('modify_type')['rt60'].agg(['mean', 'std', 'min', 'max'])
stats.to_csv('rt60_statistics_by_type.csv')
print(stats)
"""


if __name__ == "__main__":
    print(__doc__)
    
    if HAS_DEPS:
        print("\n✓ Dependencies installed - examples are ready to run!")
        print("\nTo see usage examples, edit this file and uncomment:")
        print("  - example_1_simple_extraction()")
        print("  - example_2_batch_processing()")
        print("  - example_3_direct_api()")
        print("  - example_4_analysis_and_visualization()")
    else:
        print("\n⚠ First install dependencies:")
        print("  pip install blind-rt60 librosa scipy numpy")
        print("\nThen run: python rt60_extractor.py --help")
