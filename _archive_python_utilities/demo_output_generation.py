"""
Demonstration: CSV Output Generation & VoxCeleb2 Usage

This script shows:
1. How CSV files are generated (raw Python)
2. How to use the VoxCeleb2 extractor
3. How to analyze output CSV files
"""

import csv
import tempfile
from pathlib import Path
import json


def demo_1_csv_generation():
    """Demo 1: Show exactly how CSV files are generated"""
    print("\n" + "="*70)
    print("DEMO 1: How CSV Output Files Are Generated")
    print("="*70)
    
    # Step 1: Create results as Python dictionaries
    print("\n1. Create results as list of dictionaries:")
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
        },
        {
            'video_path': '/data/video3.mp4',
            'filename': 'video3.mp4',
            'rt60': None,
            'audio_duration': 5.123,
            'status': 'failed'
        }
    ]
    
    print("   Python objects created:")
    for i, result in enumerate(results, 1):
        print(f"     {i}. {result}")
    
    # Step 2: Write to CSV using Python csv module
    print("\n2. Write to CSV using csv.DictWriter:")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', 
                                     delete=False, newline='') as f:
        csv_path = f.name
        
        # Get column names from first result
        fieldnames = results[0].keys()
        print(f"   Column names: {list(fieldnames)}")
        
        # Create CSV writer
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header (column names)
        print(f"   Writing header row...")
        writer.writeheader()
        
        # Write data rows
        print(f"   Writing {len(results)} data rows...")
        writer.writerows(results)
    
    # Step 3: Read and display the file
    print(f"\n3. Read CSV file contents:")
    print(f"   File saved to: {csv_path}")
    
    with open(csv_path, 'r') as f:
        content = f.read()
        print(f"\n   Raw CSV content:")
        print("   " + "-" * 66)
        for line in content.split('\n'):
            if line:
                print(f"   {line}")
        print("   " + "-" * 66)
    
    # Step 4: Parse back to Python
    print(f"\n4. Read CSV back to Python:")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"   Loaded {len(rows)} rows:")
        for i, row in enumerate(rows, 1):
            print(f"     Row {i}: {dict(row)}")
    
    # Cleanup
    Path(csv_path).unlink()


def demo_2_voxceleb2_format():
    """Demo 2: Show VoxCeleb2 CSV format with real-world structure"""
    print("\n" + "="*70)
    print("DEMO 2: VoxCeleb2 Output Format")
    print("="*70)
    
    print("""
VoxCeleb2 has hierarchical structure:
  /data/VoxCeleb2/dev/mp4/
    ├── id10001/          (Speaker 1)
    │   ├── 1HDQeFPvL6c/  (Utterance 1)
    │   │   ├── 00001.mp4
    │   │   ├── 00002.mp4
    │   └── 2aCdaNq7p3K/  (Utterance 2)
    │       └── 00001.mp4
    └── id10002/          (Speaker 2)
        └── 3bCdaNq7p3K/
            └── 00001.mp4

RT60 extraction outputs speaker_id and utterance_id:
    """)
    
    voxceleb2_results = [
        {
            'video_path': '/data/VoxCeleb2/dev/mp4/id10001/1HDQeFPvL6c/00001.mp4',
            'relative_path': 'dev/mp4/id10001/1HDQeFPvL6c/00001.mp4',
            'video_filename': '00001.mp4',
            'speaker_id': 'id10001',
            'utterance_id': '1HDQeFPvL6c',
            'rt60': 0.456,
            'audio_duration': 10.234,
            'status': 'success'
        },
        {
            'video_path': '/data/VoxCeleb2/dev/mp4/id10001/2aCdaNq7p3K/00001.mp4',
            'relative_path': 'dev/mp4/id10001/2aCdaNq7p3K/00001.mp4',
            'video_filename': '00001.mp4',
            'speaker_id': 'id10001',
            'utterance_id': '2aCdaNq7p3K',
            'rt60': 0.523,
            'audio_duration': 8.765,
            'status': 'success'
        },
        {
            'video_path': '/data/VoxCeleb2/dev/mp4/id10002/3bCdaNq7p3K/00001.mp4',
            'relative_path': 'dev/mp4/id10002/3bCdaNq7p3K/00001.mp4',
            'video_filename': '00001.mp4',
            'speaker_id': 'id10002',
            'utterance_id': '3bCdaNq7p3K',
            'rt60': 0.312,
            'audio_duration': 11.234,
            'status': 'success'
        }
    ]
    
    # Save to CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                     delete=False, newline='') as f:
        csv_path = f.name
        fieldnames = voxceleb2_results[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(voxceleb2_results)
    
    # Display
    print("VoxCeleb2 output CSV:")
    print("-" * 120)
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                # Format header
                print(line.rstrip())
            else:
                # Format data rows for readability
                print(line.rstrip())
    print("-" * 120)
    
    # Cleanup
    Path(csv_path).unlink()


def demo_3_analysis_workflow():
    """Demo 3: Show how to analyze CSV results"""
    print("\n" + "="*70)
    print("DEMO 3: Analyzing CSV Results with Pandas")
    print("="*70)
    
    print("""
After running the pipeline and getting a CSV file, you can analyze it:

    import pandas as pd
    
    # Load results
    df = pd.read_csv('voxceleb2_rt60_results.csv')
    
    # Basic statistics
    print(f"Total videos: {len(df)}")
    print(f"Mean RT60: {df['rt60'].mean():.3f}s")
    print(f"Std RT60: {df['rt60'].std():.3f}s")
    
    # Filter by speaker
    speaker_1 = df[df['speaker_id'] == 'id10001']
    print(f"Videos for id10001: {len(speaker_1)}")
    print(f"Mean RT60 for id10001: {speaker_1['rt60'].mean():.3f}s")
    
    # Find failed extractions
    failed = df[df['status'] != 'success']
    print(f"Failed videos: {len(failed)}")
    
    # Sort by RT60
    sorted_df = df.sort_values('rt60', ascending=False)
    print("Top 5 videos with highest RT60:")
    print(sorted_df[['speaker_id', 'utterance_id', 'rt60']].head())
    
    # Visualize
    import matplotlib.pyplot as plt
    df['rt60'].hist(bins=50)
    plt.xlabel('RT60 (seconds)')
    plt.ylabel('Count')
    plt.title('RT60 Distribution - VoxCeleb2')
    plt.savefig('voxceleb2_rt60_distribution.png')
    """)


def demo_4_file_generation_code():
    """Demo 4: Show the actual code in rt60_extractor.py"""
    print("\n" + "="*70)
    print("DEMO 4: Actual Code for CSV Generation")
    print("="*70)
    
    print("""
From rt60_extractor.py, the CSV generation code is:

    @staticmethod
    def _save_to_csv(results: List[Dict], output_path: str) -> None:
        '''Save results to CSV file.'''
        if not results:
            return
        
        fieldnames = results[0].keys()
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

How it works:
  1. results = [
       {'video_path': '...', 'filename': '...', 'rt60': 0.456, ...},
       {'video_path': '...', 'filename': '...', 'rt60': 0.382, ...},
       ...
     ]
  
  2. fieldnames = results[0].keys()
     → Gets column names: ['video_path', 'filename', 'rt60', ...]
  
  3. writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     → Creates CSV writer with those columns
  
  4. writer.writeheader()
     → Writes: video_path,filename,rt60,audio_duration,status
  
  5. writer.writerows(results)
     → Writes each dictionary as a CSV row
    """)


def demo_5_voxceleb2_usage():
    """Demo 5: Show how to use VoxCeleb2 extractor"""
    print("\n" + "="*70)
    print("DEMO 5: Using VoxCeleb2 RT60 Extractor")
    print("="*70)
    
    print("""
Command line usage:

    python voxceleb2_rt60_extractor.py \\
        --voxceleb2-root /path/to/VoxCeleb2 \\
        --split dev \\
        --limit 100 \\
        --output voxceleb2_rt60_sample.csv

This will:
  1. Discover videos in /path/to/VoxCeleb2/dev/mp4/
  2. Process first 100 videos
  3. Extract audio from each video
  4. Estimate RT60 using blind_rt60
  5. Save results to voxceleb2_rt60_sample.csv
  6. Print statistics (mean, std, min, max RT60)

Programmatic usage:

    from voxceleb2_rt60_extractor import VoxCeleb2RT60Extractor
    
    extractor = VoxCeleb2RT60Extractor('/path/to/VoxCeleb2')
    
    # Option 1: Batch process
    extractor.process_batch(split='dev', limit=100, output_csv='results.csv')
    
    # Option 2: Manual process per video
    videos = extractor.discover_videos('dev', limit=10)
    for video_path in videos:
        result = extractor.process_video(video_path)
        if result:
            print(f"RT60: {result['rt60']:.3f}s")
    """)


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║  CSV Generation & VoxCeleb2 Extraction Demonstration          ║")
    print("╚" + "="*68 + "╝")
    
    demo_1_csv_generation()
    demo_2_voxceleb2_format()
    demo_3_analysis_workflow()
    demo_4_file_generation_code()
    demo_5_voxceleb2_usage()
    
    print("\n" + "="*70)
    print("Demonstration Complete!")
    print("="*70)
    print("""
Key Takeaways:

1. CSV OUTPUT GENERATION:
   - Results stored as Python dictionaries
   - csv.DictWriter converts to CSV format
   - One row per dictionary, columns from keys
   
2. VOXCELEB2 STRUCTURE:
   - Organized by speaker_id and utterance_id
   - CSV includes these IDs for organization
   - Easy to group and filter by speaker

3. USAGE:
   - Command line: python voxceleb2_rt60_extractor.py --voxceleb2-root /path
   - Programming: Import VoxCeleb2RT60Extractor class
   - Analysis: Load CSV with pandas, query/visualize

4. FILES CREATED:
   ✓ voxceleb2_rt60_extractor.py (main pipeline)
   ✓ VOXCELEB2_GUIDE.md (detailed documentation)
   ✓ This demo script (examples and walkthroughs)
    """)
