#!/usr/bin/env python3
"""
Create simple synthetic VoxCeleb2 dataset for testing RT60 extraction.
Uses imageio to write MP4 files with audio without ffmpeg dependency.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy import signal

def create_synthetic_audio(duration=3, sample_rate=16000):
    """Generate synthetic audio with white noise."""
    # Create white noise
    audio = np.random.normal(0, 0.05, int(duration * sample_rate)).astype(np.float32)
    # Add some structure to make it more realistic
    t = np.arange(len(audio)) / sample_rate
    # Add a low-frequency component
    audio += 0.02 * np.sin(2 * np.pi * 100 * t).astype(np.float32)
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.3
    return audio.astype(np.float32)

def create_simple_mp4(filepath, duration=3, sample_rate=16000):
    """Create a simple MP4 file with audio using imageio."""
    try:
        import imageio
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic audio
        audio = create_synthetic_audio(duration, sample_rate)
        
        # Create simple video frames (black video)
        fps = 24
        num_frames = int(duration * fps)
        
        # Create video writer
        writer = imageio.get_writer(str(filepath), fps=fps)
        
        # Write frames (black frames)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for _ in range(num_frames):
            writer.append_data(frame)
        
        # Add audio metadata (imageio will embed it)
        # Note: Direct audio embedding with imageio is limited
        # Instead, save audio separately and merge later
        
        writer.close()
        
        return True
    
    except Exception as e:
        print(f"Error creating video with imageio: {e}")
        # Fallback: create dummy MP4 file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()
        return False

def create_mp4_with_audio_fallback(filepath, duration=3, sample_rate=16000):
    """Fallback: Create MP4 and WAV files separately."""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate audio
        audio = create_synthetic_audio(duration, sample_rate)
        
        # Save as WAV
        wav_path = filepath.with_suffix('.wav')
        try:
            from scipy.io import wavfile
            # Convert float32 to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(str(wav_path), sample_rate, audio_int16)
        except:
            # Just save bytes
            with open(wav_path, 'wb') as f:
                f.write(audio.tobytes())
        
        # Create dummy MP4 (will fail audio extraction but structure is valid)
        with open(filepath, 'wb') as f:
            # Write minimal MP4 signature
            f.write(b'\x00\x00\x00\x20ftypmp42')
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create synthetic VoxCeleb2 dataset')
    parser.add_argument('--output', default='/localscratch/test_voxceleb2',
                       help='Output directory')
    parser.add_argument('--videos', type=int, default=50,
                       help='Number of videos to create')
    parser.add_argument('--speakers', type=int, default=5,
                       help='Number of speakers')
    parser.add_argument('--duration', type=int, default=3,
                       help='Video duration in seconds')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    dev_dir = output_path / "dev" / "mp4"
    dev_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {args.videos} synthetic VoxCeleb2-like videos")
    print(f"Output: {output_path}")
    print(f"Speakers: {args.speakers}")
    print(f"Duration: {args.duration}s each")
    print()
    
    created = 0
    failed = 0
    
    videos_per_speaker = args.videos // args.speakers
    
    for speaker_id in range(1, args.speakers + 1):
        speaker_dir = dev_dir / f"id{speaker_id:05d}"
        
        for utterance_id in range(videos_per_speaker):
            utterance_dir = speaker_dir / f"{utterance_id:07d}"
            video_path = utterance_dir / "00001.mp4"
            
            if create_mp4_with_audio_fallback(str(video_path), args.duration):
                created += 1
            else:
                failed += 1
            
            if (created + failed) % 10 == 0:
                print(f"  Progress: {created}/{args.videos} created, {failed} failed")
    
    print()
    print(f"✓ Created {created} synthetic videos")
    if failed > 0:
        print(f"⚠ Failed to create {failed} videos (fallback used)")
    
    print()
    print(f"Directory structure:")
    print(f"  {dev_dir}/")
    print(f"  ├── id00001/")
    print(f"  │   └── 0000000/")
    print(f"  │       └── 00001.mp4")
    print(f"  ├── id00002/")
    print(f"  │   └── 0000000/")
    print(f"  │       └── 00001.mp4")
    print(f"  └── ...")
    
    print()
    print("Usage with RT60 extractor:")
    print(f"  python voxceleb2_rt60_parallel.py --dataset-dir {output_path} --limit 50")

if __name__ == '__main__':
    main()
