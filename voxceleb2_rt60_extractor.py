"""
VoxCeleb2 RT60 Extraction

This script handles RT60 extraction from VoxCeleb2 dataset videos.
VoxCeleb2 contains 1M+ utterances from 6112 speakers with both audio and video.

Directory Structure:
    VoxCeleb2/
    ├── dev/
    │   ├── aac/              # Audio files
    │   │   ├── id10001/
    │   │   │   ├── 1HDQeFPvL6c/
    │   │   │   │   ├── 00001.m4a
    │   │   ├── id10002/
    │   ├── mp4/              # Video files
    │   │   ├── id10001/
    │   │   │   ├── 1HDQeFPvL6c/
    │   │   │   │   ├── 00001.mp4
    ├── test/
    │   ├── aac/
    │   ├── mp4/
"""

import os
import csv
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json

import numpy as np
import librosa
from blind_rt60 import BlindRT60

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoxCeleb2RT60Extractor:
    """Extract RT60 from VoxCeleb2 dataset."""

    def __init__(self, voxceleb2_root: str, target_sr: int = 16000):
        """
        Initialize VoxCeleb2 extractor.
        
        Args:
            voxceleb2_root: Root directory of VoxCeleb2 dataset
            target_sr: Target sampling rate
        """
        self.voxceleb2_root = Path(voxceleb2_root)
        self.target_sr = target_sr
        self.estimator = BlindRT60()
        
        # Validate structure
        if not self.voxceleb2_root.exists():
            raise FileNotFoundError(f"VoxCeleb2 root not found: {voxceleb2_root}")
    
    def discover_videos(self, split: str = "dev", limit: Optional[int] = None) -> List[Path]:
        """
        Discover all video files in VoxCeleb2.
        
        Args:
            split: "dev" or "test"
            limit: Maximum number of videos to return. None = all.
            
        Returns:
            List of video file paths
        """
        mp4_dir = self.voxceleb2_root / split / "mp4"
        
        if not mp4_dir.exists():
            logger.error(f"Directory not found: {mp4_dir}")
            return []
        
        video_files = sorted(mp4_dir.glob("**/*.mp4"))
        
        if limit:
            video_files = video_files[:limit]
        
        logger.info(f"Found {len(video_files)} videos in {split} split")
        return video_files
    
    def extract_audio(self, video_path: Path) -> Optional[np.ndarray]:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Audio signal or None if extraction failed
        """
        try:
            audio, sr = librosa.load(
                str(video_path),
                sr=self.target_sr,
                mono=True
            )
            return audio
        except Exception as e:
            logger.error(f"Failed to extract audio from {video_path.name}: {e}")
            return None
    
    def estimate_rt60(self, audio: np.ndarray) -> Optional[float]:
        """
        Estimate RT60 from audio signal.
        
        Args:
            audio: Audio signal
            
        Returns:
            RT60 in seconds or None
        """
        try:
            if len(audio) < self.target_sr:  # Less than 1 second
                logger.warning("Audio too short (< 1s)")
                return None
            
            rt60 = self.estimator(audio, self.target_sr)
            return float(rt60)
        except Exception as e:
            logger.error(f"RT60 estimation failed: {e}")
            return None
    
    def extract_speaker_id(self, video_path: Path) -> str:
        """Extract speaker ID from video path (e.g., id10001)."""
        return video_path.parents[1].name
    
    def extract_utterance_id(self, video_path: Path) -> str:
        """Extract utterance ID from video path (e.g., 1HDQeFPvL6c)."""
        return video_path.parent.name
    
    def process_video(self, video_path: Path) -> Optional[Dict]:
        """
        Process a single video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with results or None if failed
        """
        logger.info(f"Processing: {video_path.relative_to(self.voxceleb2_root)}")
        
        # Extract audio
        audio = self.extract_audio(video_path)
        if audio is None:
            return None
        
        # Estimate RT60
        rt60 = self.estimate_rt60(audio)
        if rt60 is None:
            return None
        
        return {
            'video_path': str(video_path),
            'relative_path': str(video_path.relative_to(self.voxceleb2_root)),
            'video_filename': video_path.name,
            'speaker_id': self.extract_speaker_id(video_path),
            'utterance_id': self.extract_utterance_id(video_path),
            'rt60': rt60,
            'audio_duration': len(audio) / self.target_sr,
            'status': 'success'
        }
    
    def process_batch(
        self,
        split: str = "dev",
        limit: Optional[int] = None,
        output_csv: str = "voxceleb2_rt60_results.csv"
    ) -> None:
        """
        Process multiple videos from VoxCeleb2.
        
        Args:
            split: "dev" or "test"
            limit: Maximum videos to process
            output_csv: Output CSV filename
        """
        # Discover videos
        video_paths = self.discover_videos(split, limit)
        
        if not video_paths:
            logger.error("No videos found to process")
            return
        
        results = []
        failed = []
        
        logger.info(f"Processing {len(video_paths)} videos from {split} split...")
        
        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"[{i}/{len(video_paths)}]")
            
            result = self.process_video(video_path)
            if result:
                results.append(result)
            else:
                failed.append(str(video_path))
        
        # Save results
        if results:
            self._save_to_csv(results, output_csv)
            logger.info(f"\n{'='*60}")
            logger.info(f"Results saved to {output_csv}")
            logger.info(f"Total processed: {len(video_paths)}")
            logger.info(f"Successful: {len(results)}")
            logger.info(f"Failed: {len(failed)}")
            
            # Print statistics
            rt60_values = [r['rt60'] for r in results]
            logger.info(f"\nRT60 Statistics:")
            logger.info(f"  Mean: {np.mean(rt60_values):.3f}s")
            logger.info(f"  Std: {np.std(rt60_values):.3f}s")
            logger.info(f"  Min: {np.min(rt60_values):.3f}s")
            logger.info(f"  Max: {np.max(rt60_values):.3f}s")
            logger.info(f"{'='*60}")
    
    @staticmethod
    def _save_to_csv(results: List[Dict], output_path: str) -> None:
        """Save results to CSV file."""
        if not results:
            return
        
        fieldnames = results[0].keys()
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"CSV saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract RT60 from VoxCeleb2 dataset"
    )
    parser.add_argument(
        "--voxceleb2-root",
        type=str,
        required=True,
        help="Root directory of VoxCeleb2 dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Dataset split (default: dev)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos (default: process all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="voxceleb2_rt60_results.csv",
        help="Output CSV filename"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = VoxCeleb2RT60Extractor(args.voxceleb2_root)
    
    # Process batch
    extractor.process_batch(
        split=args.split,
        limit=args.limit,
        output_csv=args.output
    )
