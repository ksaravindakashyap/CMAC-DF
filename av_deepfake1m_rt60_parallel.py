#!/usr/bin/env python3
"""
Parallelized RT60 extraction for AV-Deepfake1M on Tinkercliff cluster.
Uses multiprocessing to process videos in parallel across multiple cores.

AV-Deepfake1M structure:
  dataset/
  ├── train/
  │   ├── original/
  │   │   └── video_files.mp4
  │   └── deepfake/
  │       └── video_files.mp4
  └── test/
      ├── original/
      └── deepfake/

Usage (on login node):
    python av_deepfake1m_rt60_parallel.py --dataset-dir /path/to/AV-Deepfake1M --split train --category original --output results.csv

Usage (via SLURM):
    sbatch submit_rt60_job_av_deepfake.sh
"""

import os
import sys
import csv
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback

try:
    import librosa
    from blind_rt60 import BlindRT60
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package. Install with: pip install librosa blind-rt60 numpy scipy")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AVDeepfake1MParallelExtractor:
    """Parallel RT60 extraction for AV-Deepfake1M dataset."""
    
    def __init__(self, dataset_dir: str):
        """Initialize extractor with AV-Deepfake1M root directory."""
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset root not found: {dataset_dir}")
        
        self.statistics = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
        }
    
    def discover_videos(self, split: str = 'train', category: Optional[str] = None, 
                       limit: int = None) -> List[str]:
        """
        Discover all MP4 videos in AV-Deepfake1M dataset.
        
        Args:
            split: 'train' or 'test'
            category: 'original', 'deepfake', or None for both
            limit: Max videos to process (for testing)
        
        Returns:
            List of absolute paths to video files
        """
        split_dir = self.dataset_dir / split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        videos = []
        
        # Determine which categories to search
        if category:
            categories = [category]
        else:
            categories = ['original', 'deepfake']
        
        for cat in categories:
            cat_dir = split_dir / cat
            if cat_dir.exists():
                cat_videos = sorted(cat_dir.rglob('*.mp4'))
                videos.extend(cat_videos)
                logger.info(f"Found {len(cat_videos)} videos in {cat} category")
        
        if limit:
            videos = videos[:limit]
        
        logger.info(f"Discovered {len(videos)} total videos from {split} split")
        return [str(v) for v in videos]
    
    @staticmethod
    def _extract_category(video_path: str) -> str:
        """Extract category (original/deepfake) from video path."""
        parts = Path(video_path).parts
        if 'original' in parts:
            return 'original'
        elif 'deepfake' in parts:
            return 'deepfake'
        return "unknown"
    
    @staticmethod
    def _extract_audio(video_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Extract audio from video file.
        Tries to load WAV file first (if available), then MP4.
        
        Args:
            video_path: Path to video file
            sr: Sampling rate (default 16000 Hz)
        
        Returns:
            Tuple of (audio array, sample rate)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Try to use WAV file if it exists (same path, different extension)
        wav_path = Path(video_path).with_suffix('.wav')
        if wav_path.exists():
            audio, sr_loaded = librosa.load(str(wav_path), sr=sr, mono=True)
            return audio, sr_loaded
        
        # Fall back to MP4
        audio, sr_loaded = librosa.load(video_path, sr=sr, mono=True)
        return audio, sr_loaded
    
    @staticmethod
    def _estimate_rt60(audio: np.ndarray, sr: int = 16000) -> float:
        """
        Estimate RT60 from audio signal using blind RT60 estimation.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sampling rate
        
        Returns:
            RT60 value in seconds
        """
        if len(audio) < sr:  # Less than 1 second
            return -1.0  # Invalid
        
        estimator = BlindRT60(fs=int(sr))
        result = estimator.estimate(audio, fs=int(sr))
        return float(result)
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process single video: extract audio and estimate RT60.
        
        Args:
            video_path: Absolute path to video file
        
        Returns:
            Dictionary with results
        """
        result = {
            'video_path': video_path,
            'relative_path': str(Path(video_path).relative_to(self.dataset_dir))
                            if Path(video_path).is_relative_to(self.dataset_dir) else video_path,
            'video_filename': Path(video_path).name,
            'category': self._extract_category(video_path),
            'rt60': -1.0,
            'audio_duration': -1.0,
            'status': 'error'
        }
        
        try:
            # Extract audio
            audio, sr = self._extract_audio(video_path, sr=16000)
            result['audio_duration'] = float(len(audio) / sr)
            
            # Estimate RT60
            rt60_value = self._estimate_rt60(audio, sr=sr)
            result['rt60'] = rt60_value
            result['status'] = 'success'
            
        except FileNotFoundError:
            result['status'] = 'file_not_found'
        except Exception as e:
            result['status'] = f'error: {type(e).__name__}'
            logger.debug(f"Error processing {video_path}: {e}")
        
        return result
    
    def process_batch(self, video_paths: List[str], num_workers: int = None,
                      output_csv: str = None) -> Dict:
        """
        Process batch of videos in parallel.
        
        Args:
            video_paths: List of video file paths
            num_workers: Number of parallel workers (default: num CPUs)
            output_csv: Optional path to save results to CSV
        
        Returns:
            Dictionary with results and statistics
        """
        if num_workers is None:
            num_workers = cpu_count()
        
        logger.info(f"Starting parallel processing with {num_workers} workers")
        logger.info(f"Processing {len(video_paths)} videos")
        
        results = []
        processed = 0
        
        # Use multiprocessing pool for parallel processing
        with Pool(processes=num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(
                partial(self.process_video), 
                video_paths,
                chunksize=max(1, len(video_paths) // (num_workers * 4))
            ), 1):
                results.append(result)
                
                # Update statistics
                if result['status'] == 'success':
                    self.statistics['success'] += 1
                else:
                    self.statistics['failed'] += 1
                self.statistics['total'] += 1
                
                # Log progress every 100 videos
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(video_paths)} videos "
                               f"({self.statistics['success']} successful)")
        
        # Save to CSV if requested
        if output_csv:
            self._save_to_csv(results, output_csv)
            logger.info(f"Results saved to {output_csv}")
        
        # Log final statistics
        success_rate = (self.statistics['success'] / self.statistics['total'] * 100
                       if self.statistics['total'] > 0 else 0)
        logger.info(f"Completed {self.statistics['total']} videos")
        logger.info(f"Success: {self.statistics['success']}, Failed: {self.statistics['failed']}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        return {
            'results': results,
            'statistics': self.statistics,
        }
    
    @staticmethod
    def _save_to_csv(results: List[Dict], output_path: str) -> None:
        """Save results to CSV file."""
        if not results:
            logger.warning("No results to save")
            return
        
        fieldnames = results[0].keys()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"Saved {len(results)} results to {output_path}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Parallel RT60 extraction for AV-Deepfake1M on Tinkercliff',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process original videos from training set
  python av_deepfake1m_rt60_parallel.py --dataset-dir /mnt/av_deepfake1m --split train --category original --output results_original.csv
  
  # Process deepfake videos from test set
  python av_deepfake1m_rt60_parallel.py --dataset-dir /mnt/av_deepfake1m --split test --category deepfake --output results_deepfake.csv
  
  # Process both categories with all available cores
  python av_deepfake1m_rt60_parallel.py --dataset-dir /mnt/av_deepfake1m --split train --output results_all.csv --workers 16
        """
    )
    
    parser.add_argument('--dataset-dir', required=True,
                       help='Path to AV-Deepfake1M root directory')
    parser.add_argument('--split', default='train', choices=['train', 'test'],
                       help='Dataset split to process (train or test)')
    parser.add_argument('--category', default=None, choices=['original', 'deepfake', None],
                       help='Video category (original, deepfake, or both)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of videos to process (for testing)')
    parser.add_argument('--output', default='av_deepfake1m_rt60.csv',
                       help='Output CSV file path')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: number of CPUs)')
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = AVDeepfake1MParallelExtractor(args.dataset_dir)
        
        # Discover videos
        videos = extractor.discover_videos(split=args.split, category=args.category, limit=args.limit)
        
        if not videos:
            logger.error("No videos found to process")
            return 1
        
        # Process videos in parallel
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = extractor.process_batch(
            videos,
            num_workers=args.workers,
            output_csv=str(output_path)
        )
        
        return 0
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
