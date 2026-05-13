"""
RT60 Extraction Pipeline for AV-Deepfake1M Dataset

Extracts reverberation time (RT60) from audio channels of deepfake videos
using the blind_rt60 library.
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
import subprocess
import tempfile

import numpy as np
import librosa
from blind_rt60 import BlindRT60

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RT60Extractor:
    """Extract RT60 from video audio channels."""

    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize RT60 extractor.
        
        Args:
            temp_dir: Directory for temporary audio files. 
                     Defaults to system temp directory.
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.estimator = BlindRT60()
        self.target_sr = 16000  # Standard sampling rate
        
    def extract_audio_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract audio from video file using ffmpeg.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Audio signal as numpy array, or None if extraction failed
        """
        try:
            # Use ffmpeg to extract audio to WAV
            audio_data, sr = librosa.load(
                video_path, 
                sr=self.target_sr, 
                mono=True
            )
            logger.info(f"Extracted audio from {Path(video_path).name}")
            return audio_data
        except Exception as e:
            logger.error(f"Failed to extract audio from {video_path}: {e}")
            return None

    def estimate_rt60(
        self, 
        audio: np.ndarray, 
        sr: int = 16000
    ) -> Optional[float]:
        """
        Estimate RT60 from audio signal using blind_rt60.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sampling rate (default: 16000 Hz)
            
        Returns:
            RT60 value in seconds, or None if estimation failed
        """
        try:
            if len(audio) < sr:  # Audio too short (less than 1 second)
                logger.warning("Audio too short for reliable RT60 estimation")
                return None
                
            rt60 = self.estimator(audio, sr)
            logger.info(f"Estimated RT60: {rt60:.3f} seconds")
            return float(rt60)
        except Exception as e:
            logger.error(f"RT60 estimation failed: {e}")
            return None

    def process_video(self, video_path: str) -> Optional[Dict]:
        """
        Extract RT60 from a single video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with results or None if processing failed
        """
        logger.info(f"Processing: {video_path}")
        
        # Extract audio
        audio = self.extract_audio_from_video(video_path)
        if audio is None:
            return None
        
        # Estimate RT60
        rt60 = self.estimate_rt60(audio, self.target_sr)
        if rt60 is None:
            return None
        
        return {
            'video_path': video_path,
            'filename': Path(video_path).name,
            'rt60': rt60,
            'audio_duration': len(audio) / self.target_sr,
            'status': 'success'
        }

    def process_batch(
        self, 
        video_paths: List[str],
        output_csv: str = "rt60_results.csv"
    ) -> None:
        """
        Process multiple videos and save results to CSV.
        
        Args:
            video_paths: List of video file paths
            output_csv: Path to output CSV file
        """
        results = []
        failed = []
        
        logger.info(f"Processing {len(video_paths)} videos...")
        
        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"[{i}/{len(video_paths)}] {Path(video_path).name}")
            
            result = self.process_video(video_path)
            if result:
                results.append(result)
            else:
                failed.append(video_path)
        
        # Save results to CSV
        if results:
            self._save_to_csv(results, output_csv)
            logger.info(f"Results saved to {output_csv}")
        
        # Log summary
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing Summary:")
        logger.info(f"  Total videos: {len(video_paths)}")
        logger.info(f"  Successful: {len(results)}")
        logger.info(f"  Failed: {len(failed)}")
        if failed:
            logger.warning(f"Failed videos: {failed}")

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


def load_dataset_metadata(metadata_path: str) -> List[str]:
    """
    Load video paths from AV-Deepfake1M metadata JSON.
    
    Args:
        metadata_path: Path to metadata.json file
        
    Returns:
        List of video file paths
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract file paths from metadata
        video_paths = [item['file'] for item in metadata]
        logger.info(f"Loaded {len(video_paths)} videos from metadata")
        return video_paths
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return []


if __name__ == "__main__":
    """
    Example usage:
    
    python rt60_extractor.py \
        --metadata path/to/train_metadata.json \
        --dataset-dir path/to/dataset \
        --output results.csv \
        --limit 100
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract RT60 from AV-Deepfake1M video audio"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to metadata JSON file"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Base directory of dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rt60_results.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to process (for testing)"
    )
    parser.add_argument(
        "--video-list",
        type=str,
        help="Path to text file with video paths (one per line)"
    )
    
    args = parser.parse_args()
    
    # Get video paths
    video_paths = []
    
    if args.metadata:
        video_paths = load_dataset_metadata(args.metadata)
        
        # Make absolute paths
        if args.dataset_dir:
            base_dir = Path(args.dataset_dir)
            video_paths = [str(base_dir / Path(vp).name) for vp in video_paths]
    
    elif args.video_list:
        with open(args.video_list, 'r') as f:
            video_paths = [line.strip() for line in f if line.strip()]
    
    # Limit videos if specified
    if args.limit:
        video_paths = video_paths[:args.limit]
    
    if not video_paths:
        logger.error("No video paths found. Provide --metadata or --video-list")
        exit(1)
    
    # Process
    extractor = RT60Extractor()
    extractor.process_batch(video_paths, args.output)
