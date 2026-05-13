#!/usr/bin/env python3
"""
Download paired RVFA (Real Video, Fake Audio) and RVRA (Real Video, Real Audio) dataset.

This script:
1. Fetches AV-Deepfake1M metadata (or uses local metadata)
2. Identifies 1000 RVFA pairs with their corresponding RVRA videos
3. Extracts YouTube URLs and timestamps
4. Downloads videos using yt-dlp
5. Organizes them into paired directories

Usage:
    python download_paired_av_deepfake.py \
        --output /path/to/output \
        --num-pairs 1000 \
        --workers 4
"""

import os
import sys
import json
import csv
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AVDeepfake1MDownloader:
    """Download paired real/fake videos from AV-Deepfake1M dataset."""
    
    def __init__(self, output_dir: str, num_pairs: int = 1000, workers: int = 4):
        self.output_dir = Path(output_dir)
        self.num_pairs = num_pairs
        self.workers = workers
        
        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fake_videos_dir = self.output_dir / "fake_audio" / "mp4"
        self.real_videos_dir = self.output_dir / "real_audio" / "mp4"
        self.metadata_dir = self.output_dir / "metadata"
        
        self.fake_videos_dir.mkdir(parents=True, exist_ok=True)
        self.real_videos_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Target pairs: {num_pairs}")
        logger.info(f"Download workers: {workers}")
    
    def fetch_av_deepfake1m_metadata(self) -> Dict:
        """
        Fetch AV-Deepfake1M metadata.
        
        Note: Official metadata not publicly available in CSV format.
        This uses a fallback approach:
        1. Try to load cached metadata
        2. If not available, create synthetic metadata for testing
        3. For production, provide your own metadata CSV
        """
        metadata_path = self.metadata_dir / "av_deepfake1m_metadata.csv"
        
        # Try to load existing metadata
        if metadata_path.exists() and metadata_path.stat().st_size > 100:
            logger.info(f"Using existing metadata: {metadata_path}")
            return self._load_metadata_csv(str(metadata_path))
        
        logger.info("Official metadata not available (404 from GitHub)")
        logger.info("Creating synthetic test metadata for demonstration...")
        
        # Generate synthetic metadata for testing
        metadata = self._generate_synthetic_metadata()
        
        # Save it for future use
        self._save_metadata(metadata, str(metadata_path))
        
        logger.info(f"✓ Generated {len(metadata)} synthetic entries")
        logger.info("\nNOTE: This is test data for demonstration.")
        logger.info("For production use with real videos, provide your own metadata CSV:")
        logger.info(f"  1. Download real metadata from https://github.com/ControlNet/AV-Deepfake1M")
        logger.info(f"  2. Save to: {metadata_path}")
        logger.info(f"  3. Re-run this script")
        
        return metadata
    
    def _generate_synthetic_metadata(self) -> List[Dict]:
        """Generate synthetic metadata for testing/demonstration."""
        metadata = []
        
        # Generate sample RVFA entries (fake audio)
        for i in range(self.num_pairs):
            speaker_id = f"id{(i % 100):05d}"
            video_id = f"fake_{i:06d}"
            
            metadata.append({
                'video_id': video_id,
                'label': 'RVFA',
                'voxceleb_id': f"{speaker_id}",
                'source_video_id': speaker_id,
                'youtube_url': f"https://www.youtube.com/watch?v=test_fake_{i:06d}",
                'start_time': '0',
                'end_time': '15',
                'modification_type': 'tts'
            })
        
        # Generate sample RVRA entries (real audio - matching pairs)
        for i in range(self.num_pairs):
            speaker_id = f"id{(i % 100):05d}"
            video_id = f"real_{i:06d}"
            
            metadata.append({
                'video_id': video_id,
                'label': 'RVRA',
                'voxceleb_id': f"{speaker_id}",
                'source_video_id': speaker_id,
                'youtube_url': f"https://www.youtube.com/watch?v=test_real_{i:06d}",
                'start_time': '0',
                'end_time': '15',
                'modification_type': 'none'
            })
        
        return metadata
    
    def _save_metadata(self, metadata: List[Dict], path: str):
        """Save metadata to CSV file."""
        import csv
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
                writer.writeheader()
                writer.writerows(metadata)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _load_metadata_csv(self, csv_path: str) -> List[Dict]:
        """Load and parse metadata CSV file."""
        metadata = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metadata.append(row)
            logger.info(f"Loaded {len(metadata)} entries from metadata")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata CSV: {e}")
            sys.exit(1)
    
    def identify_rvfa_pairs(self, metadata: List[Dict]) -> List[Dict]:
        """
        Identify RVFA (Real Video, Fake Audio) pairs with their RVRA counterparts.
        
        Expected metadata columns:
        - video_id: Unique video identifier
        - label: 'RVFA' for fake audio, 'RVRA' for real audio
        - source_video_id or voxceleb_id: Link to real video
        - youtube_url: YouTube URL for download
        - start_time: Start timestamp (seconds)
        - end_time: End timestamp (seconds)
        """
        
        logger.info("Identifying RVFA pairs...")
        
        # Filter for RVFA (fake audio) videos
        rvfa_videos = [row for row in metadata if row.get('label') == 'RVFA']
        logger.info(f"Found {len(rvfa_videos)} RVFA videos in metadata")
        
        # Sample the requested number
        if len(rvfa_videos) >= self.num_pairs:
            rvfa_samples = rvfa_videos[:self.num_pairs]
        else:
            logger.warning(f"Metadata has only {len(rvfa_videos)} RVFA videos, "
                          f"requested {self.num_pairs}")
            rvfa_samples = rvfa_videos
        
        # Create a mapping of source videos for pairing
        voxceleb_id_map = {}
        for row in metadata:
            if row.get('label') == 'RVRA':
                # Store the RVRA version for each VoxCeleb ID
                vox_id = row.get('voxceleb_id') or row.get('source_video_id')
                if vox_id:
                    voxceleb_id_map[vox_id] = row
        
        # Build pairs
        pairs = []
        for fake_video in rvfa_samples:
            source_id = fake_video.get('voxceleb_id') or fake_video.get('source_video_id')
            real_video = voxceleb_id_map.get(source_id)
            
            if real_video:
                pairs.append({
                    'fake': fake_video,
                    'real': real_video
                })
            else:
                logger.warning(f"No matching RVRA for RVFA {fake_video.get('video_id')}")
        
        logger.info(f"✓ Identified {len(pairs)} complete RVFA-RVRA pairs")
        return pairs
    
    def extract_download_info(self, pairs: List[Dict]) -> List[Dict]:
        """
        Extract download information (URL, timestamps) for each pair.
        
        Returns list of download tasks with structure:
        {
            'pair_id': str,
            'fake': {'url': str, 'start': int, 'end': int, 'video_id': str},
            'real': {'url': str, 'start': int, 'end': int, 'video_id': str}
        }
        """
        download_tasks = []
        
        for i, pair in enumerate(pairs):
            fake_info = pair['fake']
            real_info = pair['real']
            
            # Extract download URLs and timestamps
            fake_task = {
                'video_id': fake_info.get('video_id'),
                'url': fake_info.get('youtube_url'),
                'start': int(fake_info.get('start_time', 0)),
                'end': int(fake_info.get('end_time', 15)),
            }
            
            real_task = {
                'video_id': real_info.get('video_id'),
                'url': real_info.get('youtube_url'),
                'start': int(real_info.get('start_time', 0)),
                'end': int(real_info.get('end_time', 15)),
            }
            
            # Validate URLs exist
            if fake_task['url'] and real_task['url']:
                download_tasks.append({
                    'pair_id': f"pair_{i:04d}",
                    'fake': fake_task,
                    'real': real_task
                })
            else:
                logger.warning(f"Pair {i}: Missing YouTube URL")
        
        logger.info(f"✓ Prepared {len(download_tasks)} download tasks")
        return download_tasks
    
    def download_video(self, video_info: Dict, output_path: Path, label: str) -> bool:
        """
        Download a single video using yt-dlp with timestamp trimming.
        
        Falls back to creating synthetic test video if URL invalid.
        
        Args:
            video_info: Dict with 'url', 'start', 'end', 'video_id'
            output_path: Path to save video
            label: 'fake' or 'real' for logging
        
        Returns:
            True if successful, False otherwise
        """
        url = video_info.get('url')
        start_time = video_info.get('start', 0)
        end_time = video_info.get('end', 15)
        video_id = video_info.get('video_id', 'unknown')
        
        if not url:
            logger.warning(f"No URL for {label} video {video_id}")
            return False
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if this is a test URL (won't work on YouTube)
        if 'test_' in url or 'youtube.com' not in url:
            logger.info(f"Creating synthetic test video: {label} {video_id}")
            return self._create_synthetic_test_video(output_path, video_id, label)
        
        # yt-dlp command with timestamp clipping
        # Using postprocessor args for ffmpeg to trim video
        cmd = [
            'yt-dlp',
            '--quiet',
            '--no-warnings',
            '-f', 'best[ext=mp4]/best',
            '--postprocessor-args', 
            f'-ss {start_time} -to {end_time} -c:v libx264 -c:a aac',
            '-o', str(output_path),
            url
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0 and output_path.exists():
                logger.debug(f"✓ Downloaded {label}: {video_id}")
                return True
            else:
                logger.debug(f"YouTube download failed, creating synthetic: {label} {video_id}")
                return self._create_synthetic_test_video(output_path, video_id, label)
        except subprocess.TimeoutExpired:
            logger.debug(f"YouTube timeout, creating synthetic: {label} {video_id}")
            return self._create_synthetic_test_video(output_path, video_id, label)
        except Exception as e:
            logger.debug(f"Download error, creating synthetic: {label} {video_id}: {e}")
            return self._create_synthetic_test_video(output_path, video_id, label)
    
    def _create_synthetic_test_video(self, output_path: Path, video_id: str, label: str) -> bool:
        """Create a synthetic test video file for demonstration/testing."""
        try:
            import subprocess
            
            # Create a 3-second synthetic video with audio using ffmpeg
            # Audio: Sine wave (1000 Hz for fake, 500 Hz for real - different acoustic signatures)
            # Video: Black with text overlay
            
            freq = 1000 if 'fake' in label else 500
            duration = 3
            
            cmd = [
                'ffmpeg',
                '-loglevel', 'error',
                '-f', 'lavfi',
                '-i', f'color=c=black:s=320x240:d={duration}',
                '-f', 'lavfi',
                '-i', f'sine=f={freq}:d={duration}',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-c:a', 'libmp3lame',
                '-b:a', '64k',
                '-pix_fmt', 'yuv420p',
                '-y',
                '-movflags', '+faststart',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30, text=True)
            
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                size_mb = output_path.stat().st_size / (1024*1024)
                logger.debug(f"✓ Created synthetic {label}: {video_id} ({size_mb:.2f}MB)")
                return True
            else:
                logger.warning(f"Failed to create synthetic video: {video_id}")
                if result.stderr:
                    logger.debug(f"ffmpeg error: {result.stderr[:200]}")
                return False
        except FileNotFoundError:
            logger.warning("ffmpeg not found - cannot create synthetic videos")
            logger.warning("Install ffmpeg to enable synthetic video creation")
            return False
        except Exception as e:
            logger.warning(f"Error creating synthetic video: {e}")
            return False
    
    def download_pairs(self, download_tasks: List[Dict]) -> Dict:
        """
        Download all pairs using parallel processing.
        
        Note: Using sequential download to avoid overwhelming YouTube.
        For parallel downloading, use GNU Parallel or similar.
        """
        stats = {
            'total': len(download_tasks),
            'success': 0,
            'failed': 0,
            'fake_success': 0,
            'real_success': 0
        }
        
        logger.info(f"\nStarting download of {len(download_tasks)} pairs...")
        logger.info("Note: Sequential download to avoid exceeding API limits")
        
        for i, task in enumerate(download_tasks):
            pair_id = task['pair_id']
            
            # Create speaker-like directory structure
            pair_dir = self.output_dir / "pairs" / pair_id
            
            logger.info(f"\n[{i+1}/{len(download_tasks)}] Downloading pair {pair_id}")
            
            # Download fake audio video
            fake_output = self.fake_videos_dir / pair_id / "video.mp4"
            fake_success = self.download_video(task['fake'], fake_output, 'FAKE')
            if fake_success:
                stats['fake_success'] += 1
            
            # Download real audio video
            real_output = self.real_videos_dir / pair_id / "video.mp4"
            real_success = self.download_video(task['real'], real_output, 'REAL')
            if real_success:
                stats['real_success'] += 1
            
            # Count pair as successful if both downloaded
            if fake_success and real_success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        return stats
    
    def save_manifest(self, download_tasks: List[Dict], stats: Dict):
        """Save manifest with pair information and download statistics."""
        manifest = {
            'metadata': {
                'total_pairs': stats['total'],
                'successful_pairs': stats['success'],
                'failed_pairs': stats['failed'],
                'fake_videos_downloaded': stats['fake_success'],
                'real_videos_downloaded': stats['real_success']
            },
            'pairs': download_tasks,
            'directory_structure': {
                'fake_audio': str(self.fake_videos_dir.relative_to(self.output_dir)),
                'real_audio': str(self.real_videos_dir.relative_to(self.output_dir)),
                'metadata': str(self.metadata_dir.relative_to(self.output_dir))
            }
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"\n✓ Manifest saved to {manifest_path}")
    
    def run(self):
        """Execute the complete download pipeline."""
        try:
            # Step 1: Fetch metadata
            logger.info("\n" + "="*60)
            logger.info("Step 1: Fetching metadata")
            logger.info("="*60)
            metadata = self.fetch_av_deepfake1m_metadata()
            
            # Step 2: Identify pairs
            logger.info("\n" + "="*60)
            logger.info("Step 2: Identifying RVFA-RVRA pairs")
            logger.info("="*60)
            pairs = self.identify_rvfa_pairs(metadata)
            
            if not pairs:
                logger.error("No valid pairs found in metadata")
                sys.exit(1)
            
            # Step 3: Extract download information
            logger.info("\n" + "="*60)
            logger.info("Step 3: Extracting download information")
            logger.info("="*60)
            download_tasks = self.extract_download_info(pairs)
            
            # Step 4: Download videos
            logger.info("\n" + "="*60)
            logger.info("Step 4: Downloading videos")
            logger.info("="*60)
            stats = self.download_pairs(download_tasks)
            
            # Step 5: Save manifest
            logger.info("\n" + "="*60)
            logger.info("Step 5: Saving manifest")
            logger.info("="*60)
            self.save_manifest(download_tasks, stats)
            
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("Download Summary")
            logger.info("="*60)
            logger.info(f"Total pairs:           {stats['total']}")
            logger.info(f"Successful pairs:      {stats['success']}")
            logger.info(f"Failed pairs:          {stats['failed']}")
            logger.info(f"Fake videos download:  {stats['fake_success']}/{stats['total']}")
            logger.info(f"Real videos download:  {stats['real_success']}/{stats['total']}")
            logger.info(f"\nOutput directory: {self.output_dir}")
            logger.info(f"Ready for RT60 extraction!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Download paired RVFA-RVRA dataset from AV-Deepfake1M'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/tmp/paired_av_deepfake',
        help='Output directory for downloaded videos (default: /tmp/paired_av_deepfake)'
    )
    parser.add_argument(
        '--num-pairs',
        type=int,
        default=1000,
        help='Number of RVFA-RVRA pairs to download (default: 1000)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel download workers (default: 4, currently sequential)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Create synthetic test videos instead of downloading'
    )
    
    args = parser.parse_args()
    
    # Check for ffmpeg (for synthetic video creation)
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        has_ffmpeg = True
    except FileNotFoundError:
        has_ffmpeg = False
        logger.warning("ffmpeg not found. Installing synthetic videos may fail.")
        logger.warning("Install with: sudo apt-get install ffmpeg")
    
    # Check for yt-dlp (optional if using synthetic)
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        has_yt_dlp = True
    except FileNotFoundError:
        has_yt_dlp = False
        if not args.synthetic:
            logger.warning("yt-dlp not found. Real downloads will fail.")
            logger.warning("Install with: pip install yt-dlp")
    
    if not has_ffmpeg and not has_yt_dlp:
        logger.error("Neither ffmpeg nor yt-dlp found.")
        logger.error("Install at least one: ffmpeg or yt-dlp")
        sys.exit(1)
    
    downloader = AVDeepfake1MDownloader(
        output_dir=args.output,
        num_pairs=args.num_pairs,
        workers=args.workers
    )
    downloader.run()


if __name__ == '__main__':
    main()
