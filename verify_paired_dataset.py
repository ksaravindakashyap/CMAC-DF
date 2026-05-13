#!/usr/bin/env python3
"""
Verify and prepare paired dataset for RT60 extraction.

This script:
1. Validates downloaded videos (checks for corruption)
2. Creates video pair lists for RT60 extraction
3. Generates metadata mapping
4. Identifies failed downloads for retry

Usage:
    python verify_paired_dataset.py \
        --dataset-dir /path/to/paired_dataset \
        --output rt60_paired_videos.txt
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PairedDatasetVerifier:
    """Verify and prepare paired dataset for RT60 extraction."""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.fake_dir = self.dataset_dir / "fake_audio" / "mp4"
        self.real_dir = self.dataset_dir / "real_audio" / "mp4"
        self.manifest_path = self.dataset_dir / "manifest.json"
        
        if not self.dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            sys.exit(1)
    
    def check_tools(self):
        """Verify required tools are available."""
        try:
            subprocess.run(['ffprobe', '-version'], 
                          capture_output=True, check=True)
        except FileNotFoundError:
            logger.error("ffprobe not found. Install ffmpeg:")
            logger.error("  sudo apt-get install ffmpeg")
            sys.exit(1)
    
    def verify_video(self, video_path: Path) -> Tuple[bool, str]:
        """
        Verify video file is valid and readable.
        
        Returns:
            (is_valid, error_message)
        """
        if not video_path.exists():
            return False, "File not found"
        
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', 
                 '-show_entries', 'stream=codec_type',
                 '-of', 'default=noprint_wrappers=1:nokey=1:nk=1',
                 str(video_path)],
                capture_output=True,
                timeout=10,
                text=True
            )
            
            if result.returncode == 0:
                # Check for both video and audio streams
                output = result.stdout.strip()
                has_video = 'video' in output
                has_audio = 'audio' in output
                
                if has_video and has_audio:
                    return True, "OK"
                else:
                    missing = []
                    if not has_video:
                        missing.append('video')
                    if not has_audio:
                        missing.append('audio')
                    return False, f"Missing: {', '.join(missing)}"
            else:
                return False, "ffprobe error"
        except subprocess.TimeoutExpired:
            return False, "Timeout during verification"
        except Exception as e:
            return False, str(e)
    
    def scan_directory(self, video_dir: Path, label: str) -> dict:
        """Scan directory and verify all videos."""
        logger.info(f"\nScanning {label} videos...")
        
        stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'missing_pair': 0,
            'videos': {}
        }
        
        pair_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
        
        for pair_dir in pair_dirs:
            video_path = pair_dir / "video.mp4"
            pair_id = pair_dir.name
            stats['total'] += 1
            
            is_valid, error = self.verify_video(video_path)
            
            if is_valid:
                stats['valid'] += 1
                stats['videos'][pair_id] = {
                    'path': str(video_path.relative_to(self.dataset_dir)),
                    'status': 'valid',
                    'size_mb': video_path.stat().st_size / (1024*1024)
                }
                logger.debug(f"✓ {pair_id}: {stats['videos'][pair_id]['size_mb']:.1f}MB")
            else:
                stats['invalid'] += 1
                stats['videos'][pair_id] = {
                    'status': 'invalid',
                    'error': error
                }
                logger.warning(f"✗ {pair_id}: {error}")
        
        logger.info(f"{label} videos: {stats['valid']}/{stats['total']} valid")
        return stats
    
    def match_pairs(self, fake_stats: dict, real_stats: dict) -> dict:
        """
        Verify that fake and real videos are properly paired.
        
        Returns:
            {
                'complete_pairs': count,
                'fake_only': [pair_ids],
                'real_only': [pair_ids],
                'mismatched': [pair_ids],
                'pairs': [{'id': str, 'fake': Path, 'real': Path}]
            }
        """
        logger.info("\nMatching fake-real pairs...")
        
        result = {
            'complete_pairs': 0,
            'fake_only': [],
            'real_only': [],
            'mismatched': [],
            'pairs': []
        }
        
        fake_ids = set(fake_stats['videos'].keys())
        real_ids = set(real_stats['videos'].keys())
        
        # Find complete pairs
        for pair_id in sorted(fake_ids & real_ids):
            fake_video = fake_stats['videos'][pair_id]
            real_video = real_stats['videos'][pair_id]
            
            # Check both are valid
            if (fake_video.get('status') == 'valid' and 
                real_video.get('status') == 'valid'):
                result['complete_pairs'] += 1
                result['pairs'].append({
                    'id': pair_id,
                    'fake': fake_video['path'],
                    'real': real_video['path'],
                    'fake_size_mb': fake_video['size_mb'],
                    'real_size_mb': real_video['size_mb']
                })
            else:
                result['mismatched'].append(pair_id)
        
        # Unpaired videos
        for pair_id in fake_ids - real_ids:
            if fake_stats['videos'][pair_id].get('status') == 'valid':
                result['fake_only'].append(pair_id)
        
        for pair_id in real_ids - fake_ids:
            if real_stats['videos'][pair_id].get('status') == 'valid':
                result['real_only'].append(pair_id)
        
        logger.info(f"✓ Complete pairs: {result['complete_pairs']}")
        if result['fake_only']:
            logger.warning(f"  Fake-only: {len(result['fake_only'])} orphaned")
        if result['real_only']:
            logger.warning(f"  Real-only: {len(result['real_only'])} orphaned")
        
        return result
    
    def create_rt60_list(self, pair_info: dict, output_file: str):
        """
        Create paired video list for RT60 extraction.
        
        Format: fake_video\treal_video (tab-separated)
        """
        logger.info(f"\nCreating RT60 pair list: {output_file}")
        
        with open(output_file, 'w') as f:
            for pair in pair_info['pairs']:
                fake_path = pair['fake']
                real_path = pair['real']
                f.write(f"{fake_path}\t{real_path}\n")
        
        logger.info(f"✓ Wrote {len(pair_info['pairs'])} pairs to {output_file}")
    
    def create_individual_lists(self, pair_info: dict, output_dir: str = None):
        """
        Create separate lists for fake and real videos.
        
        Useful for running RT60 extraction on each separately.
        """
        if output_dir is None:
            output_dir = self.dataset_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fake_list = output_dir / "rt60_fake_videos.txt"
        real_list = output_dir / "rt60_real_videos.txt"
        
        logger.info(f"\nCreating individual video lists...")
        
        with open(fake_list, 'w') as f:
            for pair in pair_info['pairs']:
                f.write(f"{pair['fake']}\n")
        
        with open(real_list, 'w') as f:
            for pair in pair_info['pairs']:
                f.write(f"{pair['real']}\n")
        
        logger.info(f"✓ Fake videos: {fake_list}")
        logger.info(f"✓ Real videos: {real_list}")
        
        return fake_list, real_list
    
    def generate_report(self, 
                       fake_stats: dict, 
                       real_stats: dict,
                       pair_info: dict,
                       output_file: str = None):
        """Generate comprehensive verification report."""
        
        if output_file is None:
            output_file = self.dataset_dir / "verification_report.json"
        
        report = {
            'timestamp': str(Path(__file__).parent),
            'summary': {
                'fake_videos_valid': fake_stats['valid'],
                'fake_videos_total': fake_stats['total'],
                'fake_videos_invalid': fake_stats['invalid'],
                'real_videos_valid': real_stats['valid'],
                'real_videos_total': real_stats['total'],
                'real_videos_invalid': real_stats['invalid'],
                'complete_pairs': pair_info['complete_pairs'],
                'orphaned_fake': len(pair_info['fake_only']),
                'orphaned_real': len(pair_info['real_only']),
                'mismatched': len(pair_info['mismatched']),
            },
            'failed_pairs': {
                'fake_only': pair_info['fake_only'],
                'real_only': pair_info['real_only'],
                'mismatched': pair_info['mismatched']
            },
            'fake_videos': fake_stats['videos'],
            'real_videos': real_stats['videos'],
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Report saved: {output_file}")
        return report
    
    def print_summary(self, fake_stats: dict, real_stats: dict, pair_info: dict):
        """Print human-readable summary."""
        total_size = sum(p.get('fake_size_mb', 0) + p.get('real_size_mb', 0) 
                        for p in pair_info['pairs'])
        
        logger.info("\n" + "="*60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("="*60)
        logger.info(f"\nFake Audio Videos:")
        logger.info(f"  Valid:   {fake_stats['valid']}/{fake_stats['total']}")
        logger.info(f"  Invalid: {fake_stats['invalid']}")
        
        logger.info(f"\nReal Audio Videos:")
        logger.info(f"  Valid:   {real_stats['valid']}/{real_stats['total']}")
        logger.info(f"  Invalid: {real_stats['invalid']}")
        
        logger.info(f"\nPaired Videos:")
        logger.info(f"  Complete pairs:  {pair_info['complete_pairs']}")
        logger.info(f"  Fake orphaned:   {len(pair_info['fake_only'])}")
        logger.info(f"  Real orphaned:   {len(pair_info['real_only'])}")
        logger.info(f"  Mismatched:      {len(pair_info['mismatched'])}")
        
        logger.info(f"\nStorage Usage:")
        logger.info(f"  Total size: {total_size/1024:.1f} GB")
        if pair_info['complete_pairs'] > 0:
            logger.info(f"  Avg per pair: {total_size/pair_info['complete_pairs']:.1f} MB")
        else:
            logger.warning(f"  No complete pairs found - check videos are valid")
        
        logger.info(f"\n✓ Ready for RT60 extraction: {pair_info['complete_pairs']} pairs")
    
    def run(self, output_list: str = None):
        """Execute complete verification and preparation."""
        logger.info("="*60)
        logger.info("DATASET VERIFICATION & PREPARATION")
        logger.info("="*60)
        
        # Check tools
        self.check_tools()
        
        # Scan directories
        fake_stats = self.scan_directory(self.fake_dir, "FAKE")
        real_stats = self.scan_directory(self.real_dir, "REAL")
        
        # Match pairs
        pair_info = self.match_pairs(fake_stats, real_stats)
        
        # Create RT60 lists
        if output_list is None:
            output_list = self.dataset_dir / "rt60_paired_videos.txt"
        
        self.create_rt60_list(pair_info, str(output_list))
        self.create_individual_lists(pair_info)
        
        # Generate reports
        self.generate_report(fake_stats, real_stats, pair_info)
        
        # Print summary
        self.print_summary(fake_stats, real_stats, pair_info)
        
        return pair_info


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify and prepare paired dataset for RT60 extraction'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        required=True,
        help='Path to paired dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for RT60 paired video list'
    )
    
    args = parser.parse_args()
    
    verifier = PairedDatasetVerifier(args.dataset_dir)
    verifier.run(args.output)


if __name__ == '__main__':
    main()
