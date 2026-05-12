"""Compatibility check for AVDeepfake1M with the visual RIR pipeline.

This script samples AVDeepfake1M clips from a local dataset tree, runs the
existing visual RIR pipeline, and summarizes whether the dataset is suitable
for extracting RIR/RT60/DRR values.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from visual_rir_estimator import (
    check_frame_has_room_context,
    compute_drr_from_rir,
    extract_frame,
    get_rir_rt60_from_video,
)

LOGGER = logging.getLogger("check_avdeepfake1m_compatibility")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
KNOWN_CLIP_TYPES = {
    "real.mp4": "real",
    "real_video_fake_audio.mp4": "real_video_fake_audio",
    "fake_video_fake_audio.mp4": "fake_video_fake_audio",
}


@dataclass
class SampleResult:
    """Per-clip compatibility result."""

    video_path: str
    clip_type: str
    scene_frame_path: str
    scene_context_pass: bool
    used_frame_path: str
    used_scene_frame: bool
    sample_rate: int | None
    rir_length: int | None
    rt60_seconds: float | None
    drr_db: float | None
    status: str
    error: str | None = None


def _configure_logging(verbose: bool = False) -> None:
    """Configure module logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def discover_videos(dataset_root: str, extensions: Iterable[str] = VIDEO_EXTENSIONS) -> list[Path]:
    """Find candidate video clips beneath a dataset root.

    Args:
        dataset_root: Root folder containing AVDeepfake1M clips.
        extensions: Allowed video file extensions.

    Returns:
        Sorted list of video paths.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    extension_set = {extension.lower() for extension in extensions}
    videos = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extension_set
    ]
    return sorted(videos)


def classify_clip(video_path: Path) -> str:
    """Classify a clip from its filename."""
    return KNOWN_CLIP_TYPES.get(video_path.name, "unknown")


def sample_videos(video_paths: list[Path], sample_count: int, seed: int) -> list[Path]:
    """Select up to sample_count videos deterministically."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not video_paths:
        return []

    if len(video_paths) <= sample_count:
        return video_paths

    rng = random.Random(seed)
    return sorted(rng.sample(video_paths, sample_count))


def run_compatibility_check(dataset_root: str, output_dir: str, sample_count: int = 100, seed: int = 7) -> list[SampleResult]:
    """Run the AVDeepfake1M compatibility test on a sample of clips.

    Args:
        dataset_root: Root directory containing AVDeepfake1M video clips.
        output_dir: Directory used to store per-sample outputs and summaries.
        sample_count: Number of clips to process.
        seed: Random seed used for sampling.

    Returns:
        Per-clip results.
    """
    root = Path(dataset_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = discover_videos(dataset_root)
    sampled_videos = sample_videos(video_paths, sample_count=sample_count, seed=seed)

    LOGGER.info("Discovered %d video clips under %s", len(video_paths), root)
    LOGGER.info("Sampling %d clips for compatibility testing", len(sampled_videos))
    if not sampled_videos:
        LOGGER.warning("No AVDeepfake1M video clips were found under %s", root)

    results: list[SampleResult] = []
    for index, video_path in enumerate(sampled_videos, start=1):
        clip_type = classify_clip(video_path)
        sample_dir = out_dir / f"sample_{index:03d}_{video_path.stem}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("[%d/%d] Processing %s", index, len(sampled_videos), video_path)
        try:
            scene_frame_path = extract_frame(str(video_path), strategy="scene")
            scene_context_pass = check_frame_has_room_context(scene_frame_path)
            rir, sample_rate, rt60_seconds, used_frame_path = get_rir_rt60_from_video(str(video_path), str(sample_dir))
            drr_db = compute_drr_from_rir(rir, sample_rate)

            result = SampleResult(
                video_path=str(video_path),
                clip_type=clip_type,
                scene_frame_path=scene_frame_path,
                scene_context_pass=scene_context_pass,
                used_frame_path=used_frame_path,
                used_scene_frame=Path(used_frame_path).name.endswith("_scene.png"),
                sample_rate=int(sample_rate),
                rir_length=int(rir.size),
                rt60_seconds=float(rt60_seconds),
                drr_db=float(drr_db),
                status="ok",
            )
        except Exception as error:  # pragma: no cover - compatibility test should report failures rather than crash
            LOGGER.exception("Failed to process %s", video_path)
            result = SampleResult(
                video_path=str(video_path),
                clip_type=clip_type,
                scene_frame_path="",
                scene_context_pass=False,
                used_frame_path="",
                used_scene_frame=False,
                sample_rate=None,
                rir_length=None,
                rt60_seconds=None,
                drr_db=None,
                status="error",
                error=str(error),
            )

        results.append(result)

    _write_outputs(results, out_dir)
    _log_summary(results)
    return results


def _write_outputs(results: list[SampleResult], out_dir: Path) -> None:
    """Write summary JSON and CSV outputs."""
    summary_path = out_dir / "avdeepfake1m_compatibility_summary.json"
    records_path = out_dir / "avdeepfake1m_compatibility_records.csv"

    summary = _build_summary(results)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()) if results else [])
        if results:
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))

    LOGGER.info("Saved summary to %s", summary_path)
    LOGGER.info("Saved records to %s", records_path)


def _build_summary(results: list[SampleResult]) -> dict[str, object]:
    """Build aggregate statistics for the compatibility run."""
    ok_results = [result for result in results if result.status == "ok"]
    scene_pass_count = sum(1 for result in ok_results if result.scene_context_pass)
    fallback_count = sum(1 for result in ok_results if not result.used_scene_frame)

    rt60_values = [result.rt60_seconds for result in ok_results if result.rt60_seconds is not None]
    drr_values = [result.drr_db for result in ok_results if result.drr_db is not None]
    sample_rates = [result.sample_rate for result in ok_results if result.sample_rate is not None]
    label_counts: dict[str, int] = {}
    for result in results:
        label_counts[result.clip_type] = label_counts.get(result.clip_type, 0) + 1

    return {
        "sample_count": len(results),
        "ok_count": len(ok_results),
        "error_count": len(results) - len(ok_results),
        "scene_context_pass_count": scene_pass_count,
        "scene_context_pass_rate": (scene_pass_count / len(ok_results)) if ok_results else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate": (fallback_count / len(ok_results)) if ok_results else 0.0,
        "sample_rate_values": sample_rates,
        "clip_type_counts": label_counts,
        "rt60_stats": _stats(rt60_values),
        "drr_stats": _stats(drr_values),
        "errors": [result.error for result in results if result.error],
    }


def _stats(values: list[float]) -> dict[str, float | None]:
    """Compute simple descriptive statistics for a list of values."""
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}

    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _log_summary(results: list[SampleResult]) -> None:
    """Log a compact summary to the console."""
    summary = _build_summary(results)
    LOGGER.info(
        "Compatibility summary: %d clips, %d ok, %d errors, scene-pass rate %.2f%%, fallback rate %.2f%%",
        summary["sample_count"],
        summary["ok_count"],
        summary["error_count"],
        100.0 * float(summary["scene_context_pass_rate"]),
        100.0 * float(summary["fallback_rate"]),
    )
    LOGGER.info("Clip type counts: %s", summary["clip_type_counts"])
    LOGGER.info("RT60 stats: %s", summary["rt60_stats"])
    LOGGER.info("DRR stats: %s", summary["drr_stats"])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compatibility check for AVDeepfake1M with the visual RIR pipeline")
    parser.add_argument("--dataset_root", required=True, help="Root directory of the AVDeepfake1M video clips")
    parser.add_argument("--output_dir", default="./avdeepfake1m_check", help="Directory for output artifacts")
    parser.add_argument("--sample_count", type=int, default=100, help="Number of clips to sample")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    _configure_logging(args.verbose)
    run_compatibility_check(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
