"""Standalone DRR estimator for a previously generated RIR waveform.

This helper is intended for downstream use after the visual pipeline has
produced a numpy array or WAV file containing the RIR.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from rt60_from_rir import load_rir
from visual_rir_estimator import compute_drr_from_rir

LOGGER = logging.getLogger("drr_from_rir")


def _configure_logging(verbose: bool = False) -> None:
    """Configure module logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def estimate_drr_from_rir_file(rir_path: str, sample_rate: int | None = None, direct_window_ms: float = 2.5) -> float:
    """Estimate DRR from a saved RIR file.

    Args:
        rir_path: Path to a WAV or NPY file.
        sample_rate: Optional sample rate for NPY input.
        direct_window_ms: Width of the direct-path window in milliseconds.

    Returns:
        Estimated DRR in decibels.
    """
    rir, sr = load_rir(rir_path, sample_rate=sample_rate)
    return compute_drr_from_rir(rir, sr, direct_window_ms=direct_window_ms)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Estimate DRR from a saved RIR waveform")
    parser.add_argument("--rir", required=True, help="Path to RIR .wav or .npy file")
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=None,
        help="Sample rate for .npy input. Not needed for .wav input.",
    )
    parser.add_argument(
        "--direct_window_ms",
        type=float,
        default=2.5,
        help="Width of the direct-path window around the main peak in milliseconds.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save the DRR result as JSON.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    _configure_logging(args.verbose)

    rir, sr = load_rir(args.rir, sample_rate=args.sample_rate)
    drr_db = compute_drr_from_rir(rir, sr, direct_window_ms=args.direct_window_ms)

    payload = {
        "rir_path": str(args.rir),
        "sample_rate": int(sr),
        "direct_window_ms": float(args.direct_window_ms),
        "drr_db": float(drr_db),
    }

    LOGGER.info("DRR estimate: %.4f dB", drr_db)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Saved DRR JSON to %s", output_path)


if __name__ == "__main__":
    main()
