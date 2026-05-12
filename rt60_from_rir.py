"""Standalone RT60 estimator for a previously generated RIR waveform.

This helper is intended for downstream use after the visual pipeline has
produced a numpy array or WAV file containing the RIR.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from visual_rir_estimator import compute_rt60_from_rir

LOGGER = logging.getLogger("rt60_from_rir")


def _configure_logging(verbose: bool = False) -> None:
    """Configure module logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_rir(rir_path: str, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Load an RIR waveform from a WAV or NPY file.

    Args:
        rir_path: Path to a WAV or NPY file containing the RIR waveform.
        sample_rate: Required when loading from NPY.

    Returns:
        Tuple of (rir waveform, sample rate).
    """
    path = Path(rir_path)
    if not path.exists():
        raise FileNotFoundError(f"RIR file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".wav":
        rir, sr = sf.read(path, dtype="float32")
        if rir.ndim > 1:
            rir = rir[:, 0]
        return np.asarray(rir, dtype=np.float32).reshape(-1), int(sr)

    if suffix == ".npy":
        if sample_rate is None:
            raise ValueError("sample_rate is required when loading from .npy")
        rir = np.load(path).astype(np.float32).reshape(-1)
        return rir, int(sample_rate)

    raise ValueError("rir_path must point to a .wav or .npy file")


def estimate_rt60_from_rir_file(rir_path: str, sample_rate: int | None = None, method: str = "auto") -> float:
    """Estimate RT60 from a saved RIR file.

    Args:
        rir_path: Path to a WAV or NPY file.
        sample_rate: Optional sample rate for NPY input.
        method: RT60 fit method, one of "t20", "t30", or "auto".

    Returns:
        Estimated RT60 in seconds.
    """
    rir, sr = load_rir(rir_path, sample_rate=sample_rate)
    return compute_rt60_from_rir(rir, sr, method=method)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Estimate RT60 from a saved RIR waveform")
    parser.add_argument("--rir", required=True, help="Path to RIR .wav or .npy file")
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=None,
        help="Sample rate for .npy input. Not needed for .wav input.",
    )
    parser.add_argument(
        "--method",
        choices=("t20", "t30", "auto"),
        default="auto",
        help="RT60 estimation method.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save the RT60 result as JSON.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    _configure_logging(args.verbose)

    rir, sr = load_rir(args.rir, sample_rate=args.sample_rate)
    rt60_seconds = compute_rt60_from_rir(rir, sr, method=args.method)

    payload = {
        "rir_path": str(args.rir),
        "sample_rate": int(sr),
        "method": args.method,
        "rt60_seconds": float(rt60_seconds),
    }

    LOGGER.info("RT60 estimate: %.4f s", rt60_seconds)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Saved RT60 JSON to %s", output_path)


if __name__ == "__main__":
    main()