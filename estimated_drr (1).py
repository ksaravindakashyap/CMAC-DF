from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

EPS = 1e-10


# -----------------------------
# Audio loading
# -----------------------------
def load_mono(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    return x.astype(np.float64), sr


def energy(x: np.ndarray) -> float:
    return float(np.sum(np.square(x)))


# -----------------------------
# Lag alignment
# -----------------------------
def overlap_for_lag(raw: np.ndarray, direct: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return overlapping raw/direct regions for a proposed lag.

    lag > 0 means direct is shifted right relative to raw.
    lag < 0 means direct is shifted left relative to raw.
    """
    n = min(len(raw), len(direct))

    raw = raw[:n]
    direct = direct[:n]

    if lag > 0:
        return raw[lag:], direct[: n - lag]
    elif lag < 0:
        lag_abs = -lag
        return raw[: n - lag_abs], direct[lag_abs:]
    else:
        return raw, direct


def estimate_best_lag(raw: np.ndarray, direct: np.ndarray, sr: int, max_lag_ms: float = 30.0) -> int:
    """
    Brute-force normalized cross-correlation over a small lag range.
    This is simple and robust for short clips.
    """
    max_lag = int(sr * max_lag_ms / 1000.0)
    best_lag = 0
    best_score = -np.inf

    # Remove DC
    raw0 = raw - np.mean(raw)
    direct0 = direct - np.mean(direct)

    for lag in range(-max_lag, max_lag + 1):
        r, d = overlap_for_lag(raw0, direct0, lag)

        if len(r) < sr // 10:
            continue

        denom = np.sqrt(np.sum(r * r) * np.sum(d * d)) + EPS
        score = float(np.sum(r * d) / denom)

        if score > best_score:
            best_score = score
            best_lag = lag

    return best_lag


# -----------------------------
# Gain alignment
# -----------------------------
def estimate_gain(raw: np.ndarray, direct: np.ndarray, min_gain: float = 0.05, max_gain: float = 20.0, gain_mode: str = "least_squares",) -> float:
    """
    gain_mode:
      - least_squares: fit gain so gain * direct best matches raw
      - none: force gain = 1.0
    """
    if gain_mode == "none":
        return 1.0

    if gain_mode != "least_squares":
        raise ValueError(f"Unknown gain_mode: {gain_mode}")

    gain = float(np.sum(raw * direct) / (np.sum(direct * direct) + EPS))
    gain = max(min_gain, min(max_gain, gain))
    return gain

# -----------------------------
# Speech-active mask
# -----------------------------
def frame_signal(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(x) < frame_len:
        return np.empty((0, frame_len), dtype=x.dtype)

    n_frames = 1 + (len(x) - frame_len) // hop_len
    frames = np.zeros((n_frames, frame_len), dtype=x.dtype)

    for i in range(n_frames):
        start = i * hop_len
        frames[i] = x[start : start + frame_len]

    return frames


def speech_active_mask(
    raw: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    db_below_peak: float = 40.0,
    min_active_ratio: float = 0.05,
) -> np.ndarray:
    """
    Simple energy-based speech activity mask.
    Marks samples active if their frame energy is within db_below_peak of the max frame energy.
    """
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))

    frames = frame_signal(raw, frame_len, hop_len)

    if len(frames) == 0:
        return np.ones(len(raw), dtype=bool)

    frame_energy = np.mean(frames ** 2, axis=1) + EPS
    frame_db = 10.0 * np.log10(frame_energy)
    threshold = np.max(frame_db) - db_below_peak

    active_frames = frame_db >= threshold

    # If the threshold is too strict, fall back to keeping the loudest frames.
    if np.mean(active_frames) < min_active_ratio:
        cutoff = np.quantile(frame_energy, 1.0 - min_active_ratio)
        active_frames = frame_energy >= cutoff

    mask = np.zeros(len(raw), dtype=bool)

    for i, active in enumerate(active_frames):
        if active:
            start = i * hop_len
            end = min(start + frame_len, len(raw))
            mask[start:end] = True

    if not np.any(mask):
        mask[:] = True

    return mask


# -----------------------------
# DRR calculation
# -----------------------------
def compute_drr(direct: np.ndarray, reverb: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, float]:
    if mask is not None:
        direct = direct[mask]
        reverb = reverb[mask]

    e_direct = energy(direct)
    e_reverb = energy(reverb)
    drr_db = 10.0 * math.log10((e_direct + EPS) / (e_reverb + EPS))

    return {
        "direct_energy": e_direct,
        "reverb_energy": e_reverb,
        "drr_db": drr_db,
    }


def windowed_drr_stats(
    direct: np.ndarray,
    reverb: np.ndarray,
    active_mask: np.ndarray,
    sr: int,
    win_sec: float = 1.0,
    hop_sec: float = 0.5,
    min_active_fraction: float = 0.30,
) -> Dict[str, float]:
    win = max(1, int(win_sec * sr))
    hop = max(1, int(hop_sec * sr))

    vals = []

    if len(direct) < win:
        stats = compute_drr(direct, reverb, active_mask)
        vals = [stats["drr_db"]]
    else:
        for start in range(0, len(direct) - win + 1, hop):
            end = start + win
            m = active_mask[start:end]

            if np.mean(m) < min_active_fraction:
                continue

            stats = compute_drr(direct[start:end], reverb[start:end], m)
            vals.append(stats["drr_db"])

    if len(vals) == 0:
        return {
            "window_count": 0,
            "drr_win_mean_db": "",
            "drr_win_median_db": "",
            "drr_win_std_db": "",
            "drr_win_min_db": "",
            "drr_win_max_db": "",
        }

    vals = np.array(vals, dtype=np.float64)

    return {
        "window_count": int(len(vals)),
        "drr_win_mean_db": float(np.mean(vals)),
        "drr_win_median_db": float(np.median(vals)),
        "drr_win_std_db": float(np.std(vals)),
        "drr_win_min_db": float(np.min(vals)),
        "drr_win_max_db": float(np.max(vals)),
    }


def process_pair(raw_path: Path, direct_path: Path, raw_root: Path, args) -> Dict:
    raw, sr_raw = load_mono(raw_path)
    direct, sr_direct = load_mono(direct_path)

    if sr_raw != sr_direct:
        raise ValueError(f"sample-rate mismatch: raw={sr_raw}, direct={sr_direct}")

    best_lag = estimate_best_lag(raw, direct, sr_raw, max_lag_ms=args.max_lag_ms)
    raw_aligned, direct_aligned = overlap_for_lag(raw, direct, best_lag)

    gain = estimate_gain(raw_aligned, direct_aligned, min_gain=args.min_gain, max_gain=args.max_gain, gain_mode=args.gain_mode,)
    direct_scaled = gain * direct_aligned

    reverb = raw_aligned - direct_scaled

    active_mask = speech_active_mask(
        raw_aligned,
        sr_raw,
        db_below_peak=args.vad_db_below_peak,
        min_active_ratio=args.min_active_ratio,
    )

    global_all = compute_drr(direct_scaled, reverb, mask=None)
    global_active = compute_drr(direct_scaled, reverb, mask=active_mask)

    win_stats = windowed_drr_stats(
        direct_scaled,
        reverb,
        active_mask,
        sr_raw,
        win_sec=args.window_sec,
        hop_sec=args.window_hop_sec,
    )

    return {
        "relative_file": str(raw_path.relative_to(raw_root)),
        "raw_wav": str(raw_path),
        "direct_wav": str(direct_path),
        "sample_rate": sr_raw,
        "num_samples_aligned": int(len(raw_aligned)),
        "lag_samples": int(best_lag),
        "lag_ms": float(1000.0 * best_lag / sr_raw),
        "direct_gain": float(gain),
        "active_ratio": float(np.mean(active_mask)),
        "raw_energy_aligned": energy(raw_aligned),

        # Original whole-signal style DRR after alignment/gain correction
        "drr_all_db": global_all["drr_db"],
        "direct_energy_all": global_all["direct_energy"],
        "reverb_energy_all": global_all["reverb_energy"],

        # Recommended DRR: speech-active only
        "drr_active_db": global_active["drr_db"],
        "direct_energy_active": global_active["direct_energy"],
        "reverb_energy_active": global_active["reverb_energy"],

        **win_stats,
        "error": "",
    }


def save_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "relative_file",
        "raw_wav",
        "direct_wav",
        "sample_rate",
        "num_samples_aligned",
        "lag_samples",
        "lag_ms",
        "direct_gain",
        "active_ratio",
        "raw_energy_aligned",
        "drr_all_db",
        "direct_energy_all",
        "reverb_energy_all",
        "drr_active_db",
        "direct_energy_active",
        "reverb_energy_active",
        "window_count",
        "drr_win_mean_db",
        "drr_win_median_db",
        "drr_win_std_db",
        "drr_win_min_db",
        "drr_win_max_db",
        "error",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Improved estimated DRR: alignment + gain scaling + speech-active masking + window stats."
    )

    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--direct_root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="AVDeepfake1M_local/drr_improved_results.csv")
    parser.add_argument("--out_json", type=str, default="AVDeepfake1M_local/drr_improved_results.json")
    parser.add_argument("--gain_mode", type=str, default="least_squares", choices=["least_squares", "none"])
    parser.add_argument("--min_gain", type=float, default=0.05)
    parser.add_argument("--max_gain", type=float, default=20.0)

    parser.add_argument("--max_lag_ms", type=float, default=30.0)
    parser.add_argument("--vad_db_below_peak", type=float, default=40.0)
    parser.add_argument("--min_active_ratio", type=float, default=0.05)
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--window_hop_sec", type=float, default=0.5)

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    direct_root = Path(args.direct_root)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    raw_files = sorted(raw_root.rglob("*.wav"))
    if not raw_files:
        raise FileNotFoundError(f"No raw WAVs found under {raw_root}")

    rows = []

    for raw_path in raw_files:
        rel = raw_path.relative_to(raw_root)
        direct_path = direct_root / rel

        if not direct_path.exists():
            rows.append({
                "relative_file": str(rel),
                "raw_wav": str(raw_path),
                "direct_wav": str(direct_path),
                "error": "missing_direct_wav",
            })
            continue

        try:
            row = process_pair(raw_path, direct_path, raw_root, args)
            rows.append(row)
            print(f"[ok] {rel} | DRR active = {row['drr_active_db']:.3f} dB")
        except Exception as e:
            rows.append({
                "relative_file": str(rel),
                "raw_wav": str(raw_path),
                "direct_wav": str(direct_path),
                "error": f"{type(e).__name__}: {e}",
            })
            print(f"[err] {rel}: {type(e).__name__}: {e}")

    save_csv(rows, out_csv)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    valid = [r for r in rows if not r.get("error")]

    print("\nDone.")
    print(f"Total raw files: {len(rows)}")
    print(f"Valid DRR rows: {len(valid)}")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")

    if valid:
        vals = np.array([r["drr_active_db"] for r in valid], dtype=np.float64)
        print("\nSpeech-active estimated DRR summary:")
        print(f"Mean:   {vals.mean():.4f} dB")
        print(f"Std:    {vals.std():.4f} dB")
        print(f"Min:    {vals.min():.4f} dB")
        print(f"Median: {np.median(vals):.4f} dB")
        print(f"Max:    {vals.max():.4f} dB")


if __name__ == "__main__":
    main()