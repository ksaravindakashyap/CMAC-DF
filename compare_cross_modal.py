"""
Cross-modal acoustic consistency comparison.
Joins video-derived RT60/DRR (Image2Reverb), audio-derived DRR (SepFormer),
and audio-derived RT60 (blind estimation), then computes the discrepancy score.
"""
import pandas as pd
import numpy as np

# ── 1. Load CSVs ─────────────────────────────────────────────────────────────

vid = pd.read_csv(
    r"E:\CAMC-DF\CMAC-DF\resultsavdeepfake1m_full\avdeepfake1m_compatibility_records.csv"
)
aud_drr = pd.read_csv(
    r"E:\CAMC-DF\CMAC-DF\resultsavdeepfake1m_full\drr_i_600_results_audio.csv"
)
aud_rt60 = pd.read_csv(
    r"C:\Users\arvis\Downloads\rt60_results_videos_with_paths.csv"
)

# ── 2. Build a common join key: "train/.../clip_type.mp4" ────────────────────

# Video CSV: strip everything up to and including "videos_subset_600/"
vid["key"] = (
    vid["video_path"]
    .str.replace(r".*videos_subset_600[/\\]", "", regex=True)
    .str.replace("\\", "/", regex=False)
)

# Audio DRR CSV: relative_file uses .wav — swap to .mp4
aud_drr["key"] = (
    aud_drr["relative_file"]
    .str.replace("\\", "/", regex=False)
    .str.replace(r"\.wav$", ".mp4", regex=True)
)

# Audio RT60 CSV: strip "AVDeepfake1M_local/videos_subset_600/"
aud_rt60["key"] = (
    aud_rt60["video_path"]
    .str.replace(r".*videos_subset_600[/\\]", "", regex=True)
    .str.replace("\\", "/", regex=False)
)

# ── 3. Merge ─────────────────────────────────────────────────────────────────

df = vid[["key", "clip_type", "rt60_seconds", "drr_db"]].rename(
    columns={"rt60_seconds": "rt60_video", "drr_db": "drr_video"}
)
df = df.merge(
    aud_drr[["key", "drr_active_db"]].rename(columns={"drr_active_db": "drr_audio"}),
    on="key", how="inner"
)
df = df.merge(
    aud_rt60[["key", "rt60"]].rename(columns={"rt60": "rt60_audio"}),
    on="key", how="inner"
)

print(f"Merged rows: {len(df)}  (expected 600)")
print(f"Missing after merge: {600 - len(df)}")

# ── 4. Discrepancy scores ─────────────────────────────────────────────────────

df["delta_rt60"] = (df["rt60_audio"] - df["rt60_video"]).abs()
df["delta_drr"]  = (df["drr_audio"]  - df["drr_video"]).abs()

# L2 discrepancy in raw units (RT60 in seconds, DRR in dB — different scales)
df["discrepancy_raw"] = np.sqrt(df["delta_rt60"]**2 + df["delta_drr"]**2)

# Normalised discrepancy: z-score each dimension by its overall std
std_rt60 = df["delta_rt60"].std()
std_drr  = df["delta_drr"].std()
df["discrepancy_norm"] = np.sqrt(
    (df["delta_rt60"] / std_rt60)**2 + (df["delta_drr"] / std_drr)**2
)

# ── 5. Per-class summary ──────────────────────────────────────────────────────

label_map = {
    "real":                  "real",
    "real_video_fake_audio": "audio_fake",
    "fake_video_fake_audio": "both_fake",
}
df["label"] = df["clip_type"].map(label_map)

cols = ["rt60_audio", "rt60_video", "drr_audio", "drr_video",
        "delta_rt60", "delta_drr", "discrepancy_raw", "discrepancy_norm"]

print("\n-- Per-class means --")
summary = df.groupby("label")[cols].agg(["mean", "median", "std"]).round(3)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
print(summary)

# ── 6. Threshold suggestion ───────────────────────────────────────────────────

print("\n-- Threshold analysis (discrepancy_norm) --")
for label in ["real", "audio_fake", "both_fake"]:
    sub = df[df["label"] == label]["discrepancy_norm"]
    print(f"  {label:12s}  mean={sub.mean():.3f}  median={sub.median():.3f}  "
          f"std={sub.std():.3f}  min={sub.min():.3f}  max={sub.max():.3f}")

real_mean = df[df["label"] == "real"]["discrepancy_norm"].mean()
real_std  = df[df["label"] == "real"]["discrepancy_norm"].std()

thresholds = {}
for k in [0.5, 1.0, 1.5, 2.0]:
    thr = real_mean + k * real_std
    thresholds[k] = thr
    preds = (df["discrepancy_norm"] > thr).astype(int)
    # ground truth: 0=real, 1=fake (audio manipulated)
    truth = (df["label"] != "real").astype(int)
    tp = ((preds == 1) & (truth == 1)).sum()
    fp = ((preds == 1) & (truth == 0)).sum()
    fn = ((preds == 0) & (truth == 1)).sum()
    tn = ((preds == 0) & (truth == 0)).sum()
    acc  = (tp + tn) / len(df)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"\n  Threshold = mean + {k}*std = {thr:.3f}")
    print(f"    Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")

# ── 7. Save merged output ─────────────────────────────────────────────────────

out_path = r"E:\CAMC-DF\CMAC-DF\resultsavdeepfake1m_full\cross_modal_comparison.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved merged comparison to: {out_path}")
