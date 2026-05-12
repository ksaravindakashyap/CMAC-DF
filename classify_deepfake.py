"""
Deepfake classification using cross-modal acoustic discrepancy.

Formula:
    score = |RT60_audio - RT60_video| + |DRR_audio - DRR_video|
    if score > threshold -> FAKE  else -> REAL

Also tries audio-only signals for comparison.
"""
import pandas as pd
import numpy as np

# Load merged data
df = pd.read_csv(
    r"E:\CAMC-DF\CMAC-DF\resultsavdeepfake1m_full\cross_modal_comparison.csv"
)

# Also bring in window DRR std from audio (noted as "most promising" signal)
aud_drr = pd.read_csv(
    r"E:\CAMC-DF\CMAC-DF\resultsavdeepfake1m_full\drr_i_600_results_audio.csv"
)
aud_drr["key"] = (
    aud_drr["relative_file"]
    .str.replace(r"\.wav$", ".mp4", regex=True)
    .str.replace("\\", "/", regex=False)
)
df = df.merge(aud_drr[["key", "drr_win_std_db"]], on="key", how="left")

# Ground truth: 0 = real, 1 = fake (any audio manipulation)
df["truth"] = (df["label"] != "real").astype(int)

def best_threshold(scores, truth):
    """Scan all midpoints and return threshold with best accuracy."""
    best_acc, best_thr, best_dir = 0, 0, ">"
    vals = np.sort(scores.unique())
    thresholds = (vals[:-1] + vals[1:]) / 2
    for thr in thresholds:
        for direction in [">", "<"]:
            preds = (scores > thr) if direction == ">" else (scores < thr)
            acc = (preds.astype(int) == truth).mean()
            if acc > best_acc:
                best_acc, best_thr, best_dir = acc, thr, direction
    return best_thr, best_dir, best_acc

def evaluate(scores, truth, thr, direction):
    preds = (scores > thr).astype(int) if direction == ">" else (scores < thr).astype(int)
    tp = ((preds == 1) & (truth == 1)).sum()
    fp = ((preds == 1) & (truth == 0)).sum()
    fn = ((preds == 0) & (truth == 1)).sum()
    tn = ((preds == 0) & (truth == 0)).sum()
    acc  = (tp + tn) / len(truth)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

truth = df["truth"]

signals = {
    "Cross-modal (RT60+DRR abs diff)":
        df["delta_rt60"] + df["delta_drr"],
    "Cross-modal L2 discrepancy":
        df["discrepancy_raw"],
    "Cross-modal DRR diff only":
        df["delta_drr"],
    "Cross-modal RT60 diff only":
        df["delta_rt60"],
    "Audio DRR (active)":
        df["drr_audio"],
    "Audio RT60":
        df["rt60_audio"],
    "Audio DRR window std":
        df["drr_win_std_db"],
}

results = []
for name, scores in signals.items():
    thr, direction, acc = best_threshold(scores, truth)
    m = evaluate(scores, truth, thr, direction)
    results.append({
        "Signal": name,
        "Threshold": f"{direction} {thr:.3f}",
        "Accuracy": f"{m['acc']:.3f}",
        "Precision": f"{m['prec']:.3f}",
        "Recall": f"{m['rec']:.3f}",
        "F1": f"{m['f1']:.3f}",
    })

res_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\n=== Signal comparison (sorted by F1) ===")
print(res_df.to_string(index=False))

# ── Best signal: detailed breakdown ──────────────────────────────────────────
best_row = res_df.iloc[0]
print(f"\n=== Best signal: {best_row['Signal']} ===")
print(f"    Rule: if score {best_row['Threshold']} -> FAKE else REAL")
print(f"    Accuracy={best_row['Accuracy']}  Precision={best_row['Precision']}  Recall={best_row['Recall']}  F1={best_row['F1']}")

# Show per-class breakdown for the best signal
best_name = best_row["Signal"]
best_scores = signals[best_name]
direction = best_row["Threshold"].split()[0]
thr_val = float(best_row["Threshold"].split()[1])
df["prediction"] = ((best_scores > thr_val) if direction == ">" else (best_scores < thr_val)).map({True: "FAKE", False: "REAL"})

print("\n=== Per-class prediction breakdown ===")
breakdown = df.groupby(["label", "prediction"]).size().unstack(fill_value=0)
print(breakdown)

# ── Recommended formula ───────────────────────────────────────────────────────
print("\n=== Recommended formula ===")
print(f"  score = |RT60_audio - RT60_video| + |DRR_audio - DRR_video|")
print(f"  if score > threshold -> FAKE, else -> REAL")

# Find the best threshold for the cross-modal sum signal specifically
cm_signal = df["delta_rt60"] + df["delta_drr"]
thr, direction, acc = best_threshold(cm_signal, truth)
m = evaluate(cm_signal, truth, thr, direction)
print(f"\n  Optimal threshold = {thr:.2f}")
print(f"  Accuracy={m['acc']:.3f}  Precision={m['prec']:.3f}  Recall={m['rec']:.3f}  F1={m['f1']:.3f}")
print(f"  TP={m['tp']} (fake correctly detected)  FP={m['fp']} (real wrongly flagged)")
print(f"  FN={m['fn']} (fake missed)             TN={m['tn']} (real correctly passed)")
