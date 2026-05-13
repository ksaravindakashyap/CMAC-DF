# Cross-Modal Acoustic Consistency for Audio-Visual Deepfake Detection

This repository implements the visual and cross-modal acoustic pipeline for the CMAC-DF project. The core idea is that authentic audio-visual recordings must be physically consistent — the room acoustics heard in the audio (RT60, DRR) should match the acoustics predicted from the visible scene geometry. Manipulated videos systematically violate this constraint.

**Authors:** Sneha Aggarwal, Matthew Nissen, Aravinda Kashyap — Virginia Tech

---

## Overview

The pipeline has two streams:

- **Visual stream** — extracts a representative frame from the video, runs it through Image2Reverb (a GAN-based neural network) to synthesize the expected Room Impulse Response (RIR), then computes RT60 and DRR from that RIR.
- **Audio stream** — estimates RT60 from the speech signal using blind estimation, and estimates DRR by separating the direct and reverberant waveforms using SpeechBrain SepFormer.

The **cross-modal discrepancy score** is then computed as:

```
score = |RT60_audio - RT60_video| + |DRR_audio - DRR_video|
```

A high score indicates the audio acoustics do not match the visual scene — a signal of manipulation.

---

## Acoustic Parameters

| Parameter | Description |
|-----------|-------------|
| **RT60** | Reverberation time — how long it takes for sound to decay 60 dB. Larger rooms with hard surfaces (tile, concrete) yield longer RT60; smaller rooms with absorptive materials (carpet, curtains) yield shorter RT60. |
| **DRR** | Direct-to-Reverberant Ratio — energy of the direct sound path vs. reflected components, in dB. High DRR = anechoic/close-mic conditions. TTS/VC synthetic audio typically has abnormally high DRR. |

---

## Setup

Download the required pretrained models (Image2Reverb checkpoint, ResNet50/Places365, Monodepth2):

```powershell
python setup_models.py --models_dir models --install_deps
```

Omit `--install_deps` if dependencies are already installed.

**Dependencies:** `torch`, `torchvision`, `torchaudio`, `pytorch-lightning`, `opencv-python`, `Pillow`, `scipy`, `numpy`, `soundfile`, `librosa`, `pyroomacoustics`

---

## Scripts

### `visual_rir_estimator.py`
Core pipeline. Extracts a frame from a video, runs Image2Reverb inference, and computes RT60 and DRR from the synthesized RIR.

```powershell
python visual_rir_estimator.py --video path\to\video.mp4 --output_dir results
```

Outputs `rir.npy`, `rir.wav`, `rt60.json`, `drr.json` in the output directory.

### `check_avdeepfake1m_compatibility.py`
Runs the visual pipeline on the AVDeepfake1M dataset and produces per-clip RT60/DRR estimates across the three clip types (`real`, `real_video_fake_audio`, `fake_video_fake_audio`).

```powershell
python check_avdeepfake1m_compatibility.py \
  --dataset_root C:\path\to\videos_subset_600 \
  --sample_count 1000 \
  --output_dir results\avdeepfake1m_full \
  --verbose
```

Use `--sample_count 1000` (or any number above the total clip count) to process all clips.

### `compare_cross_modal.py`
Joins the visual RT60/DRR results with audio-derived RT60 and DRR CSVs, computes per-clip cross-modal discrepancy scores, and saves a merged comparison CSV.

```powershell
python compare_cross_modal.py
```

Output: `resultsavdeepfake1m_full\cross_modal_comparison.csv`

### `classify_deepfake.py`
Evaluates several acoustic signals (audio-only, video-only, and cross-modal) as deepfake classifiers. For each signal, finds the optimal decision threshold by grid search and reports accuracy, precision, recall, and F1.

```powershell
python classify_deepfake.py
```

### `rt60_from_rir.py` / `drr_from_rir.py`
Standalone helpers to estimate RT60 or DRR from a saved RIR file.

```powershell
python rt60_from_rir.py --rir path\to\rir.wav --output_json results\rt60.json
python drr_from_rir.py --rir path\to\rir.wav --output_json results\drr.json
```

For `.npy` input, add `--sample_rate 22050`.

---

## Dataset — AVDeepfake1M

**Path:** `videos_subset_600/train/<speaker_id>/<session>/<clip>/`

Each clip directory contains three files:

| File | Description |
|------|-------------|
| `real.mp4` | Real video, real audio |
| `real_video_fake_audio.mp4` | Real video, TTS/VC synthesized audio |
| `fake_video_fake_audio.mp4` | Face-swapped video, TTS/VC synthesized audio |

**Subset used:** 600 clips (200 per class, balanced).

---

## Results — Full 600-Clip Run

**Visual pipeline** (Image2Reverb on video frames):

| Clip Type | RT60 Mean | RT60 Median | DRR Mean | DRR Median |
|-----------|-----------|-------------|----------|------------|
| real | 5.20s | 2.49s | −16.16 dB | −16.53 dB |
| real_video_fake_audio | 4.97s | 2.46s | −16.13 dB | −16.19 dB |
| fake_video_fake_audio | 5.17s | 2.56s | −15.99 dB | −16.29 dB |

**Cross-modal classification** (best threshold, grid search):

| Signal | Rule | Accuracy | Precision | Recall | F1 |
|--------|------|----------|-----------|--------|----|
| Audio RT60 | < 3.94s → FAKE | 66.8% | 0.668 | 1.000 | 0.801 |
| Audio DRR | > −12.72 dB → FAKE | 66.8% | 0.669 | 0.995 | 0.800 |
| Cross-modal sum | > 0.39 → FAKE | 66.5% | 0.666 | 0.998 | 0.799 |

**Key finding:** All signals achieve ~66.8% accuracy, equal to the majority-class baseline (400/600 clips are fake). The classifier flags nearly every clip as fake regardless of signal, indicating no discriminative separation between real and manipulated clips under the current setup.

**Root causes:**
1. **Visual stream degradation** — 93% of clips triggered face-shot fallback (scene quality check failed), so Image2Reverb received face images instead of room geometry, producing near-constant and uninformative predictions.
2. **Audio signal collapse** — blind RT60 and DRR estimates from speech are statistically indistinguishable across all three clip types.

---

## Limitations and Future Work

- Filter to clips where sufficient room context is visible (the ~7% that pass the scene quality check) for a meaningful visual stream evaluation.
- Transition from static 2D image inference to temporal video-based RIR extraction.
- Project audio-derived and video-derived features into a shared latent embedding space using contrastive learning rather than comparing scalar parameters directly.
