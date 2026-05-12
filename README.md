# CMAC-DF Visual RIR Pipeline

This repository includes a visual room impulse response pipeline built on Image2Reverb, plus a standalone RT60 estimator for downstream use.

## Setup

Download the required pretrained models:

```powershell
python setup_models.py --models_dir models --install_deps
```

If you already have the dependencies installed, you can omit `--install_deps`.

## Video to RIR to RT60

Generate an RIR from a video and estimate RT60 from that RIR in one step:

```powershell
python visual_rir_estimator.py --video path\to\video.mp4 --output_dir results
```

This will:

1. Extract a representative frame from the video.
2. Run Image2Reverb to generate an RIR waveform.
3. Estimate RT60 from the RIR.
4. Save `rir.npy`, `rir.wav`, and `rt60.json` in the output directory.

## RT60 From an Existing RIR

If you already have a generated RIR file, use the standalone helper:

```powershell
python rt60_from_rir.py --rir path\to\rir.wav --output_json results\rt60.json
```

For `.npy` input, pass the sample rate explicitly:

```powershell
python rt60_from_rir.py --rir path\to\rir.npy --sample_rate 22050 --output_json results\rt60.json
```

## DRR From an Existing RIR

If you want direct-to-reverberant ratio from a saved RIR waveform, use the DRR helper:

```powershell
python drr_from_rir.py --rir path\to\rir.wav --output_json results\drr.json
```

For `.npy` input, pass the sample rate explicitly:

```powershell
python drr_from_rir.py --rir path\to\rir.npy --sample_rate 22050 --output_json results\drr.json
```

## Example Image Self-Test

You can verify the pipeline with the bundled Image2Reverb example image path:

```powershell
python visual_rir_estimator.py --self_test --output_dir results
```

If the example image exists at `image2reverb/datasets/examples/bedroom-1/test/input.png`, the script will generate an example RIR and RT60 estimate.

## VoxCeleb1 Compatibility Check

If you have VoxCeleb1 downloaded locally, you can run a 100-clip compatibility check to see whether the visual pipeline is extracting meaningful values:

```powershell
python check_voxceleb1_compatibility.py --dataset_root path\to\VoxCeleb1 --sample_count 100 --output_dir results\voxceleb1_check
```

The script samples up to 600 clips, runs the scene-frame quality check, generates RIR/RT60/DRR outputs, and writes summary JSON/CSV files.

## AVDeepfake1M Compatibility Check

If your local dataset is AVDeepfake1M, use the dataset-specific checker instead:

```powershell
python check_avdeepfake1m_compatibility.py --dataset_root C:\Users\arvis\Downloads\avdeepfake1m_660_videos\AVDeepfake1M_local\videos_subset_600 --sample_count 100 --output_dir results\avdeepfake1m_check
```

The script samples up to 600 clips from the local tree, preserves the three clip types in the subset (`real`, `real_video_fake_audio`, and `fake_video_fake_audio`), runs the visual RIR pipeline, and writes summary JSON/CSV files.
