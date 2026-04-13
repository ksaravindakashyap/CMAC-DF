"""Visual RIR estimation wrapper around the original Image2Reverb repository.

This module extracts a representative video frame, verifies scene quality,
and estimates an RIR waveform from the selected image.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal, Tuple

import cv2
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from torchvision import transforms

from setup_models import setup_models

LOGGER = logging.getLogger("visual_rir_estimator")
SAMPLE_RATE = 22050
TARGET_SIZE = (512, 512)

ModelCacheType = Tuple[torch.nn.Module, object, torch.device]
_MODEL_CACHE: ModelCacheType | None = None


def _configure_logging(verbose: bool = False) -> None:
    """Configure logging for module and CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_workspace_root() -> Path:
    """Resolve workspace root based on this file location."""
    return Path(__file__).resolve().parent


def _prepare_repo_imports(repo_root: Path) -> None:
    """Ensure Image2Reverb package path is importable."""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _get_model_paths(models_dir: Path) -> dict[str, Path]:
    """Ensure required checkpoints are available and return their paths."""
    paths = setup_models(models_dir, install_deps=False)
    return {
        "places365": Path(paths["places365"]),
        "monodepth2_dir": Path(paths["monodepth2_dir"]),
        "image2reverb_ckpt": Path(paths["image2reverb_ckpt"]),
    }


def _load_model(repo_root: Path, models_dir: Path) -> ModelCacheType:
    """Load and cache Image2Reverb model and STFT inverter."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    _prepare_repo_imports(repo_root)

    from image2reverb.model import Image2Reverb  # pylint: disable=import-error
    from image2reverb.stft import STFT  # pylint: disable=import-error

    paths = _get_model_paths(models_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Loading Image2Reverb model on device: %s", device)
    model = Image2Reverb(
        str(paths["places365"]),
        str(paths["monodepth2_dir"]),
        spec="stft",
    )
    ckpt = torch.load(paths["image2reverb_ckpt"], map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    stft = STFT()
    _MODEL_CACHE = (model, stft, device)
    return _MODEL_CACHE


def _image_to_tensor(image_path: str) -> torch.Tensor:
    """Load an image and convert it to normalized tensor expected by Image2Reverb."""
    image = Image.open(image_path).convert("RGB").resize(TARGET_SIZE)
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return tfm(image).unsqueeze(0)


def _score_frame_for_scene_context(frame: np.ndarray) -> float:
    """Score frame suitability using edge density and foreground-face penalty."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    center = gray[gray.shape[0] // 4 : (3 * gray.shape[0]) // 4, gray.shape[1] // 4 : (3 * gray.shape[1]) // 4]
    center_var = float(np.var(center)) / 255.0

    face_ratio = _largest_face_ratio(frame)
    face_penalty = 0.7 * face_ratio

    return edge_density + (0.15 * center_var) - face_penalty


def _largest_face_ratio(frame: np.ndarray) -> float:
    """Estimate largest detected face area ratio in a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return 0.0

    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return 0.0

    frame_area = float(frame.shape[0] * frame.shape[1])
    max_area = max(w * h for (_, _, w, h) in faces)
    return float(max_area / frame_area)


def extract_frame(video_path: str, strategy: Literal["middle", "first", "scene"] = "middle") -> str:
    """Extract a representative frame from a video and save as a 512x512 PNG.

    Args:
        video_path: Path to input video file.
        strategy: Frame selection strategy: "first", "middle", or "scene".

    Returns:
        Path to extracted frame image.
    """
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    frame_dir = _resolve_workspace_root() / "tmp_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError("Video appears to have no frames.")

    if strategy == "first":
        selected_index = 0
    elif strategy == "middle":
        selected_index = frame_count // 2
    elif strategy == "scene":
        sample_count = min(64, frame_count)
        sample_indices = np.linspace(0, frame_count - 1, num=sample_count, dtype=int)
        best_score = -1e9
        selected_index = int(sample_indices[len(sample_indices) // 2])

        for idx in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            score = _score_frame_for_scene_context(frame)
            if score > best_score:
                best_score = score
                selected_index = int(idx)

        LOGGER.info("Scene strategy chose frame %d/%d", selected_index, frame_count)
    else:
        capture.release()
        raise ValueError("strategy must be one of: first, middle, scene")

    capture.set(cv2.CAP_PROP_POS_FRAMES, int(selected_index))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read selected frame index {selected_index}")

    resized = cv2.resize(frame, TARGET_SIZE)
    output_path = frame_dir / f"{video.stem}_{strategy}.png"
    cv2.imwrite(str(output_path), resized)
    LOGGER.info("Saved extracted frame to %s", output_path)
    return str(output_path)


def check_frame_has_room_context(image_path: str) -> bool:
    """Check whether a frame likely contains enough room context.

    Heuristics combine edge density (scene structure) and largest face ratio
    (close-up penalty).

    Args:
        image_path: Path to image file.

    Returns:
        True if frame likely has room context, otherwise False.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.mean(edges > 0))
    face_ratio = _largest_face_ratio(frame)

    has_context = edge_density >= 0.02 and face_ratio <= 0.45
    LOGGER.info(
        "Quality check edge_density=%.4f face_ratio=%.4f pass=%s",
        edge_density,
        face_ratio,
        has_context,
    )
    return has_context


def image_to_rir(image_path: str, output_dir: str) -> tuple[np.ndarray, int]:
    """Estimate RIR waveform from an image via Image2Reverb inference.

    Args:
        image_path: Path to input image (will be resized to 512x512).
        output_dir: Directory to save debug artifacts.

    Returns:
        Tuple containing (rir waveform as float32 numpy array, sample rate).
    """
    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workspace_root = _resolve_workspace_root()
    repo_root = workspace_root / "image2reverb"
    if not repo_root.exists():
        raise FileNotFoundError(
            "Image2Reverb repository not found at ./image2reverb. "
            "Clone it first: git clone https://github.com/nikhilsinghmus/image2reverb.git"
        )

    models_dir = workspace_root / "models"
    model, stft, device = _load_model(repo_root, models_dir)

    img_tensor = _image_to_tensor(str(image)).to(device)
    with torch.no_grad():
        spec = model(img_tensor)

    rir = stft.inverse(spec[0].squeeze())
    rir = np.asarray(rir, dtype=np.float32).reshape(-1)

    wav_path = out_dir / f"{image.stem}_rir.wav"
    sf.write(wav_path, rir, SAMPLE_RATE)
    LOGGER.info("Saved debug RIR WAV to %s", wav_path)

    return rir, SAMPLE_RATE


def get_rir_from_video(video_path: str, output_dir: str) -> tuple[np.ndarray, int, str]:
    """Run frame extraction + quality check + Image2Reverb to estimate RIR.

    Pipeline:
    1) Extract frame with strategy="scene"
    2) Run quality check; if it fails, fallback to strategy="middle"
    3) Estimate RIR and save .wav/.npy in output_dir

    Args:
        video_path: Path to source video.
        output_dir: Directory for output files.

    Returns:
        Tuple (rir array, sample rate, frame path used for inference).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_path = extract_frame(video_path, strategy="scene")
    if not check_frame_has_room_context(frame_path):
        LOGGER.warning("Scene frame failed quality check, falling back to middle frame.")
        frame_path = extract_frame(video_path, strategy="middle")

    rir, sample_rate = image_to_rir(frame_path, output_dir)

    rir_npy_path = out_dir / "rir.npy"
    np.save(rir_npy_path, rir.astype(np.float32))

    rir_wav_path = out_dir / "rir.wav"
    sf.write(rir_wav_path, rir, sample_rate)

    LOGGER.info("Saved RIR numpy to %s", rir_npy_path)
    LOGGER.info("Saved RIR wav to %s", rir_wav_path)

    return rir.astype(np.float32), int(sample_rate), frame_path


def _run_example_image_test(output_dir: str) -> None:
    """Run a setup verification pass on the repo example image path."""
    workspace_root = _resolve_workspace_root()
    example_image = (
        workspace_root
        / "image2reverb"
        / "datasets"
        / "examples"
        / "bedroom-1"
        / "test"
        / "input.png"
    )

    if not example_image.exists():
        LOGGER.error(
            "Example image not found at %s. "
            "Place the sample there or run with --video to test end-to-end.",
            example_image,
        )
        return

    rir, sample_rate = image_to_rir(str(example_image), output_dir)

    out_dir = Path(output_dir)
    npy_path = out_dir / "example_rir.npy"
    wav_path = out_dir / "example_rir.wav"
    np.save(npy_path, rir.astype(np.float32))
    sf.write(wav_path, rir, sample_rate)

    LOGGER.info("Example image test complete. rir shape=%s sample_rate=%d", rir.shape, sample_rate)


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for video inference or self-test."""
    parser = argparse.ArgumentParser(description="Estimate visual RIR from video using Image2Reverb")
    parser.add_argument("--video", type=str, default=None, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument(
        "--self_test",
        action="store_true",
        help="Run setup verification on Image2Reverb example image path.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for visual RIR estimation."""
    args = _parse_args()
    _configure_logging(args.verbose)

    if args.video:
        rir, sample_rate, frame_path = get_rir_from_video(args.video, args.output_dir)
        LOGGER.info("Output RIR shape=%s sample_rate=%d frame=%s", rir.shape, sample_rate, frame_path)
        return

    if args.self_test or not args.video:
        LOGGER.info("No --video provided. Running example-image self-test.")
        _run_example_image_test(args.output_dir)


if __name__ == "__main__":
    main()
