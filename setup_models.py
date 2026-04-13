

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

LOGGER = logging.getLogger("setup_models")

PLACES365_URL = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
MONODEPTH2_URL = "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip"
IMAGE2REVERB_CKPT_URL = "https://media.mit.edu/~nsingh1/image2reverb/model.ckpt"

MANUAL_DOWNLOAD_HELP = {
    "places365": (
        "Download resnet50_places365.pth.tar manually from "
        "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar "
        "and place it at models/resnet50_places365.pth.tar"
    ),
    "monodepth2": (
        "Download mono_640x192.zip manually from "
        "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip, "
        "extract encoder.pth and depth.pth, and place them under models/mono_640x192/"
    ),
    "image2reverb": (
        "Download model.ckpt manually from "
        "https://media.mit.edu/~nsingh1/image2reverb/model.ckpt "
        "and place it at models/model.ckpt"
    ),
}

PIP_DEPENDENCIES = [
    "numpy",
    "Pillow",
    "opencv-python",
    "soundfile",
    "librosa",
    "scipy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "torch",
    "torchvision",
    "torchaudio",
    "pytorch-lightning",
    "pyroomacoustics",
]


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging for setup steps."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def install_dependencies() -> None:
    """Install Python dependencies with pip in the current environment."""
    cmd = [sys.executable, "-m", "pip", "install", *PIP_DEPENDENCIES]
    LOGGER.info("Installing dependencies with pip...")
    subprocess.run(cmd, check=True)


def _download_file(url: str, destination: Path, model_name: str) -> None:
    """Download a file from URL to destination with safe temp-file handling."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info("Found existing %s at %s", model_name, destination)
        return

    LOGGER.info("Downloading %s from %s", model_name, url)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="download_", suffix=".tmp")
    os.close(tmp_fd)

    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        shutil.move(tmp_path, destination)
        LOGGER.info("Saved %s to %s", model_name, destination)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        help_msg = MANUAL_DOWNLOAD_HELP.get(model_name, "")
        raise RuntimeError(
            f"Failed to download '{model_name}' from {url}. {help_msg}"
        ) from exc


def ensure_places365(models_dir: Path) -> Path:
    """Ensure the Places365 checkpoint exists and return its path."""
    destination = models_dir / "resnet50_places365.pth.tar"
    _download_file(PLACES365_URL, destination, "places365")
    return destination


def ensure_monodepth2(models_dir: Path) -> Path:
    """Ensure Monodepth2 encoder/depth checkpoints exist and return directory path."""
    mono_dir = models_dir / "mono_640x192"
    encoder_path = mono_dir / "encoder.pth"
    depth_path = mono_dir / "depth.pth"

    if encoder_path.exists() and depth_path.exists():
        LOGGER.info("Found existing monodepth2 weights in %s", mono_dir)
        return mono_dir

    mono_dir.mkdir(parents=True, exist_ok=True)
    archive_path = models_dir / "mono_640x192.zip"
    _download_file(MONODEPTH2_URL, archive_path, "monodepth2")

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.namelist()
            encoder_member = next(m for m in members if m.endswith("/encoder.pth"))
            depth_member = next(m for m in members if m.endswith("/depth.pth"))
            archive.extract(encoder_member, path=models_dir)
            archive.extract(depth_member, path=models_dir)

            extracted_root = Path(encoder_member).parts[0]
            extracted_dir = models_dir / extracted_root
            shutil.move(str(extracted_dir / "encoder.pth"), encoder_path)
            shutil.move(str(extracted_dir / "depth.pth"), depth_path)
            shutil.rmtree(extracted_dir, ignore_errors=True)

        LOGGER.info("Prepared monodepth2 checkpoints in %s", mono_dir)
    except (StopIteration, zipfile.BadZipFile, OSError) as exc:
        raise RuntimeError(
            "Failed to extract monodepth2 checkpoints. "
            + MANUAL_DOWNLOAD_HELP["monodepth2"]
        ) from exc
    finally:
        if archive_path.exists():
            archive_path.unlink()

    return mono_dir


def ensure_image2reverb_ckpt(models_dir: Path) -> Path:
    """Ensure the Image2Reverb checkpoint exists and return its path."""
    destination = models_dir / "model.ckpt"
    _download_file(IMAGE2REVERB_CKPT_URL, destination, "image2reverb")
    return destination


def setup_models(models_dir: Path, install_deps: bool = False) -> dict[str, Path]:
    """Download all required model files and optionally install dependencies."""
    if install_deps:
        install_dependencies()

    models_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "places365": ensure_places365(models_dir),
        "monodepth2_dir": ensure_monodepth2(models_dir),
        "image2reverb_ckpt": ensure_image2reverb_ckpt(models_dir),
    }

    LOGGER.info("All models are ready.")
    return paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model setup."""
    parser = argparse.ArgumentParser(description="Set up Image2Reverb models")
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=Path("models"),
        help="Directory where model checkpoints are stored.",
    )
    parser.add_argument(
        "--install_deps",
        action="store_true",
        help="Install required Python dependencies via pip before downloading models.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for model setup."""
    args = parse_args()
    configure_logging(args.verbose)
    setup_models(args.models_dir, install_deps=args.install_deps)


if __name__ == "__main__":
    main()
