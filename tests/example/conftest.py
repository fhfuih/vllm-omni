"""
Shared fixtures, helpers, and path constants for tests/example/.

All offline and online diffusion example tests import from here.
"""

import base64
import io
import os
import pathlib
import signal
import subprocess
import sys
import threading
import urllib.request
from io import BytesIO
from pathlib import Path

import imageio.v3 as iio
import pytest
import soundfile as sf
from PIL import Image

# ---------------------------------------------------------------------------
# Add L4 markers to all tests in this directory and subdirectories.
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(session, config, items):
    rootdir = pathlib.Path(config.rootdir)
    for item in items:
        rel_path = pathlib.Path(item.fspath).relative_to(rootdir)
        if rel_path.parts[:2] == ("tests", "example"):
            item.add_marker("advanced_model")
            item.add_marker("example")


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples" / "offline_inference"

T2I_SCRIPT = EXAMPLES / "text_to_image" / "text_to_image.py"
I2I_SCRIPT = EXAMPLES / "image_to_image" / "image_edit.py"
T2V_SCRIPT = EXAMPLES / "text_to_video" / "text_to_video.py"
I2V_SCRIPT = EXAMPLES / "image_to_video" / "image_to_video.py"
T2A_SCRIPT = EXAMPLES / "text_to_audio" / "text_to_audio.py"
BAGEL_SCRIPT = EXAMPLES / "bagel" / "end2end.py"
BAGEL_CLIENT = REPO_ROOT / "examples" / "online_serving" / "bagel" / "openai_chat_client.py"
T2I_ONLINE_CLIENT = REPO_ROOT / "examples" / "online_serving" / "text_to_image" / "openai_chat_client.py"
I2I_ONLINE_CLIENT = REPO_ROOT / "examples" / "online_serving" / "image_to_image" / "openai_chat_client.py"

BEAR_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png"
HORSE_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/horsepony.jpg"

for path in (T2I_SCRIPT, I2I_SCRIPT, T2V_SCRIPT, I2V_SCRIPT, T2A_SCRIPT, BAGEL_SCRIPT, BAGEL_CLIENT):
    assert path.exists(), f"Example script not found: {path}"

# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_prefix() -> str:
    """Optional model-path prefix from MODEL_PREFIX env var."""
    prefix = os.environ.get("MODEL_PREFIX", "")
    return f"{prefix.rstrip('/')}/" if prefix else ""


@pytest.fixture(scope="session")
def assets_dir(tmp_path_factory) -> Path:
    """Download shared image assets once per session."""
    prefix = os.environ.get("ASSETS_DIR")
    if prefix is not None:
        d = REPO_ROOT / prefix
        d.mkdir(parents=True, exist_ok=True)
    else:
        d: Path = tmp_path_factory.mktemp("assets")
    urllib.request.urlretrieve(BEAR_IMAGE_URL, str(d / "qwen-bear.png"))
    urllib.request.urlretrieve(HORSE_IMAGE_URL, str(d / "horsepony.jpg"))
    return d


@pytest.fixture(scope="session")
def bear(assets_dir) -> str:
    """Path to the qwen-bear.png asset (string, for passing to subprocesses)."""
    return str(assets_dir / "qwen-bear.png")


@pytest.fixture(scope="session")
def horse(assets_dir) -> str:
    """Path to the horsepony.jpg asset (string, for passing to subprocesses)."""
    return str(assets_dir / "horsepony.jpg")


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory) -> Path:
    """Shared output directory for offline tests."""
    prefix = os.environ.get("OUTPUT_DIR")
    if prefix is not None:
        d = REPO_ROOT / prefix
        d.mkdir(parents=True, exist_ok=True)
    else:
        d = tmp_path_factory.mktemp("outputs")
    return d


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------


def run_script(script: Path, *args: str) -> None:
    """Run an example Python script as a subprocess; assert zero exit code.

    Output is tee'd: written to the console in real time (visible with pytest
    -s) and also buffered so the last 2000 chars appear in the assertion
    message on failure.

    Uses PIPE + reader threads rather than communicate() to avoid the classic
    hang: grandchild worker processes inherit the pipe write-end, so
    communicate() would block on EOF forever if the main child crashes.
    Instead we use proc.wait() (direct-child only), then killpg to terminate
    the whole process group, which closes all write-ends and lets the reader
    threads finish naturally.
    """
    proc = subprocess.Popen(
        [sys.executable, str(script), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,  # new process group → killpg cleans up workers
    )

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    def _tee(src, console, buf):
        for raw_line in src:
            line = raw_line.decode(errors="replace")
            console.write(line)
            console.flush()
            buf.write(line)

    t_out = threading.Thread(target=_tee, args=(proc.stdout, sys.stdout, stdout_buf), daemon=True)
    t_err = threading.Thread(target=_tee, args=(proc.stderr, sys.stderr, stderr_buf), daemon=True)
    t_out.start()
    t_err.start()

    try:
        returncode = proc.wait()
    finally:
        # Kill surviving grandchild workers; this closes their pipe write-ends
        # so the reader threads reach EOF and exit.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    t_out.join(timeout=10)
    t_err.join(timeout=10)

    assert returncode == 0, (
        f"{script.name} failed (exit {returncode}):\n"
        f"--- STDOUT (last 2000 chars) ---\n{stdout_buf.getvalue()[-2000:]}\n"
        f"--- STDERR (last 2000 chars) ---\n{stderr_buf.getvalue()[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Output validation helpers
# ---------------------------------------------------------------------------


def assert_image_valid(path: Path, *, width: int | None = None, height: int | None = None) -> Image.Image:
    """Assert the file is a loadable image with optional exact dimensions."""
    assert path.exists(), f"Image not found: {path}"
    img = Image.open(path)
    img.load()
    assert img.width > 0 and img.height > 0
    if width is not None:
        assert img.width == width, f"Expected width={width}, got {img.width} in {path.name}"
    if height is not None:
        assert img.height == height, f"Expected height={height}, got {img.height} in {path.name}"
    return img


def assert_video_valid(path: Path, *, width: int, height: int, num_frames: int) -> None:
    """Assert the MP4 has the expected resolution and exact frame count."""
    assert path.exists(), f"Video not found: {path}"
    # shape: (num_frames, height, width, channels)
    frames = iio.imread(str(path), plugin="pyav", index=None)
    assert frames.shape[0] == num_frames, f"Expected {num_frames} frames, got {frames.shape[0]}"
    assert frames.shape[1] == height, f"Expected height={height}, got {frames.shape[1]}"
    assert frames.shape[2] == width, f"Expected width={width}, got {frames.shape[2]}"


def assert_audio_valid(path: Path, *, sample_rate: int, channels: int, duration_s: float) -> None:
    """Assert the WAV has the expected sample rate, channel count, and duration."""
    assert path.exists(), f"Audio not found: {path}"
    info = sf.info(str(path))
    assert info.samplerate == sample_rate, f"Expected sample_rate={sample_rate}, got {info.samplerate}"
    assert info.channels == channels, f"Expected {channels} channel(s), got {info.channels}"
    expected_frames = int(duration_s * sample_rate)
    assert info.frames == expected_frames, (
        f"Expected {expected_frames} frames ({duration_s}s @ {sample_rate} Hz), got {info.frames}"
    )


def decode_b64_png(b64: str) -> Image.Image:
    img = Image.open(BytesIO(base64.b64decode(b64)))
    img.load()
    return img
