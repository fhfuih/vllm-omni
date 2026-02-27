"""
Pytest version of tests/e2e/example/test_diffusion.sh.

Covers:
  - Offline text-to-image  (multiple models, parallelism variants, cache backends)
  - Offline image editing
  - Offline text-to-video  (Wan2.2 T2V)
  - Offline image-to-video (Wan2.2 TI2V)
  - Offline text-to-audio  (Stable Audio Open)
  - Offline BAGEL 2-stage  (text2img / img2img)
  - Online Z-Image-Turbo, Qwen-Image-Edit, BAGEL, Wan2.2 T2V/TI2V

Each test asserts:
  1. The inference script / HTTP call succeeds without error.
  2. The output artifact has the expected deterministic properties
     (image dimensions, video frame count / resolution, audio sample-rate /
     channel count / duration).
"""

import base64
import os
import signal
import subprocess
import sys
import tempfile
import urllib.request
from io import BytesIO
from pathlib import Path

import pytest
import requests
from PIL import Image

from tests.conftest import OmniServer

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "examples" / "offline_inference"

T2I_SCRIPT = EXAMPLES / "text_to_image" / "text_to_image.py"
I2I_SCRIPT = EXAMPLES / "image_to_image" / "image_edit.py"
T2V_SCRIPT = EXAMPLES / "text_to_video" / "text_to_video.py"
I2V_SCRIPT = EXAMPLES / "image_to_video" / "image_to_video.py"
T2A_SCRIPT = EXAMPLES / "text_to_audio" / "text_to_audio.py"
BAGEL_SCRIPT = EXAMPLES / "bagel" / "end2end.py"
BAGEL_CLIENT = REPO_ROOT / "examples" / "online_serving" / "bagel" / "openai_chat_client.py"

BEAR_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png"
HORSE_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/horsepony.jpg"

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
    d = tmp_path_factory.mktemp("assets")
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
    return tmp_path_factory.mktemp("outputs")


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------


def run_script(script: Path, *args: str) -> None:
    """Run an example Python script as a subprocess; assert zero exit code.

    Uses temp files for stdout/stderr instead of pipes so that grandchild
    worker processes (which inherit file descriptors) do not cause an
    indefinite hang: proc.wait() returns as soon as the direct child exits,
    regardless of whether grandchildren are still alive.  start_new_session
    puts the whole tree in its own process group so orphaned workers are
    cleaned up via killpg after the direct child exits.
    """
    with tempfile.TemporaryFile() as stdout_f, tempfile.TemporaryFile() as stderr_f:
        proc = subprocess.Popen(
            [sys.executable, str(script), *args],
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,  # new process group → killpg cleans up workers
        )
        try:
            returncode = proc.wait()
        finally:
            # Kill any surviving grandchild workers so they don't linger.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        stdout_f.seek(0)
        stderr_f.seek(0)
        stdout = stdout_f.read().decode(errors="replace")
        stderr = stderr_f.read().decode(errors="replace")

    assert returncode == 0, (
        f"{script.name} failed (exit {returncode}):\n"
        f"--- STDOUT (last 2000 chars) ---\n{stdout[-2000:]}\n"
        f"--- STDERR (last 2000 chars) ---\n{stderr[-2000:]}"
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
    import imageio.v3 as iio

    # shape: (num_frames, height, width, channels)
    frames = iio.imread(str(path), plugin="pyav", index=None)
    assert frames.shape[0] == num_frames, f"Expected {num_frames} frames, got {frames.shape[0]}"
    assert frames.shape[1] == height, f"Expected height={height}, got {frames.shape[1]}"
    assert frames.shape[2] == width, f"Expected width={width}, got {frames.shape[2]}"


def assert_audio_valid(path: Path, *, sample_rate: int, channels: int, duration_s: float) -> None:
    """Assert the WAV has the expected sample rate, channel count, and duration."""
    assert path.exists(), f"Audio not found: {path}"
    import soundfile as sf

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


# ---------------------------------------------------------------------------
# Offline: Text-to-Image
# ---------------------------------------------------------------------------

_T2I_PROMPT = "a cup of coffee on the table"
_T2I_COMMON = [
    "--seed", "42",
    "--cfg-scale", "4.0",
    "--num-images-per-prompt", "1",
    "--num-inference-steps", "50",
    "--height", "1024",
    "--width", "1024",
]  # fmt: skip

# (case_id, model, extra_cli_args)
_T2I_CASES = [
    ("qwen_image", "Qwen/Qwen-Image", []),
    ("qwen_image_cache_dit", "Qwen/Qwen-Image", ["--cache-backend", "cache_dit"]),
    ("qwen_image_tea_cache", "Qwen/Qwen-Image", ["--cache-backend", "tea_cache"]),
    ("qwen_image_ulysses2", "Qwen/Qwen-Image", ["--ulysses-degree", "2"]),
    ("qwen_image_ring2", "Qwen/Qwen-Image", ["--ring-degree", "2"]),
    ("qwen_image_cfg_par2", "Qwen/Qwen-Image", ["--cfg-parallel-size", "2"]),
    ("qwen_image_tp2", "Qwen/Qwen-Image", ["--tensor-parallel-size", "2"]),
    ("zimage", "Tongyi-MAI/Z-Image-Turbo", []),
    ("ovis_image", "AIDC-AI/Ovis-Image-7B", []),
    ("longcat_image", "meituan-longcat/LongCat-Image", []),
    ("sd3", "stabilityai/stable-diffusion-3.5-medium", []),
    ("flux1_dev", "black-forest-labs/FLUX.1-dev", []),
    ("flux2_klein", "black-forest-labs/FLUX.2-klein-9B", []),
    ("glm_image", "zai-org/GLM-Image", []),
    ("omnigen2", "OmniGen2/OmniGen2", []),
    ("nextstep", "stepfun-ai/NextStep-1.1", []),
]


@pytest.mark.parametrize("case_id,model,extra_args", _T2I_CASES, ids=[c[0] for c in _T2I_CASES])
def test_offline_text_to_image(case_id, model, extra_args, model_prefix, output_dir):
    out = output_dir / f"t2i_{case_id}.png"
    run_script(
        T2I_SCRIPT,
        "--model", f"{model_prefix}{model}",
        "--prompt", _T2I_PROMPT,
        *_T2I_COMMON,
        *extra_args,
        "--output", str(out),
    )  # fmt: skip
    assert_image_valid(out, width=1024, height=1024)


# ---------------------------------------------------------------------------
# Offline: Image Editing
# ---------------------------------------------------------------------------

_EDIT_PROMPT = "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'"
_MULTI_PROMPT = "Make the bear ride the horse,  with smooth blending between them"
_I2I_COMMON = ["--num-inference-steps", "50", "--cfg-scale", "4.0"]

# Single-image editing: (case_id, model, extra_args)
_I2I_SINGLE_CASES = [
    ("qwen_image_edit", "Qwen/Qwen-Image-Edit", []),
    ("longcat_image_edit", "meituan-longcat/LongCat-Image-Edit", []),
    ("flux2_klein_edit", "black-forest-labs/FLUX.2-klein-9B", []),
    ("glm_image_edit", "zai-org/GLM-Image", []),
    ("omnigen2_edit", "OmniGen2/OmniGen2", []),
]


@pytest.mark.parametrize("case_id,model,extra_args", _I2I_SINGLE_CASES, ids=[c[0] for c in _I2I_SINGLE_CASES])
def test_offline_image_edit_single(case_id, model, extra_args, model_prefix, output_dir, bear):
    out = output_dir / f"i2i_{case_id}.png"
    run_script(
        I2I_SCRIPT,
        "--model", f"{model_prefix}{model}",
        "--prompt", _EDIT_PROMPT,
        "--image", bear,
        *_I2I_COMMON,
        *extra_args,
        "--output", str(out),
    )  # fmt: skip
    assert_image_valid(out)


# Multi-image editing: (case_id, model, extra_args)
_I2I_MULTI_CASES = [
    ("qwen_image_edit_2509", "Qwen/Qwen-Image-Edit-2509", []),
    ("omnigen2_multi", "OmniGen2/OmniGen2", []),
]


@pytest.mark.parametrize("case_id,model,extra_args", _I2I_MULTI_CASES, ids=[c[0] for c in _I2I_MULTI_CASES])
def test_offline_image_edit_multi(case_id, model, extra_args, model_prefix, output_dir, bear, horse):
    out = output_dir / f"i2i_multi_{case_id}.png"
    run_script(
        I2I_SCRIPT,
        "--model", f"{model_prefix}{model}",
        "--prompt", _MULTI_PROMPT,
        "--image", bear, horse,
        "--guidance-scale", "1.0",
        *_I2I_COMMON,
        *extra_args,
        "--output", str(out),
    )  # fmt: skip
    assert_image_valid(out)


def test_offline_image_layered(model_prefix, output_dir, bear):
    """Qwen-Image-Layered decomposes an image into N RGBA layer outputs."""
    n_layers = 4
    out_prefix = output_dir / "qwen_image_layered"
    run_script(
        I2I_SCRIPT,
        "--model", f"{model_prefix}Qwen/Qwen-Image-Layered",
        "--prompt", "Decompose the image into layered RGBA outputs",
        "--image", bear,
        "--layers", str(n_layers),
        "--color-format", "RGBA",
        *_I2I_COMMON,
        "--output", str(out_prefix),
    )  # fmt: skip
    # image_edit.py saves layers as {prefix}_{i}.png (no extension on prefix)
    for i in range(n_layers):
        layer = out_prefix.parent / f"{out_prefix.name}_{i}.png"
        assert_image_valid(layer)


# ---------------------------------------------------------------------------
# Offline: Text-to-Video  (Wan2.2 T2V-A14B)
# ---------------------------------------------------------------------------

_WAN_T2V_NUM_FRAMES = 32
_WAN_T2V_FPS = 16
_T2V_PROMPT = "Two anthropomorphic cats in comfy boxing gear fight intensely on a spotlighted stage."


def test_offline_text_to_video(model_prefix, output_dir):
    out = output_dir / "wan22_t2v.mp4"
    run_script(
        T2V_SCRIPT,
        "--model", f"{model_prefix}Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "--prompt", _T2V_PROMPT,
        "--negative-prompt", "",
        "--height", "480",
        "--width", "640",
        "--num-frames", str(_WAN_T2V_NUM_FRAMES),
        "--guidance-scale", "4.0",
        "--guidance-scale-high", "3.0",
        "--num-inference-steps", "40",
        "--enable-cpu-offload",
        "--fps", str(_WAN_T2V_FPS),
        "--output", str(out),
    )  # fmt: skip
    assert_video_valid(out, width=640, height=480, num_frames=_WAN_T2V_NUM_FRAMES)


# ---------------------------------------------------------------------------
# Offline: Image-to-Video  (Wan2.2 TI2V-5B)
# ---------------------------------------------------------------------------

_WAN_I2V_NUM_FRAMES = 48
_WAN_I2V_FPS = 16
_I2V_PROMPT = "A bear playing with yarn, smooth motion"


def test_offline_image_to_video(model_prefix, output_dir, bear):
    out = output_dir / "wan22_ti2v.mp4"
    run_script(
        I2V_SCRIPT,
        "--model", f"{model_prefix}Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "--prompt", _I2V_PROMPT,
        "--image", bear,
        "--negative-prompt", "",
        "--height", "480",
        "--width", "832",
        "--num-frames", str(_WAN_I2V_NUM_FRAMES),
        "--guidance-scale", "4.0",
        "--num-inference-steps", "40",
        "--flow-shift", "12.0",
        "--enable-cpu-offload",
        "--fps", str(_WAN_I2V_FPS),
        "--output", str(out),
    )  # fmt: skip
    assert_video_valid(out, width=832, height=480, num_frames=_WAN_I2V_NUM_FRAMES)


# ---------------------------------------------------------------------------
# Offline: Text-to-Audio  (Stable Audio Open 1.0)
# ---------------------------------------------------------------------------

_AUDIO_LENGTH_S = 10.0
_AUDIO_SAMPLE_RATE = 44100
_AUDIO_CHANNELS = 2  # stable-audio-open produces stereo output


def test_offline_text_to_audio(model_prefix, output_dir):
    out = output_dir / "stable_audio.wav"
    run_script(
        T2A_SCRIPT,
        "--model", f"{model_prefix}stabilityai/stable-audio-open-1.0",
        "--prompt", "The sound of a hammer hitting a wooden surface.",
        "--negative-prompt", "Low quality.",
        "--seed", "42",
        "--guidance-scale", "7.0",
        "--audio-length", str(_AUDIO_LENGTH_S),
        "--num-inference-steps", "100",
        "--num-waveforms", "1",
        "--output", str(out),
    )  # fmt: skip
    assert_audio_valid(out, sample_rate=_AUDIO_SAMPLE_RATE, channels=_AUDIO_CHANNELS, duration_s=_AUDIO_LENGTH_S)


# ---------------------------------------------------------------------------
# Offline: BAGEL 2-stage
# ---------------------------------------------------------------------------

_BAGEL_CASES = [
    ("text2img", ["--prompts", "A cute cat", "--modality", "text2img"], False),
    ("img2img", ["--prompts", _EDIT_PROMPT, "--modality", "img2img"], True),
]


@pytest.mark.parametrize("case_id,extra_args,needs_image", _BAGEL_CASES, ids=[c[0] for c in _BAGEL_CASES])
def test_offline_bagel(case_id, extra_args, needs_image, model_prefix, bear):
    args = list(extra_args)
    if needs_image:
        args += ["--image-path", bear]
    run_script(
        BAGEL_SCRIPT,
        "--model", f"{model_prefix}ByteDance-Seed/BAGEL-7B-MoT",
        *args,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Online: Z-Image-Turbo
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zimage_server(model_prefix):
    with OmniServer(f"{model_prefix}Tongyi-MAI/Z-Image-Turbo", []) as server:
        yield server


def test_online_zimage_images_generations(zimage_server):
    """POST /v1/images/generations → 1024×1024 PNG."""
    resp = requests.post(
        f"http://{zimage_server.host}:{zimage_server.port}/v1/images/generations",
        json={
            "prompt": _T2I_PROMPT,
            "size": "1024x1024",
            "seed": 42,
        },
        timeout=600,
    )
    assert resp.status_code == 200
    img = decode_b64_png(resp.json()["data"][0]["b64_json"])
    assert img.size == (1024, 1024)


def test_online_zimage_chat_completions(zimage_server):
    """POST /v1/chat/completions → 1024×1024 PNG."""
    resp = requests.post(
        f"http://{zimage_server.host}:{zimage_server.port}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": _T2I_PROMPT}],
            "extra_body": {"height": 1024, "width": 1024, "num_inference_steps": 50, "true_cfg_scale": 4.0, "seed": 42},
        },
        timeout=600,
    )
    assert resp.status_code == 200
    data_url = resp.json()["choices"][0]["message"]["content"][0]["image_url"]["url"]
    img = decode_b64_png(data_url.split(",", 1)[1])
    assert img.size == (1024, 1024)


# ---------------------------------------------------------------------------
# Online: Qwen-Image-Edit
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen_edit_server(model_prefix):
    with OmniServer(f"{model_prefix}Qwen/Qwen-Image-Edit", []) as server:
        yield server


def test_online_qwen_image_edit(qwen_edit_server, assets_dir):
    """POST /v1/chat/completions with base64 image → valid PNG."""
    bear_b64 = base64.b64encode((assets_dir / "qwen-bear.png").read_bytes()).decode()
    resp = requests.post(
        f"http://{qwen_edit_server.host}:{qwen_edit_server.port}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _EDIT_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bear_b64}"}},
                    ],
                }
            ],
            "extra_body": {"num_inference_steps": 50, "guidance_scale": 1, "seed": 42},
        },
        timeout=600,
    )
    assert resp.status_code == 200
    data_url = resp.json()["choices"][0]["message"]["content"][0]["image_url"]["url"]
    img = decode_b64_png(data_url.split(",", 1)[1])
    assert img.width > 0 and img.height > 0


# ---------------------------------------------------------------------------
# Online: BAGEL
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bagel_server(model_prefix):
    with OmniServer(f"{model_prefix}ByteDance-Seed/BAGEL-7B-MoT", []) as server:
        yield server


def test_online_bagel_text2img(bagel_server, tmp_path):
    """openai_chat_client.py text2img → valid PNG output."""
    out = tmp_path / "bagel_online.png"
    run_script(
        BAGEL_CLIENT,
        "--prompt", _T2I_PROMPT,
        "--modality", "text2img",
        "--output", str(out),
        "--server", f"http://{bagel_server.host}:{bagel_server.port}",
    )  # fmt: skip
    assert_image_valid(out)


# ---------------------------------------------------------------------------
# Online: Wan2.2 T2V
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wan_t2v_server(model_prefix):
    with OmniServer(f"{model_prefix}Wan-AI/Wan2.2-T2V-A14B-Diffusers", []) as server:
        yield server


def test_online_wan_t2v(wan_t2v_server, tmp_path):
    """POST /v1/videos → MP4 with width=640, height=480, 32 frames @ 16 fps."""
    resp = requests.post(
        f"http://{wan_t2v_server.host}:{wan_t2v_server.port}/v1/videos",
        data={
            "prompt": _T2V_PROMPT,
            "negative_prompt": "",
            "width": "640",
            "height": "480",
            "num_frames": "32",
            "fps": "16",
            "num_inference_steps": "40",
            "guidance_scale": "4.0",
            "guidance_scale_2": "3.0",
            "seed": "42",
        },
        headers={"Accept": "application/json"},
        timeout=600,
    )
    assert resp.status_code == 200
    mp4_bytes = base64.b64decode(resp.json()["data"][0]["b64_json"])
    out = tmp_path / "wan22_t2v_online.mp4"
    out.write_bytes(mp4_bytes)
    assert_video_valid(out, width=640, height=480, num_frames=32)


# ---------------------------------------------------------------------------
# Online: Wan2.2 TI2V
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wan_ti2v_server(model_prefix):
    with OmniServer(f"{model_prefix}Wan-AI/Wan2.2-TI2V-5B-Diffusers", []) as server:
        yield server


def test_online_wan_ti2v(wan_ti2v_server, assets_dir, tmp_path):
    """POST /v1/videos with reference image → MP4 with width=832, height=480, 48 frames @ 16 fps."""
    bear_path = assets_dir / "qwen-bear.png"
    with open(bear_path, "rb") as img_file:
        resp = requests.post(
            f"http://{wan_ti2v_server.host}:{wan_ti2v_server.port}/v1/videos",
            data={
                "prompt": _I2V_PROMPT,
                "negative_prompt": "",
                "width": "832",
                "height": "480",
                "num_frames": "48",
                "fps": "16",
                "num_inference_steps": "40",
                "guidance_scale": "4.0",
                "flow_shift": "12.0",
                "seed": "42",
            },
            files={"input_reference": ("qwen-bear.png", img_file, "image/png")},
            headers={"Accept": "application/json"},
            timeout=600,
        )
    assert resp.status_code == 200
    mp4_bytes = base64.b64decode(resp.json()["data"][0]["b64_json"])
    out = tmp_path / "wan22_ti2v_online.mp4"
    out.write_bytes(mp4_bytes)
    assert_video_valid(out, width=832, height=480, num_frames=48)
