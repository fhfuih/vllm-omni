"""
Online serving tests: text-to-video.

Covers:
  - Online Wan2.2 T2V-A14B  (/v1/videos)

Doc-linked tests (test_<subsection>_<id>) mirror every executable snippet in
docs/user_guide/examples/online_serving/text_to_video.md.
"""

import base64

import imageio.v3 as iio
import pytest
import requests

from tests.conftest import OmniServer
from tests.examples.conftest import assert_video_valid

# ---------------------------------------------------------------------------
# Doc-linked tests: docs/user_guide/examples/online_serving/text_to_video.md
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wan_t2v_server(model_prefix):
    with OmniServer(f"{model_prefix}Wan-AI/Wan2.2-T2V-A14B-Diffusers", []) as server:
        yield server


# --- ### Method 1: Using curl ---


def test_method_1_using_curl_1(wan_t2v_server, output_dir):
    """POST /v1/videos with full params from doc → MP4 with width=832, height=480, 33 frames."""
    resp = requests.post(
        f"http://{wan_t2v_server.host}:{wan_t2v_server.port}/v1/videos",
        data={
            "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
            "negative_prompt": "",
            "width": "832",
            "height": "480",
            "num_frames": "33",
            "fps": "16",
            "num_inference_steps": "40",
            "guidance_scale": "4.0",
            "guidance_scale_2": "4.0",
            "boundary_ratio": "0.875",
            "seed": "42",
        },
        headers={"Accept": "application/json"},
        timeout=600,
    )
    assert resp.status_code == 200
    mp4_bytes = base64.b64decode(resp.json()["data"][0]["b64_json"])
    out = output_dir / "doc-t2v-method_1_using_curl_1.mp4"
    out.write_bytes(mp4_bytes)
    assert_video_valid(out, width=832, height=480, num_frames=33)


# --- ### Simple Text-to-Video Generation ---


def test_simple_text_to_video_generation_1(wan_t2v_server, output_dir):
    """POST /v1/videos with prompt only → non-empty MP4."""
    resp = requests.post(
        f"http://{wan_t2v_server.host}:{wan_t2v_server.port}/v1/videos",
        data={"prompt": "A cinematic view of a futuristic city at sunset"},
        headers={"Accept": "application/json"},
        timeout=600,
    )
    assert resp.status_code == 200
    mp4_bytes = base64.b64decode(resp.json()["data"][0]["b64_json"])
    out = output_dir / "doc-t2v-simple_text_to_video_generation_1.mp4"
    out.write_bytes(mp4_bytes)
    frames = iio.imread(str(out), plugin="pyav", index=None)
    assert frames.shape[0] >= 1


# --- ### Generation with Parameters ---


def test_generation_with_parameters_1(wan_t2v_server, output_dir):
    """POST /v1/videos with all generation params → MP4 with width=832, height=480, 33 frames."""
    resp = requests.post(
        f"http://{wan_t2v_server.host}:{wan_t2v_server.port}/v1/videos",
        data={
            "prompt": "A cinematic view of a futuristic city at sunset",
            "width": "832",
            "height": "480",
            "num_frames": "33",
            "negative_prompt": "low quality, blurry, static",
            "fps": "16",
            "num_inference_steps": "40",
            "guidance_scale": "4.0",
            "guidance_scale_2": "4.0",
            "boundary_ratio": "0.875",
            "flow_shift": "5.0",
            "seed": "42",
        },
        headers={"Accept": "application/json"},
        timeout=600,
    )
    assert resp.status_code == 200
    mp4_bytes = base64.b64decode(resp.json()["data"][0]["b64_json"])
    out = output_dir / "doc-t2v-generation_with_parameters_1.mp4"
    out.write_bytes(mp4_bytes)
    assert_video_valid(out, width=832, height=480, num_frames=33)
