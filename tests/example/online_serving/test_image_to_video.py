"""
Online serving tests: image-to-video.

Covers:
  - Online Wan2.2 TI2V-5B  (/v1/videos with reference image)

Doc-linked tests (test_<subsection>_<id>) mirror every executable snippet in
docs/user_guide/examples/online_serving/image_to_video.md.
"""

import base64

import pytest
import requests

from tests.conftest import OmniServer
from tests.example.conftest import assert_video_valid

_I2V_PROMPT = "A bear playing with yarn, smooth motion"


# ---------------------------------------------------------------------------
# Doc-linked tests: docs/user_guide/examples/online_serving/image_to_video.md
# (doc uses Wan2.2-I2V-A14B-Diffusers, a MoE model distinct from TI2V-5B above)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wan_i2v_a14b_server(model_prefix):
    with OmniServer(f"{model_prefix}Wan-AI/Wan2.2-I2V-A14B-Diffusers", []) as server:
        yield server


# --- ### Method 1: Using curl ---


def test_method_1_using_curl_1(wan_i2v_a14b_server, assets_dir, output_dir):
    """POST /v1/videos with image and full params → MP4 width=832, height=480, 33 frames."""
    bear_path = assets_dir / "qwen-bear.png"
    with open(bear_path, "rb") as img_file:
        resp = requests.post(
            f"http://{wan_i2v_a14b_server.host}:{wan_i2v_a14b_server.port}/v1/videos",
            data={
                "prompt": _I2V_PROMPT,
                "negative_prompt": "low quality, blurry, static",
                "width": "832",
                "height": "480",
                "num_frames": "33",
                "fps": "16",
                "num_inference_steps": "40",
                "guidance_scale": "1.0",
                "guidance_scale_2": "1.0",
                "boundary_ratio": "0.875",
                "flow_shift": "12.0",
                "seed": "42",
            },
            files={"input_reference": ("qwen-bear.png", img_file, "image/png")},
            headers={"Accept": "application/json"},
            timeout=600,
        )
    assert resp.status_code == 200
    mp4_bytes = base64.b64decode(resp.json()["data"][0]["b64_json"])
    out = output_dir / "doc-i2v-method_1_using_curl_1.mp4"
    out.write_bytes(mp4_bytes)
    assert_video_valid(out, width=832, height=480, num_frames=33)


# --- ### Required Fields ---


def test_required_fields_1(wan_i2v_a14b_server, assets_dir, output_dir):
    """POST /v1/videos with only required fields (prompt + image) → non-empty MP4."""
    bear_path = assets_dir / "qwen-bear.png"
    with open(bear_path, "rb") as img_file:
        resp = requests.post(
            f"http://{wan_i2v_a14b_server.host}:{wan_i2v_a14b_server.port}/v1/videos",
            data={
                "prompt": _I2V_PROMPT,
                "negative_prompt": "low quality, blurry, static",
            },
            files={"input_reference": ("qwen-bear.png", img_file, "image/png")},
            headers={"Accept": "application/json"},
            timeout=600,
        )
    assert resp.status_code == 200
    mp4_bytes = base64.b64decode(resp.json()["data"][0]["b64_json"])
    out = output_dir / "doc-i2v-required_fields_1.mp4"
    out.write_bytes(mp4_bytes)
    assert len(mp4_bytes) > 0


# --- ### Generation with Parameters ---


def test_generation_with_parameters_1(wan_i2v_a14b_server, assets_dir, output_dir):
    """POST /v1/videos with all generation parameters → MP4 width=832, height=480, 33 frames."""
    bear_path = assets_dir / "qwen-bear.png"
    with open(bear_path, "rb") as img_file:
        resp = requests.post(
            f"http://{wan_i2v_a14b_server.host}:{wan_i2v_a14b_server.port}/v1/videos",
            data={
                "prompt": _I2V_PROMPT,
                "negative_prompt": "low quality, blurry, static",
                "width": "832",
                "height": "480",
                "num_frames": "33",
                "fps": "16",
                "num_inference_steps": "40",
                "guidance_scale": "1.0",
                "guidance_scale_2": "1.0",
                "boundary_ratio": "0.875",
                "flow_shift": "12.0",
                "seed": "42",
            },
            files={"input_reference": ("qwen-bear.png", img_file, "image/png")},
            headers={"Accept": "application/json"},
            timeout=600,
        )
    assert resp.status_code == 200
    mp4_bytes = base64.b64decode(resp.json()["data"][0]["b64_json"])
    out = output_dir / "doc-i2v-generation_with_parameters_1.mp4"
    out.write_bytes(mp4_bytes)
    assert_video_valid(out, width=832, height=480, num_frames=33)
