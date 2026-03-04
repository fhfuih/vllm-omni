"""
Online serving tests: image-to-video.

Covers:
  - Online Wan2.2 TI2V-5B  (/v1/videos with reference image)

Doc-linked tests (test_<subsection>_<id>) mirror every executable snippet in
docs/user_guide/examples/online_serving/image_to_video.md.
"""

import pytest

from tests.conftest import OmniServer
from tests.examples.conftest import assert_video_valid, run_command_with_successful_return

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


def test_method_1_using_curl_1(wan_i2v_a14b_server, bear, output_dir):
    url = f"http://{wan_i2v_a14b_server.host}:{wan_i2v_a14b_server.port}/v1/videos"
    out = output_dir / "doc-i2v-method_1_using_curl_1.mp4"
    run_command_with_successful_return(
        [
            f"SERVER='{url}'",
            f"INPUT_IMAGE='{bear}'",
            f"OUTPUT_PATH='{out}'",
            "bash",
            "run_curl_image_to_video.sh",
        ]
    )
    assert_video_valid(out, width=832, height=480, num_frames=33)


def test_method_1_using_curl_2(wan_i2v_a14b_server, bear, output_dir):
    url = f"http://{wan_i2v_a14b_server.host}:{wan_i2v_a14b_server.port}/v1/videos"
    out = output_dir / "doc-i2v-method_1_using_curl_2.mp4"
    run_command_with_successful_return([
        "bash", "-c",
        f"curl -X POST '{url}'"
        " -H 'Accept: application/json'"
        f" -F 'prompt={_I2V_PROMPT}'"
        " -F 'negative_prompt=low quality, blurry, static'"
        f" -F 'input_reference=@{bear}'"
        " -F 'width=832'"
        " -F 'height=480'"
        " -F 'num_frames=33'"
        " -F 'fps=16'"
        " -F 'num_inference_steps=40'"
        " -F 'guidance_scale=1.0'"
        " -F 'guidance_scale_2=1.0'"
        " -F 'boundary_ratio=0.875'"
        " -F 'flow_shift=12.0'"
        " -F 'seed=42'"
        f" | jq -r '.data[0].b64_json' | base64 -d > '{out}'",
    ])  # fmt: skip
    assert_video_valid(out, width=832, height=480, num_frames=33)


# --- ### Required Fields ---


def test_required_fields_1(wan_i2v_a14b_server, bear, output_dir):
    url = f"http://{wan_i2v_a14b_server.host}:{wan_i2v_a14b_server.port}/v1/videos"
    out = output_dir / "doc-i2v-required_fields_1.mp4"
    run_command_with_successful_return([
        "bash", "-c",
        f"curl -X POST '{url}'"
        f" -F 'prompt={_I2V_PROMPT}'"
        " -F 'negative_prompt=low quality, blurry, static'"
        f" -F 'input_reference=@{bear}'"
        f" | jq -r '.data[0].b64_json' | base64 -d > '{out}'",
    ])  # fmt: skip
    # Just verify the file is non-empty. Because this script only shows required fields, and optional fields are not set
    assert out.stat().st_size > 0


# --- ### Generation with Parameters ---


def test_generation_with_parameters_1(wan_i2v_a14b_server, bear, output_dir):
    url = f"http://{wan_i2v_a14b_server.host}:{wan_i2v_a14b_server.port}/v1/videos"
    out = output_dir / "doc-i2v-generation_with_parameters_1.mp4"
    run_command_with_successful_return([
        "bash", "-c",
        f"curl -X POST '{url}'"
        f" -F 'prompt={_I2V_PROMPT}'"
        " -F 'negative_prompt=low quality, blurry, static'"
        f" -F 'input_reference=@{bear}'"
        " -F 'width=832'"
        " -F 'height=480'"
        " -F 'num_frames=33'"
        " -F 'fps=16'"
        " -F 'num_inference_steps=40'"
        " -F 'guidance_scale=1.0'"
        " -F 'guidance_scale_2=1.0'"
        " -F 'boundary_ratio=0.875'"
        " -F 'flow_shift=12.0'"
        " -F 'seed=42'"
        f" | jq -r '.data[0].b64_json' | base64 -d > '{out}'",
    ])  # fmt: skip
    assert_video_valid(out, width=832, height=480, num_frames=33)
