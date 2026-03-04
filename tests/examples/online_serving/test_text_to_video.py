"""
Online serving tests: text-to-video.

Covers:
  - Online Wan2.2 T2V-A14B  (/v1/videos)

Doc-linked tests (test_<subsection>_<id>) mirror every executable snippet in
docs/user_guide/examples/online_serving/text_to_video.md.
"""

import imageio.v3 as iio
import pytest

from tests.conftest import OmniServer
from tests.examples.conftest import assert_video_valid, run_command_with_successful_return

# ---------------------------------------------------------------------------
# Doc-linked tests: docs/user_guide/examples/online_serving/text_to_video.md
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wan_t2v_server(model_prefix):
    with OmniServer(f"{model_prefix}Wan-AI/Wan2.2-T2V-A14B-Diffusers", []) as server:
        yield server


# --- ### Method 1: Using curl ---


def test_method_1_using_curl_1(wan_t2v_server, output_dir):
    url = f"http://{wan_t2v_server.host}:{wan_t2v_server.port}/v1/videos"
    out = output_dir / "doc-t2v-method_1_using_curl_1.mp4"
    run_command_with_successful_return([
        "bash", "-c",
        f"curl -s '{url}'"
        " -H 'Accept: application/json'"
        " -F 'prompt=Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.'"
        " -F 'width=832'"
        " -F 'height=480'"
        " -F 'num_frames=33'"
        " -F 'negative_prompt=色调艳丽 ，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'"
        " -F 'fps=16'"
        " -F 'num_inference_steps=40'"
        " -F 'guidance_scale=4.0'"
        " -F 'guidance_scale_2=4.0'"
        " -F 'boundary_ratio=0.875'"
        " -F 'seed=42'"
        f" | jq -r '.data[0].b64_json' | base64 -d > '{out}'",
    ])  # fmt: skip
    assert_video_valid(out, width=832, height=480, num_frames=33)


# --- ### Simple Text-to-Video Generation ---


def test_simple_text_to_video_generation_1(wan_t2v_server, output_dir):
    url = f"http://{wan_t2v_server.host}:{wan_t2v_server.port}/v1/videos"
    out = output_dir / "doc-t2v-simple_text_to_video_generation_1.mp4"
    run_command_with_successful_return([
        "bash", "-c",
        f"curl -X POST '{url}'"
        " -F 'prompt=A cinematic view of a futuristic city at sunset'"
        f" | jq -r '.data[0].b64_json' | base64 -d > '{out}'",
    ])  # fmt: skip
    frames = iio.imread(str(out), plugin="pyav", index=None)
    assert frames.shape[0] >= 1


# --- ### Generation with Parameters ---


def test_generation_with_parameters_1(wan_t2v_server, output_dir):
    url = f"http://{wan_t2v_server.host}:{wan_t2v_server.port}/v1/videos"
    out = output_dir / "doc-t2v-generation_with_parameters_1.mp4"
    run_command_with_successful_return([
        "bash", "-c",
        f"curl -X POST '{url}'"
        " -F 'prompt=A cinematic view of a futuristic city at sunset'"
        " -F 'width=832'"
        " -F 'height=480'"
        " -F 'num_frames=33'"
        " -F 'negative_prompt=low quality, blurry, static'"
        " -F 'fps=16'"
        " -F 'num_inference_steps=40'"
        " -F 'guidance_scale=4.0'"
        " -F 'guidance_scale_2=4.0'"
        " -F 'boundary_ratio=0.875'"
        " -F 'flow_shift=5.0'"
        " -F 'seed=42'"
        f" | jq -r '.data[0].b64_json' | base64 -d > '{out}'",
    ])  # fmt: skip
    assert_video_valid(out, width=832, height=480, num_frames=33)
