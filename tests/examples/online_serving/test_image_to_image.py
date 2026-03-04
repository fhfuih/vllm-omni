"""
Online serving tests: image-to-image.

Covers:
  - Online Qwen-Image-Edit  (/v1/chat/completions with base64 image input)

Doc-linked tests (test_<subsection>_<id>) mirror every executable snippet in
docs/user_guide/examples/online_serving/image_to_image.md.
"""

import base64

import pytest
import requests

from tests.conftest import OmniServer
from tests.examples.conftest import (
    I2I_ONLINE_CLIENT,
    assert_image_valid,
    decode_b64_png,
    run_script_with_successful_return,
)

# ---------------------------------------------------------------------------
# Doc-linked tests: docs/user_guide/examples/online_serving/image_to_image.md
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen_edit_server(model_prefix):
    with OmniServer(f"{model_prefix}Qwen/Qwen-Image-Edit", []) as server:
        yield server


# --- ### Method 1: Using curl (Image Editing) ---


def test_method_1_using_curl_image_editing_1(qwen_edit_server, assets_dir):
    """POST /v1/chat/completions with base64 image and explicit params → valid PNG."""
    bear_b64 = base64.b64encode((assets_dir / "qwen-bear.png").read_bytes()).decode()
    resp = requests.post(
        f"http://{qwen_edit_server.host}:{qwen_edit_server.port}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Convert this image to watercolor style"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bear_b64}"}},
                    ],
                }
            ],
            "extra_body": {"height": 1024, "width": 1024, "num_inference_steps": 50, "guidance_scale": 1, "seed": 42},
        },
        timeout=600,
    )
    assert resp.status_code == 200
    data_url = resp.json()["choices"][0]["message"]["content"][0]["image_url"]["url"]
    img = decode_b64_png(data_url.split(",", 1)[1])
    assert img.width > 0 and img.height > 0


# --- ### Method 2: Using Python Client ---


def test_method_2_using_python_client_1(qwen_edit_server, output_dir, bear):
    """openai_chat_client.py single image → valid PNG."""
    out = output_dir / "doc-i2i-method_2_using_python_client_1.png"
    run_script_with_successful_return(
        I2I_ONLINE_CLIENT,
        "--input", bear,
        "--prompt", "Convert to oil painting style",
        "--output", str(out),
        "--server", f"http://{qwen_edit_server.host}:{qwen_edit_server.port}",
    )  # fmt: skip
    assert_image_valid(out)


@pytest.fixture(scope="module")
def qwen_edit_2509_server(model_prefix):
    with OmniServer(f"{model_prefix}Qwen/Qwen-Image-Edit-2509", []) as server:
        yield server


def test_method_2_using_python_client_2(qwen_edit_2509_server, output_dir, bear, horse):
    """openai_chat_client.py multi-image (Qwen-Image-Edit-2509 server) → valid PNG."""
    out = output_dir / "doc-i2i-method_2_using_python_client_2.png"
    run_script_with_successful_return(
        I2I_ONLINE_CLIENT,
        "--input", bear, horse,
        "--prompt", "Combine these images into a single scene",
        "--output", str(out),
        "--server", f"http://{qwen_edit_2509_server.host}:{qwen_edit_2509_server.port}",
    )  # fmt: skip
    assert_image_valid(out)
