"""
Online serving tests: image-to-image.

Covers:
  - Online Qwen-Image-Edit  (/v1/chat/completions with base64 image input)

Doc-linked tests (test_<subsection>_<id>) mirror every executable snippet in
docs/user_guide/examples/online_serving/image_to_image.md.
"""

import base64
import json

import pytest

from tests.conftest import OmniServer
from tests.examples.conftest import (
    I2I_ONLINE_CLIENT,
    assert_image_valid,
    run_command_with_successful_return,
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


def test_method_1_using_curl_image_editing_1(qwen_edit_server, assets_dir, output_dir):
    url = f"http://{qwen_edit_server.host}:{qwen_edit_server.port}"
    out = output_dir / "doc-i2i-method_1_using_curl_image_editing_1.png"
    run_command_with_successful_return(
        [
            f"SERVER='{url}'",
            "bash",
            "run_curl_image_edit.sh",
            "input.png",
            "'Convert this image to watercolor style'",
            out,
        ]
    )
    assert_image_valid(out)


def test_method_1_using_curl_image_editing_2(qwen_edit_server, assets_dir, output_dir):
    url = f"http://{qwen_edit_server.host}:{qwen_edit_server.port}/v1/chat/completions"
    out = output_dir / "doc-i2i-method_1_using_curl_image_editing_2.png"
    bear_b64 = base64.b64encode((assets_dir / "qwen-bear.png").read_bytes()).decode()
    req_file = output_dir / "req_i2i_curl_2.json"
    req_file.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Convert this image to watercolor style"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bear_b64}"}},
                        ],
                    }
                ],
                "extra_body": {
                    "height": 1024,
                    "width": 1024,
                    "num_inference_steps": 50,
                    "guidance_scale": 1,
                    "seed": 42,
                },
            }
        )
    )
    run_command_with_successful_return([
        "bash", "-c",
        f"curl -s '{url}'"
        " -H 'Content-Type: application/json'"
        f" -d '@{req_file}'"
        " | jq -r '.choices[0].message.content[0].image_url.url'"
        f" | cut -d',' -f2- | base64 -d > '{out}'",
    ])  # fmt: skip
    assert_image_valid(out)


# --- ### Method 2: Using Python Client ---


def test_method_2_using_python_client_1(qwen_edit_server, output_dir, bear):
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
    out = output_dir / "doc-i2i-method_2_using_python_client_2.png"
    run_script_with_successful_return(
        I2I_ONLINE_CLIENT,
        "--input", bear, horse,
        "--prompt", "Combine these images into a single scene",
        "--output", str(out),
        "--server", f"http://{qwen_edit_2509_server.host}:{qwen_edit_2509_server.port}",
    )  # fmt: skip
    assert_image_valid(out)
