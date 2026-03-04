"""
Online serving tests: text-to-image.

Covers:
  - Online Z-Image-Turbo  (/v1/images/generations and /v1/chat/completions)
  - Online BAGEL text2img (via openai_chat_client.py)

Doc-linked tests (test_<subsection>_<id>) mirror every executable snippet in
docs/user_guide/examples/online_serving/text_to_image.md.
"""

import pytest
import requests

from tests.conftest import OmniServer
from tests.example.conftest import (
    T2I_ONLINE_CLIENT,
    assert_image_valid,
    decode_b64_png,
    run_script,
)

# ---------------------------------------------------------------------------
# Doc-linked tests: docs/user_guide/examples/online_serving/text_to_image.md
# ---------------------------------------------------------------------------


# Qwen/Qwen-Image server (different from Z-Image-Turbo used above)
@pytest.fixture(scope="module")
def qwen_image_server(model_prefix):
    with OmniServer(f"{model_prefix}Qwen/Qwen-Image", []) as server:
        yield server


# --- ### Method 1: Using curl ---


def test_method_1_using_curl_1(qwen_image_server):
    """POST /v1/chat/completions with extra_body params → 1024×1024 PNG."""
    resp = requests.post(
        f"http://{qwen_image_server.host}:{qwen_image_server.port}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "A beautiful landscape painting"}],
            "extra_body": {"height": 1024, "width": 1024, "num_inference_steps": 50, "true_cfg_scale": 4.0, "seed": 42},
        },
        timeout=600,
    )
    assert resp.status_code == 200
    data_url = resp.json()["choices"][0]["message"]["content"][0]["image_url"]["url"]
    img = decode_b64_png(data_url.split(",", 1)[1])
    assert img.size == (1024, 1024)


# --- ### Method 2: Using Python Client ---


def test_method_2_using_python_client_1(qwen_image_server, output_dir):
    """openai_chat_client.py → valid 1024×1024 PNG."""
    out = output_dir / "doc-t2i-method_2_using_python_client_1.png"
    run_script(
        T2I_ONLINE_CLIENT,
        "--prompt", "A beautiful landscape painting",
        "--output", str(out),
        "--server", f"http://{qwen_image_server.host}:{qwen_image_server.port}",
    )  # fmt: skip
    assert_image_valid(out)
