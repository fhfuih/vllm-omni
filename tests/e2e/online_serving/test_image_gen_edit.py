# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online serving test for Qwen-Image-Edit-2509 multi-image input.
"""

import base64
import os
import threading
from io import BytesIO
from typing import Any

import openai
import pytest
import requests
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionUserMessageParam
from PIL import Image
from vllm.assets.image import ImageAsset

from tests.conftest import OmniServer
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# Increase timeout for downloading assets from S3 (default 5s is too short for CI)
os.environ.setdefault("VLLM_IMAGE_FETCH_TIMEOUT", "60")

models = ["Qwen/Qwen-Image-Edit-2509"]
test_params = models
t2i_models = ["Tongyi-MAI/Z-Image-Turbo"]


@pytest.fixture
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights."""
    model = request.param
    with OmniServer(model, ["--num-gpus", "1"]) as server:
        yield server


@pytest.fixture
def client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


@pytest.fixture(scope="session")
def base64_encoded_images() -> list[str]:
    """Base64 encoded PNG images for testing."""
    images = [
        ImageAsset("cherry_blossom").pil_image.convert("RGB"),
        ImageAsset("stop_sign").pil_image.convert("RGB"),
    ]
    encoded: list[str] = []
    for img in images:
        with BytesIO() as buffer:
            img.save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return encoded


def dummy_messages_from_image_data(
    image_data_urls: list[str],
    content_text: str = "Combine these two images into one scene.",
) -> list[ChatCompletionUserMessageParam]:
    """Create messages with image data URLs for OpenAI API."""
    content: list[ChatCompletionContentPartParam] = [{"type": "text", "text": content_text}]
    for image_url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return [{"role": "user", "content": content}]


def _extract_image_data_url(message_content) -> str:
    assert isinstance(message_content, list) and len(message_content) >= 1
    content_part = message_content[0]
    if isinstance(content_part, dict):
        image_url = content_part.get("image_url", {}).get("url", "")
    else:
        image_url_obj = getattr(content_part, "image_url", None)
        if isinstance(image_url_obj, dict):
            image_url = image_url_obj.get("url", "")
        else:
            image_url = getattr(image_url_obj, "url", "")
    assert isinstance(image_url, str) and image_url
    return image_url


def _decode_data_url_to_image_bytes(data_url: str) -> bytes:
    assert data_url.startswith("data:image")
    _, b64_data = data_url.split(",", 1)
    return base64.b64decode(b64_data)


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_i2i_multi_image_input_qwen_image_edit_2509(
    omni_server,
    base64_encoded_images: list[str],
) -> None:
    """Test multi-image input editing via OpenAI API with concurrent requests."""
    image_data_urls = [f"data:image/png;base64,{img}" for img in base64_encoded_images]
    messages = dummy_messages_from_image_data(image_data_urls)

    barrier = threading.Barrier(2)
    results: list[tuple[int, int]] = []

    def _call_chat(width: int, height: int) -> None:
        client = openai.OpenAI(
            base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
            api_key="EMPTY",
        )
        barrier.wait()
        chat_completion = client.chat.completions.create(
            model=omni_server.model,
            messages=messages,
            extra_body={
                "height": height,
                "width": width,
                "num_inference_steps": 2,
                "guidance_scale": 0.0,
                "seed": 42,
            },
        )

        assert len(chat_completion.choices) == 1
        choice = chat_completion.choices[0]
        assert choice.finish_reason == "stop"
        assert choice.message.role == "assistant"

        image_data_url = _extract_image_data_url(choice.message.content)
        image_bytes = _decode_data_url_to_image_bytes(image_data_url)
        img = Image.open(BytesIO(image_bytes))
        img.load()
        results.append(img.size)

    threads = [
        threading.Thread(target=_call_chat, args=(1248, 832)),
        threading.Thread(target=_call_chat, args=(1024, 768)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # TODO @ZJY
    # assert (1248, 832) in results
    # assert (1024, 768) in results


@pytest.mark.parametrize("omni_server", t2i_models, indirect=True)
def test_t2i_concurrent_requests_different_sizes(omni_server) -> None:
    """Test /v1/images/generations concurrent requests with different sizes."""
    base_url = f"http://{omni_server.host}:{omni_server.port}"
    url = f"{base_url}/v1/images/generations"

    barrier = threading.Barrier(2)
    results: list[tuple[int, int]] = []

    def _call_generate(size: str) -> None:
        payload: dict[str, Any] = {
            "prompt": "cute cat playing with a ball",
            "n": 1,
            "size": size,
            "response_format": "b64_json",
            "num_inference_steps": 2,
        }
        barrier.wait()
        response = requests.post(url, json=payload, timeout=120)
        assert response.status_code == 200
        data = response.json()
        image_b64 = data["data"][0]["b64_json"]
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(BytesIO(image_bytes))
        img.load()
        results.append(img.size)

    threads = [
        threading.Thread(target=_call_generate, args=("512x512",)),
        threading.Thread(target=_call_generate, args=("768x512",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert (512, 512) in results
    assert (768, 512) in results
