"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following models:
- Qwen-Image-Edit: single image input
- Qwen-Image-Edit-2509: two image inputs
"""

import time

import pytest
import requests

from tests.conftest import assert_image_valid, decode_b64_image, generate_synthetic_image
from tests.utils import hardware_marks

EDIT_PROMPT = "Transform this image of colorful geometric shapes into a Piet Mondrian style abstract painting."
MULTI_EDIT_PROMPT = (
    "Transform the first image of colorful geometric shapes into a Piet Mondrian style abstract painting. "
    "Transform the second image of colorful geometric shapes into a Vincent van Gogh style painting. "
    "Then juxtapose the two transformed images into a single artwork for visual contrast."
)
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"}, num_cards=2, parallel=True)


# This test file targets two models, so I write a helper function.
# If a similar test only involves one model, one can just define a global list variable.
def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            (
                model,
                None,
                ["--cache-backend", "tea_cache"],
            ),  # This tuple's structure corresponds to `omni_server`'s `request.param`. See tests/conftest.py.
            id="cache_tea_cache",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            (model, None, ["--cache-backend", "cache_dit"]), id="cache_cache_dit", marks=SINGLE_CARD_FEATURE_MARKS
        ),
        pytest.param((model, None, ["--ulysses-degree", "2"]), id="ulysses_2", marks=PARALLEL_FEATURE_MARKS),
        pytest.param((model, None, ["--ring", "2"]), id="ring_2", marks=PARALLEL_FEATURE_MARKS),
        pytest.param((model, None, ["--cfg-parallel-size", "2"]), id="cfg_parallel_2", marks=PARALLEL_FEATURE_MARKS),
        pytest.param((model, None, ["--enable-cpu-offload"]), id="cpu_offload", marks=SINGLE_CARD_FEATURE_MARKS),
        pytest.param(
            (model, None, ["--enable-layerwise-offload"]), id="layerwise_offload", marks=SINGLE_CARD_FEATURE_MARKS
        ),
        # pytest.param((model, None, ["--vae-use-slicing"]), id="vae_slicing", marks=SINGLE_CARD_FEATURE_MARKS),
        # pytest.param((model, None, ["--vae-use-tiling"]), id="vae_tiling", marks=SINGLE_CARD_FEATURE_MARKS),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("Qwen/Qwen-Image-Edit"),
    indirect=True,
)
def test_qwen_image_edit_single(omni_server):
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    start_time = time.perf_counter()

    resp = requests.post(
        f"http://{omni_server.host}:{omni_server.port}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EDIT_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "extra_body": {
                "height": 512,
                "width": 512,
                "num_inference_steps": 50,
                "guidance_scale": 1,
                "seed": 42,
            },
        },
        timeout=600,  # Same as the OpenAI python client's default timeout
    )
    assert resp.status_code == 200, f"Request failed: {resp.text}"

    e2e_latency = time.perf_counter() - start_time
    print(f"the avg e2e is: {e2e_latency}")

    data_url = resp.json()["choices"][0]["message"]["content"][0]["image_url"]["url"]
    img = decode_b64_image(data_url.split(",", 1)[1])
    assert_image_valid(img, width=512, height=512)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("Qwen/Qwen-Image-Edit-2509"),
    indirect=True,
)
def test_qwen_image_edit_multi(omni_server):
    image_data_url_1 = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"
    image_data_url_2 = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    start_time = time.perf_counter()

    resp = requests.post(
        f"http://{omni_server.host}:{omni_server.port}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": MULTI_EDIT_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_data_url_1}},
                        {"type": "image_url", "image_url": {"url": image_data_url_2}},
                    ],
                }
            ],
            "extra_body": {
                "height": 512,
                "width": 512,
                "num_inference_steps": 50,
                "guidance_scale": 1,
                "seed": 42,
            },
        },
        timeout=600,  # Same as the OpenAI python client's default timeout
    )
    assert resp.status_code == 200, f"Request failed: {resp.text}"

    e2e_latency = time.perf_counter() - start_time
    print(f"the avg e2e is: {e2e_latency}")

    data_url = resp.json()["choices"][0]["message"]["content"][0]["image_url"]["url"]
    img = decode_b64_image(data_url.split(",", 1)[1])
    assert_image_valid(img, width=512, height=512)
