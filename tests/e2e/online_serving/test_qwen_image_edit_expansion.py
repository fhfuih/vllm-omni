"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following models:
- Qwen-Image-Edit: single image input
- Qwen-Image-Edit-2509: two image inputs
"""

import time

import pytest
import requests

from tests.conftest import OmniServer, assert_image_valid, decode_b64_image, generate_synthetic_image
from tests.utils import hardware_marks

EDIT_PROMPT = "Transform this image of colorful geometric shapes into a Piet Mondrian style abstract painting."
MULTI_EDIT_PROMPT = (
    "Transform the first image of colorful geometric shapes into a Piet Mondrian style abstract painting. "
    "Transform the second image of colorful geometric shapes into a Vincent van Gogh style painting. "
    "Then juxtapose the two transformed images into a single artwork for visual contrast."
)
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"}, num_cards=2, parallel=True)


_DIFFUSION_FEATURE_CASES = [
    pytest.param("cache_tea_cache", ["--cache-backend", "tea_cache"], marks=SINGLE_CARD_FEATURE_MARKS),
    pytest.param("cache_cache_dit", ["--cache-backend", "cache_dit"], marks=SINGLE_CARD_FEATURE_MARKS),
    pytest.param("ulysses_2", ["--ulysses-degree", "2"], marks=PARALLEL_FEATURE_MARKS),
    pytest.param("ring_2", ["--ring", "2"], marks=PARALLEL_FEATURE_MARKS),
    pytest.param("cfg_parallel_2", ["--cfg-parallel-size", "2"], marks=PARALLEL_FEATURE_MARKS),
    pytest.param("cpu_offload", ["--enable-cpu-offload"], marks=SINGLE_CARD_FEATURE_MARKS),
    pytest.param("layerwise_offload", ["--enable-layerwise-offload"], marks=SINGLE_CARD_FEATURE_MARKS),
    # pytest.param("vae_slicing", ["--vae-use-slicing"], marks=SINGLE_CARD_FEATURE_MARKS),
    # pytest.param("vae_tiling", ["--vae-use-tiling"], marks=SINGLE_CARD_FEATURE_MARKS),
]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    ("case_id", "extra_args"),
    _DIFFUSION_FEATURE_CASES,
    ids=[c.values[0] for c in _DIFFUSION_FEATURE_CASES],
)
def test_qwen_image_edit_single(case_id, extra_args, model_prefix):
    model = f"{model_prefix}Qwen/Qwen-Image-Edit"
    with OmniServer(model, extra_args) as server:
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

        start_time = time.perf_counter()

        resp = requests.post(
            f"http://{server.host}:{server.port}/v1/chat/completions",
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
    ("case_id", "extra_args"),
    _DIFFUSION_FEATURE_CASES,
    ids=[c.values[0] for c in _DIFFUSION_FEATURE_CASES],
)
def test_qwen_image_edit_multi(case_id, extra_args, model_prefix):
    model = f"{model_prefix}Qwen/Qwen-Image-Edit-2509"
    with OmniServer(model, extra_args) as server:
        image_data_url_1 = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"
        image_data_url_2 = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

        start_time = time.perf_counter()

        resp = requests.post(
            f"http://{server.host}:{server.port}/v1/chat/completions",
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
