"""
Comprehensive tests of diffusion features that are *ONLY* available in offline inference mode
(i.e., not supported in online serving) and are supported by the following models:
- Qwen-Image-Edit: single image input
- Qwen-Image-Edit-2509: two image inputs
"""

import time

import PIL
import pytest

from tests.conftest import assert_image_valid, generate_synthetic_image
from tests.utils import hardware_marks
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

EDIT_PROMPT = "Transform this image of colorful geometric shapes into a Piet Mondrian style abstract painting."
MULTI_EDIT_PROMPT = (
    "Transform the first image of colorful geometric shapes into a Piet Mondrian style abstract painting. "
    "Transform the second image of colorful geometric shapes into a Vincent van Gogh style painting. "
    "Then juxtapose the two transformed images into a single artwork for visual contrast."
)
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"}, num_cards=2, parallel=True)


# This test file targets two models, so I write a helper function.
# If a similar test only involves one model, one can just define a global list variable.
def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            model,
            None,
            {
                "parallel_config": DiffusionParallelConfig(
                    tensor_parallel_size=2,
                ),
            },
            marks=PARALLEL_FEATURE_MARKS,
            id="tp_2",
        ),
        pytest.param(
            model,
            None,
            {
                "parallel_config": DiffusionParallelConfig(
                    vae_patch_parallel_size=2,
                ),
            },
            id="vae_parallel_2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_runner",
    _get_diffusion_feature_cases("Qwen/Qwen-Image-Edit"),
    indirect=True,
)
def test_qwen_image_edit_single(omni_runner):
    synthetic_image = generate_synthetic_image(512, 512, save_to_file=True)
    pil_image = PIL.Image.open(synthetic_image["file_path"]).convert("RGB")

    start_time = time.perf_counter()

    output = omni_runner.generate(
        {
            "prompt": EDIT_PROMPT,
            "multi_modal_data": {
                "image": pil_image,
            },
        },
        OmniDiffusionSamplingParams(
            width=512,
            height=512,
            num_inference_steps=50,
            guidance_scale=1,
            seed=42,
        ),
    )

    e2e_latency = time.perf_counter() - start_time
    print(f"the avg e2e is: {e2e_latency}")

    img = output[0].request_output[0].images[0]
    assert_image_valid(img, width=512, height=512)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_runner",
    _get_diffusion_feature_cases("Qwen/Qwen-Image-Edit-2509"),
    indirect=True,
)
def test_qwen_image_edit_multi(omni_runner):
    synthetic_image_1 = generate_synthetic_image(512, 512, save_to_file=True)
    synthetic_image_2 = generate_synthetic_image(512, 512, save_to_file=True)
    pil_image_1 = PIL.Image.open(synthetic_image_1["file_path"]).convert("RGB")
    pil_image_2 = PIL.Image.open(synthetic_image_2["file_path"]).convert("RGB")

    start_time = time.perf_counter()

    output = omni_runner.generate(
        {
            "prompt": MULTI_EDIT_PROMPT,
            "multi_modal_data": {
                "image": [
                    pil_image_1,
                    pil_image_2,
                ],
            },
        },
        OmniDiffusionSamplingParams(
            width=512,
            height=512,
            num_inference_steps=50,
            guidance_scale=1,
            seed=42,
        ),
    )

    e2e_latency = time.perf_counter() - start_time
    print(f"the avg e2e is: {e2e_latency}")

    img = output[0].request_output[0].images[0]
    assert_image_valid(img, width=512, height=512)
