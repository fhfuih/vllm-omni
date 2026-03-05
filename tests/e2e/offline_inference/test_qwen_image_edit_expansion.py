"""
Comprehensive tests of diffusion features that are *ONLY* available in offline inference mode
(i.e., not supported in online serving) and are supported by the following models:
- Qwen-Image-Edit: single image input
- Qwen-Image-Edit-2509: two image inputs
"""

import PIL
import pytest

from tests.conftest import assert_image_valid, generate_synthetic_image
from tests.utils import hardware_marks
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

EDIT_PROMPT = "Transform this image of colorful geometric shapes into a Piet Mondrian style abstract painting."
MULTI_EDIT_PROMPT = (
    "Transform the first image of colorful geometric shapes into a Piet Mondrian style abstract painting. "
    "Transform the second image of colorful geometric shapes into a Vincent van Gogh style painting. "
    "Then juxtapose the two transformed images into a single artwork for visual contrast."
)
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"}, num_cards=2, parallel=True)


_DIFFUSION_FEATURE_CASES = [
    pytest.param(
        "tp_2",
        {
            "parallel_config": DiffusionParallelConfig(
                tensor_parallel_size=2,
            ),
        },
        marks=PARALLEL_FEATURE_MARKS,
    ),
    pytest.param(
        "vae_parallel_2",
        {
            "parallel_config": DiffusionParallelConfig(
                vae_patch_parallel_size=2,
            ),
        },
        marks=PARALLEL_FEATURE_MARKS,
    ),
]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    ("case_id", "extra_args"),
    _DIFFUSION_FEATURE_CASES,
    ids=[c.values[0] for c in _DIFFUSION_FEATURE_CASES],
)
def test_qwen_image_edit_single(case_id, extra_args, model_prefix):
    synthetic_image = generate_synthetic_image(512, 512, save_to_file=True)
    pil_image = PIL.Image.open(synthetic_image["file_path"]).convert("RGB")
    model = f"{model_prefix}Qwen/Qwen-Image-Edit"
    omni = Omni(model=model, seed=42, stage_init_timeout=300, **extra_args)
    req_output = omni.generate(
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
    img = req_output[0].images[0]
    assert_image_valid(img, width=512, height=512)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    ("case_id", "extra_args"),
    _DIFFUSION_FEATURE_CASES,
    ids=[c.values[0] for c in _DIFFUSION_FEATURE_CASES],
)
def test_qwen_image_edit_multi(case_id, extra_args, model_prefix):
    synthetic_image_1 = generate_synthetic_image(512, 512, save_to_file=True)
    synthetic_image_2 = generate_synthetic_image(512, 512, save_to_file=True)
    pil_image_1 = PIL.Image.open(synthetic_image_1["file_path"]).convert("RGB")
    pil_image_2 = PIL.Image.open(synthetic_image_2["file_path"]).convert("RGB")
    model = f"{model_prefix}Qwen/Qwen-Image-Edit-2509"
    omni = Omni(model=model, seed=42, stage_init_timeout=300, **extra_args)
    req_output = omni.generate(
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
    img = req_output[0].images[0]
    assert_image_valid(img, width=512, height=512)
