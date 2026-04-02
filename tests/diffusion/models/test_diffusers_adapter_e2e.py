# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end tests for DiffusersAdapterPipeline.

These tests require a GPU and download model weights from HuggingFace.
They are intended to run in nightly CI, not per-PR.

Run with:
    pytest tests/diffusion/models/test_diffusers_adapter_e2e.py \
        -v -s --gpu
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.diffusion,
    pytest.mark.nightly,
]


def _make_od_config(model, dtype=torch.float16, **overrides):
    """Create an OmniDiffusionConfig for e2e testing."""
    return OmniDiffusionConfig(
        model=model,
        model_class_name="DiffusersAdapterPipeline",
        dtype=dtype,
        diffusion_load_format="diffusers",
        diffusers_load_kwargs={},
        diffusers_call_kwargs={},
        output_type="pil",
        parallel_config=DiffusionParallelConfig(
            cfg_parallel_size=1,
            sequence_parallel_size=1,
        ),
        cache_backend="none",
        **overrides,
    )


def _make_request(prompt, **overrides):
    """Create an OmniDiffusionRequest for e2e testing."""
    return OmniDiffusionRequest(
        prompts=[{"prompt": prompt}],
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=4,
            guidance_scale=7.5,
            height=64,
            width=64,
            num_frames=1,
            num_outputs_per_prompt=1,
            seed=42,
            output_type="np",
            latents=None,
            generator=None,
        ),
        **overrides,
    )


@pytest.mark.skip(reason="Requires GPU and model download; run nightly only")
class TestDiffusersAdapterE2E:
    def test_e2e_sd15_text2img(self):
        """Stable Diffusion 1.5: prompt -> image, verify shape/range."""
        from vllm_omni.diffusion.data import DiffusionOutput
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        od_config = _make_od_config(
            model="stable-diffusion-v1-5/stable-diffusion-v1-5",
            dtype=torch.float16,
        )
        adapter = DiffusersAdapterPipeline(od_config=od_config)
        adapter.load_weights()

        req = _make_request(
            prompt="a photo of an astronaut riding a horse on mars",
            sampling_params=SimpleNamespace(
                num_inference_steps=4,
                guidance_scale=7.5,
                height=64,
                width=64,
                num_frames=1,
                num_outputs_per_prompt=1,
                seed=42,
                output_type="np",
                latents=None,
                generator=None,
            ),
        )
        output = adapter.forward(req)

        assert isinstance(output, DiffusionOutput)
        assert isinstance(output.output, np.ndarray)
        assert output.output.ndim == 4
        assert output.output.shape[-2:] == (64, 64)
        assert output.output.min() >= 0.0
        assert output.output.max() <= 1.0

    def test_e2e_wan_text2video(self):
        """Wan2.1 T2V via diffusers: prompt -> video, verify shape/range."""
        from vllm_omni.diffusion.data import DiffusionOutput
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        od_config = _make_od_config(
            model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            dtype=torch.bfloat16,
        )
        adapter = DiffusersAdapterPipeline(od_config=od_config)
        adapter.load_weights()

        req = _make_request(
            prompt="a cat walking on the grass",
            sampling_params=SimpleNamespace(
                num_inference_steps=4,
                guidance_scale=7.5,
                height=64,
                width=64,
                num_frames=8,
                num_outputs_per_prompt=1,
                seed=42,
                output_type="np",
                latents=None,
                generator=None,
            ),
        )
        output = adapter.forward(req)

        assert isinstance(output, DiffusionOutput)
        assert isinstance(output.output, np.ndarray)
        assert output.output.ndim == 4
        assert output.output.shape[0] == 8
        assert output.output.shape[-2:] == (64, 64)
        assert output.output.min() >= 0.0
        assert output.output.max() <= 1.0
