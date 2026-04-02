# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DiffusersAdapterPipeline.

These tests do NOT require a GPU or network access. They use mocks to
simulate diffusers pipeline behavior.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_od_config(**overrides):
    """Create a minimal OmniDiffusionConfig for testing."""
    defaults = {
        "model": "test/model",
        "model_class_name": "DiffusersAdapterPipeline",
        "dtype": torch.float16,
        "diffusion_load_format": "diffusers",
        "diffusers_load_kwargs": {},
        "diffusers_call_kwargs": {},
        "output_type": "pil",
        "parallel_config": SimpleNamespace(
            cfg_parallel_size=1,
            sequence_parallel_size=1,
        ),
        "cache_backend": "none",
    }
    defaults.update(overrides)
    return OmniDiffusionConfig(**defaults)


def _make_request(**overrides):
    """Create a minimal OmniDiffusionRequest for testing."""
    prompt = overrides.pop("prompt", "a test prompt")
    neg_prompt = overrides.pop("negative_prompt", None)
    prompt_obj = {"prompt": prompt}
    if neg_prompt:
        prompt_obj["negative_prompt"] = neg_prompt

    defaults = {
        "prompts": [prompt_obj],
        "sampling_params": SimpleNamespace(
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512,
            num_frames=1,
            num_outputs_per_prompt=1,
            seed=42,
            output_type="pil",
            latents=None,
            generator=None,
        ),
    }
    defaults.update(overrides)
    return OmniDiffusionRequest(**defaults)


class _MockPipeline:
    """Simulates a diffusers pipeline.__call__ signature."""

    def __init__(self, class_name="StableDiffusionPipeline"):
        self.__class__.__name__ = class_name
        self.components = {}
        self.call_args: dict[str, Any] = {}

    def __call__(self, **kwargs):
        self.call_args = kwargs
        return SimpleNamespace(images=[np.zeros((64, 64, 3), dtype=np.uint8)])


class TestDiffusersAdapterInit:
    def test_adapter_init_empty_pipeline(self):
        """Adapter initializes with no pipeline loaded."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        assert adapter._pipeline is None

    @patch("vllm_omni.diffusion.models.diffusers_adapter.DiffusionPipeline")
    def test_adapter_load_weights_sets_pipeline(self, mock_dp):
        """load_weights() creates a diffusers pipeline."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.components = {}
        mock_dp.from_pretrained.return_value = mock_pipeline

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        adapter.load_weights()

        mock_dp.from_pretrained.assert_called_once()
        assert adapter._pipeline is mock_pipeline

    @patch("vllm_omni.diffusion.models.diffusers_adapter.DiffusionPipeline")
    def test_adapter_load_weights_cpu_offload(self, mock_dp):
        """CPU offload is only enabled when explicitly requested."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.components = {}
        mock_dp.from_pretrained.return_value = mock_pipeline

        config = _make_od_config(diffusers_load_kwargs={"enable_cpu_offload": True})
        adapter = DiffusersAdapterPipeline(od_config=config)
        adapter.load_weights()

        mock_pipeline.enable_model_cpu_offload.assert_called_once()

    @patch("vllm_omni.diffusion.models.diffusers_adapter.DiffusionPipeline")
    def test_adapter_load_weights_no_cpu_offload_default(self, mock_dp):
        """CPU offload is NOT enabled by default."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.components = {}
        mock_dp.from_pretrained.return_value = mock_pipeline

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        adapter.load_weights()

        mock_pipeline.enable_model_cpu_offload.assert_not_called()
        mock_pipeline.to.assert_called_once()


class TestDiffusersAdapterCapabilities:
    @patch("vllm_omni.diffusion.models.diffusers_adapter.DiffusionPipeline")
    def test_detect_text2img_pipeline(self, mock_dp):
        """Text2img pipeline has no image_encoder."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        mock_pipe = MagicMock()
        mock_pipe.components = {"vae": MagicMock(), "text_encoder": MagicMock()}
        mock_pipe.__class__.__name__ = "StableDiffusionPipeline"
        mock_dp.from_pretrained.return_value = mock_pipe

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        adapter.load_weights()

    @patch("vllm_omni.diffusion.models.diffusers_adapter.DiffusionPipeline")
    def test_detect_img2img_pipeline(self, mock_dp):
        """Img2img pipeline has image_encoder."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        mock_pipe = MagicMock()
        mock_pipe.components = {"image_encoder": MagicMock()}
        mock_pipe.__class__.__name__ = "StableDiffusionImg2ImgPipeline"
        mock_dp.from_pretrained.return_value = mock_pipe

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        adapter.load_weights()


class TestDiffusersAdapterForward:
    def test_forward_returns_diffusion_output(self):
        """Forward pass returns a valid DiffusionOutput."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        adapter._pipeline = _MockPipeline()

        req = _make_request()
        output = adapter.forward(req)

        assert isinstance(output, DiffusionOutput)
        assert output.output is not None

    def test_forward_passes_prompt_to_pipeline(self):
        """Prompt is correctly passed to diffusers __call__."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        mock_pipe = _MockPipeline()
        adapter._pipeline = mock_pipe

        req = _make_request(prompt="a cat on mars")
        adapter.forward(req)

        assert mock_pipe.call_args["prompt"] == "a cat on mars"

    def test_forward_passes_negative_prompt(self):
        """Negative prompt is passed when present."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        mock_pipe = _MockPipeline()
        adapter._pipeline = mock_pipe

        req = _make_request(
            prompt="a cat",
            negative_prompt="ugly, blurry",
        )
        adapter.forward(req)

        assert mock_pipe.call_args["negative_prompt"] == "ugly, blurry"

    def test_forward_passes_sampling_params(self):
        """Sampling params are mapped to diffusers kwargs."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        mock_pipe = _MockPipeline()
        adapter._pipeline = mock_pipe

        req = _make_request(
            prompt="test",
            sampling_params=SimpleNamespace(
                num_inference_steps=30,
                guidance_scale=5.0,
                height=768,
                width=1024,
                num_frames=1,
                num_outputs_per_prompt=1,
                seed=123,
                output_type="np",
                latents=None,
                generator=None,
            ),
        )
        adapter.forward(req)
        kwargs = mock_pipe.call_args

        assert kwargs["num_inference_steps"] == 30
        assert kwargs["guidance_scale"] == 5.0
        assert kwargs["height"] == 768
        assert kwargs["width"] == 1024
        assert kwargs["output_type"] == "np"
        assert isinstance(kwargs["generator"], torch.Generator)


class TestDiffusersAdapterGuards:
    @pytest.mark.parametrize(
        "feature_id,cfg_sp,cache_backend",
        [
            ("cfg_parallel", 2, "none"),
            ("sequence_parallel", 1, "none"),
            ("teacache", 1, "tea_cache"),
            ("cache_dit", 1, "cache_dit"),
        ],
    )
    def test_guard_unsupported_feature(self, feature_id, cfg_sp, cache_backend):
        """Each unsupported feature raises ValueError."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        pc = SimpleNamespace(
            cfg_parallel_size=cfg_sp if feature_id == "cfg_parallel" else 1,
            sequence_parallel_size=(cfg_sp if feature_id == "sequence_parallel" else 1),
        )
        config = _make_od_config(
            parallel_config=pc,
            cache_backend=cache_backend,
        )
        adapter = DiffusersAdapterPipeline(od_config=config)
        adapter._pipeline = _MockPipeline()

        req = _make_request()
        with pytest.raises(ValueError):
            adapter.forward(req)

    def test_guard_unknown_output_type(self):
        """Unknown diffusers output type raises ValueError."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())

        class WeirdOutput:
            pass

        with pytest.raises(ValueError, match="Unknown diffusers output type"):
            adapter._wrap_output(WeirdOutput())


class TestDiffusersAdapterStepWiseRejection:
    def test_supports_step_execution_returns_false(self):
        """Step-wise execution flag is False."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        assert adapter.supports_step_execution is False

    @pytest.mark.parametrize(
        "method",
        ["prepare_encode", "denoise_step", "step_scheduler", "post_decode"],
    )
    def test_step_methods_raise_not_implemented(self, method):
        """All step methods raise NotImplementedError."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        with pytest.raises(NotImplementedError, match="Step-wise execution is not supported"):
            getattr(adapter, method)(None)


class TestDiffusersAdapterBuildKwargs:
    def test_build_call_kwargs_defaults(self):
        """Default kwargs are set when sampling params are None."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        req = _make_request()
        kwargs = adapter._build_call_kwargs(req)

        assert "prompt" in kwargs
        assert kwargs["num_inference_steps"] == 20
        assert kwargs["guidance_scale"] == 7.5

    def test_build_call_kwargs_call_kwargs_merge(self):
        """diffusers_call_kwargs are merged into pipeline call kwargs."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        config = _make_od_config(diffusers_call_kwargs={"guidance_scale": 9.0, "eta": 0.5})
        adapter = DiffusersAdapterPipeline(od_config=config)
        req = _make_request()
        kwargs = adapter._build_call_kwargs(req)

        assert kwargs["guidance_scale"] == 9.0
        assert kwargs["eta"] == 0.5

    def test_build_call_kwargs_video_pipeline(self):
        """num_frames is passed for video pipelines."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        req = _make_request(
            sampling_params=SimpleNamespace(
                num_inference_steps=20,
                guidance_scale=7.5,
                height=480,
                width=832,
                num_frames=49,
                num_outputs_per_prompt=1,
                seed=42,
                output_type="pil",
                latents=None,
                generator=None,
            ),
        )
        kwargs = adapter._build_call_kwargs(req)

        assert kwargs["num_frames"] == 49

    def test_extract_prompt_string(self):
        """String prompt is returned as-is."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        assert DiffusersAdapterPipeline._extract_prompt("hello") == "hello"

    def test_extract_prompt_dict(self):
        """Dict prompt extracts 'prompt' key."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        assert DiffusersAdapterPipeline._extract_prompt({"prompt": "hi", "negative_prompt": "no"}) == "hi"

    def test_extract_negative_prompt_dict(self):
        """Dict prompt extracts 'negative_prompt' key."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        assert DiffusersAdapterPipeline._extract_negative_prompt({"prompt": "hi", "negative_prompt": "no"}) == "no"

    def test_extract_negative_prompt_none(self):
        """Missing negative_prompt returns None."""
        from vllm_omni.diffusion.models.diffusers_adapter import (
            DiffusersAdapterPipeline,
        )

        assert DiffusersAdapterPipeline._extract_negative_prompt({"prompt": "hi"}) is None


class TestConversionHelpers:
    def test_images_to_tensor_numpy(self):
        """Numpy image list is converted to (N, C, H, W) tensor."""
        from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import (
            _images_to_tensor,
        )

        images = [np.zeros((64, 64, 3), dtype=np.uint8)]
        tensor = _images_to_tensor(images)

        assert tensor.shape == (1, 3, 64, 64)
        assert tensor.dtype == torch.float32

    def test_images_to_tensor_tensor_passthrough(self):
        """Tensor images are returned as-is."""
        from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import (
            _images_to_tensor,
        )

        original = torch.randn(2, 3, 64, 64)
        result = _images_to_tensor(original)

        assert result is original

    def test_frames_to_tensor_numpy(self):
        """Numpy frame list is converted to (T, C, H, W) tensor."""
        from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import (
            _frames_to_tensor,
        )

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(16)]
        tensor = _frames_to_tensor(frames)

        assert tensor.shape == (16, 3, 64, 64)
        assert tensor.dtype == torch.float32

    def test_frames_to_tensor_nested_list(self):
        """Nested list [[PIL], [PIL]] is handled."""
        from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import (
            _frames_to_tensor,
        )

        frames = [[Image.new("RGB", (64, 64)) for _ in range(8)]]
        tensor = _frames_to_tensor(frames)

        assert tensor.shape == (8, 3, 64, 64)
