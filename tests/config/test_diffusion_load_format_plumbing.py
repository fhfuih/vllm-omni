# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for diffusion_load_format and split diffusers kwargs plumbing."""

import pytest

from vllm_omni.config.stage_config import StageConfig, StageConfigFactory, StageType
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.stage_init_utils import StageMetadata, build_diffusion_config
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_create_default_diffusion_stage_cfg_plumbs_cli_kwargs():
    kwargs = {
        "tensor_parallel_size": 1,
        "diffusion_load_format": "diffusers",
        "diffusers_load_kwargs": {"use_safetensors": True},
        "diffusers_call_kwargs": {"num_inference_steps": 3},
        "cache_backend": "none",
    }
    cfg_list = AsyncOmniEngine._create_default_diffusion_stage_cfg(kwargs)
    ea = cfg_list[0]["engine_args"]
    assert ea["diffusion_load_format"] == "diffusers"
    assert ea["diffusers_load_kwargs"]["use_safetensors"] is True
    assert ea["diffusers_call_kwargs"]["num_inference_steps"] == 3


def test_create_default_diffusion_omits_none_load_format():
    kwargs = {"tensor_parallel_size": 1, "diffusion_load_format": None, "cache_backend": "none"}
    cfg_list = AsyncOmniEngine._create_default_diffusion_stage_cfg(kwargs)
    assert cfg_list[0]["engine_args"]["diffusion_load_format"] == "default"


def test_build_diffusion_config_from_engine_args():
    """engine_args (e.g. from stage YAML) must populate od_config."""
    from types import SimpleNamespace

    stage_cfg = SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        engine_args={
            "model": "dummy/model",
            "tensor_parallel_size": 1,
            "diffusion_load_format": "diffusers",
            "diffusers_call_kwargs": {"guidance_scale": 8.0},
        },
        runtime={"process": True, "devices": "0"},
    )
    metadata = StageMetadata(
        stage_id=0,
        stage_type="diffusion",
        engine_output_type=None,
        is_comprehension=False,
        requires_multimodal_data=False,
        engine_input_source=[],
        final_output=True,
        final_output_type="image",
        default_sampling_params=OmniDiffusionSamplingParams(),
        custom_process_input_func=None,
        model_stage=None,
        runtime_cfg={},
    )
    od = build_diffusion_config("dummy/model", stage_cfg, metadata)
    assert od.diffusion_load_format == "diffusers"
    assert od.diffusers_call_kwargs["guidance_scale"] == 8.0


def test_stage_config_factory_reads_diffusers_from_engine_args(tmp_path):
    """Pipeline YAML: split diffusers kwargs live under engine_args."""
    p = tmp_path / "pipeline.yaml"
    p.write_text(
        """
model_type: dummy
stages:
  - stage_id: 0
    model_stage: diffusion
    stage_type: diffusion
    input_sources: []
    engine_args:
      diffusion_load_format: diffusers
      diffusers_load_kwargs:
        use_safetensors: true
      diffusers_call_kwargs:
        num_inference_steps: 9
""",
        encoding="utf-8",
    )
    mp = StageConfigFactory._parse_pipeline_yaml(p, "dummy")
    assert len(mp.stages) == 1
    st = mp.stages[0]
    assert st.yaml_engine_args["diffusion_load_format"] == "diffusers"
    assert st.yaml_engine_args["diffusers_load_kwargs"]["use_safetensors"] is True
    assert st.yaml_engine_args["diffusers_call_kwargs"]["num_inference_steps"] == 9


def test_stage_config_has_no_diffusion_backend_field():
    sc = StageConfig(stage_id=0, model_stage="diffusion", stage_type=StageType.DIFFUSION)
    assert not hasattr(sc, "diffusion_backend")
