# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Sequence
from dataclasses import fields
from typing import Any

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
from vllm_omni.outputs import OmniRequestOutput

# TODO configure logging properly
logging.basicConfig(level=logging.INFO)

logger = init_logger(__name__)


def prepare_requests(sp_dict: dict[str, Any]) -> OmniDiffusionSamplingParams:
    field_names = {f.name for f in fields(OmniDiffusionSamplingParams)}

    init_kwargs = {}

    for key, value in sp_dict.items():
        if key in field_names:
            init_kwargs[key] = value

    if "guidance_scale" in sp_dict:
        init_kwargs["guidance_scale_provided"] = True

    return OmniDiffusionSamplingParams(**init_kwargs)


class OmniDiffusion:
    """
    It is the main class to interact with vLLM-Omni diffusion models.
    It acts as a high-level interface that prepares requests and
    delegates the actual diffusion process to the DiffusionEngine.

    You can pass either an `OmniDiffusionConfig` via `od_config`, or
    pass kwargs such as `model="Qwen/Qwen-Image"`,
    which will be forwarded to `OmniDiffusionConfig.from_kwargs`.
    """

    def __init__(self, od_config: OmniDiffusionConfig | None = None, **kwargs):
        if od_config is None:
            od_config = OmniDiffusionConfig.from_kwargs(**kwargs)
        elif isinstance(od_config, dict):
            od_config = OmniDiffusionConfig.from_kwargs(**od_config)

        self.od_config = od_config

        # Diffusers-style models expose `model_index.json` with `_class_name`.
        # Bagel models (and other non-diffusers) typically expose `config.json`.
        try:
            config_dict = get_hf_file_to_dict(
                "model_index.json",
                od_config.model,
            )
            od_config.model_class_name = config_dict.get("_class_name", None)
            od_config.update_multimodal_support()

            tf_config_dict = get_hf_file_to_dict(
                "transformer/config.json",
                od_config.model,
            )
            od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
        except (AttributeError, OSError, ValueError):
            cfg = get_hf_file_to_dict("config.json", od_config.model)
            if cfg is None:
                raise ValueError(f"Could not find config.json or model_index.json for model {od_config.model}")

            model_type = cfg.get("model_type")
            architectures = cfg.get("architectures") or []
            if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
                od_config.model_class_name = "BagelPipeline"
                od_config.tf_model_config = TransformerConfig()
                od_config.update_multimodal_support()
            else:
                raise

        self.engine: DiffusionEngine = DiffusionEngine.make_engine(od_config)

    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params: dict[str, Any] = {},
    ) -> list[OmniRequestOutput]:
        if isinstance(prompts, (str, dict)):
            prompts = [prompts]

        omni_diffusion_sp = prepare_requests(sampling_params)
        request = OmniDiffusionRequest(list(prompts), omni_diffusion_sp)
        return self._run_engine(request)

    def _run_engine(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        return self.engine.step(request)

    def close(self) -> None:
        self.engine.close()

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
