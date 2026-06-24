# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import torch

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import QwenImageEditPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)


class CustomPipeline(QwenImageEditPipeline):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Forward pass for image editing with dummy trajectory data."""
        # Call parent's forward to get the normal output
        output = super().forward(req=req)

        # Get actual num_inference_steps used
        actual_num_steps = req.sampling_params.num_inference_steps or 50

        # Create dummy trajectory data
        dummy_trajectory_latents = torch.randn(actual_num_steps, 1, 16, 64, 64, dtype=torch.float32)

        # Inject dummy trajectory data into output
        output.trajectory_latents = dummy_trajectory_latents

        return output
