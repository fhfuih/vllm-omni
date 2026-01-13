# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType


@dataclass
class OmniDiffusionRequest:
    """
    Complete state passed through the pipeline execution.

    This dataclass contains all information needed during the diffusion pipeline
    execution, allowing methods to update specific components without needing
    to manage numerous individual parameters.
    """

    # TODO(will): double check that args are separate from server_args
    # properly. Also maybe think about providing an abstraction for pipeline
    # specific arguments.
    # data_type: DataType

    prompts: list[OmniPromptType]
    sampling_params: OmniDiffusionSamplingParams

    request_id: str | None = None

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""
        for p in self.prompts:
            if not isinstance(p, str) and "negative_prompt_embeds" in p and p["negative_prompt_embeds"] is None:
                p.negative_prompt_embeds = []  # type: ignore # already ensure that p is a prompt type with "negative_prompt_embeds" key

        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.sampling_params.guidance_scale > 1.0 and any(
            (not isinstance(p, str) and p.get("negative_prompt")) for p in self.prompts
        ):
            self.sampling_params.do_classifier_free_guidance = True
        if self.sampling_params.guidance_scale_2 is None:
            self.sampling_params.guidance_scale_2 = self.sampling_params.guidance_scale

        # Moved from omni_diffusion.py prepare_requests(), for stable audio open support
        if self.sampling_params.guidance_scale:
            self.sampling_params.guidance_scale_provided = True
