# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

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
