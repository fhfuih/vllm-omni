# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry of mid-way interaction handlers during a diffusion generation.
A pipeline only declares the modalities that it supports.

Outer key: pipeline architecture name (``od_config.model_class_name``).
Inner key: modality name (e.g. ``\"camera\"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler

STRUCTURED_HANDLER_REGISTRY: dict[str, dict[str, type[InteractionHandler]]] = {
    # Pipeline class name -> modality -> handler class. Examples:
    ### "SomeCameraPipeline": {
    ###     "camera": SE3DeltaCameraHandler,
    ### },
    ### "SomeControlEventPipeline": {
    ###     "camera": WASDEventCameraHandler,
    ### },
}
