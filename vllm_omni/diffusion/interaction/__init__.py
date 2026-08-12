# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Midway interaction handlers for chunked diffusion step execution."""

from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator
from vllm_omni.diffusion.interaction.mixin import InteractionMixin
from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler
from vllm_omni.diffusion.interaction.modality_handlers.camera import (
    CameraModalityHandler,
    CameraSession,
    QueuedCameraEvent,
    SE3DeltaCameraHandler,
    WASDEventCameraHandler,
)
from vllm_omni.diffusion.interaction.modality_handlers.prompt import (
    DEFAULT_TRANSITION_CHUNKS,
    PromptInteractionHandler,
    PromptSession,
    QueuedPromptEvent,
    prompt_update_versions,
)
from vllm_omni.diffusion.interaction.registry import STRUCTURED_HANDLER_REGISTRY
from vllm_omni.diffusion.interaction.types import (
    INTERACTION_MODES,
    ChunkMediaSpec,
    InteractionBoundaryContext,
    InteractionChunkMetadata,
    InteractionEvent,
    InteractionMode,
    InteractionPayload,
    InteractionSession,
    OmniTimestampedInteractionPrompt,
    merge_interaction_metadata,
    resolve_event_frame_offset,
)

# Backward-compatible alias for pipelines that previously inherited PromptUpdateMixin.
PromptUpdateMixin = InteractionMixin

__all__ = [
    "CameraModalityHandler",
    "CameraSession",
    "ChunkMediaSpec",
    "DEFAULT_TRANSITION_CHUNKS",
    "INTERACTION_MODES",
    "InteractionBoundaryContext",
    "InteractionChunkMetadata",
    "InteractionCoordinator",
    "InteractionEvent",
    "InteractionHandler",
    "InteractionMixin",
    "InteractionMode",
    "InteractionPayload",
    "InteractionSession",
    "OmniTimestampedInteractionPrompt",
    "PromptInteractionHandler",
    "PromptSession",
    "PromptUpdateMixin",
    "QueuedCameraEvent",
    "QueuedPromptEvent",
    "SE3DeltaCameraHandler",
    "STRUCTURED_HANDLER_REGISTRY",
    "WASDEventCameraHandler",
    "merge_interaction_metadata",
    "prompt_update_versions",
    "resolve_event_frame_offset",
]
