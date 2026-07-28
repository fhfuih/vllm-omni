# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interaction-handling mixin for diffusion pipelines. Process interactions at chunk boundaries."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vllm_omni.diffusion.interaction.modality_handlers.prompt import PromptInteractionHandler
from vllm_omni.diffusion.interaction.types import (
    InteractionChunkMetadata,
    merge_interaction_metadata,
)
from vllm_omni.diffusion.worker.utils import StepRequestState

if TYPE_CHECKING:
    from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator


class InteractionMixin:
    """Unified chunk-boundary interaction hook.

    Pipelines call only ``apply_interaction_at_chunk_boundary``; they do not
    branch on prompt, camera, or any other modality.

    Camera interaction handlers own mode/data validation, timeline
    integration/interpolation, and projection into model-specific tensors, for
    example dense SE(3) deltas shaped ``[num_frames_per_chunk, 4, 4]`` or sparse
    control events shaped ``[num_events_captured_in_this_chunk, 8]`` (WASDIJKL).
    Camera session state stays on ``StepRequestState.extra``; handler strategy
    objects live on the pipeline/runner ``InteractionCoordinator``.
    """

    _interaction_coordinator: InteractionCoordinator | None = None

    def apply_interaction_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        chunk_index: int | None = None,
        num_frames: int | None = None,
        fps: float | None = None,
    ) -> None:
        """Advance all active interaction tracks before the next chunk."""
        if chunk_index is None or num_frames is None or fps is None:
            chunk_index, num_frames, fps = self._default_chunk_parameters(state)

        boundary_at = time.monotonic()

        metas: list[InteractionChunkMetadata] = []
        coordinator = self._interaction_coordinator
        if coordinator is not None:
            handlers = coordinator.handlers_in_apply_order()
        else:
            handlers = [self._get_prompt_handler()]

        for handler in handlers:
            meta = handler.apply_at_chunk_boundary(
                state,
                chunk_index=chunk_index,
                num_frames=num_frames,
                fps=fps,
                boundary_at=boundary_at,
            )
            if meta is not None:
                metas.append(meta)

        state.extra["interaction_chunk_metadata"] = merge_interaction_metadata(metas).as_dict()

    def _get_prompt_handler(self) -> PromptInteractionHandler:
        coordinator = self._interaction_coordinator
        if coordinator is not None and coordinator.has_modality("prompt"):
            handler = coordinator.get_handler("prompt")
            assert isinstance(handler, PromptInteractionHandler)
            return handler
        return PromptInteractionHandler.from_pipeline(self)

    def _default_chunk_parameters(self, state: StepRequestState) -> tuple[int, int, float]:
        num_frames_raw = state.extra.get("window_num_frames")
        if num_frames_raw is None:
            num_frames_raw = state.extra.get("num_latent_frames_per_chunk")
        num_frames = int(num_frames_raw)  # pyright: ignore[reportArgumentType] # intentionally raise on format mismatch
        fps = 0.0
        if state.sampling.fps:  # not None and not zero
            fps = float(state.sampling.fps)
        return state.chunk_index, num_frames, fps
