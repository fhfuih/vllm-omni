# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interaction-handling mixin for diffusion pipelines. Process interactions at chunk boundaries."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.worker.utils import StepRequestState

if TYPE_CHECKING:
    from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator


class InteractionMixin:
    """Unified chunk-boundary interaction hook.

    Pipelines call only ``apply_interaction_at_chunk_boundary`` without knowing which modality is being interacted with.

    Per-modality session state lives on ``StepRequestState.interaction_sessions``;
    handler strategy objects live on the pipeline/runner ``InteractionCoordinator``.
    """

    _interaction_coordinator: InteractionCoordinator | None = None

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        """Return the media timeline for the current/next chunk.

        Concrete pipelines must override this. Interaction apply uses it when
        ``num_frames`` / ``fps`` are not passed explicitly.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement peek_chunk_media() for interaction apply")

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
            media = self.peek_chunk_media(state)
            if chunk_index is None:
                chunk_index = state.chunk_index
            if num_frames is None:
                num_frames = media.num_frames
            if fps is None:
                fps = media.fps

        if self._interaction_coordinator is None:
            raise RuntimeError(
                "interaction coordinator is not initialized; "
                "DiffusionModelRunner.load_model must wire InteractionCoordinator "
                "onto the pipeline before chunked generation"
            )

        merged = self._interaction_coordinator.apply_at_chunk_boundary(
            state,
            chunk_index=chunk_index,
            num_frames=num_frames,
            fps=fps,
            boundary_at=time.monotonic(),
        )
        state.interaction_chunk_metadata = merged

    def prepare_next_chunk(self, state: StepRequestState) -> None:
        """Set up pipeline state for the next chunk after interaction apply.

        Default no-op. Pipelines may override this if needed.
        """
        pass
