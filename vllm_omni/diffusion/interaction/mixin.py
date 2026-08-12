# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interaction-handling mixin for diffusion pipelines. Process interactions at chunk boundaries."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

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

    def _default_chunk_parameters(self, state: StepRequestState) -> tuple[int, int, float]:
        num_frames_raw = state.extra.get("window_num_frames")
        if num_frames_raw is None:
            num_frames_raw = state.extra.get("num_latent_frames_per_chunk")
        num_frames = int(num_frames_raw)  # pyright: ignore[reportArgumentType] # intentionally raise on format mismatch
        fps = 0.0
        if state.sampling.fps:  # not None and not zero
            fps = float(state.sampling.fps)
        return state.chunk_index, num_frames, fps
