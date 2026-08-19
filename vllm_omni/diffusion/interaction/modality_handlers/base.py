# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base interaction handler interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Self

from vllm_omni.diffusion.interaction.types import (
    InteractionChunkMetadata,
    InteractionPayload,
)
from vllm_omni.diffusion.worker.utils import StepRequestState


class InteractionHandler(ABC):
    """Strategy object for one interaction modality.

    Handler instances are pipeline/runner-owned and request-agnostic. Per-request
    session state lives on ``StepRequestState.interaction_sessions``.
    Each concrete modality handler decides how to handle timing from ``received_at``.
    """

    modality: ClassVar[str]

    @classmethod
    def from_pipeline(cls, pipeline: object) -> Self:
        """Bind a request-agnostic handler to ``pipeline`` when construction needs it.

        Default construction ignores the pipeline. Prompt handlers override this to
        capture ``encode_prompt`` / device / dtype.
        """
        del pipeline
        return cls()

    @abstractmethod
    def enqueue(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        received_at: float,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        """Validate and queue this track on request-local state."""

    @abstractmethod
    def apply_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        chunk_index: int,
        num_frames: int,
        fps: float,
        boundary_at: float,
    ) -> InteractionChunkMetadata | None:
        """Advance request-local state and materialize this chunk's effects."""
