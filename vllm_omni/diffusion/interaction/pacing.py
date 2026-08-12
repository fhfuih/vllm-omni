# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observation-window state and eligibility helpers for streaming pacing."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.diffusion.interaction.types import ChunkMediaSpec, InteractionMode
from vllm_omni.diffusion.worker.utils import StepRequestState


@dataclass
class ObservationWindow:
    """Observation window that collects interactions for one upcoming chunk.

    Window ``W(k+1)`` is open while chunk ``k`` is generating and collects
    inputs that will be applied to chunk ``k+1``.
    """

    index: int
    collecting_for_chunk: int
    opened_at: float
    window_end_at: float
    media: ChunkMediaSpec


@dataclass
class StreamingPacingState:
    """Request-local pacing timeline (windows / visibility).

    Whether pacing is active is owned by ``OmniDiffusionConfig.streaming_pacing``;
    this object is only allocated when that flag is on.
    """

    current_window: ObservationWindow | None = None
    chunk_gen_started_at: dict[int, float] = field(default_factory=dict)  # chunk_index -> opened_at (monotonic)
    chunk_visible_from: dict[int, float] = field(default_factory=dict)  # chunk_index -> visible_from (monotonic)


def ensure_pacing_state(state: StepRequestState) -> StreamingPacingState:
    if state.streaming_pacing_state is None:
        state.streaming_pacing_state = StreamingPacingState()
    return state.streaming_pacing_state


def open_observation_window(
    pacing: StreamingPacingState,
    *,
    collecting_for_chunk: int,
    opened_at: float,
    media: ChunkMediaSpec,
) -> ObservationWindow:
    """Open ``W(collecting_for_chunk)`` sized by ``media``."""
    window = ObservationWindow(
        index=collecting_for_chunk,
        collecting_for_chunk=collecting_for_chunk,
        opened_at=opened_at,
        window_end_at=opened_at + media.duration_s,
        media=media,
    )
    pacing.current_window = window
    return window


def close_observation_window(pacing: StreamingPacingState) -> ObservationWindow | None:
    return pacing.current_window


def is_event_eligible(
    *,
    received_at: float,
    window: ObservationWindow | None,
    mode: InteractionMode | None,
    visible_from: float | None = None,
) -> bool:
    """Return whether an event may be applied at the current boundary.

    * All events require ``received_at >= window.opened_at``.
    * When ``visible_from`` is set (first/watched media became available in the
      worker), also require ``received_at >= visible_from`` so interactions
      before any generated media are ignored.
    * When ``visible_from`` is ``None``, no media has been produced yet → reject.
    * Timed/trajectory (``velocity``) events are also capped at ``window_end_at``.
    * Target / LWW events (``target`` or ``mode is None`` for prompt) have no
      upper window cutoff and remain eligible until boundary apply.
    """
    if visible_from is None:
        return False
    if window is None:
        return False
    if received_at < window.opened_at:
        return False
    if received_at < visible_from:
        return False
    if mode == "velocity":
        return received_at <= window.window_end_at
    return True
