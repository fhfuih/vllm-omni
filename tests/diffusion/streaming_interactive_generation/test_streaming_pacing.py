# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for streaming pacing observation windows and eligibility."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.interaction.pacing import (
    close_observation_window,
    ensure_pacing_state,
    is_event_eligible,
    open_observation_window,
)
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _state() -> StepRequestState:
    return StepRequestState(
        request_id="req-1",
        sampling=SimpleNamespace(fps=10.0),  # pyright: ignore[reportArgumentType]
    )


class TestObservationWindow:
    def test_open_window_sized_by_upcoming_chunk_media(self) -> None:
        state = _state()
        pacing = ensure_pacing_state(state)
        assert state.streaming_pacing_state is pacing
        media = ChunkMediaSpec(num_frames=12, fps=16.0)
        window = open_observation_window(
            pacing,
            collecting_for_chunk=1,
            opened_at=100.0,
            media=media,
        )
        assert window.collecting_for_chunk == 1
        assert window.window_end_at == pytest.approx(100.0 + 12 / 16.0)
        assert pacing.current_window is window

    def test_pending_chunk_boundary_flag(self) -> None:
        state = _state()
        state.chunk_index = 0
        state.pending_chunk_boundary = True
        assert state.pending_chunk_boundary is True
        # While pending, chunk_index still names the completed chunk.
        assert state.chunk_index + 1 == 1
        state.pending_chunk_boundary = False
        assert state.pending_chunk_boundary is False

    def test_close_returns_current_window(self) -> None:
        pacing = ensure_pacing_state(_state())
        window = open_observation_window(
            pacing,
            collecting_for_chunk=1,
            opened_at=0.0,
            media=ChunkMediaSpec(num_frames=10, fps=10.0),
        )
        closed = close_observation_window(pacing)
        assert closed is window
        assert pacing.current_window is window


class TestEligibility:
    def _window(self):
        return open_observation_window(
            ensure_pacing_state(_state()),
            collecting_for_chunk=1,
            opened_at=10.0,
            media=ChunkMediaSpec(num_frames=10, fps=10.0),
        )

    def test_requires_visible_from(self) -> None:
        window = self._window()
        assert not is_event_eligible(
            received_at=10.5,
            window=window,
            mode="target",
            visible_from=None,
        )
        assert not is_event_eligible(
            received_at=10.5,
            window=window,
            mode="target",
            visible_from=10.6,
        )

    def test_requires_opened_at_lower_bound(self) -> None:
        window = self._window()
        assert not is_event_eligible(
            received_at=9.9,
            window=window,
            mode="target",
            visible_from=0.0,
        )

    def test_velocity_capped_at_window_end(self) -> None:
        window = self._window()
        assert is_event_eligible(
            received_at=11.0,
            window=window,
            mode="velocity",
            visible_from=0.0,
        )
        assert not is_event_eligible(
            received_at=11.1,
            window=window,
            mode="velocity",
            visible_from=0.0,
        )

    def test_target_and_prompt_remain_eligible_after_window_end(self) -> None:
        window = self._window()
        assert is_event_eligible(
            received_at=12.5,
            window=window,
            mode="target",
            visible_from=0.0,
        )
        assert is_event_eligible(
            received_at=12.5,
            window=window,
            mode=None,
            visible_from=0.0,
        )


class TestStreamingPacingConfig:
    def test_default_off(self) -> None:
        cfg = OmniDiffusionConfig(model="test")
        assert cfg.streaming_pacing is False

    def test_requires_streaming_output(self) -> None:
        with pytest.raises(ValueError, match="streaming_output=True"):
            OmniDiffusionConfig(model="test", streaming_pacing=True, streaming_output=False)

    def test_requires_max_num_seqs_one(self) -> None:
        with pytest.raises(ValueError, match="max_num_seqs=1"):
            OmniDiffusionConfig(
                model="test",
                streaming_pacing=True,
                streaming_output=True,
                max_num_seqs=2,
            )

    def test_valid_pacing_config(self) -> None:
        cfg = OmniDiffusionConfig(
            model="test",
            streaming_pacing=True,
            streaming_output=True,
            max_num_seqs=1,
        )
        assert cfg.streaming_pacing is True

    def test_alias_normalization(self) -> None:
        from vllm_omni.diffusion.data import normalize_omni_diffusion_kwargs

        normalized = normalize_omni_diffusion_kwargs(
            {
                "diffusion_streaming_output": True,
                "diffusion_streaming_pacing": True,
                "max_num_seqs": 1,
            }
        )
        assert normalized["streaming_output"] is True
        assert normalized["streaming_pacing"] is True
        assert "diffusion_streaming_pacing" not in normalized
