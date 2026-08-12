# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for unified interaction handlers, coordinator, and camera projectors."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator
from vllm_omni.diffusion.interaction.mixin import InteractionMixin
from vllm_omni.diffusion.interaction.modality_handlers.camera import SE3DeltaCameraHandler, WASDEventCameraHandler
from vllm_omni.diffusion.interaction.registry import STRUCTURED_HANDLER_REGISTRY
from vllm_omni.diffusion.interaction.types import resolve_event_frame_offset
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Test-only registry entries; production STRUCTURED_HANDLER_REGISTRY stays empty until real models land.
_FAKE_CAMERA_REGISTRY = {
    "SomeCameraPipeline": {"camera": SE3DeltaCameraHandler},
    "SomeControlEventPipeline": {"camera": WASDEventCameraHandler},
}


class _FakePromptPipeline(InteractionMixin):
    """Minimal pipeline stub with prompt-update support."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.transformer = SimpleNamespace(dtype=torch.float32)
        self.encode_prompt = MagicMock(return_value=(torch.full((1, 4, 2), 2.0), None))
        self._interaction_coordinator = None


def _make_state(*, request_id: str = "req-1") -> StepRequestState:
    state = StepRequestState(
        request_id=request_id,
        sampling=SimpleNamespace(num_outputs_per_prompt=1, max_sequence_length=226),  # pyright: ignore[reportArgumentType]
        prompt="hello",
    )
    state.prompt_embeds = torch.zeros(1, 4, 2)
    state.extra = {}
    return state


def _make_prompt_pipeline() -> _FakePromptPipeline:
    return _FakePromptPipeline()


def _boundary_ctx(
    boundary_at: float,
    *,
    window_opened_at: float | None = None,
    window_end_at: float | None = None,
    visible_from: float | None = None,
):
    from vllm_omni.diffusion.interaction.types import InteractionBoundaryContext

    return InteractionBoundaryContext(
        boundary_at=boundary_at,
        window_opened_at=window_opened_at,
        window_end_at=window_end_at,
        visible_from=visible_from,
    )


def _boundary_at(previous_boundary_at: float | None, num_frames: int, fps: float) -> float:
    if previous_boundary_at is None:
        return 0.0
    if fps <= 0:
        return previous_boundary_at
    return previous_boundary_at + num_frames / fps


class TestCoordinatorResolution:
    def test_helios_has_prompt_but_no_camera(self) -> None:
        pipeline = _make_prompt_pipeline()
        od_config = SimpleNamespace(model_class_name="HeliosPipeline")
        coordinator = InteractionCoordinator.build(pipeline, od_config)

        assert coordinator.has_modality("prompt")
        assert not coordinator.has_modality("camera")
        assert "HeliosPipeline" not in STRUCTURED_HANDLER_REGISTRY

    def test_camera_pipeline_resolves_from_literal_registry(self, mocker) -> None:
        pipeline = _make_prompt_pipeline()
        od_config = SimpleNamespace(model_class_name="SomeCameraPipeline")
        mocker.patch(
            "vllm_omni.diffusion.interaction.coordinator.STRUCTURED_HANDLER_REGISTRY",
            _FAKE_CAMERA_REGISTRY,
        )
        coordinator = InteractionCoordinator.build(pipeline, od_config)

        assert coordinator.has_modality("camera")
        assert isinstance(coordinator.get_handler("camera"), SE3DeltaCameraHandler)

    def test_unsupported_modality_includes_model_context(self) -> None:
        pipeline = _make_prompt_pipeline()
        od_config = SimpleNamespace(model_class_name="HeliosPipeline")
        coordinator = InteractionCoordinator.build(pipeline, od_config)

        with pytest.raises(ValueError, match="HeliosPipeline"):
            coordinator.enqueue(
                _make_state(),
                modality="camera",
                event_id="cam-1",
                received_at=0.0,
                payload={"mode": "target", "data": {"translation": [0.0, 0.0, 1.0]}},
                transition_chunks=None,
            )


class TestCameraHandlers:
    def test_target_transition_progress_and_se3_projection(self) -> None:
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_id="cam-1",
            received_at=0.0,
            payload={
                "mode": "target",
                "data": {"translation": [0.0, 0.0, 3.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
            },
            transition_chunks=2,
        )

        meta = handler.apply_at_chunk_boundary(
            state,
            chunk_index=0,
            num_frames=4,
            fps=16.0,
            boundary_ctx=_boundary_ctx(0.25),
        )
        tensor = state.conditioning["camera"]
        assert meta is not None
        assert meta.started_event_ids == ["cam-1"]
        assert meta.active_event_ids == ["cam-1"]
        assert meta.completed_event_ids == []
        assert tensor.shape == (4, 4, 4)

        meta2 = handler.apply_at_chunk_boundary(
            state,
            chunk_index=1,
            num_frames=4,
            fps=16.0,
            boundary_ctx=_boundary_ctx(0.5),
        )
        assert meta2 is not None
        assert meta2.started_event_ids == []
        assert meta2.active_event_ids == []
        assert meta2.completed_event_ids == ["cam-1"]
        assert state.interaction_sessions["camera"].active_event is None
        assert state.interaction_sessions["camera"].current_pose.translation[2] == pytest.approx(3.0)

        # Finished targets must not reappear on later chunks.
        meta3 = handler.apply_at_chunk_boundary(
            state,
            chunk_index=2,
            num_frames=4,
            fps=16.0,
            boundary_ctx=_boundary_ctx(0.75),
        )
        assert meta3 is not None
        assert meta3.started_event_ids == []
        assert meta3.active_event_ids == []
        assert meta3.completed_event_ids == []
        assert state.interaction_sessions["camera"].current_pose.translation[2] == pytest.approx(3.0)

    def test_single_chunk_target_completes_once(self) -> None:
        """transition_chunks=1 may start+complete in one chunk, then stay idle."""
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_id="cam-fast",
            received_at=0.0,
            payload={
                "mode": "target",
                "data": {"translation": [1.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
            },
            transition_chunks=1,
        )

        meta = handler.apply_at_chunk_boundary(
            state,
            chunk_index=0,
            num_frames=8,
            fps=16.0,
            boundary_ctx=_boundary_ctx(0.5),
        )
        assert meta is not None
        assert meta.started_event_ids == ["cam-fast"]
        assert meta.active_event_ids == []
        assert meta.completed_event_ids == ["cam-fast"]
        assert state.interaction_sessions["camera"].active_event is None

        meta2 = handler.apply_at_chunk_boundary(
            state,
            chunk_index=1,
            num_frames=8,
            fps=16.0,
            boundary_ctx=_boundary_ctx(1.0),
        )
        assert meta2 is not None
        assert meta2.started_event_ids == []
        assert meta2.active_event_ids == []
        assert meta2.completed_event_ids == []

    def test_velocity_integrates_across_chunks(self) -> None:
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_id="vel-1",
            received_at=0.0,
            payload={
                "mode": "velocity",
                "data": {"translation": [0.0, 0.0, 1.0]},
            },
            transition_chunks=None,
        )

        meta0 = handler.apply_at_chunk_boundary(
            state, chunk_index=0, num_frames=3, fps=16.0, boundary_ctx=_boundary_ctx(0.0)
        )
        assert meta0 is not None
        assert meta0.started_event_ids == ["vel-1"]
        assert meta0.active_event_ids == ["vel-1"]
        assert meta0.completed_event_ids == []
        assert state.interaction_sessions["camera"].current_pose.translation[2] == pytest.approx(3.0)

        meta1 = handler.apply_at_chunk_boundary(
            state, chunk_index=1, num_frames=2, fps=16.0, boundary_ctx=_boundary_ctx(0.125)
        )
        assert meta1 is not None
        assert meta1.started_event_ids == []
        assert meta1.active_event_ids == ["vel-1"]
        assert meta1.completed_event_ids == []
        assert state.interaction_sessions["camera"].current_pose.translation[2] == pytest.approx(5.0)

    def test_velocity_completed_when_replaced_by_target(self) -> None:
        """A later target cancels the active velocity and emits it in completed_event_ids."""
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_id="vel-1",
            received_at=0.0,
            payload={"mode": "velocity", "data": {"translation": [0.0, 0.0, 0.1]}},
            transition_chunks=None,
        )
        meta0 = handler.apply_at_chunk_boundary(
            state,
            chunk_index=0,
            num_frames=4,
            fps=16.0,
            boundary_ctx=_boundary_ctx(0.25),
        )
        assert meta0 is not None
        assert meta0.active_event_ids == ["vel-1"]

        handler.enqueue(
            state,
            event_id="tgt-1",
            received_at=0.25,
            payload={
                "mode": "target",
                "data": {"translation": [1.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
            },
            transition_chunks=1,
        )
        meta1 = handler.apply_at_chunk_boundary(
            state,
            chunk_index=1,
            num_frames=4,
            fps=16.0,
            boundary_ctx=_boundary_ctx(0.5),
        )
        assert meta1 is not None
        assert meta1.started_event_ids == ["tgt-1"]
        assert "vel-1" in meta1.completed_event_ids
        assert "tgt-1" in meta1.completed_event_ids
        assert meta1.active_event_ids == []
        assert state.interaction_sessions["camera"].active_event is None

    def test_wasd_projects_to_event_matrix(self) -> None:
        handler = WASDEventCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_id="wasd-1",
            received_at=0.0,
            payload={
                "mode": "velocity",
                "data": {"translation": [1.0, 0.0, 0.5], "rotation": [0.0, 0.09983341664, 0.0, 0.99500416527]},
            },
            transition_chunks=None,
        )
        handler.apply_at_chunk_boundary(state, chunk_index=0, num_frames=3, fps=16.0, boundary_ctx=_boundary_ctx(0.0))
        tensor = state.conditioning["camera"]
        assert tensor.ndim == 2
        assert tensor.shape[1] == 8
        assert tensor.shape[0] >= 1

    def test_distinct_intra_chunk_schedules_differ(self) -> None:
        """up@2,left@5 must differ from up@5,left@10 in dense SE3 conditioning."""
        fps = 16.0
        boundary = 100.0
        num_frames = 33

        def _run(offsets: list[tuple[str, int, list[float]]]) -> torch.Tensor:
            handler = SE3DeltaCameraHandler()
            state = _make_state()
            for event_id, frame, translation in offsets:
                handler.enqueue(
                    state,
                    event_id=event_id,
                    received_at=boundary + frame / fps,
                    payload={"mode": "velocity", "data": {"translation": translation}},
                    transition_chunks=None,
                )
            state.interaction_sessions["camera"].last_boundary_at = boundary
            handler.apply_at_chunk_boundary(
                state,
                chunk_index=1,
                num_frames=num_frames,
                fps=fps,
                boundary_ctx=_boundary_ctx(_boundary_at(boundary, num_frames, fps)),
            )
            return state.conditioning["camera"]

        early = _run(
            [
                ("up", 2, [0.0, 0.0, 1.0]),
                ("left", 5, [-1.0, 0.0, 0.0]),
            ]
        )
        late = _run(
            [
                ("up", 5, [0.0, 0.0, 1.0]),
                ("left", 10, [-1.0, 0.0, 0.0]),
            ]
        )
        assert early.shape == (num_frames, 4, 4)
        assert late.shape == (num_frames, 4, 4)
        assert not torch.allclose(early, late)

    def test_equal_timestamp_preserves_arrival_order(self) -> None:
        """When two events arrive at the same timestamp, the order of arrival is preserved."""
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        t = 10.5
        handler.enqueue(
            state,
            event_id="first",
            received_at=t,
            payload={"mode": "velocity", "data": {"translation": [0.0, 0.0, 1.0]}},
            transition_chunks=None,
        )
        handler.enqueue(
            state,
            event_id="second",
            received_at=t,
            payload={"mode": "velocity", "data": {"translation": [-1.0, 0.0, 0.0]}},
            transition_chunks=None,
        )
        state.interaction_sessions["camera"].last_boundary_at = 10.0
        meta = handler.apply_at_chunk_boundary(
            state,
            chunk_index=0,
            num_frames=4,
            fps=16.0,
            boundary_ctx=_boundary_ctx(_boundary_at(10.0, 4, 16.0)),
        )
        assert meta is not None
        assert meta.started_event_ids == ["first", "second"]
        assert meta.completed_event_ids == ["first"]
        assert meta.active_event_ids == ["second"]
        # Last equal-timestamp command wins for the remaining frames.
        assert state.interaction_sessions["camera"].active_event.event_id == "second"

    def test_frame_offset_clamps_and_first_boundary_fallback(self) -> None:
        """Frame offsets are clamped to the chunk boundaries and the first boundary is used as fallback."""
        assert resolve_event_frame_offset(received_at=-1.0, previous_boundary_at=0.0, num_frames=10, fps=10.0) == 0
        assert resolve_event_frame_offset(received_at=0.25, previous_boundary_at=0.0, num_frames=10, fps=10.0) == 2
        assert resolve_event_frame_offset(received_at=99.0, previous_boundary_at=0.0, num_frames=10, fps=10.0) == 9
        assert resolve_event_frame_offset(received_at=5.0, previous_boundary_at=None, num_frames=10, fps=10.0) == 0

    def test_mid_chunk_target_replaces_velocity(self) -> None:
        """When a "target"-mode event arrives, it replaces the previous "velocity"-mode event."""
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        boundary = 0.0
        fps = 10.0
        handler.enqueue(
            state,
            event_id="vel",
            received_at=0.0,
            payload={"mode": "velocity", "data": {"translation": [0.0, 0.0, 1.0]}},
            transition_chunks=None,
        )
        handler.enqueue(
            state,
            event_id="tgt",
            received_at=0.5,
            payload={
                "mode": "target",
                "data": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
            },
            transition_chunks=0,
        )
        state.interaction_sessions["camera"].last_boundary_at = boundary
        meta = handler.apply_at_chunk_boundary(
            state,
            chunk_index=0,
            num_frames=10,
            fps=fps,
            boundary_ctx=_boundary_ctx(_boundary_at(boundary, 10, fps)),
        )
        assert meta is not None
        assert meta.started_event_ids == ["vel", "tgt"]
        assert meta.completed_event_ids == ["vel", "tgt"]
        assert meta.active_event_ids == []
        assert state.interaction_sessions["camera"].active_event is None
        # Instant target (transition_chunks=0) finishes on the activation frame.
        assert state.interaction_sessions["camera"].current_pose.translation[2] == pytest.approx(0.0)

    def test_pacing_drops_late_velocity_keeps_late_target(self) -> None:
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_id="late-vel",
            received_at=1.5,
            payload={"mode": "velocity", "data": {"translation": [0.0, 0.0, 1.0]}},
            transition_chunks=None,
        )
        handler.enqueue(
            state,
            event_id="late-tgt",
            received_at=1.5,
            payload={
                "mode": "target",
                "data": {"translation": [1.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
            },
            transition_chunks=0,
        )
        meta = handler.apply_at_chunk_boundary(
            state,
            chunk_index=1,
            num_frames=10,
            fps=10.0,
            boundary_ctx=_boundary_ctx(
                2.0,
                window_opened_at=0.0,
                window_end_at=1.0,
                visible_from=0.0,
            ),
            pacing_enabled=True,
        )
        assert meta is not None
        assert meta.started_event_ids == ["late-tgt"]
        assert "late-vel" not in meta.started_event_ids

    def test_pacing_drops_invisible_events(self) -> None:
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_id="blind",
            received_at=0.2,
            payload={
                "mode": "target",
                "data": {"translation": [1.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
            },
            transition_chunks=0,
        )
        meta = handler.apply_at_chunk_boundary(
            state,
            chunk_index=1,
            num_frames=10,
            fps=10.0,
            boundary_ctx=_boundary_ctx(
                1.0,
                window_opened_at=0.0,
                window_end_at=1.0,
                visible_from=None,
            ),
            pacing_enabled=True,
        )
        assert meta is not None
        assert meta.started_event_ids == []
