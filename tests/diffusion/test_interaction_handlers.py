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
from vllm_omni.diffusion.interaction.modality_handlers.prompt import PromptInteractionHandler
from vllm_omni.diffusion.interaction.registry import STRUCTURED_HANDLER_REGISTRY
from vllm_omni.diffusion.interaction.types import (
    InteractionEventArrival,
    resolve_event_frame_offset,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
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


def _event_ctx(event_id: str, *, received_at: float = 0.0) -> InteractionEventArrival:
    return InteractionEventArrival(event_id=event_id, received_at=received_at)


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
                event_arrival=_event_ctx("cam-1"),
                payload={"mode": "target", "data": {"translation": [0.0, 0.0, 1.0]}},
                transition_chunks=None,
            )


class TestCameraHandlers:
    def test_target_transition_progress_and_se3_projection(self) -> None:
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_arrival=_event_ctx("cam-1"),
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
            boundary_at=0.25,
        )
        tensor = state.extra["conditioning"]["camera"]
        assert meta is not None
        assert meta.started_event_ids == ["cam-1"]
        assert tensor.shape == (4, 4, 4)

        meta2 = handler.apply_at_chunk_boundary(
            state,
            chunk_index=1,
            num_frames=4,
            fps=16.0,
            boundary_at=0.5,
        )
        assert meta2 is not None
        assert "cam-1" in meta2.completed_event_ids
        assert state.extra["camera_session"].current_pose.translation[2] == pytest.approx(3.0)

    def test_velocity_integrates_across_chunks(self) -> None:
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_arrival=_event_ctx("vel-1"),
            payload={
                "mode": "velocity",
                "data": {"translation": [0.0, 0.0, 1.0]},
            },
            transition_chunks=None,
        )

        handler.apply_at_chunk_boundary(state, chunk_index=0, num_frames=3, fps=16.0, boundary_at=0.0)
        assert state.extra["camera_session"].current_pose.translation[2] == pytest.approx(3.0)
        handler.apply_at_chunk_boundary(state, chunk_index=1, num_frames=2, fps=16.0, boundary_at=0.125)
        assert state.extra["camera_session"].current_pose.translation[2] == pytest.approx(5.0)

    def test_wasd_projects_to_event_matrix(self) -> None:
        handler = WASDEventCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_arrival=_event_ctx("wasd-1"),
            payload={
                "mode": "velocity",
                "data": {"translation": [1.0, 0.0, 0.5], "rotation": [0.0, 0.09983341664, 0.0, 0.99500416527]},
            },
            transition_chunks=None,
        )
        handler.apply_at_chunk_boundary(state, chunk_index=0, num_frames=3, fps=16.0, boundary_at=0.0)
        tensor = state.extra["conditioning"]["camera"]
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
                    event_arrival=_event_ctx(event_id, received_at=boundary + frame / fps),
                    payload={"mode": "velocity", "data": {"translation": translation}},
                    transition_chunks=None,
                )
            state.extra["camera_session"].last_boundary_at = boundary
            handler.apply_at_chunk_boundary(
                state,
                chunk_index=1,
                num_frames=num_frames,
                fps=fps,
                boundary_at=_boundary_at(boundary, num_frames, fps),
            )
            return state.extra["conditioning"]["camera"]

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
            event_arrival=_event_ctx("first", received_at=t),
            payload={"mode": "velocity", "data": {"translation": [0.0, 0.0, 1.0]}},
            transition_chunks=None,
        )
        handler.enqueue(
            state,
            event_arrival=_event_ctx("second", received_at=t),
            payload={"mode": "velocity", "data": {"translation": [-1.0, 0.0, 0.0]}},
            transition_chunks=None,
        )
        state.extra["camera_session"].last_boundary_at = 10.0
        meta = handler.apply_at_chunk_boundary(
            state,
            chunk_index=0,
            num_frames=4,
            fps=16.0,
            boundary_at=_boundary_at(10.0, 4, 16.0),
        )
        assert meta is not None
        assert meta.started_event_ids == ["first", "second"]
        # Last equal-timestamp command wins for the remaining frames.
        assert state.extra["camera_session"].active_velocity.event_id == "second"

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
            event_arrival=_event_ctx("vel", received_at=0.0),
            payload={"mode": "velocity", "data": {"translation": [0.0, 0.0, 1.0]}},
            transition_chunks=None,
        )
        handler.enqueue(
            state,
            event_arrival=_event_ctx("tgt", received_at=0.5),
            payload={
                "mode": "target",
                "data": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
            },
            transition_chunks=0,
        )
        state.extra["camera_session"].last_boundary_at = boundary
        meta = handler.apply_at_chunk_boundary(
            state,
            chunk_index=0,
            num_frames=10,
            fps=fps,
            boundary_at=_boundary_at(boundary, 10, fps),
        )
        assert meta is not None
        assert meta.started_event_ids == ["vel", "tgt"]
        assert state.extra["camera_session"].active_velocity is None
        assert state.extra["camera_session"].active_target is not None
        assert state.extra["camera_session"].current_pose.translation[2] == pytest.approx(0.0)


class TestRunnerCameraRouting:
    def test_helios_rejects_camera_interaction(self) -> None:
        pipeline = _make_prompt_pipeline()
        runner = object.__new__(DiffusionModelRunner)
        runner.pipeline = pipeline
        runner.state_cache = {"req-1": _make_state()}
        runner.od_config = SimpleNamespace(model_class_name="HeliosPipeline", streaming_output=True)
        runner._supports_step_mode = lambda: True
        runner._interaction_coordinator = None

        with pytest.raises(ValueError, match="camera"):
            runner.submit_interaction(
                "req-1",
                {
                    "event_id": "cam-1",
                    "event": {
                        "multi_modal_data": {
                            "camera": {
                                "mode": "target",
                                "data": {"translation": [0, 0, 1]},
                            }
                        }
                    },
                },
            )

    def test_camera_pipeline_accepts_camera_and_prompt(self, mocker) -> None:
        pipeline = _make_prompt_pipeline()
        runner = object.__new__(DiffusionModelRunner)
        runner.pipeline = pipeline
        state = _make_state()
        runner.state_cache = {"req-1": state}
        runner.od_config = SimpleNamespace(model_class_name="SomeCameraPipeline", streaming_output=True)
        runner._supports_step_mode = lambda: True
        mocker.patch(
            "vllm_omni.diffusion.interaction.coordinator.STRUCTURED_HANDLER_REGISTRY",
            _FAKE_CAMERA_REGISTRY,
        )
        runner._interaction_coordinator = InteractionCoordinator.build(pipeline, runner.od_config)

        runner.submit_interaction(
            "req-1",
            {
                "event_id": "combo-1",
                "event": {
                    "prompt": "new scene",
                    "multi_modal_data": {
                        "camera": {
                            "mode": "target",
                            "data": {"translation": [0, 0, 1]},
                        }
                    },
                },
                "transition_chunks": 1,
            },
        )

        assert "pending_prompt_update" in state.extra
        assert len(state.extra["camera_session"].pending_events) == 1


class TestInteractionMixinApply:
    def test_unified_apply_updates_prompt_metadata(self) -> None:
        pipeline = _make_prompt_pipeline()
        state = _make_state()
        handler = PromptInteractionHandler.from_pipeline(pipeline)
        handler.enqueue(
            state,
            event_arrival=_event_ctx("ui-1"),
            payload={"prompt": "new scene"},
            transition_chunks=1,
        )
        pipeline.apply_interaction_at_chunk_boundary(state, chunk_index=0, num_frames=1, fps=16.0)
        assert state.extra["prompt_update_version"] == 1
        assert state.extra["interaction_chunk_metadata"]["started_event_ids"] == ["ui-1"]
