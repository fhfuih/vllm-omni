# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for midway prompt update across runner, batch, and pipeline layers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.diffusion.models.helios.pipeline_helios import HeliosPipeline
from vllm_omni.diffusion.prompt_update import PROMPT_UPDATE_VERSION_KEY, PromptUpdatePayload, get_prompt_update_state
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


class _UnsupportedPipeline:
    supports_step_execution = True


def _make_helios_pipeline() -> HeliosPipeline:
    pipeline = object.__new__(HeliosPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = SimpleNamespace(dtype=torch.float32)
    pipeline.encode_prompt = MagicMock(
        return_value=(
            torch.full((1, 4, 2), 2.0),
            None,
        )
    )
    pipeline._prepare_next_chunk = MagicMock()
    return pipeline


def _make_state(*, prompt_embeds: torch.Tensor | None = None) -> DiffusionRequestState:
    state = DiffusionRequestState(
        request_id="req-1",
        sampling=SimpleNamespace(num_outputs_per_prompt=1, max_sequence_length=226),
        prompts=["hello"],
    )
    state.prompt_embeds = prompt_embeds if prompt_embeds is not None else torch.zeros(1, 4, 2)
    state.extra = {"dtype": torch.float32}
    return state


def _make_prompt_update_runner(*, pipeline, streaming_output: bool = True) -> DiffusionModelRunner:
    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = pipeline
    runner.state_cache = {}
    runner.od_config = SimpleNamespace(
        model_class_name="HeliosPipeline",
        streaming_output=streaming_output,
    )
    runner.supports_step_mode = lambda: True
    return runner


# ---- DiffusionModelRunner ----


def test_runner_prompt_update_delegates_to_helios_pipeline() -> None:
    pipeline = _make_helios_pipeline()
    runner = _make_prompt_update_runner(pipeline=pipeline)
    state = _make_state()
    runner.state_cache["req-1"] = state

    assert runner.prompt_update("req-1", {"prompt": "updated", "transition_duration_chunks": 2}) is True

    pipeline.encode_prompt.assert_called_once()
    pending = state.extra[pipeline._PENDING_PROMPT_UPDATE_KEY]
    assert pending["transition_duration_chunks"] == 2
    assert torch.equal(pending["target_prompt_embeds"], torch.full((1, 4, 2), 2.0))


def test_runner_prompt_update_rejects_unsupported_pipeline() -> None:
    runner = _make_prompt_update_runner(pipeline=_UnsupportedPipeline())
    with pytest.raises(ValueError, match="not supported"):
        runner.prompt_update("req-1", {"prompt": "updated"})


def test_runner_prompt_update_rejects_missing_request() -> None:
    runner = _make_prompt_update_runner(pipeline=_make_helios_pipeline())
    with pytest.raises(ValueError, match="No active request state"):
        runner.prompt_update("missing", {"prompt": "updated"})


# ---- InputBatch static-field refresh ----


def test_input_batch_refreshes_prompt_embeds_on_version_change() -> None:
    state = DiffusionRequestState(
        request_id="req-1",
        sampling=SimpleNamespace(),
        prompts=["hello"],
    )
    state.prompt_embeds = torch.zeros(1, 2, 3)
    state.latents = torch.zeros(1, 2)
    state.timesteps = torch.tensor([1.0])
    state.extra[PROMPT_UPDATE_VERSION_KEY] = 0

    batch = InputBatch.make_batch([state])
    assert torch.equal(batch.prompt_embeds, torch.zeros(1, 2, 3))

    state.prompt_embeds = torch.ones(1, 2, 3)
    state.extra[PROMPT_UPDATE_VERSION_KEY] = 1
    refreshed = InputBatch.make_batch([state], cached_batch=batch)
    assert torch.equal(refreshed.prompt_embeds, torch.ones(1, 2, 3))


# ---- HeliosPipeline chunk-boundary behavior ----


def test_helios_prepare_prompt_update_queues_pending_target() -> None:
    pipeline = _make_helios_pipeline()
    state = _make_state()

    pipeline.prepare_prompt_update(state, PromptUpdatePayload(prompt="new scene", transition_duration_chunks=2))

    pending = state.extra[pipeline._PENDING_PROMPT_UPDATE_KEY]
    assert torch.equal(pending["target_prompt_embeds"], torch.full((1, 4, 2), 2.0))
    assert pending["transition_duration_chunks"] == 2
    assert torch.equal(state.prompt_embeds, torch.zeros(1, 4, 2))


def test_helios_apply_prompt_update_at_chunk_boundary_starts_transition() -> None:
    pipeline = _make_helios_pipeline()
    state = _make_state()
    pipeline.prepare_prompt_update(state, PromptUpdatePayload(prompt="new scene", transition_duration_chunks=2))

    pipeline._apply_prompt_update_at_chunk_boundary(state)

    assert get_prompt_update_state(state) is not None
    assert torch.equal(state.prompt_embeds, torch.zeros(1, 4, 2))
    assert state.extra[PROMPT_UPDATE_VERSION_KEY] == 1


def test_helios_apply_prompt_update_advances_transition_over_chunks() -> None:
    pipeline = _make_helios_pipeline()
    state = _make_state()
    pipeline.prepare_prompt_update(state, PromptUpdatePayload(prompt="new scene", transition_duration_chunks=1))
    pipeline._apply_prompt_update_at_chunk_boundary(state)

    pipeline._apply_prompt_update_at_chunk_boundary(state)

    assert torch.allclose(state.prompt_embeds, torch.full((1, 4, 2), 2.0))
    assert state.extra[PROMPT_UPDATE_VERSION_KEY] == 2
