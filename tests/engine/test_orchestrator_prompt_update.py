# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm_omni.engine.messages import PromptUpdateMessage
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeDiffusionClient:
    def __init__(self) -> None:
        self.od_config = SimpleNamespace(streaming_output=True)
        self.prompt_update_calls: list[tuple[str, dict]] = []

    async def prompt_update_async(self, request_id: str, update: dict, timeout=None) -> None:
        del timeout
        self.prompt_update_calls.append((request_id, update))


class _FakeStagePool:
    def __init__(self, client: _FakeDiffusionClient) -> None:
        self.stage_type = "diffusion"
        self.clients = [client]
        self.submit_update_calls = 0
        self.submit_prompt_update = AsyncMock(side_effect=self._submit_prompt_update)

    def get_bound_replica_id(self, request_id: str) -> int:
        del request_id
        return 0

    async def _submit_prompt_update(self, request_id, req_state, *, prompt, transition_duration_chunks):
        del req_state
        await self.clients[0].prompt_update_async(
            request_id,
            {"prompt": prompt, "transition_duration_chunks": transition_duration_chunks},
        )


@pytest.mark.asyncio
async def test_handle_prompt_update_routes_to_submit_prompt_update() -> None:
    client = _FakeDiffusionClient()
    pool = _FakeStagePool(client)
    orchestrator = object.__new__(Orchestrator)
    orchestrator.num_stages = 1
    orchestrator.stage_pools = [pool]
    orchestrator.request_states = {
        "req-1": OrchestratorRequestState(
            request_id="req-1",
            prompt="hello",
            sampling_params_list=[SimpleNamespace()],
            final_stage_id=0,
            final_output_stage_ids={0},
        )
    }
    orchestrator.output_async_queue = AsyncMock()

    await orchestrator._handle_prompt_update(
        PromptUpdateMessage(
            request_id="req-1",
            prompt="updated prompt",
            transition_duration_chunks=3,
        )
    )

    pool.submit_prompt_update.assert_awaited_once()
    assert client.prompt_update_calls == [("req-1", {"prompt": "updated prompt", "transition_duration_chunks": 3})]
    orchestrator.output_async_queue.put.assert_not_called()


@pytest.mark.asyncio
async def test_handle_prompt_update_does_not_call_submit_update() -> None:
    client = _FakeDiffusionClient()
    pool = _FakeStagePool(client)
    pool.submit_update = AsyncMock()
    orchestrator = object.__new__(Orchestrator)
    orchestrator.num_stages = 1
    orchestrator.stage_pools = [pool]
    orchestrator.request_states = {
        "req-1": OrchestratorRequestState(
            request_id="req-1",
            prompt="hello",
            sampling_params_list=[SimpleNamespace()],
            final_stage_id=0,
            final_output_stage_ids={0},
        )
    }
    orchestrator.output_async_queue = AsyncMock()

    await orchestrator._handle_prompt_update(PromptUpdateMessage(request_id="req-1", prompt="updated prompt"))

    pool.submit_update.assert_not_called()
