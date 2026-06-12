# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.client_request_state import ClientRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_async_omni(*, num_stages: int = 1, stage_type: str = "diffusion") -> AsyncOmni:
    omni = object.__new__(AsyncOmni)
    omni.log_stats = False
    omni.request_states = {
        "external-abc-uuid-1": ClientRequestState(
            request_id="external-abc-uuid-1",
            external_request_id="external-abc",
            queue=AsyncMock(),
        ),
        "external-abc-uuid-2": ClientRequestState(
            request_id="external-abc-uuid-2",
            external_request_id="external-abc",
            queue=AsyncMock(),
        ),
        "other-req-uuid": ClientRequestState(
            request_id="other-req-uuid",
            external_request_id="other-req",
            queue=AsyncMock(),
        ),
    }
    omni.engine = SimpleNamespace(
        num_stages=num_stages,
        get_stage_metadata=lambda stage_id: SimpleNamespace(stage_type=stage_type),
        add_prompt_update_async=AsyncMock(),
    )
    return omni


@pytest.mark.asyncio
async def test_add_prompt_update_async_maps_external_to_internal_id() -> None:
    omni = _make_async_omni()
    with pytest.raises(ValueError, match="exactly one active request"):
        await omni.add_prompt_update_async("external-abc", prompt="new prompt")

    omni.request_states.pop("external-abc-uuid-2")
    await omni.add_prompt_update_async(
        "external-abc",
        prompt="new prompt",
        transition_duration_chunks=2,
    )
    omni.engine.add_prompt_update_async.assert_awaited_once_with(
        "external-abc-uuid-1",
        prompt="new prompt",
        transition_duration_chunks=2,
    )


@pytest.mark.asyncio
async def test_add_prompt_update_async_rejects_inactive_request() -> None:
    omni = _make_async_omni()
    omni.request_states.clear()
    with pytest.raises(ValueError, match="No active request"):
        await omni.add_prompt_update_async("missing", prompt="new prompt")


@pytest.mark.asyncio
async def test_add_prompt_update_async_rejects_non_diffusion() -> None:
    omni = _make_async_omni(stage_type="llm")
    omni.request_states = {
        "req-uuid": ClientRequestState(
            request_id="req-uuid",
            external_request_id="req",
            queue=AsyncMock(),
        ),
    }
    with pytest.raises(ValueError, match="requires a diffusion stage"):
        await omni.add_prompt_update_async("req", prompt="new prompt")
