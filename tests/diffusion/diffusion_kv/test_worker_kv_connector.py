# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata, DiffusionKVSequenceMetadata
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _metadata(request_id: str = "req-0") -> DiffusionKVMetadata:
    return DiffusionKVMetadata(
        request_id=request_id,
        allocation_generation=1,
        sequences=(
            DiffusionKVSequenceMetadata(
                sequence_id=0,
                prefix_len=4,
                target_len=2,
                seq_len=8,
                block_ids=([1, 2],),
            ),
        ),
    )


def _scheduler_output(*, metadata=None, request_id: str = "req-0") -> DiffusionSchedulerOutput:
    req = SimpleNamespace(
        request_id=request_id,
        kv_sender_info=None,
        kv_transfer_params={"transfer_id": "xfer-req-0", "do_remote_prefill": True},
        sampling_params=SimpleNamespace(),
        prompt="hi",
        prepared_layout=None,
    )
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[
            NewRequestData(request_id=request_id, req=req, diffusion_kv_metadata=_metadata(request_id))
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
        kv_connector_metadata=metadata,
    )


def test_start_remote_kv_load_binds_metadata(monkeypatch) -> None:
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = SimpleNamespace()
    start = Mock()
    monkeypatch.setattr("vllm_omni.diffusion.worker.diffusion_model_runner.start_load_kv", start)
    sched = _scheduler_output(metadata=object())
    runner._maybe_start_remote_kv_load(sched)

    start.assert_called_once()
