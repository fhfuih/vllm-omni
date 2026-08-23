# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.outputs import KVConnectorOutput

import vllm_omni.diffusion.diffusion_kv.kv_connector as connector_module
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.kv_connector import (
    KVConnectorFatalError,
    wait_for_remote_kv_before_forward,
)
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
        kv_connector_request_ids=frozenset({f"{request_id}/diffusion-kv/0"}),
    )


def test_model_runner_prepare_kv_for_forward_installs_bindings_and_waits(monkeypatch) -> None:
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = SimpleNamespace(diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.kv_connector = Mock()
    runtime = SimpleNamespace(
        install_allocations=Mock(return_value=True),
        remove_diffusion_kv_requests=Mock(return_value=0),
    )
    runner.diffusion_kv_backend = runtime
    wait = Mock(
        return_value=KVConnectorOutput(
            finished_recving={"req-0/diffusion-kv/0"},
        )
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_model_runner.wait_for_remote_kv_before_forward",
        wait,
    )
    sched = _scheduler_output(metadata=object())
    result = runner.prepare_kv_for_forward(sched, 9.0, rank=3)

    runtime.install_allocations.assert_called_once_with(sched.scheduled_new_reqs[0].diffusion_kv_metadata)
    wait.assert_called_once_with(
        runner.kv_connector,
        sched,
        timeout_s=9.0,
        rank=3,
    )
    assert result.kv_connector_output.finished_recving == {"req-0/diffusion-kv/0"}


def test_wait_for_remote_kv_rejects_invalid_blocks() -> None:
    connector = Mock()
    connector.post_forward.return_value = KVConnectorOutput(invalid_block_ids={7})

    with pytest.raises(KVConnectorFatalError, match="invalid target blocks"):
        wait_for_remote_kv_before_forward(
            connector,
            _scheduler_output(metadata=object()),
            timeout_s=1.0,
            rank=1,
        )


def test_wait_for_remote_kv_times_out(monkeypatch) -> None:
    connector = Mock()
    connector.post_forward.return_value = KVConnectorOutput()
    monotonic = iter((0.0, 1.0))
    monkeypatch.setattr(connector_module.time, "monotonic", lambda: next(monotonic))

    with pytest.raises(KVConnectorFatalError, match="timed out.*rank 4"):
        wait_for_remote_kv_before_forward(
            connector,
            _scheduler_output(metadata=object()),
            timeout_s=0.5,
            rank=4,
        )
