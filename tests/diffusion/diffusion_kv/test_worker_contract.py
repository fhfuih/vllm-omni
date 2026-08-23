# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.metadata import (
    DiffusionKVMetadata,
    DiffusionKVSequenceMetadata,
)
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def make_metadata(request_id: str = "req-0") -> DiffusionKVMetadata:
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


def make_scheduler_output(*new_reqs: NewRequestData) -> DiffusionSchedulerOutput:
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=len(new_reqs),
        num_waiting_reqs=0,
    )


def make_executor(
    mode: DiffusionKVCacheMode = DiffusionKVCacheMode.DENSE_LEGACY,
) -> tuple[MultiprocDiffusionExecutor, list[tuple]]:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(
        enable_distributed_layerwise_offload=False,
        dlo_use_allgather=True,
        diffusion_kv_mode=mode,
        dist_timeout=1.0,
    )
    executor._ensure_open = lambda: None
    calls: list[tuple] = []

    def collective_rpc(method, *, args, **kwargs):
        calls.append((method, args, kwargs))
        if method == "prepare_kv_for_forward":
            return None
        return DiffusionOutput(output=None)

    executor.collective_rpc = collective_rpc
    return executor, calls


def test_new_request_data_carries_scheduler_allocation_atomically() -> None:
    req = SimpleNamespace(request_id="req-0")
    state = SimpleNamespace(request_id="req-0", req=req)
    metadata = make_metadata()

    new_req = NewRequestData.from_state(state, diffusion_kv_metadata=metadata)

    assert new_req.req is req
    assert new_req.diffusion_kv_metadata is metadata


@pytest.mark.parametrize(
    ("state_request_id", "forwarded_request_id", "metadata_request_id"),
    [
        ("state-id", "forwarded-id", "state-id"),
        ("state-id", "state-id", "metadata-id"),
    ],
)
def test_scheduler_owns_new_request_identity_validation(
    state_request_id: str,
    forwarded_request_id: str,
    metadata_request_id: str,
) -> None:
    state = SimpleNamespace(
        request_id=state_request_id,
        req=SimpleNamespace(request_id=forwarded_request_id),
    )

    with pytest.raises(ValueError, match="request identity mismatch"):
        NewRequestData.from_state(
            state,
            diffusion_kv_metadata=make_metadata(metadata_request_id),
        )


@pytest.mark.parametrize(
    ("mode", "metadata", "message"),
    [
        (DiffusionKVCacheMode.PAGED_SCHEDULER, None, "requires Diffusion KV metadata"),
        (DiffusionKVCacheMode.DENSE_LEGACY, make_metadata(), "dense_legacy.*must not carry"),
        (DiffusionKVCacheMode.PAGED_SCHEDULER, make_metadata("stale"), "metadata request mismatch"),
    ],
)
def test_model_runner_validates_allocation_before_forward(
    mode: DiffusionKVCacheMode,
    metadata: DiffusionKVMetadata | None,
    message: str,
) -> None:
    runtime = SimpleNamespace(
        install_allocations=Mock(),
        remove_diffusion_kv_requests=Mock(),
    )
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = SimpleNamespace(diffusion_kv_mode=mode)
    runner.diffusion_kv_backend = runtime
    runner.kv_connector = None
    new_req = NewRequestData(
        request_id="req-0",
        req=SimpleNamespace(request_id="req-0"),
        diffusion_kv_metadata=metadata,
    )

    with pytest.raises(ValueError, match=message):
        runner.prepare_kv_for_forward(make_scheduler_output(new_req), timeout_s=1.0, rank=0)

    runtime.remove_diffusion_kv_requests.assert_not_called()
    runtime.install_allocations.assert_not_called()


def test_dense_request_rpc_keeps_legacy_positional_shape() -> None:
    executor, calls = make_executor()
    prepared_layout = object()
    req = SimpleNamespace(request_id="req-0", prepared_layout=prepared_layout)
    new_req = NewRequestData(request_id="req-0", req=req)

    result = executor.execute_request(make_scheduler_output(new_req))

    assert result.request_ids == ["req-0"]
    assert [(method, args) for method, args, _ in calls] == [
        ("execute_model", (req, executor.od_config, None)),
    ]
    assert calls[0][1][0].prepared_layout is prepared_layout


def test_request_rpc_prepares_allocations_then_keeps_runner_signature_small() -> None:
    executor, calls = make_executor(DiffusionKVCacheMode.PAGED_SCHEDULER)
    req_0 = SimpleNamespace(request_id="req-0")
    req_1 = SimpleNamespace(request_id="req-1")
    metadata_0 = make_metadata("req-0")
    metadata_1 = make_metadata("req-1")
    new_reqs = (
        NewRequestData(request_id="req-0", req=req_0, diffusion_kv_metadata=metadata_0),
        NewRequestData(request_id="req-1", req=req_1, diffusion_kv_metadata=metadata_1),
    )

    result = executor.execute_request(make_scheduler_output(*new_reqs))

    assert result.request_ids == ["req-0", "req-1"]
    assert [(method, args) for method, args, _ in calls] == [
        ("prepare_kv_for_forward", (make_scheduler_output(*new_reqs), 1.0)),
        ("execute_model", (req_0, executor.od_config, None)),
        ("execute_model", (req_1, executor.od_config, None)),
    ]


def test_dense_worker_call_does_not_extend_model_runner_signature() -> None:
    calls: list[tuple] = []

    class DenseModelRunner:
        def execute_model(self, req, kv_prefetch_job=None):
            calls.append((req, kv_prefetch_job))
            return DiffusionOutput(output=None)

    worker = object.__new__(DiffusionWorker)
    worker.model_runner = DenseModelRunner()
    worker.lora_manager = None
    worker._get_profiler = lambda: None
    req = SimpleNamespace(request_id="req-0")

    worker.execute_model(req, SimpleNamespace())

    assert calls == [(req, None)]


def test_dlo_worker_selects_request_without_forwarding_allocation(monkeypatch) -> None:
    calls: list[tuple] = []

    class ModelRunner:
        def execute_model(self, req, **kwargs):
            calls.append((req, kwargs))
            return DiffusionOutput(output=None)

    worker = object.__new__(DiffusionWorker)
    worker.model_runner = ModelRunner()
    worker.lora_manager = None
    worker._get_profiler = lambda: None
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_data_parallel_rank",
        lambda: 1,
    )
    req_0 = SimpleNamespace(request_id="req-0")
    req_1 = SimpleNamespace(request_id="req-1")
    metadata_0 = make_metadata("req-0")
    metadata_1 = make_metadata("req-1")
    envelopes = [
        NewRequestData(request_id="req-0", req=req_0, diffusion_kv_metadata=metadata_0),
        NewRequestData(request_id="req-1", req=req_1, diffusion_kv_metadata=metadata_1),
    ]

    result = worker.execute_model(envelopes, SimpleNamespace())

    assert calls == [
        (
            req_1,
            {
                "kv_prefetch_job": None,
            },
        )
    ]
    assert result["dp_rank"] == 1
