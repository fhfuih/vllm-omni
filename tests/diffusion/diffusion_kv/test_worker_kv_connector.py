# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput

from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata, DiffusionKVSequenceMetadata
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

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


def test_set_kv_cache_configs_registers_backend_pages(monkeypatch) -> None:
    worker = object.__new__(DiffusionWorker)
    worker.rank = 0
    worker.od_config = SimpleNamespace(num_gpus=1)
    worker.vllm_config = SimpleNamespace(kv_transfer_config=object())
    fake_backend = SimpleNamespace(kv_caches_by_layer={"layer.0": torch.zeros(1)})
    runner = object.__new__(DiffusionModelRunner)
    runner.diffusion_kv_backend = fake_backend
    runner.set_kv_cache_config = Mock()
    runner.initialize_kv_cache = Mock()
    worker.model_runner = runner

    register = Mock()
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.has_kv_transfer_group",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.init_worker_kv_connector_v1",
        Mock(),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.maybe_register_vllm_kv_caches",
        register,
    )

    worker.set_kv_cache_configs([KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=[])])

    runner.set_kv_cache_config.assert_called_once()
    runner.initialize_kv_cache.assert_called_once()
    register.assert_called_once_with({"layer.0": fake_backend.kv_caches_by_layer["layer.0"]})


def test_empty_kv_caches_by_layer_skips_register(monkeypatch) -> None:
    worker = object.__new__(DiffusionWorker)
    worker.rank = 0
    worker.od_config = SimpleNamespace(num_gpus=1)
    worker.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner = object.__new__(DiffusionModelRunner)
    runner.diffusion_kv_backend = None
    runner.set_kv_cache_config = Mock()
    runner.initialize_kv_cache = Mock()
    worker.model_runner = runner
    register = Mock()
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.maybe_register_vllm_kv_caches",
        register,
    )

    worker.set_kv_cache_configs([KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])])

    register.assert_called_once_with({})


def test_execute_stepwise_starts_load_and_skips_omnikv(monkeypatch) -> None:
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = SimpleNamespace(
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        kv_transfer_config=object(),
    )
    runner.kv_transfer_manager = SimpleNamespace(
        config=SimpleNamespace(recv_timeout=1.0),
        receive_multi_kv_cache_distributed=Mock(),
    )
    start = Mock()
    wait = Mock(return_value=KVConnectorOutput(finished_recving={"req-0"}))
    monkeypatch.setattr("vllm_omni.diffusion.worker.diffusion_model_runner.start_load_kv", start)
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_model_runner.wait_remote_kv_before_forward",
        wait,
    )
    sched = _scheduler_output(metadata=object())
    output = runner._maybe_load_remote_kv(sched)
    start.assert_called_once()
    wait.assert_called_once()
    assert output.finished_recving == {"req-0"}
    assert runner._should_receive_omnikv(sched.scheduled_new_reqs[0].req, scheduler_output=sched) is False
    runner.kv_transfer_manager.receive_multi_kv_cache_distributed.assert_not_called()


def test_omnikv_bypass_when_metadata_present() -> None:
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = SimpleNamespace(
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        kv_transfer_config=object(),
    )
    sched = _scheduler_output(metadata=object())
    assert runner._should_receive_omnikv(sched.scheduled_new_reqs[0].req, scheduler_output=sched) is False


def test_all_rank_aggregate_merges_finished_recving() -> None:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor._kv_output_aggregator = None
    rank0 = BatchRunnerOutput.from_list(
        [RunnerOutput(request_id="req-0", finished=True)],
        kv_connector_output=KVConnectorOutput(finished_recving={"req-0"}),
    )
    rank1 = BatchRunnerOutput.from_list(
        [RunnerOutput(request_id="req-0", finished=True)],
        kv_connector_output=KVConnectorOutput(finished_recving={"req-0"}),
    )

    merged = executor._merge_native_kv_rank_outputs([rank0, rank1])

    assert merged is rank0
    assert merged.kv_connector_output is not None
    assert merged.kv_connector_output.finished_recving == {"req-0"}
