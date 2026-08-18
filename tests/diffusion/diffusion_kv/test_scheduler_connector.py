# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import vllm.v1.core.single_type_kv_cache_manager as native_kv_managers
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

KV_TRANSFER_YAML = {
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_consumer",
    "engine_id": "dit-engine-1",
    "kv_connector_extra_config": {"mooncake_protocol": "rdma"},
}

_FAKE_TRANSFER_PARAMS = {
    "do_remote_prefill": True,
    "remote_engine_id": "ar-engine-1",
    "remote_bootstrap_addr": "tcp://127.0.0.1:9999",
    "transfer_id": "xfer-1",
}


class _ConcreteScheduler(BaseScheduler):
    def update_from_output(self, sched_output, output) -> set[str]:
        del sched_output, output
        return set()


def _kv_cache_config(*, num_blocks: int = 64) -> KVCacheConfig:
    native_kv_managers.register_all_kvcache_specs(None)
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=["layer0"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer0"], kv_cache_spec=spec)],
    )


def _initialize_paged_scheduler(
    scheduler: BaseScheduler,
    *,
    kv_transfer_config: KVTransferConfig | None = None,
    num_blocks: int = 64,
    max_num_seqs: int = 1,
) -> KVCacheConfig:
    config = _kv_cache_config(num_blocks=num_blocks)
    scheduler.initialize(
        SimpleNamespace(
            kv_transfer_config=kv_transfer_config,
            diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
            max_model_len=64,
            max_num_seqs=max_num_seqs,
        ),
        kv_cache_config=config,
        scheduler_block_size=4,
        hash_block_size=4,
        kv_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=64),
            max_in_flight_tokens=64,
            kv_transfer_config=kv_transfer_config,
        ),
    )
    return config


def _make_request(
    req_id: str,
    *,
    num_sequences: int = 1,
    seq_len: int = 8,
    kv_transfer_params: dict | None = None,
) -> OmniDiffusionRequest:
    sequences = tuple(
        DiffusionKVRequest(
            f"{req_id}/diffusion-kv/{sequence_id}",
            sequence_id=sequence_id,
            prefix_len=4,
            target_len=4,
            seq_len=seq_len,
        )
        for sequence_id in range(num_sequences)
    )
    return OmniDiffusionRequest(
        prompt=f"prompt_{req_id}",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_id=req_id,
        diffusion_kv_requests=sequences,
        kv_transfer_params=kv_transfer_params,
    )


def _fake_connector(*, ext_tokens: int | None = 8, load_async: bool = True) -> Mock:
    connector = Mock(spec=KVConnectorBase_V1)
    connector.get_num_new_matched_tokens.return_value = (ext_tokens, load_async)
    connector.build_connector_meta.return_value = object()
    return connector


def test_no_connector_allocates_and_runs_without_metadata() -> None:
    scheduler = _ConcreteScheduler()
    _initialize_paged_scheduler(scheduler)
    request = _make_request("req-0")

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.kv_connector_metadata is None
    assert [req.request_id for req in output.scheduled_new_reqs] == ["req-0"]
    assert scheduler.get_request_state("req-0").status is DiffusionRequestStatus.RUNNING
    assert scheduler._diffusion_kv_manager.has_request("req-0")


def test_no_params_skips_connector_metadata() -> None:
    scheduler = _ConcreteScheduler()
    _initialize_paged_scheduler(scheduler)
    connector = _fake_connector(ext_tokens=0, load_async=False)
    scheduler._kv_connector_v1 = connector
    request = _make_request("req-0")

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.kv_connector_metadata is None
    assert scheduler.get_request_state("req-0").status is DiffusionRequestStatus.RUNNING
    connector.build_connector_meta.assert_not_called()
    assert connector.get_num_new_matched_tokens.call_count == 1
    connector.update_state_after_alloc.assert_called_once()


def test_cfg_sequences_each_call_connector_and_build_meta_once() -> None:
    scheduler = _ConcreteScheduler()
    _initialize_paged_scheduler(scheduler)
    connector = _fake_connector(ext_tokens=8, load_async=True)
    expected_meta = connector.build_connector_meta.return_value
    scheduler._kv_connector_v1 = connector
    request = _make_request("req-cfg", num_sequences=2, kv_transfer_params=dict(_FAKE_TRANSFER_PARAMS))

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.kv_connector_metadata is expected_meta
    assert scheduler.get_request_state("req-cfg").status is DiffusionRequestStatus.RUNNING
    assert connector.get_num_new_matched_tokens.call_count == 2
    assert connector.update_state_after_alloc.call_count == 2
    connector.build_connector_meta.assert_called_once()
    seq_ids = [call.args[0].request_id for call in connector.get_num_new_matched_tokens.call_args_list]
    assert seq_ids == ["req-cfg/diffusion-kv/0", "req-cfg/diffusion-kv/1"]
    for call in connector.get_num_new_matched_tokens.call_args_list:
        assert call.args[1] == 0


def test_load_async_still_enters_running() -> None:
    scheduler = _ConcreteScheduler()
    _initialize_paged_scheduler(scheduler)
    connector = _fake_connector(ext_tokens=8, load_async=True)
    scheduler._kv_connector_v1 = connector
    request = _make_request("req-0", kv_transfer_params=dict(_FAKE_TRANSFER_PARAMS))

    scheduler.add_request(request)
    output = scheduler.schedule()

    state = scheduler.get_request_state("req-0")
    assert state.status is DiffusionRequestStatus.RUNNING
    assert not hasattr(DiffusionRequestStatus, "WAITING_FOR_REMOTE_KVS")
    assert output.scheduled_new_reqs[0].request_id == "req-0"


def test_public_params_are_copied_per_sequence() -> None:
    scheduler = _ConcreteScheduler()
    _initialize_paged_scheduler(scheduler)
    public_params = dict(_FAKE_TRANSFER_PARAMS)
    request = _make_request("req-cfg", num_sequences=2, kv_transfer_params=public_params)
    scheduler.add_request(request)

    sequences = scheduler.get_request_state("req-cfg").diffusion_kv_requests
    assert len(sequences) == 2
    assert sequences[0].kv_transfer_params == public_params
    assert sequences[1].kv_transfer_params == public_params
    assert sequences[0].kv_transfer_params is not public_params
    assert sequences[1].kv_transfer_params is not public_params
    assert sequences[0].kv_transfer_params is not sequences[1].kv_transfer_params


def test_missing_prompt_token_ids_are_filled_to_seq_len() -> None:
    scheduler = _ConcreteScheduler()
    _initialize_paged_scheduler(scheduler)
    request = _make_request("req-0", seq_len=12)
    scheduler.add_request(request)

    seq = scheduler.get_request_state("req-0").diffusion_kv_requests[0]
    assert seq.prompt_token_ids == [0] * 12


def test_ext_tokens_none_frees_and_stays_waiting() -> None:
    scheduler = _ConcreteScheduler()
    _initialize_paged_scheduler(scheduler)
    connector = _fake_connector(ext_tokens=None, load_async=False)
    scheduler._kv_connector_v1 = connector
    request = _make_request("req-0", kv_transfer_params=dict(_FAKE_TRANSFER_PARAMS))

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.scheduled_new_reqs == []
    assert output.kv_connector_metadata is None
    assert scheduler.get_request_state("req-0").status is DiffusionRequestStatus.WAITING
    assert not scheduler._diffusion_kv_manager.has_request("req-0")
    connector.build_connector_meta.assert_not_called()


def test_finish_frees_reserved_pages() -> None:
    scheduler = _ConcreteScheduler()
    _initialize_paged_scheduler(scheduler)
    request = _make_request("req-0")
    scheduler.add_request(request)
    scheduler.schedule()

    assert scheduler._diffusion_kv_manager.has_request("req-0")
    scheduler.finish_requests("req-0", DiffusionRequestStatus.FINISHED_COMPLETED)
    assert not scheduler._diffusion_kv_manager.has_request("req-0")


def test_paged_initialize_passes_real_kv_cache_config() -> None:
    scheduler = _ConcreteScheduler()
    fake_connector = Mock()
    kv_transfer_config = KVTransferConfig(**KV_TRANSFER_YAML)
    cache_config = _kv_cache_config(num_blocks=48)
    created = {}

    def _create_connector(*, config, role, kv_cache_config):
        created["kv_cache_config"] = kv_cache_config
        created["vllm_config"] = config
        created["role"] = role
        return fake_connector

    with patch(
        "vllm_omni.diffusion.diffusion_kv.v1.connector.KVConnectorFactory.create_connector",
        side_effect=_create_connector,
    ):
        scheduler.initialize(
            SimpleNamespace(
                kv_transfer_config=kv_transfer_config,
                diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
                max_model_len=64,
                max_num_seqs=1,
            ),
            kv_cache_config=cache_config,
            scheduler_block_size=4,
            hash_block_size=4,
            kv_vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=64),
                max_in_flight_tokens=64,
                kv_transfer_config=kv_transfer_config,
            ),
        )

    assert scheduler.kv_connector_v1 is fake_connector
    assert created["kv_cache_config"] is cache_config
    assert created["kv_cache_config"].num_blocks == 48
    assert created["vllm_config"].kv_transfer_config is kv_transfer_config
