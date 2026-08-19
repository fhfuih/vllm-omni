# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import vllm.v1.core.single_type_kv_cache_manager as native_kv_managers
from vllm.config import KVTransferConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.v1.connector import (
    create_scheduler_kv_connector_v1,
    maybe_register_vllm_kv_caches,
    mint_transfer_id,
    parse_kv_transfer_config,
    shutdown_kv_connector_v1,
)
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

KV_TRANSFER_YAML = {
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_consumer",
    "engine_id": "dit-engine-1",
    "kv_connector_extra_config": {"mooncake_protocol": "rdma"},
}


def _paged_od_config(**kwargs) -> OmniDiffusionConfig:
    return OmniDiffusionConfig.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        max_model_len=64,
        kv_transfer_config=dict(KV_TRANSFER_YAML),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _fixed_master_port(monkeypatch) -> None:
    monkeypatch.setattr(OmniDiffusionConfig, "_resolve_master_port", lambda _self: 29500)


def test_kv_transfer_config_roundtrip_to_worker_vllm_config() -> None:
    od_config = _paged_od_config()

    assert od_config.kv_transfer_config is not None
    assert isinstance(od_config.kv_transfer_config, KVTransferConfig)
    assert od_config.kv_transfer_config.kv_connector == "MooncakeConnector"
    assert od_config.kv_transfer_config.kv_role == "kv_consumer"
    assert od_config.kv_transfer_config.engine_id == "dit-engine-1"
    assert od_config.kv_transfer_config.kv_connector_extra_config == {"mooncake_protocol": "rdma"}

    vllm_config = create_diffusion_vllm_config(torch.device("cpu"), od_config)
    assert vllm_config.kv_transfer_config is od_config.kv_transfer_config
    assert vllm_config.kv_transfer_config.engine_id == "dit-engine-1"
    assert vllm_config.cache_config is not None


def test_unconfigured_does_not_create_scheduler_connector() -> None:
    od_config = OmniDiffusionConfig.from_kwargs()
    assert od_config.kv_transfer_config is None
    assert create_scheduler_kv_connector_v1(od_config, kv_cache_config=None, vllm_config=None) is None


class _ConcreteScheduler(BaseScheduler):
    def update_from_output(self, sched_output, output) -> set[str]:
        del sched_output, output
        return set()


def _initialize_paged_scheduler(scheduler: BaseScheduler, od_config: OmniDiffusionConfig) -> KVCacheConfig:
    native_kv_managers.register_all_kvcache_specs(None)
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[KVCacheTensor(size=1024, shared_by=["layer.0"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)],
    )
    scheduler.initialize(
        od_config,
        kv_cache_config=kv_cache_config,
        scheduler_block_size=4,
        hash_block_size=4,
        kv_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=64),
            max_in_flight_tokens=64,
            kv_transfer_config=od_config.kv_transfer_config,
        ),
    )
    return kv_cache_config


def test_scheduler_creates_kv_connector_v1_when_configured() -> None:
    od_config = _paged_od_config()
    scheduler = _ConcreteScheduler()
    fake_connector = mock.Mock()

    with mock.patch(
        "vllm_omni.diffusion.diffusion_kv.v1.connector.KVConnectorFactory.create_connector",
        return_value=fake_connector,
    ) as create_connector:
        kv_cache_config = _initialize_paged_scheduler(scheduler, od_config)

    assert scheduler.kv_connector_v1 is fake_connector
    create_connector.assert_called_once()
    _, kwargs = create_connector.call_args
    assert kwargs["role"].name == "SCHEDULER"
    assert kwargs["kv_cache_config"] is kv_cache_config
    assert kwargs["kv_cache_config"].num_blocks == 8


def test_worker_init_device_defers_connector_until_kv_config(monkeypatch) -> None:
    od_config = _paged_od_config()
    init_mock = mock.Mock()

    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.init_worker_kv_connector_v1",
        init_mock,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.get_torch_device",
        lambda rank: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.set_device", lambda device: None
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.init_diffusion_worker_vllm_config",
        lambda config: None,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.empty_cache",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.init_distributed_environment", lambda **kwargs: None
    )
    monkeypatch.setattr("vllm_omni.diffusion.worker.diffusion_worker.initialize_model_parallel", lambda **kwargs: None)
    monkeypatch.setattr("vllm_omni.diffusion.worker.diffusion_worker.init_workspace_manager", lambda device: None)
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.MemorySnapshot",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.request_memory",
        lambda *args, **kwargs: 0,
    )

    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    worker = DiffusionWorker.__new__(DiffusionWorker)
    worker.local_rank = 0
    worker.rank = 0
    worker.od_config = od_config
    worker.device = None
    worker.vllm_config = None
    worker.model_runner = None
    worker.init_snapshot = None
    worker.requested_memory = None
    worker._sleep_saved_buffers = {}
    worker.lora_manager = None
    worker._step_lora_state = {}
    worker.stage_id = 0

    worker.init_device()
    init_mock.assert_not_called()


def test_set_kv_cache_configs_inits_worker_connector(monkeypatch) -> None:
    od_config = _paged_od_config()
    init_mock = mock.Mock()
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.init_worker_kv_connector_v1",
        init_mock,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.has_kv_transfer_group",
        lambda: False,
    )

    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    worker = DiffusionWorker.__new__(DiffusionWorker)
    worker.rank = 0
    worker.od_config = od_config
    worker.vllm_config = create_diffusion_vllm_config(torch.device("cpu"), od_config)
    runner = SimpleNamespace(set_kv_cache_config=mock.Mock())
    worker.model_runner = runner

    rank_config = KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=[])
    worker.set_kv_cache_configs([rank_config])

    runner.set_kv_cache_config.assert_called_once_with(rank_config)
    init_mock.assert_called_once_with(worker.vllm_config, kv_cache_config=rank_config)


def test_skip_register_without_layer_name_kv_caches() -> None:
    with mock.patch("vllm_omni.diffusion.diffusion_kv.v1.connector.get_kv_transfer_group") as get_group:
        maybe_register_vllm_kv_caches(None)
        maybe_register_vllm_kv_caches({})
        get_group.assert_not_called()


def test_register_vllm_kv_caches_when_dict_present() -> None:
    fake_group = mock.Mock()
    kv_caches = {"layer.0": torch.zeros(1)}

    with (
        mock.patch(
            "vllm_omni.diffusion.diffusion_kv.v1.connector.has_kv_transfer_group",
            return_value=True,
        ),
        mock.patch(
            "vllm_omni.diffusion.diffusion_kv.v1.connector.get_kv_transfer_group",
            return_value=fake_group,
        ),
    ):
        maybe_register_vllm_kv_caches(kv_caches)

    fake_group.register_kv_caches.assert_called_once_with(kv_caches)


def test_shutdown_is_idempotent() -> None:
    scheduler_connector = mock.Mock()

    with (
        mock.patch(
            "vllm_omni.diffusion.diffusion_kv.v1.connector.has_kv_transfer_group",
            side_effect=[True, False],
        ),
        mock.patch(
            "vllm_omni.diffusion.diffusion_kv.v1.connector.ensure_kv_transfer_shutdown",
        ) as ensure_shutdown,
    ):
        shutdown_kv_connector_v1(scheduler_connector=scheduler_connector)
        shutdown_kv_connector_v1(scheduler_connector=None)

    scheduler_connector.shutdown.assert_called_once()
    ensure_shutdown.assert_called_once()


def test_parse_kv_transfer_config_empty_mapping_returns_none() -> None:
    assert parse_kv_transfer_config({}) is None


def test_parse_kv_transfer_config_requires_engine_id() -> None:
    payload = dict(KV_TRANSFER_YAML)
    payload["engine_id"] = "   "
    with pytest.raises(ValueError, match="requires a non-empty engine_id"):
        parse_kv_transfer_config(payload)

    payload.pop("engine_id")
    with pytest.raises(ValueError, match="requires a non-empty engine_id"):
        parse_kv_transfer_config(payload)


def test_kv_transfer_rejects_dense_legacy() -> None:
    with pytest.raises(ValueError, match="requires diffusion_kv_mode='paged_scheduler'"):
        OmniDiffusionConfig.from_kwargs(kv_transfer_config=dict(KV_TRANSFER_YAML))


def test_scheduler_close_shuts_down_kv_connector_v1() -> None:
    od_config = _paged_od_config()
    scheduler = _ConcreteScheduler()
    fake_connector = mock.Mock()

    with mock.patch(
        "vllm_omni.diffusion.diffusion_kv.v1.connector.create_scheduler_kv_connector_v1",
        return_value=fake_connector,
    ):
        _initialize_paged_scheduler(scheduler, od_config)

    with mock.patch(
        "vllm_omni.diffusion.diffusion_kv.v1.connector.shutdown_kv_connector_v1",
    ) as shutdown_mock:
        scheduler.close()

    shutdown_mock.assert_called_once_with(scheduler_connector=fake_connector)
    assert scheduler.kv_connector_v1 is None


def test_mint_transfer_id_is_stable_per_request() -> None:
    assert mint_transfer_id("req-a") == mint_transfer_id("req-a")
    assert mint_transfer_id("req-a") != mint_transfer_id("req-b")
