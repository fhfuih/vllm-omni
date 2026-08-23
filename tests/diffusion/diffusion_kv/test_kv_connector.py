# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import vllm.distributed.parallel_state as vllm_parallel_state
from vllm.config import KVTransferConfig

from tests.diffusion.diffusion_kv.helper import (
    DEFAULT_KV_TRANSFER_YAML,
    ConcreteScheduler,
    initialize_paged_scheduler,
    make_kv_cache_config,
    make_paged_od_config,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.config import parse_kv_transfer_config
from vllm_omni.diffusion.diffusion_kv.kv_connector import (
    _use_kv_transfer_shard_group,
    shutdown_kv_connector_v1,
)
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _fixed_master_port(monkeypatch) -> None:
    monkeypatch.setattr(OmniDiffusionConfig, "_resolve_master_port", lambda _self: 29500)


def test_kv_transfer_config_roundtrip_to_worker_vllm_config() -> None:
    od_config = make_paged_od_config()

    assert od_config.kv_transfer_config is not None
    assert isinstance(od_config.kv_transfer_config, KVTransferConfig)
    assert od_config.kv_transfer_config.kv_connector == "MooncakeConnector"
    assert od_config.kv_transfer_config.kv_role == "kv_consumer"
    assert od_config.kv_transfer_config.engine_id == "dit-engine-1"
    assert od_config.kv_transfer_config.kv_connector_extra_config == {"mooncake_protocol": "rdma"}

    vllm_config = create_diffusion_vllm_config(torch.device("cpu"), od_config)
    assert vllm_config.kv_transfer_config is od_config.kv_transfer_config
    assert vllm_config.cache_config is not None


def test_scheduler_creates_kv_connector_v1_when_configured() -> None:
    od_config = make_paged_od_config()
    scheduler = ConcreteScheduler()
    fake_connector = mock.Mock()

    with mock.patch(
        "vllm_omni.diffusion.diffusion_kv.kv_connector.KVConnectorFactory.create_connector",
        return_value=fake_connector,
    ) as create_connector:
        kv_cache_config = initialize_paged_scheduler(scheduler, od_config=od_config, num_blocks=8)

    assert scheduler.get_kv_connector() is fake_connector
    create_connector.assert_called_once()
    _, kwargs = create_connector.call_args
    assert kwargs["role"].name == "SCHEDULER"
    assert kwargs["kv_cache_config"] is kv_cache_config
    assert kwargs["kv_cache_config"].num_blocks == 8
    assert kwargs["config"].kv_transfer_config is od_config.kv_transfer_config


def test_model_runner_set_kv_cache_config_inits_connector(monkeypatch) -> None:
    od_config = make_paged_od_config()
    init_mock = mock.Mock()
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_model_runner.init_worker_kv_connector_v1",
        init_mock,
    )
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

    runner = DiffusionModelRunner.__new__(DiffusionModelRunner)
    runner.vllm_config = create_diffusion_vllm_config(torch.device("cpu"), od_config)
    runner.kv_connector = None
    cache = torch.zeros(1)
    runner.diffusion_kv_backend = SimpleNamespace(
        initialize_kv_cache=mock.Mock(),
        get_kv_caches_by_layer=mock.Mock(return_value={"layer.0": cache}),
    )

    rank_config = make_kv_cache_config(num_blocks=4)
    runner.set_kv_cache_config(rank_config)

    runner.diffusion_kv_backend.initialize_kv_cache.assert_called_once_with(rank_config)
    init_mock.assert_called_once_with(
        runner.vllm_config,
        kv_cache_config=rank_config,
        kv_caches_by_layer={"layer.0": cache},
    )
    assert runner.kv_connector is init_mock.return_value


def test_worker_connector_discovers_sequence_parallel_transfer_shards(monkeypatch) -> None:
    import vllm_omni.diffusion.distributed.parallel_state as diffusion_parallel_state

    original_tp_group = object()
    sequence_parallel_group = SimpleNamespace(world_size=2)
    monkeypatch.setattr(vllm_parallel_state, "_TP", original_tp_group)
    monkeypatch.setattr(vllm_parallel_state, "_PCP", sequence_parallel_group)
    monkeypatch.setattr(diffusion_parallel_state, "_SP", sequence_parallel_group)

    with _use_kv_transfer_shard_group():
        assert vllm_parallel_state._TP is sequence_parallel_group

    assert vllm_parallel_state._TP is original_tp_group


def test_shutdown_is_idempotent() -> None:
    scheduler_connector = mock.Mock()

    with (
        mock.patch(
            "vllm_omni.diffusion.diffusion_kv.kv_connector.has_kv_transfer_group",
            side_effect=[True, False],
        ),
        mock.patch(
            "vllm_omni.diffusion.diffusion_kv.kv_connector.ensure_kv_transfer_shutdown",
        ) as ensure_shutdown,
    ):
        shutdown_kv_connector_v1(scheduler_connector=scheduler_connector)
        shutdown_kv_connector_v1(scheduler_connector=None)

    scheduler_connector.shutdown.assert_called_once()
    ensure_shutdown.assert_called_once()


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({}, None),
        ({**DEFAULT_KV_TRANSFER_YAML, "engine_id": "   "}, "requires a non-empty engine_id"),
        ({k: v for k, v in DEFAULT_KV_TRANSFER_YAML.items() if k != "engine_id"}, "requires a non-empty engine_id"),
    ],
    ids=["empty", "blank_engine_id", "missing_engine_id"],
)
def test_parse_kv_transfer_config(payload, match: str | None) -> None:
    if match is None:
        assert parse_kv_transfer_config(payload) is None
        return
    with pytest.raises(ValueError, match=match):
        parse_kv_transfer_config(payload)


def test_kv_transfer_rejects_dense_legacy() -> None:
    with pytest.raises(ValueError, match="requires diffusion_kv_mode='paged_scheduler'"):
        OmniDiffusionConfig.from_kwargs(kv_transfer_config=dict(DEFAULT_KV_TRANSFER_YAML))


def test_scheduler_close_shuts_down_kv_connector_v1() -> None:
    od_config = make_paged_od_config()
    scheduler = ConcreteScheduler()
    fake_connector = mock.Mock()

    with mock.patch(
        "vllm_omni.diffusion.sched.base_scheduler.create_scheduler_kv_connector_v1",
        return_value=fake_connector,
    ):
        initialize_paged_scheduler(scheduler, od_config=od_config, num_blocks=8)

    with mock.patch(
        "vllm_omni.diffusion.sched.base_scheduler.shutdown_kv_connector_v1",
    ) as shutdown_mock:
        scheduler.close()

    shutdown_mock.assert_called_once_with(scheduler_connector=fake_connector)
    assert scheduler.connector is None
