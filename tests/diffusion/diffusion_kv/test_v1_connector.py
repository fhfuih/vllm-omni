# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from tests.diffusion.diffusion_kv.helper import (
    DEFAULT_KV_TRANSFER_YAML,
    ConcreteScheduler,
    initialize_paged_scheduler,
    make_paged_od_config,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.v1.connector import (
    create_scheduler_kv_connector_v1,
    maybe_register_vllm_kv_caches,
    parse_kv_transfer_config,
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


def test_unconfigured_does_not_create_scheduler_connector() -> None:
    od_config = OmniDiffusionConfig.from_kwargs()
    assert od_config.kv_transfer_config is None
    assert create_scheduler_kv_connector_v1(od_config, kv_cache_config=None, vllm_config=None) is None


def test_scheduler_creates_kv_connector_v1_when_configured() -> None:
    od_config = make_paged_od_config()
    scheduler = ConcreteScheduler()
    fake_connector = mock.Mock()

    with mock.patch(
        "vllm_omni.diffusion.diffusion_kv.v1.connector.KVConnectorFactory.create_connector",
        return_value=fake_connector,
    ) as create_connector:
        kv_cache_config = initialize_paged_scheduler(scheduler, od_config=od_config, num_blocks=8)

    assert scheduler._kv_connector_v1 is fake_connector
    create_connector.assert_called_once()
    _, kwargs = create_connector.call_args
    assert kwargs["role"].name == "SCHEDULER"
    assert kwargs["kv_cache_config"] is kv_cache_config
    assert kwargs["kv_cache_config"].num_blocks == 8
    assert kwargs["config"].kv_transfer_config is od_config.kv_transfer_config


def test_set_kv_cache_configs_inits_worker_connector(monkeypatch) -> None:
    od_config = make_paged_od_config()
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


@pytest.mark.parametrize(
    "kv_caches",
    [None, {}, {"layer.0": torch.zeros(1)}],
    ids=["none", "empty", "named"],
)
def test_maybe_register_vllm_kv_caches(kv_caches) -> None:
    fake_group = mock.Mock()
    with (
        mock.patch(
            "vllm_omni.diffusion.diffusion_kv.v1.connector.has_kv_transfer_group",
            return_value=True,
        ),
        mock.patch(
            "vllm_omni.diffusion.diffusion_kv.v1.connector.get_kv_transfer_group",
            return_value=fake_group,
        ) as get_group,
    ):
        maybe_register_vllm_kv_caches(kv_caches)

    if isinstance(kv_caches, dict) and kv_caches:
        fake_group.register_kv_caches.assert_called_once_with(kv_caches)
    else:
        get_group.assert_not_called()
        fake_group.register_kv_caches.assert_not_called()


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
        "vllm_omni.diffusion.diffusion_kv.v1.connector.create_scheduler_kv_connector_v1",
        return_value=fake_connector,
    ):
        initialize_paged_scheduler(scheduler, od_config=od_config, num_blocks=8)

    with mock.patch(
        "vllm_omni.diffusion.diffusion_kv.v1.connector.shutdown_kv_connector_v1",
    ) as shutdown_mock:
        scheduler.close()

    shutdown_mock.assert_called_once_with(scheduler_connector=fake_connector)
    assert scheduler._kv_connector_v1 is None
