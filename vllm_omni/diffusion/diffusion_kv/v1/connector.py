# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV connector v1 assembly for the Diffusion stage.

Creates Scheduler- and Worker-role vLLM ``MooncakeConnector`` instances.
Scheduler admission uses the page pool when ``paged_scheduler`` is enabled.
The Worker binds connector metadata and starts a remote page load.

``v1`` here means Omni's new connector system versus the legacy Omni connector,
not upstream ``KVConnectorBase_V1``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata

    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

# Placeholder KVCacheConfig for Worker/Scheduler assembly before a real
# rank-local or Scheduler-owned config is available.
_PLACEHOLDER_BLOCK_SIZE = 16


def parse_kv_transfer_config(value: object | None) -> KVTransferConfig | None:
    """Normalize YAML / dict / dataclass into ``KVTransferConfig`` and validate it."""
    if value is None:
        return None
    if isinstance(value, KVTransferConfig):
        _validate_diffusion_kv_transfer_config(value)
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        if not payload:
            return None
        # Validate before construction: KVTransferConfig fills a random UUID
        # when engine_id is omitted, which would hide a missing handshake id.
        if not isinstance(payload.get("engine_id"), str) or not payload.get("engine_id").strip():
            raise ValueError("Diffusion stage kv_transfer_config requires a non-empty engine_id")
        kv_transfer_config = KVTransferConfig(**payload)
        _validate_diffusion_kv_transfer_config(kv_transfer_config)
        return kv_transfer_config
    raise TypeError(f"kv_transfer_config must be a mapping or KVTransferConfig, got {type(value)!r}")


def create_scheduler_kv_connector_v1(
    od_config: OmniDiffusionConfig,
    *,
    kv_cache_config: KVCacheConfig | None,
    vllm_config: VllmConfig | None,
) -> KVConnectorBase_V1 | None:
    """Create ``KVConnectorRole.SCHEDULER`` when KV connector v1 is configured.

    ``paged_scheduler`` passes the Scheduler-owned ``KVCacheConfig`` used for
    page allocation. May be omitted when the caller does not supply a config.
    """
    kv_transfer_config = od_config.kv_transfer_config
    if kv_transfer_config is None:
        return None
    if not isinstance(kv_transfer_config, KVTransferConfig):
        raise TypeError(
            f"Diffusion stage kv_transfer_config must be KVTransferConfig, got {type(kv_transfer_config)!r}"
        )

    if vllm_config is None:
        connector_vllm_config = _build_kv_connector_v1_vllm_config(od_config)
    else:
        connector_vllm_config = vllm_config
        if connector_vllm_config.kv_transfer_config is None:
            connector_vllm_config.kv_transfer_config = kv_transfer_config

    cache_config = kv_cache_config if kv_cache_config is not None else _placeholder_kv_cache_config()
    connector = KVConnectorFactory.create_connector(
        config=connector_vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=cache_config,
    )
    logger.info(
        "Created KV connector v1 (SCHEDULER role): connector=%s engine_id=%s",
        kv_transfer_config.kv_connector,
        kv_transfer_config.engine_id,
    )
    return connector


def init_worker_kv_connector_v1(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig | None = None,
) -> None:
    """Create ``KVConnectorRole.WORKER`` singleton after distributed init.

    Callers pass the rank-local ``KVCacheConfig`` from ``set_kv_cache_configs``.
    A placeholder is used only when ``kv_cache_config`` is omitted. Assumes DiT workers run in separate
    processes (multiproc). Do not call from the same process that already hosts another KV transfer agent.
    """
    if vllm_config.kv_transfer_config is None:
        return

    cache_config = kv_cache_config if kv_cache_config is not None else _placeholder_kv_cache_config()
    ensure_kv_transfer_initialized(vllm_config, cache_config)
    logger.info(
        "Initialized KV connector v1 (WORKER role): connector=%s engine_id=%s num_blocks=%s",
        vllm_config.kv_transfer_config.kv_connector,
        vllm_config.kv_transfer_config.engine_id,
        cache_config.num_blocks,
    )


def maybe_register_vllm_kv_caches(kv_caches_by_layer: dict[str, torch.Tensor] | None) -> None:
    """Register layer-name tensors in the format the vLLM KV connector expects.

    ``kv_caches_by_layer`` is the layer-name mapping to be used by vLLM's Mooncake KV connector.
    An empty or missing dict is a no-op.
    """
    if not kv_caches_by_layer:
        logger.info_once("Skipping vLLM KV cache registration: kv_caches_by_layer is empty.")
        return
    if not has_kv_transfer_group():
        return

    get_kv_transfer_group().register_kv_caches(kv_caches_by_layer)
    logger.info("Registered %d vLLM KV cache tensors with the worker connector.", len(kv_caches_by_layer))


def start_load_kv(kv_connector_metadata: KVConnectorMetadata) -> None:
    """Bind Scheduler metadata and start a Worker-role remote page load."""
    if not has_kv_transfer_group():
        return
    connector = get_kv_transfer_group()
    connector.bind_connector_metadata(kv_connector_metadata)
    connector.start_load_kv(None)
    logger.info("start_load_kv: bound connector metadata and started remote KV load")


def mint_transfer_id(request_id: str) -> str:
    """Handshake id shared by AR source and DiT target params for one request."""
    return f"xfer-{request_id}"


def build_source_kv_transfer_params(
    *,
    transfer_id: str,
    remote_engine_id: str | None,
    remote_bootstrap_addr: str | None,
) -> dict[str, object]:
    """Mooncake producer (AR) handshake bag. Do not use ``request_finished`` params."""
    params: dict[str, object] = {
        "transfer_id": transfer_id,
        "do_remote_decode": True,
        "do_remote_prefill": False,
    }
    if remote_engine_id:
        params["remote_engine_id"] = remote_engine_id
    if remote_bootstrap_addr:
        params["remote_bootstrap_addr"] = remote_bootstrap_addr
    return params


def build_target_kv_transfer_params(
    *,
    transfer_id: str,
    remote_engine_id: str | None,
    remote_bootstrap_addr: str | None,
) -> dict[str, object]:
    """Mooncake consumer (DiT) handshake bag."""
    params: dict[str, object] = {
        "transfer_id": transfer_id,
        "do_remote_prefill": True,
        "do_remote_decode": False,
    }
    if remote_engine_id:
        params["remote_engine_id"] = remote_engine_id
    if remote_bootstrap_addr:
        params["remote_bootstrap_addr"] = remote_bootstrap_addr
    return params


def bootstrap_addr_from_kv_transfer_config(kv_transfer_config: KVTransferConfig | None) -> str | None:
    """Read Mooncake bootstrap address from connector extra_config when present."""
    if kv_transfer_config is None:
        return None
    extra = kv_transfer_config.kv_connector_extra_config or {}
    for key in ("bootstrap_addr", "prefill_bootstrap_addr", "mooncake_master"):
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def shutdown_kv_connector_v1(*, scheduler_connector: KVConnectorBase_V1 | None = None) -> None:
    """Shutdown worker and scheduler KV connector v1 instances."""
    if has_kv_transfer_group():
        ensure_kv_transfer_shutdown()

    if scheduler_connector is not None:
        scheduler_connector.shutdown()


def _placeholder_kv_cache_config(*, block_size: int = _PLACEHOLDER_BLOCK_SIZE) -> KVCacheConfig:
    """Empty ``KVCacheConfig`` used when the Scheduler has no page pool."""
    del block_size
    return KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )


def _build_kv_connector_v1_vllm_config(od_config: OmniDiffusionConfig) -> VllmConfig:
    """Minimal ``VllmConfig`` for KV connector v1 factory calls."""
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

    assert isinstance(od_config.kv_transfer_config, KVTransferConfig)
    vllm_config = create_diffusion_vllm_config(torch.device("cpu"), od_config)
    vllm_config.kv_transfer_config = od_config.kv_transfer_config
    return vllm_config


def _validate_diffusion_kv_transfer_config(kv_transfer_config: KVTransferConfig) -> None:
    """Best-effort validation for DiT consumer-side vLLM KV connector config."""
    engine_id = kv_transfer_config.engine_id
    if not isinstance(engine_id, str) or not engine_id.strip():
        raise ValueError("Diffusion stage kv_transfer_config requires a non-empty engine_id")
    if kv_transfer_config.kv_role not in (None, "kv_consumer", "kv_both"):
        logger.warning(
            "Diffusion stage kv_transfer_config has kv_role=%r; "
            "expected kv_consumer for AR-to-DiT page transfer on the DiT consumer stage.",
            kv_transfer_config.kv_role,
        )
