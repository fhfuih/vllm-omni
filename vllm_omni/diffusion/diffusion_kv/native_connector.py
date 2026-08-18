# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native vLLM KV connector assembly for the Diffusion stage.

Creates Scheduler- and Worker-role ``MooncakeConnector`` instances without
switching the KV data path away from ``OmniKVTransferManager``. Real KV
sizing, admission, and ``register_kv_caches`` wiring are added by later
native-paging work.
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
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

# Temporary placeholder sizing. Native page allocation is wired later from the
# Diffusion KV cache manager once the scheduler owns real allocation metadata.
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
            raise ValueError("Diffusion stage native kv_transfer_config requires a non-empty engine_id")
        kv_transfer_config = KVTransferConfig(**payload)
        _validate_diffusion_kv_transfer_config(kv_transfer_config)
        return kv_transfer_config
    raise TypeError(f"kv_transfer_config must be a mapping or KVTransferConfig, got {type(value)!r}")


def create_scheduler_native_kv_connector(
    od_config: OmniDiffusionConfig,
) -> KVConnectorBase_V1 | None:
    """Create ``KVConnectorRole.SCHEDULER`` when native config is present."""
    kv_transfer_config = od_config.kv_transfer_config
    if kv_transfer_config is None:
        return None
    if not isinstance(kv_transfer_config, KVTransferConfig):
        raise TypeError(
            f"Diffusion stage native kv_transfer_config must be KVTransferConfig, got {type(kv_transfer_config)!r}"
        )

    vllm_config = _build_native_connector_vllm_config(od_config)
    connector = KVConnectorFactory.create_connector(
        config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=_placeholder_kv_cache_config(),
    )
    logger.info(
        "Created native KV connector (SCHEDULER role): connector=%s engine_id=%s",
        kv_transfer_config.kv_connector,
        kv_transfer_config.engine_id,
    )
    return connector


def init_worker_native_kv_connector(vllm_config: VllmConfig) -> None:
    """Create ``KVConnectorRole.WORKER`` singleton after distributed init.

    Assumes DiT workers run in separate processes (multiproc). Do not call from
    the same process that already hosts another KV transfer agent.
    """
    if vllm_config.kv_transfer_config is None:
        return

    ensure_kv_transfer_initialized(vllm_config, _placeholder_kv_cache_config())
    logger.info(
        "Initialized native KV connector (WORKER role): connector=%s engine_id=%s",
        vllm_config.kv_transfer_config.kv_connector,
        vllm_config.kv_transfer_config.engine_id,
    )


def maybe_register_native_kv_caches(kv_caches_dict: dict[str, torch.Tensor] | None) -> None:
    """Register paged KV tensors when a layer-name dict is available.

    The default Diffusion runner does not expose native page tensors yet (kv_caches_dict==None),
    so this function intentionally becomes a no-op until that wiring exists.
    """
    if not kv_caches_dict:
        logger.info_once(
            "Skipping native KV cache registration: DiffusionModelRunner has no "
            "layer-name kv_caches dict yet; registration will happen once native page tensors are exposed."
        )
        return
    if not has_kv_transfer_group():
        return

    get_kv_transfer_group().register_kv_caches(kv_caches_dict)
    logger.info("Registered %d native KV cache tensors with the worker connector.", len(kv_caches_dict))


def shutdown_native_kv_connector(*, scheduler_connector: KVConnectorBase_V1 | None = None) -> None:
    """Shutdown worker and scheduler native connectors."""
    if has_kv_transfer_group():
        ensure_kv_transfer_shutdown()

    if scheduler_connector is not None:
        scheduler_connector.shutdown()


def _placeholder_kv_cache_config(*, block_size: int = _PLACEHOLDER_BLOCK_SIZE) -> KVCacheConfig:
    """Empty ``KVCacheConfig`` used until Diffusion owns real native page sizing."""
    del block_size
    return KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )


def _build_native_connector_vllm_config(od_config: OmniDiffusionConfig) -> VllmConfig:
    """Minimal ``VllmConfig`` for native KV connector factory calls."""
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

    assert isinstance(od_config.kv_transfer_config, KVTransferConfig)
    vllm_config = create_diffusion_vllm_config(torch.device("cpu"), od_config)
    vllm_config.kv_transfer_config = od_config.kv_transfer_config
    return vllm_config


def _validate_diffusion_kv_transfer_config(kv_transfer_config: KVTransferConfig) -> None:
    """Best-effort validation for DiT consumer-side native connector config."""
    engine_id = kv_transfer_config.engine_id
    if not isinstance(engine_id, str) or not engine_id.strip():
        raise ValueError("Diffusion stage native kv_transfer_config requires a non-empty engine_id")
    if kv_transfer_config.kv_role not in (None, "kv_consumer", "kv_both"):
        logger.warning(
            "Diffusion stage native kv_transfer_config has kv_role=%r; "
            "expected kv_consumer for AR-to-DiT page transfer; producer wiring is added later.",
            kv_transfer_config.kv_role,
        )
