# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from vllm.config import KVTransferConfig


class DiffusionKVCacheMode(str, Enum):
    """Migration mode for diffusion KV ownership."""

    DENSE_LEGACY = "dense_legacy"
    PAGED_SCHEDULER = "paged_scheduler"


def parse_diffusion_kv_cache_mode(value: object) -> DiffusionKVCacheMode:
    """Parse a selectable Diffusion KV ownership mode."""

    try:
        mode = DiffusionKVCacheMode(value)
    except (TypeError, ValueError) as exc:
        supported = ", ".join(mode.value for mode in DiffusionKVCacheMode)
        raise ValueError(f"Unsupported Diffusion KV diffusion_kv_mode {value!r}; expected one of: {supported}") from exc
    return mode


def is_scheduler_paged_kv_mode(mode: DiffusionKVCacheMode) -> bool:
    """Return whether a parsed cache mode uses Scheduler-owned paging."""
    return mode is DiffusionKVCacheMode.PAGED_SCHEDULER


def parse_kv_transfer_config(value: object | None) -> KVTransferConfig | None:
    """Normalize external config input before it reaches runtime code."""
    if value is None:
        return None
    if isinstance(value, KVTransferConfig):
        validate_diffusion_kv_transfer_config(value)
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"kv_transfer_config must be a mapping or KVTransferConfig, got {type(value)!r}")

    payload = dict(value)
    if not payload:
        return None
    if not isinstance(payload.get("engine_id"), str) or not payload["engine_id"].strip():
        raise ValueError("Diffusion stage kv_transfer_config requires a non-empty engine_id")
    config = KVTransferConfig(**payload)
    validate_diffusion_kv_transfer_config(config)
    return config


def validate_diffusion_kv_transfer_config(config: KVTransferConfig) -> None:
    if config.kv_connector != "MooncakeConnector":
        raise ValueError("Diffusion KV Connector currently supports only kv_connector='MooncakeConnector'")
    if not isinstance(config.engine_id, str) or not config.engine_id.strip():
        raise ValueError("Diffusion stage kv_transfer_config requires a non-empty engine_id")
    if config.kv_role != "kv_consumer":
        raise ValueError(f"Diffusion stage kv_transfer_config requires kv_role='kv_consumer'; got {config.kv_role!r}")
