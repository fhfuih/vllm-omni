# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV Connector lifecycle and synchronous receive for the Diffusion stage.

Creates Scheduler- and Worker-role vLLM ``MooncakeConnector`` instances.
Scheduler admission uses the page pool when ``paged_scheduler`` is enabled.
The ModelRunner owns the Worker-role connector and starts each remote load.

``v1`` here means Omni's new connector system versus the legacy Omni connector,
not upstream ``KVConnectorBase_V1``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    has_kv_transfer_group,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector, get_kv_connector

from vllm_omni.diffusion.diffusion_kv.config import validate_diffusion_kv_transfer_config

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class KVConnectorFatalError(RuntimeError):
    pass


@contextmanager
def _use_kv_transfer_shard_group() -> Iterator[None]:
    import vllm.distributed.parallel_state as vllm_parallel_state

    import vllm_omni.diffusion.distributed.parallel_state as diffusion_parallel_state

    sequence_parallel_group = diffusion_parallel_state._SP
    if sequence_parallel_group is None or sequence_parallel_group.world_size == 1:
        yield
        return

    original_tp_group = vllm_parallel_state._TP
    if original_tp_group is None:
        raise RuntimeError("Tensor parallel group must be initialized before KV Connector")
    transfer_group = vllm_parallel_state._PCP or sequence_parallel_group
    vllm_parallel_state._TP = transfer_group
    try:
        yield
    finally:
        vllm_parallel_state._TP = original_tp_group


def create_scheduler_kv_connector_v1(
    od_config: OmniDiffusionConfig,
    *,
    kv_cache_config: KVCacheConfig | None,
    vllm_config: VllmConfig | None,
) -> KVConnectorBase_V1 | None:
    """Create ``KVConnectorRole.SCHEDULER`` when KV connector v1 is configured.

    ``paged_scheduler`` passes the Scheduler-owned ``KVCacheConfig`` used for
    page allocation. Native transfer cannot be assembled before sizing.
    """
    kv_transfer_config = od_config.kv_transfer_config
    if kv_transfer_config is None:
        return None
    if not isinstance(kv_transfer_config, KVTransferConfig):
        raise TypeError(
            f"Diffusion stage kv_transfer_config must be KVTransferConfig, got {type(kv_transfer_config)!r}"
        )

    if vllm_config is None or kv_cache_config is None:
        raise ValueError("Native Diffusion Mooncake requires the sized VllmConfig and Scheduler KVCacheConfig")
    if kv_cache_config.num_blocks <= 0 or not kv_cache_config.kv_cache_groups:
        raise ValueError("Native Diffusion Mooncake requires a non-empty Scheduler KV cache plan")
    connector_vllm_config = vllm_config
    if connector_vllm_config.kv_transfer_config is None:
        connector_vllm_config.kv_transfer_config = kv_transfer_config
    elif connector_vllm_config.kv_transfer_config != kv_transfer_config:
        raise ValueError("Scheduler VllmConfig carries a different KVTransferConfig")

    connector = KVConnectorFactory.create_connector(
        config=connector_vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=kv_cache_config,
    )
    logger.info(
        "Created KV connector v1 (SCHEDULER role): connector=%s engine_id=%s",
        kv_transfer_config.kv_connector,
        kv_transfer_config.engine_id,
    )
    return connector


def init_worker_kv_connector_v1(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    kv_caches_by_layer: dict[str, Any],
) -> ActiveKVConnector | None:
    if vllm_config.kv_transfer_config is None:
        return
    validate_diffusion_kv_transfer_config(vllm_config.kv_transfer_config)
    if kv_cache_config.num_blocks <= 0 or not kv_cache_config.kv_cache_groups:
        raise ValueError("Native Diffusion Mooncake requires a non-empty rank-local KV cache plan")
    if not kv_caches_by_layer:
        raise ValueError("Native Diffusion Mooncake requires initialized rank-local KV page tensors")
    expected_layers = {layer_name for group in kv_cache_config.kv_cache_groups for layer_name in group.layer_names}
    if set(kv_caches_by_layer) != expected_layers:
        raise ValueError(
            "ModelRunner KV cache mapping does not match rank-local plan: "
            f"expected={sorted(expected_layers)}, got={sorted(kv_caches_by_layer)}"
        )
    for layer_name, cache in kv_caches_by_layer.items():
        if not isinstance(cache, torch.Tensor):
            raise TypeError(f"ModelRunner KV cache {layer_name!r} must be a torch.Tensor")

    # Mooncake discovers rank-local shards through vLLM's TP group; Diffusion uses SP.
    with _use_kv_transfer_shard_group():
        ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
    connector = get_kv_connector(vllm_config, kv_caches_by_layer)
    if not isinstance(connector, ActiveKVConnector):
        ensure_kv_transfer_shutdown()
        raise RuntimeError("Configured native Diffusion transfer did not create ActiveKVConnector")
    logger.info(
        "Initialized KV connector v1 (WORKER role): connector=%s engine_id=%s num_blocks=%s",
        vllm_config.kv_transfer_config.kv_connector,
        vllm_config.kv_transfer_config.engine_id,
        kv_cache_config.num_blocks,
    )
    return connector


def wait_for_remote_kv_before_forward(
    connector: ActiveKVConnector,
    scheduler_output: Any,
    *,
    timeout_s: float,
    rank: int,
) -> KVConnectorOutput:
    expected_ids = frozenset(scheduler_output.kv_connector_request_ids)
    if not expected_ids:
        raise KVConnectorFatalError("KV Connector metadata has no expected sequence request IDs")
    if timeout_s <= 0:
        raise ValueError(f"Native KV receive timeout must be positive, got {timeout_s}")

    public_ids = tuple(req.request_id for req in scheduler_output.scheduled_new_reqs)
    transfer_ids = sorted(
        {
            str(params["transfer_id"])
            for req in scheduler_output.scheduled_new_reqs
            if (params := getattr(req.req, "kv_transfer_params", None)) and isinstance(params.get("transfer_id"), str)
        }
    )
    start = time.monotonic()
    finished_recving: set[str] = set()
    finished_sending: set[str] = set()
    last_output = KVConnectorOutput()
    logger.info(
        "Native KV receive start request_ids=%s sequence_ids=%s rank=%d transfer_ids=%s",
        public_ids,
        sorted(expected_ids),
        rank,
        transfer_ids,
    )
    try:
        connector.pre_forward(scheduler_output)
        while True:
            output = connector.post_forward(set(), wait_for_save=False)
            if output is None:
                raise KVConnectorFatalError("ActiveKVConnector returned no completion output")
            last_output = output
            finished_recving.update(output.finished_recving or ())
            finished_sending.update(output.finished_sending or ())
            if output.invalid_block_ids:
                raise KVConnectorFatalError(
                    "KV Connector receive reported invalid target blocks "
                    f"on rank {rank}: {sorted(output.invalid_block_ids)}"
                )
            if expected_ids.issubset(finished_recving):
                elapsed = time.monotonic() - start
                logger.info(
                    "Native KV receive complete request_ids=%s sequence_ids=%s rank=%d transfer_ids=%s elapsed=%.3fs",
                    public_ids,
                    sorted(expected_ids),
                    rank,
                    transfer_ids,
                    elapsed,
                )
                return KVConnectorOutput(
                    finished_sending=finished_sending or None,
                    finished_recving=finished_recving,
                    kv_connector_stats=last_output.kv_connector_stats,
                    kv_cache_events=last_output.kv_cache_events,
                    kv_connector_worker_meta=last_output.kv_connector_worker_meta,
                    invalid_block_ids=set(),
                    expected_finished_count=last_output.expected_finished_count,
                )
            elapsed = time.monotonic() - start
            if elapsed >= timeout_s:
                missing = sorted(expected_ids - finished_recving)
                raise KVConnectorFatalError(
                    f"KV Connector receive timed out after {elapsed:.3f}s on rank {rank}; "
                    f"missing sequence IDs: {missing}"
                )
            time.sleep(min(0.01, max(0.0, timeout_s - elapsed)))
    except KVConnectorFatalError:
        raise
    except Exception as exc:
        raise KVConnectorFatalError(f"KV Connector receive failed on rank {rank}: {type(exc).__name__}: {exc}") from exc


def shutdown_kv_connector_v1(
    *,
    worker_connector: ActiveKVConnector | None = None,
    scheduler_connector: KVConnectorBase_V1 | None = None,
) -> None:
    """Shutdown worker and scheduler KV connector v1 instances."""
    if worker_connector is not None:
        worker_connector.set_disabled(True)
    if has_kv_transfer_group():
        ensure_kv_transfer_shutdown()

    if scheduler_connector is not None:
        scheduler_connector.shutdown()
