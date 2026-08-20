# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared fixtures for diffusion_kv unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import vllm.v1.core.single_type_kv_cache_manager as native_kv_managers
from vllm.config import KVTransferConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

DEFAULT_KV_TRANSFER_YAML: dict[str, object] = {
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_consumer",
    "engine_id": "dit-engine-1",
    "kv_connector_extra_config": {"mooncake_protocol": "rdma"},
}

FAKE_TRANSFER_PARAMS: dict[str, object] = {
    "do_remote_prefill": True,
    "remote_engine_id": "ar-engine-1",
    "remote_bootstrap_addr": "tcp://127.0.0.1:9999",
    "transfer_id": "xfer-1",
}


class ConcreteScheduler(BaseScheduler):
    def update_from_output(self, sched_output, output) -> set[str]:
        del sched_output, output
        return set()


def make_kv_cache_config(
    *,
    num_blocks: int = 8,
    block_size: int = 4,
    num_kv_heads: int = 2,
    head_size: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    layer_names: list[str] | None = None,
) -> KVCacheConfig:
    native_kv_managers.register_all_kvcache_specs(None)
    names = layer_names or ["layer.0"]
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=names)],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)],
    )


def make_paged_od_config(**kwargs) -> OmniDiffusionConfig:
    return OmniDiffusionConfig.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        max_model_len=64,
        kv_transfer_config=dict(DEFAULT_KV_TRANSFER_YAML),
        **kwargs,
    )


def initialize_paged_scheduler(
    scheduler: BaseScheduler,
    *,
    od_config: OmniDiffusionConfig | None = None,
    kv_transfer_config: KVTransferConfig | None = None,
    kv_cache_config: KVCacheConfig | None = None,
    num_blocks: int = 64,
    max_num_seqs: int = 1,
    max_model_len: int = 64,
    scheduler_block_size: int = 4,
    hash_block_size: int = 4,
    kv_vllm_config: object | None = None,
) -> KVCacheConfig:
    """Initialize a paged scheduler for unit tests.

    Prefer ``od_config`` when exercising real ``OmniDiffusionConfig`` wiring.
    Otherwise pass ``kv_transfer_config`` (or leave both None) for a lightweight
    ``SimpleNamespace`` stand-in.
    """
    config = kv_cache_config or make_kv_cache_config(num_blocks=num_blocks, block_size=scheduler_block_size)
    if od_config is not None:
        transfer = od_config.kv_transfer_config
        init_config: object = od_config
        model_len = od_config.max_model_len
    else:
        transfer = kv_transfer_config
        init_config = SimpleNamespace(
            kv_transfer_config=transfer,
            diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )
        model_len = max_model_len

    scheduler.initialize(
        init_config,
        kv_cache_config=config,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        kv_vllm_config=kv_vllm_config
        or SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=model_len),
            max_in_flight_tokens=model_len,
            kv_transfer_config=transfer,
        ),
    )
    return config


def make_omni_diffusion_request(
    req_id: str,
    *,
    num_sequences: int = 1,
    seq_len: int = 8,
    prefix_len: int = 4,
    target_len: int = 4,
    kv_transfer_params: dict | None = None,
) -> OmniDiffusionRequest:
    sequences = tuple(
        DiffusionKVRequest(
            f"{req_id}/diffusion-kv/{sequence_id}",
            sequence_id=sequence_id,
            prefix_len=prefix_len,
            target_len=target_len,
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
