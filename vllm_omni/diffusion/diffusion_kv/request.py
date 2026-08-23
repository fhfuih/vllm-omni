# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import RequestStatus


@dataclass(frozen=True)
class DiffusionPagedAttentionSequence:
    """One logical sequence participating in a paged attention call."""

    request_id: str
    query_len: int
    seq_len: int
    sequence_id: int
    kv_start_pos: int = 0

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("Paged attention request_id must be a non-empty string")
        if type(self.sequence_id) is not int or self.sequence_id < 0:
            raise ValueError("Paged attention sequence_id must be a non-negative integer")
        if type(self.query_len) is not int or self.query_len <= 0:
            raise ValueError("Paged attention query_len must be a positive integer")
        if type(self.seq_len) is not int or self.seq_len <= 0:
            raise ValueError("Paged attention seq_len must be a positive integer")
        if type(self.kv_start_pos) is not int or self.kv_start_pos < 0:
            raise ValueError("Paged attention kv_start_pos must be a non-negative integer")
        if self.kv_start_pos + self.query_len > self.seq_len:
            raise ValueError(
                "Paged attention write span exceeds seq_len: "
                f"start={self.kv_start_pos}, query_len={self.query_len}, seq_len={self.seq_len}"
            )

    @property
    def identity(self) -> tuple[str, int]:
        return (self.request_id, self.sequence_id)


class DiffusionKVRequest:
    """Scheduler-owned KV state for one diffusion execution sequence.

    The primary sequence follows an ordered ``[prefix | target]`` policy, such
    as one Hunyuan CFG execution sequence. ``prefix_len`` is the contiguous
    reusable prefix; ``target_len`` is overwritten by every denoise step;
    ``num_tokens`` is the complete first-step allocation boundary.

    This object also exposes the minimal mutable Request surface consumed by
    native ``KVCacheManager``.

    An empty ``block_hashes`` sequence means the prefix has no canonical cache
    identity yet. Such a request may use native request-local page allocation,
    but consumers must not publish it through ``KVCacheManager.cache_blocks``.
    Prefix publication becomes valid only after model preprocessing supplies
    one canonical hash for every cacheable full block.
    """

    def __init__(
        self,
        request_id: str,
        *,
        sequence_id: int,
        prefix_len: int,
        target_len: int,
        seq_len: int,
        block_hashes: Sequence[BlockHash] = (),
        kv_transfer_params: dict[str, Any] | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> None:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if sequence_id < 0:
            raise ValueError(f"sequence_id must be non-negative, got {sequence_id}")
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be non-negative, got {prefix_len}")
        if target_len <= 0:
            raise ValueError(f"target_len must be positive, got {target_len}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if prefix_len + target_len > seq_len:
            raise ValueError(
                "prefix_len + target_len must not exceed seq_len: "
                f"prefix_len={prefix_len}, target_len={target_len}, seq_len={seq_len}"
            )

        # Diffusion semantics consumed by the Scheduler-side facade.
        self.sequence_id = sequence_id
        self.prefix_len = prefix_len
        self.target_len = target_len

        # Native vLLM Request surface. Keep this list intentionally small and
        # cover it with real-KVCacheManager conformance tests.
        self.request_id = request_id
        self.num_tokens = seq_len
        self.num_prompt_tokens = prefix_len
        self.num_computed_tokens = 0
        self.block_hashes = list(block_hashes)
        self.skip_reading_prefix_cache = not self.block_hashes
        self.status = RequestStatus.WAITING
        self.num_preemptions = 0
        self.num_in_flight_tokens = 0
        # vLLM 0.26+ uses this optional boundary when publishing cached blocks;
        # zero means that no sparse-retention boundary is active.
        self.shared_prefix_boundary = 0
        # KV connector v1's opaque vLLM handshake data. The Scheduler copies a public
        # request bag onto each sequence that does not already have one.
        self.kv_transfer_params = kv_transfer_params
        self.prompt_token_ids = prompt_token_ids

    @property
    def seq_len(self) -> int:
        """Complete first-step sequence length and allocation boundary."""

        return self.num_tokens


def prepare_kv_connector_request(
    request: DiffusionKVRequest,
    transfer_params: Mapping[str, Any] | None,
) -> None:
    if transfer_params is None:
        return
    if request.kv_transfer_params is None:
        request.kv_transfer_params = dict(transfer_params)
    if request.prompt_token_ids is None:
        source_tokens = transfer_params.get("num_transfer_tokens")
        if source_tokens is not None and (type(source_tokens) is not int or source_tokens < 0):
            raise ValueError("num_transfer_tokens must be a non-negative integer")
        transfer_tokens = request.prefix_len if source_tokens is None else source_tokens
        request.prompt_token_ids = [0] * min(request.prefix_len, transfer_tokens)
