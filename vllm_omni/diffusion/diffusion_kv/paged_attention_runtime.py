# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables

from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import (
    PagedPiecewisePlan,
    build_paged_piecewise_plan,
    run_paged_piecewise_plan,
)
from vllm_omni.diffusion.diffusion_kv.request import DiffusionPagedAttentionSequence
from vllm_omni.diffusion.forward_context import override_paged_kv_runtime
from vllm_omni.platforms import current_omni_platform


@dataclass(frozen=True)
class DiffusionKVSequenceBinding:
    """Worker sequence and logical length installed for one allocation identity."""

    req_index: int
    max_seq_len: int
    num_computed_tokens: int = 0


DiffusionKVSequenceResolver = Callable[
    [str, int],
    DiffusionKVSequenceBinding,
]


class DiffusionPagedAttentionLayerAdapter(AttentionLayerBase):
    """Register a diffusion layer with vLLM's native cache machinery.

    This object deliberately does *not* subclass ``vllm.Attention``.  The
    latter owns a second execution path and would bypass Omni's sequence
    parallel pre/post hooks.  The wrapper only supplies the small
    ``AttentionLayerBase`` contract needed by ``init_attn_backend`` and keeps
    the platform-native attention implementation/cache view available to the
    diffusion runtime.
    """

    def __init__(
        self,
        *,
        layer_name: str,
        layer: Any,
        spec: AttentionSpec,
        vllm_config: VllmConfig,
        device: torch.device,
        ulysses_degree: int = 1,
    ) -> None:
        if type(ulysses_degree) is not int or ulysses_degree <= 0:
            raise ValueError(f"ulysses_degree must be a positive integer, got {ulysses_degree!r}")
        num_heads = int(layer.num_heads)
        num_kv_heads = int(spec.num_kv_heads)
        if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError(
                "Paged attention requires positive Q/KV heads with num_heads divisible by num_kv_heads: "
                f"num_heads={num_heads}, num_kv_heads={num_kv_heads}"
            )
        if num_heads % ulysses_degree != 0 or num_kv_heads % ulysses_degree != 0:
            raise ValueError(
                "Paged attention requires Q/KV heads divisible by ulysses_degree: "
                f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, ulysses_degree={ulysses_degree}"
            )
        num_heads //= ulysses_degree
        num_kv_heads //= ulysses_degree

        attention_config = vllm_config.attention_config
        parallel_config = getattr(vllm_config, "parallel_config", None)
        previous_backend = attention_config.backend
        previous_backend_per_kind = attention_config.backend_per_kind
        previous_pcp_size = getattr(parallel_config, "prefill_context_parallel_size", None)
        try:
            # This is a portable vLLM backend request. The active platform
            # resolves it to its native implementation, such as FlashAttention
            # on CUDA or AscendAttentionBackend on NPU.
            attention_config.backend = AttentionBackendEnum.FLASH_ATTN
            attention_config.backend_per_kind = {}
            # Omni already applies SP sharding; advertising PCP here would double-count it.
            if parallel_config is not None:
                parallel_config.prefill_context_parallel_size = 1
            with set_current_vllm_config(vllm_config):
                attn_backend = get_attn_backend(
                    head_size=spec.head_size,
                    dtype=vllm_config.model_config.dtype,
                    kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                    attn_type=AttentionType.DECODER,
                    num_heads=num_heads,
                )
        finally:
            attention_config.backend = previous_backend
            attention_config.backend_per_kind = previous_backend_per_kind
            if parallel_config is not None:
                parallel_config.prefill_context_parallel_size = previous_pcp_size
        canonical_spec = replace(
            spec,
            num_kv_heads=num_kv_heads,
            indexes_kv_by_block_stride=attn_backend.indexes_kv_by_block_stride(),
        )
        self.layer_name = layer_name
        self.spec = canonical_spec
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = int(canonical_spec.head_size)
        self.head_size_v = int(getattr(canonical_spec, "head_size_v", canonical_spec.head_size))
        if self.head_size_v != self.head_size:
            raise ValueError(
                "Diffusion native paged attention requires equal key/value head sizes; "
                f"got head_size={self.head_size}, head_size_v={self.head_size_v}"
            )
        self.softmax_scale = float(layer.softmax_scale)
        self.attn_backend = attn_backend
        self.kv_cache: torch.Tensor | None = None
        # Native attention backends consume either device tensors or host
        # scalar copies of these scales. The paged diffusion path is currently
        # unquantized, so initialize both contracts to identity.
        self._q_scale = torch.ones(1, device=device, dtype=torch.float32)
        self._k_scale = torch.ones(1, device=device, dtype=torch.float32)
        self._v_scale = torch.ones(1, device=device, dtype=torch.float32)
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0
        with set_current_vllm_config(vllm_config):
            self.impl = self._create_native_impl(vllm_config)

    def get_attn_backend(self):
        return self.attn_backend

    def _create_native_impl(self, vllm_config: VllmConfig):
        impl_cls = self.attn_backend.get_impl_cls()
        impl = impl_cls(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.softmax_scale,
            num_kv_heads=self.num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
        )
        if not callable(getattr(impl, "forward", None)):
            raise RuntimeError(
                f"Native paged-attention backend {self.attn_backend.get_name()!r} has no forward implementation"
            )
        if not self.attn_backend.forward_includes_kv_cache_update and not callable(
            getattr(impl, "do_kv_cache_update", None)
        ):
            raise RuntimeError(
                f"Native paged-attention backend {self.attn_backend.get_name()!r} cannot update the KV cache"
            )
        if hasattr(impl, "vllm_flash_attn_version") and impl.vllm_flash_attn_version is None:
            raise RuntimeError(f"Native paged-attention kernel is unavailable for diffusion layer {self.layer_name!r}")
        return impl

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> AttentionSpec:
        del vllm_config
        return self.spec


@dataclass(frozen=True)
class PreparedDiffusionPagedAttentionBatch:
    """Native metadata shared by all paged attention layers in one forward."""

    sequences: tuple[DiffusionPagedAttentionSequence, ...]
    positions: torch.Tensor
    block_tables: tuple[torch.Tensor, ...]
    slot_mappings: torch.Tensor
    attn_metadata: dict[str, Any]
    slot_mappings_by_layer: dict[str, torch.Tensor]
    num_tokens: int
    _owner: object = field(repr=False, compare=False)
    _generation: int = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DiffusionPagedAttentionContext:
    """One layer's native inputs for an Omni paged-backend invocation."""

    layer: DiffusionPagedAttentionLayerAdapter
    query: torch.Tensor
    key_write: torch.Tensor
    value_write: torch.Tensor
    slot_mapping: torch.Tensor
    native_metadata: Any
    piecewise_plan: PagedPiecewisePlan | None
    piecewise_native_metadata: tuple[Any, ...]
    query_token_shape: tuple[int, ...]
    query_has_head_dims: bool
    sequence_query_lens: tuple[int, ...]
    tail_padding: int = 0

    def run(self) -> torch.Tensor:
        layer = self.layer
        kv_cache = layer.kv_cache
        if kv_cache is None:
            raise RuntimeError(f"Native KV cache is not bound for diffusion layer {layer.layer_name!r}")
        if not layer.attn_backend.forward_includes_kv_cache_update:
            layer.impl.do_kv_cache_update(
                layer,
                self.key_write,
                self.value_write,
                kv_cache,
                self.slot_mapping,
            )

        def run_native_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            native_metadata: Any,
        ) -> torch.Tensor:
            output = torch.empty(
                (query.shape[0], layer.num_heads, layer.head_size_v),
                dtype=query.dtype,
                device=query.device,
            )
            return layer.impl.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                native_metadata,
                output,
            )

        if self.piecewise_plan is not None:
            output = run_paged_piecewise_plan(
                self.query,
                self.key_write,
                self.value_write,
                self.piecewise_plan,
                self.piecewise_native_metadata,
                run_native_attention,
            )
        else:
            output = run_native_attention(
                self.query,
                self.key_write,
                self.value_write,
                self.native_metadata,
            )
        return self.restore_output(output)

    def restore_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.tail_padding:
            batch_size, padded_query_len = self.query_token_shape
            sequences = []
            offset = 0
            for query_len in self.sequence_query_lens:
                sequence = output[offset : offset + query_len]
                padding = sequence.new_zeros((padded_query_len - query_len, *sequence.shape[1:]))
                sequences.append(torch.cat((sequence, padding), dim=0))
                offset += query_len
            if offset != output.shape[0]:
                raise ValueError(
                    "Paged attention output token count does not match the logical query sequences: "
                    f"output={output.shape[0]}, sequences={self.sequence_query_lens}"
                )
            output = torch.stack(sequences, dim=0)
        if self.query_has_head_dims:
            return output.reshape(*self.query_token_shape, self.layer.num_heads, self.layer.head_size_v)
        return output.reshape(*self.query_token_shape, self.layer.num_heads * self.layer.head_size_v)


class DiffusionPagedAttentionRuntime:
    """Translate diffusion sequences into vLLM metadata for an Omni paged backend."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        device: torch.device,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        attn_groups: list[list[Any]],
        layers: Mapping[str, DiffusionPagedAttentionLayerAdapter],
        resolve_sequence: DiffusionKVSequenceResolver,
    ) -> None:
        if not layers:
            raise ValueError("Paged attention requires at least one native attention layer")
        if len(attn_groups) != len(kv_cache_config.kv_cache_groups):
            raise ValueError(
                "Paged attention group mismatch: "
                f"builders={len(attn_groups)}, cache_groups={len(kv_cache_config.kv_cache_groups)}"
            )
        self.vllm_config = vllm_config
        self.device = torch.device(device)
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.attn_groups = attn_groups
        self.layers = dict(layers)
        self.resolve_sequence = resolve_sequence
        self._owner = object()
        self._prepare_generation = 0
        self._active_batch: PreparedDiffusionPagedAttentionBatch | None = None
        self._active_piecewise_plan: PagedPiecewisePlan | None = None
        self._active_piecewise_native_metadata: tuple[dict[str, Any], ...] | None = None
        self._causal_by_group = self._resolve_group_causality()
        self._reorder_batch_threshold = self._resolve_reorder_batch_threshold()

    def _resolve_group_causality(self) -> dict[int, bool]:
        causal_by_group: dict[int, bool] = {}
        for group_index, cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            layer_causality = {
                not bool(getattr(self.layers[layer_name].spec, "non_causal", False))
                for layer_name in cache_group.layer_names
            }
            if len(layer_causality) != 1:
                raise ValueError(f"Paged attention cache group {group_index} mixes causal and non-causal layers")
            causal_by_group[group_index] = layer_causality.pop()
        return causal_by_group

    def _resolve_reorder_batch_threshold(self) -> int | None:
        thresholds = [
            group.get_metadata_builder(0).reorder_batch_threshold
            for group_index, groups in enumerate(self.attn_groups)
            if self._causal_by_group[group_index]
            for group in groups
        ]
        concrete_thresholds = [threshold for threshold in thresholds if threshold is not None]
        return min(concrete_thresholds, default=None)

    def _validate_sequence_order(self, sequences: tuple[DiffusionPagedAttentionSequence, ...]) -> None:
        threshold = self._reorder_batch_threshold
        if threshold is None or not any(self._causal_by_group.values()):
            return

        found_long_query = False
        for sequence in sequences:
            if sequence.query_len > threshold:
                found_long_query = True
            elif found_long_query:
                raise ValueError(
                    "Causal paged attention sequences must place native decode/short-query sequences before "
                    f"prefill/long-query sequences (decode threshold={threshold})"
                )

    def _validate_sequence_capacity(
        self,
        sequences: tuple[DiffusionPagedAttentionSequence, ...],
        req_indices: list[int],
    ) -> None:
        native_num_blocks = self.block_tables.num_blocks.np
        blocks_per_kv_block = self.block_tables.blocks_per_kv_block
        for sequence, req_index in zip(sequences, req_indices, strict=True):
            if type(req_index) is not int or not 0 <= req_index < self.block_tables.max_num_reqs:
                raise ValueError(
                    f"Paged attention sequence {sequence.identity!r} resolved to invalid req_index {req_index!r}"
                )
            for group_index, (block_size, block_multiplier) in enumerate(
                zip(
                    self.block_tables.block_sizes,
                    blocks_per_kv_block,
                    strict=True,
                )
            ):
                required_blocks = cdiv(
                    sequence.seq_len,
                    block_size * self.block_tables.cp_size,
                )
                required_kernel_blocks = required_blocks * block_multiplier
                installed_kernel_blocks = int(native_num_blocks[group_index, req_index])
                if required_kernel_blocks > installed_kernel_blocks:
                    raise ValueError(
                        f"Paged attention sequence {sequence.identity!r} requires {required_blocks} blocks in "
                        f"cache group {group_index} for seq_len={sequence.seq_len}, but its installed Worker "
                        f"binding contains only {installed_kernel_blocks // block_multiplier} blocks"
                    )

    def _build_native_metadata(
        self,
        *,
        query_lens: Sequence[int],
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        positions: torch.Tensor,
        block_tables: Sequence[torch.Tensor],
        slot_mappings: torch.Tensor,
        causal: bool | Mapping[int, bool],
    ) -> dict[str, Any]:
        dcp_local_seq_lens = None
        if self.block_tables.cp_size > 1:
            dcp_local_seq_lens = get_dcp_local_seq_lens(
                seq_lens,
                dcp_size=self.block_tables.cp_size,
                dcp_rank=self.block_tables.cp_rank,
                cp_kv_cache_interleave_size=self.block_tables.cp_interleave,
            )
        with set_current_vllm_config(self.vllm_config):
            return current_omni_platform.build_diffusion_kv_attn_metadata(
                attn_groups=self.attn_groups,
                num_reqs=len(query_lens),
                num_tokens=sum(query_lens),
                query_start_loc_gpu=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                max_query_len=max(query_lens),
                seq_lens=seq_lens,
                max_seq_len=max(int(seq_len) for seq_len in seq_lens_cpu),
                block_tables=block_tables,
                slot_mappings=slot_mappings,
                kv_cache_config=self.kv_cache_config,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_cpu_upper_bound=seq_lens_cpu,
                dcp_local_seq_lens=dcp_local_seq_lens,
                positions=positions,
                causal=causal,
            )

    def prepare_batch(
        self,
        sequences: Sequence[DiffusionPagedAttentionSequence],
    ) -> PreparedDiffusionPagedAttentionBatch:
        sequences = tuple(sequences)
        if not sequences:
            raise ValueError("Paged attention batch must contain at least one sequence")
        if self._active_batch is not None:
            raise RuntimeError("Cannot prepare paged attention metadata during an active forward")
        if len(sequences) > self.block_tables.max_num_reqs:
            raise ValueError(
                f"Paged attention batch has {len(sequences)} sequences; "
                f"Worker capacity is {self.block_tables.max_num_reqs}"
            )

        identities = [sequence.identity for sequence in sequences]
        if len(set(identities)) != len(identities):
            raise ValueError("Paged attention batch contains duplicate sequence identities")
        bindings = [self.resolve_sequence(sequence.request_id, sequence.sequence_id) for sequence in sequences]
        req_indices_list = [binding.req_index for binding in bindings]
        if len(set(req_indices_list)) != len(req_indices_list):
            raise ValueError("Paged attention batch resolves multiple sequences to the same req_index")
        for sequence, binding in zip(sequences, bindings, strict=True):
            if type(binding.max_seq_len) is not int or binding.max_seq_len < 0:
                raise ValueError(
                    f"Paged attention sequence {sequence.identity!r} resolved to invalid logical capacity "
                    f"{binding.max_seq_len!r}"
                )
            if sequence.seq_len > binding.max_seq_len:
                raise ValueError(
                    f"Paged attention sequence {sequence.identity!r} uses seq_len={sequence.seq_len}, but its "
                    "installed "
                    f"allocation has logical length {binding.max_seq_len}"
                )
        self._validate_sequence_order(sequences)
        self._validate_sequence_capacity(sequences, req_indices_list)

        query_lens = [sequence.query_len for sequence in sequences]
        num_tokens = sum(query_lens)
        if num_tokens > self.block_tables.max_num_batched_tokens:
            raise ValueError(
                f"Paged attention batch has {num_tokens} tokens; Worker capacity is "
                f"{self.block_tables.max_num_batched_tokens}"
            )
        query_offsets = [0]
        for query_len in query_lens:
            query_offsets.append(query_offsets[-1] + query_len)

        # Native BlockTables reuses persistent gather/slot buffers. Starting a
        # new preparation invalidates every batch prepared before it, even if a
        # later metadata builder raises.
        self._prepare_generation += 1
        generation = self._prepare_generation
        req_indices = torch.tensor(req_indices_list, dtype=torch.int32, device=self.device)
        query_start_loc_cpu = torch.tensor(query_offsets, dtype=torch.int32)
        query_start_loc = query_start_loc_cpu.to(self.device)
        seq_lens_cpu = torch.tensor([sequence.seq_len for sequence in sequences], dtype=torch.int32)
        seq_lens = seq_lens_cpu.to(self.device)
        positions = torch.cat(
            [
                torch.arange(
                    sequence.kv_start_pos,
                    sequence.kv_start_pos + sequence.query_len,
                    dtype=torch.int64,
                    device=self.device,
                )
                for sequence in sequences
            ]
        )

        block_tables = self.block_tables.gather_block_tables(
            req_indices,
            num_reqs_padded=len(sequences),
        )
        slot_mappings = self.block_tables.compute_slot_mappings(
            req_indices,
            query_start_loc,
            positions,
            num_tokens_padded=num_tokens,
        )
        causal: bool | Mapping[int, bool]
        causal_values = set(self._causal_by_group.values())
        causal = causal_values.pop() if len(causal_values) == 1 else self._causal_by_group
        attn_metadata = self._build_native_metadata(
            query_lens=query_lens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            causal=causal,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings,
            self.kv_cache_config,
        )
        return PreparedDiffusionPagedAttentionBatch(
            sequences=sequences,
            positions=positions,
            block_tables=tuple(block_tables),
            slot_mappings=slot_mappings,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            num_tokens=num_tokens,
            _owner=self._owner,
            _generation=generation,
        )

    def invalidate_prepared_batches(self) -> None:
        """Invalidate native buffer views after BlockTable state changes."""

        if self._active_batch is not None:
            raise RuntimeError("Cannot change paged attention BlockTables during an active forward")
        self._prepare_generation += 1

    @contextmanager
    def activate(
        self,
        batch: PreparedDiffusionPagedAttentionBatch,
    ) -> Iterator[DiffusionPagedAttentionRuntime]:
        if batch._owner is not self._owner:
            raise ValueError("Prepared paged attention batch belongs to a different runtime")
        if batch._generation != self._prepare_generation:
            raise ValueError("Prepared paged attention batch is stale after a newer batch preparation")
        if self._active_batch is not None:
            raise RuntimeError("Paged attention runtime already has an active forward")
        self._active_batch = batch
        self._active_piecewise_plan = None
        self._active_piecewise_native_metadata = None
        try:
            with override_paged_kv_runtime(self), set_current_vllm_config(self.vllm_config):
                yield self
        finally:
            self._active_piecewise_plan = None
            self._active_piecewise_native_metadata = None
            self._active_batch = None

    @staticmethod
    def _flatten_tensor(
        tensor: torch.Tensor,
        *,
        num_heads: int,
        head_size: int,
        name: str,
    ) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        if tensor.ndim >= 3 and tensor.shape[-2:] == (num_heads, head_size):
            return tensor.reshape(-1, num_heads, head_size), tuple(tensor.shape[:-2]), True
        hidden_size = num_heads * head_size
        if tensor.ndim >= 1 and tensor.shape[-1] == hidden_size:
            return tensor.reshape(-1, num_heads, head_size), tuple(tensor.shape[:-1]), False
        raise ValueError(
            f"Paged attention {name} must end in ({num_heads}, {head_size}) or ({hidden_size},); "
            f"got shape={tuple(tensor.shape)}"
        )

    @staticmethod
    def _validate_token_layout(
        token_shape: tuple[int, ...],
        batch: PreparedDiffusionPagedAttentionBatch,
        *,
        name: str,
        tail_padding: int = 0,
    ) -> None:
        if len(token_shape) == 1:
            if token_shape[0] == batch.num_tokens:
                return
            raise ValueError(
                f"Paged attention {name} token count must match the prepared write batch: "
                f"tokens={token_shape[0]}, prepared={batch.num_tokens}"
            )
        if len(token_shape) == 2:
            batch_size, tokens_per_row = token_shape
            query_lens = tuple(sequence.query_len for sequence in batch.sequences)
            if batch_size == len(batch.sequences) and all(
                query_len + tail_padding == tokens_per_row for query_len in query_lens
            ):
                return
            raise ValueError(
                f"Paged attention batched {name} layout must match prepared sequences for the current write: "
                f"shape={token_shape}, sequence_query_lens={query_lens}"
            )
        raise ValueError(
            f"Paged attention {name} supports packed [T, ...] or uniform batched [B, T, ...] token layouts; "
            f"got token shape={token_shape}"
        )

    @staticmethod
    def _strip_uniform_tail_padding(
        tensor: torch.Tensor,
        token_shape: tuple[int, ...],
        batch: PreparedDiffusionPagedAttentionBatch,
        *,
        tail_padding: int,
    ) -> torch.Tensor:
        if not tail_padding:
            return tensor
        batch_size, padded_query_len = token_shape
        shaped = tensor.reshape(batch_size, padded_query_len, *tensor.shape[1:])
        return torch.cat(
            tuple(shaped[req_index, : sequence.query_len] for req_index, sequence in enumerate(batch.sequences)),
            dim=0,
        )

    def _get_piecewise_plan(
        self,
        full_attn_spans: list[list[tuple[int, int]]],
    ) -> PagedPiecewisePlan:
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("Piecewise paged attention requires an active prepared batch")
        if any(sequence.kv_start_pos + sequence.query_len != sequence.seq_len for sequence in batch.sequences):
            invalid_sequences = [
                sequence.identity
                for sequence in batch.sequences
                if sequence.kv_start_pos + sequence.query_len != sequence.seq_len
            ]
            raise ValueError(
                "Paged piecewise attention requires each query/write span to end at seq_len; "
                f"invalid sequences={invalid_sequences!r}"
            )

        cached_plan = self._active_piecewise_plan
        if cached_plan is not None:
            frozen_spans = tuple(tuple(tuple(span) for span in row_spans) for row_spans in full_attn_spans)
            if cached_plan.spans != frozen_spans:
                raise ValueError("Paged piecewise attention metadata changed between layers in one active batch")
            return cached_plan

        plan = build_paged_piecewise_plan(
            full_attn_spans,
            query_offsets=[sequence.kv_start_pos for sequence in batch.sequences],
            query_lens=[sequence.query_len for sequence in batch.sequences],
            seq_lens=[sequence.seq_len for sequence in batch.sequences],
            device=self.device,
        )
        self._active_piecewise_plan = plan
        return plan

    def _get_piecewise_native_metadata(
        self,
        plan: PagedPiecewisePlan,
    ) -> tuple[dict[str, Any], ...]:
        """Build one native metadata object per segment and cache it by batch."""

        batch = self._active_batch
        if batch is None:
            raise RuntimeError("Piecewise paged attention requires an active prepared batch")
        cached_metadata = self._active_piecewise_native_metadata
        if cached_metadata is not None:
            return cached_metadata

        metadata_by_segment: list[dict[str, Any]] = []
        for packed_segment in plan.segments:
            row_segments = packed_segment.row_segments
            query_lens = [segment.q_end - segment.q_start for segment in row_segments]
            seq_lens_values = [segment.kv_end for segment in row_segments]
            query_offsets = [0]
            for query_len in query_lens:
                query_offsets.append(query_offsets[-1] + query_len)
            query_start_loc_cpu = torch.tensor(query_offsets, dtype=torch.int32)
            seq_lens_cpu = torch.tensor(seq_lens_values, dtype=torch.int32)
            metadata_by_segment.append(
                self._build_native_metadata(
                    query_lens=query_lens,
                    query_start_loc=query_start_loc_cpu.to(self.device),
                    query_start_loc_cpu=query_start_loc_cpu,
                    seq_lens=seq_lens_cpu.to(self.device),
                    seq_lens_cpu=seq_lens_cpu,
                    positions=batch.positions.index_select(0, packed_segment.query_indices),
                    block_tables=batch.block_tables,
                    slot_mappings=batch.slot_mappings.index_select(-1, packed_segment.query_indices),
                    causal=(row_segments[0].mode == "causal"),
                )
            )
        self._active_piecewise_native_metadata = tuple(metadata_by_segment)
        return self._active_piecewise_native_metadata

    def prepare_layer_context(
        self,
        layer_name: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        omni_attn_metadata: Any | None = None,
    ) -> DiffusionPagedAttentionContext:
        if self._active_batch is None:
            raise RuntimeError("Paged attention forward must run inside runtime.activate(batch)")
        try:
            layer = self.layers[layer_name]
        except KeyError as exc:
            raise KeyError(f"Unknown diffusion paged attention layer {layer_name!r}") from exc
        batch = self._active_batch
        full_attn_spans, tail_padding = self._validate_omni_attn_metadata(omni_attn_metadata)

        query_flat, query_token_shape, query_has_head_dims = self._flatten_tensor(
            query,
            num_heads=layer.num_heads,
            head_size=layer.head_size,
            name="query",
        )
        key_flat, key_token_shape, _ = self._flatten_tensor(
            key,
            num_heads=layer.num_kv_heads,
            head_size=layer.head_size,
            name="key",
        )
        value_flat, value_token_shape, _ = self._flatten_tensor(
            value,
            num_heads=layer.num_kv_heads,
            head_size=layer.head_size_v,
            name="value",
        )
        self._validate_token_layout(query_token_shape, batch, name="query", tail_padding=tail_padding)
        self._validate_token_layout(key_token_shape, batch, name="key", tail_padding=tail_padding)
        self._validate_token_layout(value_token_shape, batch, name="value", tail_padding=tail_padding)
        query_flat = self._strip_uniform_tail_padding(
            query_flat,
            query_token_shape,
            batch,
            tail_padding=tail_padding,
        )
        key_flat = self._strip_uniform_tail_padding(
            key_flat,
            key_token_shape,
            batch,
            tail_padding=tail_padding,
        )
        value_flat = self._strip_uniform_tail_padding(
            value_flat,
            value_token_shape,
            batch,
            tail_padding=tail_padding,
        )
        if query.device != self.device or key.device != self.device or value.device != self.device:
            raise ValueError(
                f"Paged attention Q/K/V must be on {self.device}; "
                f"got query={query.device}, key={key.device}, value={value.device}"
            )
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError(f"Paged attention Q/K/V dtypes must match; got {query.dtype}, {key.dtype}, {value.dtype}")
        expected_dtype = self.vllm_config.model_config.dtype
        if query.dtype != expected_dtype:
            raise ValueError(
                f"Paged attention Q/K/V dtype must match model activation dtype {expected_dtype}; got {query.dtype}"
            )
        if not bool(getattr(layer.spec, "non_causal", False)):
            non_suffix_sequences = [
                sequence.identity
                for sequence in batch.sequences
                if sequence.kv_start_pos + sequence.query_len != sequence.seq_len
            ]
            if non_suffix_sequences:
                raise ValueError(
                    "Causal paged attention requires each query/write span to end at seq_len; "
                    f"invalid sequences={non_suffix_sequences!r}"
                )

        slot_mapping = batch.slot_mappings_by_layer.get(layer_name)
        if slot_mapping is None:
            raise KeyError(f"No native slot mapping was built for diffusion layer {layer_name!r}")
        slot_mapping = slot_mapping.reshape(-1)[: batch.num_tokens]
        if slot_mapping.numel() != batch.num_tokens:
            raise ValueError(
                f"Paged attention slot mapping for {layer_name!r} has {slot_mapping.numel()} entries; "
                f"expected {batch.num_tokens}"
            )

        # vLLM's writer consumes slot_mapping directly, so a span that starts
        # inside a physical block naturally writes only the requested suffix.
        key_flat = key_flat.contiguous()
        value_flat = value_flat.contiguous()
        try:
            native_metadata = batch.attn_metadata[layer_name]
        except KeyError as exc:
            raise KeyError(f"No native attention metadata was built for diffusion layer {layer_name!r}") from exc
        piecewise_plan: PagedPiecewisePlan | None = None
        piecewise_native_metadata: tuple[Any, ...] = ()
        if full_attn_spans is not None:
            piecewise_plan = self._get_piecewise_plan(full_attn_spans)
            native_metadata_by_segment = self._get_piecewise_native_metadata(piecewise_plan)
            try:
                piecewise_native_metadata = tuple(
                    segment_metadata[layer_name] for segment_metadata in native_metadata_by_segment
                )
            except KeyError as exc:
                raise KeyError(
                    f"No piecewise native attention metadata was built for diffusion layer {layer_name!r}"
                ) from exc
        return DiffusionPagedAttentionContext(
            layer=layer,
            query=query_flat.contiguous(),
            key_write=key_flat,
            value_write=value_flat,
            slot_mapping=slot_mapping,
            native_metadata=native_metadata,
            piecewise_plan=piecewise_plan,
            piecewise_native_metadata=piecewise_native_metadata,
            query_token_shape=query_token_shape,
            query_has_head_dims=query_has_head_dims,
            sequence_query_lens=tuple(sequence.query_len for sequence in batch.sequences),
            tail_padding=tail_padding,
        )

    @staticmethod
    def _validate_omni_attn_metadata(
        metadata: Any | None,
    ) -> tuple[list[list[tuple[int, int]]] | None, int]:
        if metadata is None:
            return None, 0
        full_attn_spans = getattr(metadata, "full_attn_spans", None)
        attn_mask = getattr(metadata, "attn_mask", None)
        unsupported_fields = [
            field_name
            for field_name in (
                "joint_attn_mask",
                "query_ranges",
                "video_layout",
            )
            if getattr(metadata, field_name, None) is not None
        ]
        if attn_mask is not None:
            if full_attn_spans is None:
                unsupported_fields.append("attn_mask")
            elif not isinstance(attn_mask, torch.Tensor) or attn_mask.ndim != 4:
                unsupported_fields.append("attn_mask (piecewise paging requires a 4D tensor)")
        extra = getattr(metadata, "extra", None)
        tail_padding = 0
        if extra:
            extra = dict(extra)
            tail_padding = extra.pop("paged_kv_tail_padding", 0)
            if type(tail_padding) is not int or tail_padding < 0:
                raise ValueError(f"paged_kv_tail_padding must be a non-negative integer, got {tail_padding!r}")
            if extra:
                unsupported_fields.append(f"extra={sorted(extra)}")
        if unsupported_fields:
            raise NotImplementedError(
                "Diffusion paged attention cannot translate Omni attention metadata fields "
                f"{unsupported_fields!r} to native FlashAttention metadata"
            )
        return full_attn_spans, tail_padding
