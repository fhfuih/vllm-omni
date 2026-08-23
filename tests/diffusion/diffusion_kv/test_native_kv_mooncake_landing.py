# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU landing test for AR → DiT native Mooncake paged KV.

The AR side injects recognizable values into GPU pages without loading model
weights.  The DiT side uses the production Scheduler, ModelRunner paged-cache
backend, native BlockTables, ModelRunner-owned ActiveKVConnector, and pre-forward
completion barrier.  A two-sequence request verifies CFG fan-out from one AR
transfer ticket into distinct DiT allocations.

Requires: CUDA GPU, ``mooncake`` TransferEngine, and a working local
``mooncake_protocol`` (defaults to ``tcp``). The test forces an eth-only
Mooncake topology so TransferEngine does not fall back to broken RDMA NICs.
Missing deps skip the test.
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing import Process, Queue, get_context
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import KVTransferConfig, set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_shutdown,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.utils.network_utils import get_open_port
from vllm.v1.core.sched.output import SchedulerOutput as NativeSchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.request import RequestStatus

from tests.diffusion.diffusion_kv.helper import ConcreteScheduler, make_kv_cache_config
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.kv_connector import shutdown_kv_connector_v1
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]

LAYER_NAME = "model.layers.0.self_attn"
BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 64
NUM_BLOCKS = 16
SEQ_LEN = 32
PREFIX_LEN = 16
TARGET_LEN = 16
SOURCE_BLOCK_IDS = [1, 2]
REQUEST_ID = "req-mooncake-landing"
AR_ENGINE_ID = "ar-engine-landing"
DIT_ENGINE_ID = "dit-engine-landing"
MOONCAKE_PROTOCOL = "tcp"
RECV_TIMEOUT_S = 60.0
DTYPE = torch.bfloat16


def test_ar_injected_pages_land_on_dit_registered_kv() -> None:
    """Verify Mooncake-transferred AR KV pages land on DiT-registered paged tensors."""
    with _mooncake_tcp_env():
        ctx = get_context("spawn")
        dist_port_ar = get_open_port()
        dist_port_dit = get_open_port()
        bootstrap_port = get_open_port()
        handshake: Queue = ctx.Queue()
        stop_queue: Queue = ctx.Queue()
        producer = ctx.Process(
            target=_run_ar_producer,
            args=(dist_port_ar, bootstrap_port, handshake, stop_queue),
            daemon=True,
        )
        producer.start()
        try:
            result = handshake.get(timeout=120)
            if not result.get("ok"):
                pytest.skip(f"Mooncake producer failed to start: {result.get('error')}")
            bootstrap_addr = result["bootstrap_addr"]
            _run_dit_consumer_and_assert(dist_port_dit, bootstrap_addr)
        finally:
            _cleanup_process(producer, stop_queue=stop_queue)


class _ProducerRequest:
    """Minimal Request surface consumed by Mooncake producer scheduler APIs."""

    def __init__(self, request_id: str, kv_transfer_params: dict[str, object], prompt_len: int) -> None:
        self.request_id = request_id
        self.kv_transfer_params = kv_transfer_params
        self.status = RequestStatus.WAITING
        self.prompt_token_ids = [0] * prompt_len
        self.num_prompt_tokens = prompt_len


def _pick_tcp_netif() -> str:
    """Pick a non-loopback netdev for Mooncake TCP topology."""
    net_dir = Path("/sys/class/net")
    if not net_dir.is_dir():
        return "lo"
    for name in sorted(p.name for p in net_dir.iterdir()):
        if name == "lo" or name.startswith("bonding") or name.startswith("veth"):
            continue
        return name
    return "lo"


@contextmanager
def _mooncake_tcp_env() -> Iterator[None]:
    """Force Mooncake onto TCP transport for same-host landing.

    With broken/unreachable RDMA NICs present, TransferEngine still
    auto-discovers HCAs even when ``mooncake_protocol=tcp`` and then fails
    QP setup. A custom eth-only topology makes initialize install TcpTransport
    instead. Also pin ``VLLM_HOST_IP`` so both processes advertise loopback.
    """
    netif = _pick_tcp_netif()
    # Visible accelerator is always :0 under device-visibility remapping.
    device_key = f"{current_omni_platform.device_type}:0"
    topo = {
        "cpu:0": [[netif]],
        device_key: [[netif]],
    }
    fd, topo_path = tempfile.mkstemp(prefix="mooncake-tcp-topo-", suffix=".json")
    os.close(fd)
    Path(topo_path).write_text(json.dumps(topo), encoding="utf-8")
    overrides = {
        "MC_CUSTOM_TOPO_JSON": topo_path,
        "VLLM_HOST_IP": "127.0.0.1",
        # Landing path does not need flashinfer kernels; bypass host mismatches.
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    }
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        Path(topo_path).unlink(missing_ok=True)


def _cleanup_process(process: Process, *, stop_queue: Queue | None = None) -> None:
    """Best-effort teardown for the spawned AR producer."""
    if stop_queue is not None and process.is_alive():
        try:
            stop_queue.put("stop")
        except (BrokenPipeError, OSError, ValueError):
            pass

    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    if process.is_alive() and process.pid is not None:
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)


def _make_kv_cache_config():
    return make_kv_cache_config(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
        layer_names=[LAYER_NAME],
    )


def _alloc_paged_kv_tensor(device: torch.device) -> torch.Tensor:
    """Allocate source-model blocks with one FullAttention page per row."""
    spec = _make_kv_cache_config().kv_cache_groups[0].kv_cache_spec
    assert isinstance(spec, FullAttentionSpec)
    page_elems = spec.page_size_bytes // spec.dtype.itemsize
    tensor = torch.zeros(NUM_BLOCKS, page_elems, dtype=spec.dtype, device=device)
    assert tensor.stride(0) * tensor.element_size() == spec.page_size_bytes
    return tensor


def _fill_source_pages(kv_tensor: torch.Tensor, block_ids: list[int]) -> None:
    for block_id in block_ids:
        kv_tensor[block_id] = float(block_id + 1)


def _expected_page_value(source_block_id: int) -> float:
    return float(source_block_id + 1)


def _configure_source_attention_geometry(vllm_config: object) -> None:
    """Provide the fake source model's geometry through the production API."""

    model_config = vllm_config.model_config  # type: ignore[attr-defined]
    model_config.hf_config = SimpleNamespace(model_type="mooncake_landing_source")
    model_config.set_attention_geometry(
        num_heads=NUM_KV_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )


def _native_empty_scheduler_output() -> NativeSchedulerOutput:
    return NativeSchedulerOutput.make_empty()


class _LandingPipeline(nn.Module):
    """Minimal model tree with one real cache-enabled Omni Attention layer."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = Attention(
            num_heads=NUM_KV_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            causal=False,
            softmax_scale=HEAD_SIZE**-0.5,
            prefix=LAYER_NAME,
            paged_kv_cache_role="primary",
            paged_kv_cache_dtype=DTYPE,
        )


def _make_runner_kv_cache_config(runner: DiffusionModelRunner) -> KVCacheConfig:
    specs = runner.get_kv_cache_spec()
    assert set(specs) == {LAYER_NAME}
    spec = specs[LAYER_NAME]
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * NUM_BLOCKS,
                shared_by=[LAYER_NAME],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[LAYER_NAME],
                kv_cache_spec=spec,
            )
        ],
    )


def _init_single_gpu_parallel(master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{master_port}",
        local_rank=0,
        backend="nccl",
    )
    ensure_model_parallel_initialized(1, 1)


def _destroy_parallel() -> None:
    destroy_model_parallel()
    destroy_distributed_environment()


def _kv_transfer_config(*, engine_id: str, kv_role: str, extra: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "kv_connector": "MooncakeConnector",
        "kv_role": kv_role,
        "engine_id": engine_id,
        "kv_connector_extra_config": {
            "mooncake_protocol": MOONCAKE_PROTOCOL,
            **(extra or {}),
        },
    }
    return payload


def _transfer_id() -> str:
    return f"xfer-{REQUEST_ID}-s0-t1-a0"


def _run_ar_producer(
    dist_port: int,
    bootstrap_port: int,
    handshake: Queue,
    stop_queue: Queue,
) -> None:
    """Producer process: real MooncakeConnector, fake KV pages, no AR model."""
    os.environ["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(bootstrap_port)
    device = current_omni_platform.get_torch_device(0)
    current_omni_platform.set_device(device)
    try:
        od_config = OmniDiffusionConfig.from_kwargs(
            diffusion_kv_mode="paged_scheduler",
            diffusion_kv_max_sequences_per_request=1,
            max_model_len=64,
        )
        # Spawned producer has no pytest ``default_vllm_config`` fixture;
        # ``ensure_model_parallel_initialized`` requires a current config.
        vllm_config = create_diffusion_vllm_config(device, od_config)
        vllm_config.kv_transfer_config = KVTransferConfig(
            **_kv_transfer_config(
                engine_id=AR_ENGINE_ID,
                kv_role="kv_producer",
            )
        )
        _configure_source_attention_geometry(vllm_config)
        with set_current_vllm_config(vllm_config):
            kv_cache_config = _make_kv_cache_config()
            _init_single_gpu_parallel(dist_port)
            scheduler = KVConnectorFactory.create_connector(
                config=vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=kv_cache_config,
            )
            worker = KVConnectorFactory.create_connector(
                config=vllm_config,
                role=KVConnectorRole.WORKER,
                kv_cache_config=kv_cache_config,
            )
            ar_kv = _alloc_paged_kv_tensor(device)
            _fill_source_pages(ar_kv, SOURCE_BLOCK_IDS)
            current_omni_platform.synchronize()
            worker.register_kv_caches({LAYER_NAME: ar_kv})

            bootstrap_addr = f"http://127.0.0.1:{bootstrap_port}"
            params: dict[str, object] = {
                "transfer_id": _transfer_id(),
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": DIT_ENGINE_ID,
                "remote_bootstrap_addr": bootstrap_addr,
            }
            request = _ProducerRequest(REQUEST_ID, params, SEQ_LEN)
            # update_state_after_alloc enqueues an empty send slot; request_finished
            # later fills block IDs and marks ready. Both reach the worker via
            # async record_send_reqs — wait so the empty slot exists before the
            # finished path KeyErrors on a missing transfer_id.
            scheduler.update_state_after_alloc(request, SimpleNamespace(), 0)  # pyright: ignore[reportArgumentType]
            meta = scheduler.build_connector_meta(_native_empty_scheduler_output())
            worker.bind_connector_metadata(meta)
            worker.start_load_kv(None)  # pyright: ignore[reportArgumentType]
            _wait_producer_send_slot(worker, _transfer_id())

            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            scheduler.request_finished(request, SOURCE_BLOCK_IDS)  # pyright: ignore[reportArgumentType]
            meta = scheduler.build_connector_meta(_native_empty_scheduler_output())
            worker.bind_connector_metadata(meta)
            worker.start_load_kv(None)  # pyright: ignore[reportArgumentType]
            _wait_producer_send_ready(worker, _transfer_id())

            handshake.put({"ok": True, "bootstrap_addr": bootstrap_addr})
            stop_queue.get(timeout=RECV_TIMEOUT_S + 30)
            worker.shutdown()
            scheduler.shutdown()
    except Exception as exc:  # noqa: BLE001 — report to parent for skip/fail
        handshake.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            _destroy_parallel()
        except Exception:  # noqa: BLE001
            pass


def _producer_worker_side(worker: object):
    connector_worker = worker.connector_worker  # type: ignore[attr-defined]
    if connector_worker is None:
        raise RuntimeError("Mooncake producer worker role is not initialized")
    return connector_worker


def _wait_producer_send_slot(worker: object, transfer_id: str, *, timeout_s: float = 10.0) -> None:
    connector_worker = _producer_worker_side(worker)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if transfer_id in connector_worker.reqs_need_send:
            return
        time.sleep(0.05)
    raise TimeoutError(f"producer send slot for transfer_id={transfer_id!r} not recorded in time")


def _wait_producer_send_ready(worker: object, transfer_id: str, *, timeout_s: float = 10.0) -> None:
    connector_worker = _producer_worker_side(worker)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        send_meta = connector_worker.reqs_need_send.get(transfer_id)
        if send_meta is not None and send_meta.ready.is_set() and send_meta.local_block_ids:
            return
        time.sleep(0.05)
    raise TimeoutError(f"producer send ready for transfer_id={transfer_id!r} not set in time")


def _run_dit_consumer_and_assert(dist_port: int, bootstrap_addr: str) -> None:
    device = current_omni_platform.get_torch_device(0)
    current_omni_platform.set_device(device)
    od_config = OmniDiffusionConfig.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_sequences_per_request=2,
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        kv_transfer_config=_kv_transfer_config(engine_id=DIT_ENGINE_ID, kv_role="kv_consumer"),
    )
    scheduler: BaseScheduler | None = None
    dit_worker: DiffusionWorker | None = None
    try:
        vllm_config = create_diffusion_vllm_config(device, od_config)
        with set_current_vllm_config(vllm_config):
            _init_single_gpu_parallel(dist_port)
            runner = DiffusionModelRunner(vllm_config, od_config, device)
            with set_current_diffusion_config(od_config):
                runner.pipeline = _LandingPipeline().to(device)
            vllm_config.model_config.hf_config = SimpleNamespace(model_type="mooncake_landing_target")
            kv_cache_config = _make_runner_kv_cache_config(runner)
            source_spec = _make_kv_cache_config().kv_cache_groups[0].kv_cache_spec
            target_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
            assert source_spec.page_size_bytes == target_spec.page_size_bytes

            runner.set_kv_cache_config(kv_cache_config)
            kv_caches_by_layer = runner.diffusion_kv_backend.get_kv_caches_by_layer()
            dit_kv = kv_caches_by_layer[LAYER_NAME]
            assert dit_kv.stride(0) * dit_kv.element_size() == target_spec.page_size_bytes

            active_connector = runner.kv_connector
            assert active_connector is not None
            dit_worker = object.__new__(DiffusionWorker)
            dit_worker.rank = 0
            dit_worker.od_config = od_config
            dit_worker.model_runner = runner

            scheduler = ConcreteScheduler()
            scheduler.initialize(
                od_config,
                kv_cache_config=kv_cache_config,
                scheduler_block_size=BLOCK_SIZE,
                hash_block_size=BLOCK_SIZE,
                kv_vllm_config=vllm_config,
            )
            request = OmniDiffusionRequest(
                prompt="landing-test",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
                request_id=REQUEST_ID,
                diffusion_kv_requests=tuple(
                    DiffusionKVRequest(
                        f"{REQUEST_ID}/diffusion-kv/{sequence_id}",
                        sequence_id=sequence_id,
                        prefix_len=PREFIX_LEN,
                        target_len=TARGET_LEN,
                        seq_len=SEQ_LEN,
                    )
                    for sequence_id in range(2)
                ),
                kv_transfer_params={
                    "transfer_id": _transfer_id(),
                    "do_remote_prefill": True,
                    "do_remote_decode": False,
                    "remote_engine_id": AR_ENGINE_ID,
                    "remote_bootstrap_addr": bootstrap_addr,
                    "num_transfer_tokens": SEQ_LEN,
                },
            )
            scheduler.add_request(request)
            sched_output = scheduler.schedule()
            assert sched_output.kv_connector_metadata is not None
            assert sched_output.scheduled_new_reqs
            metadata = sched_output.scheduled_new_reqs[0].diffusion_kv_metadata
            assert metadata is not None
            assert len(metadata.sequences) == 2
            target_blocks_by_sequence = [sequence.block_ids[0] for sequence in metadata.sequences]
            assert all(len(target_block_ids) == len(SOURCE_BLOCK_IDS) for target_block_ids in target_blocks_by_sequence)
            assert set(target_blocks_by_sequence[0]).isdisjoint(target_blocks_by_sequence[1])
            num_prefix_blocks = PREFIX_LEN // BLOCK_SIZE
            target_prefix_blocks_by_sequence = [
                target_block_ids[:num_prefix_blocks] for target_block_ids in target_blocks_by_sequence
            ]
            source_prefix_blocks = SOURCE_BLOCK_IDS[-num_prefix_blocks:]

            output = dit_worker.prepare_kv_for_forward(
                sched_output,
                timeout_s=RECV_TIMEOUT_S,
            )
            expected_sequence_ids = {f"{REQUEST_ID}/diffusion-kv/{sequence_id}" for sequence_id in range(2)}
            assert output.kv_connector_output is not None
            assert expected_sequence_ids.issubset(output.kv_connector_output.finished_recving or set())
            assert (
                runner.diffusion_kv_backend.resolve_sequence_binding(REQUEST_ID, 0).req_index
                != runner.diffusion_kv_backend.resolve_sequence_binding(REQUEST_ID, 1).req_index
            )

            current_omni_platform.synchronize()
            landed_page_means = [dit_kv[block_id].float().mean().item() for block_id in range(NUM_BLOCKS)]
            for sequence_id, target_prefix_blocks in enumerate(target_prefix_blocks_by_sequence):
                for dest_block, source_block in zip(
                    target_prefix_blocks,
                    source_prefix_blocks,
                    strict=True,
                ):
                    got = dit_kv[dest_block].float().mean().item()
                    assert got == pytest.approx(_expected_page_value(source_block), abs=1e-2), (
                        "landed KV mismatch "
                        f"sequence_id={sequence_id} dest_block={dest_block} "
                        f"source_block={source_block} got={got}; "
                        f"all_page_means={landed_page_means}"
                    )
                for target_block in target_blocks_by_sequence[sequence_id][num_prefix_blocks:]:
                    assert landed_page_means[target_block] == 0.0
    finally:
        if dit_worker is not None:
            dit_worker.shutdown_kv_connector()
        if scheduler is not None:
            shutdown_kv_connector_v1(scheduler_connector=scheduler.connector)
        elif has_kv_transfer_group():
            ensure_kv_transfer_shutdown()
        _destroy_parallel()
