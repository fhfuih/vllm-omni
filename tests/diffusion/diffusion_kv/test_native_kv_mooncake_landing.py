# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU landing test for AR → DiT native Mooncake paged KV.

This module checks that fake KV pages written on a Mooncake producer (AR) can
land on DiT-side tensors after the PR-1 Omni control path:

1. DiT Scheduler admission (reserve + ``update_state_after_alloc`` +
   ``build_connector_meta``)
2. Worker ``init_worker_kv_connector_v1`` + ``start_load_kv``

It does **not** load Hunyuan/DiT weights and does **not** run prompt inference.
The AR side injects a recognizable numeric pattern into GPU pages.

#6102 stubs
-----------
#6102 owns ``DiffusionKVModelRunnerBackend.initialize_kv_cache``, BlockTables,
and the Worker ``maybe_register_vllm_kv_caches`` call site. Those are not
landed on this branch, so this test **does not** call them. Instead it:

- allocates GPU paged tensors from ``KVCacheConfig`` (stand-in for
  ``initialize_kv_cache``)
- calls ``maybe_register_vllm_kv_caches`` directly (stand-in for the Worker
  hook after backend init)
- skips ``install_diffusion_kv_metadata`` / paged-attention forward and
  asserts landing on the registered tensors

After #6102 merges, replace ``_stub_6102_allocate_and_register_pages`` with
the real backend init + Worker registration, and optionally assert through
installed BlockTables rather than raw tensors.

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
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.utils.network_utils import get_open_port
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import RequestStatus

from tests.diffusion.diffusion_kv.helper import ConcreteScheduler, make_kv_cache_config
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
from vllm_omni.diffusion.diffusion_kv.v1.connector import (
    init_worker_kv_connector_v1,
    maybe_register_vllm_kv_caches,
    mint_transfer_id,
    shutdown_kv_connector_v1,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]

LAYER_NAME = "model.layers.0.self_attn"
BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 8
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
    # Visible CUDA device is always cuda:0 under CUDA_VISIBLE_DEVICES remapping.
    topo = {
        "cpu:0": [[netif]],
        "cuda:0": [[netif]],
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
    """Blocks-first GPU tensor whose stride(0) matches one FullAttention page.

    Stand-in for #6102 ``initialize_kv_cache`` physical allocation.
    """
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


def _patch_model_config_for_mooncake(model_config: object) -> None:
    """Supply Mooncake worker lookups missing on Omni's diffusion ModelConfig stub."""
    if not hasattr(model_config, "get_head_size"):
        model_config.get_head_size = lambda: HEAD_SIZE  # type: ignore[attr-defined]
    if not hasattr(model_config, "get_total_num_kv_heads"):
        model_config.get_total_num_kv_heads = lambda: NUM_KV_HEADS  # type: ignore[attr-defined]


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


def _stub_6102_allocate_and_register_pages(device: torch.device) -> torch.Tensor:
    """Allocate and register DiT pages without #6102 backend init.

    TODO(#6102): replace with ``initialize_kv_cache`` then the Worker
    ``maybe_register_vllm_kv_caches(backend.kv_caches_by_layer)`` call.
    """
    kv_tensor = _alloc_paged_kv_tensor(device)
    maybe_register_vllm_kv_caches({LAYER_NAME: kv_tensor})
    return kv_tensor


def _run_ar_producer(
    dist_port: int,
    bootstrap_port: int,
    handshake: Queue,
    stop_queue: Queue,
) -> None:
    """Producer process: real MooncakeConnector, fake KV pages, no AR model."""
    os.environ["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(bootstrap_port)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    try:
        od_config = OmniDiffusionConfig.from_kwargs(
            diffusion_kv_mode="paged_scheduler",
            max_model_len=64,
            kv_transfer_config=_kv_transfer_config(engine_id=AR_ENGINE_ID, kv_role="kv_producer"),
        )
        # Spawned producer has no pytest ``default_vllm_config`` fixture;
        # ``ensure_model_parallel_initialized`` requires a current config.
        vllm_config = create_diffusion_vllm_config(device, od_config)
        _patch_model_config_for_mooncake(vllm_config.model_config)
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
            worker.register_kv_caches({LAYER_NAME: ar_kv})

            bootstrap_addr = f"http://127.0.0.1:{bootstrap_port}"
            params: dict[str, object] = {
                "transfer_id": mint_transfer_id(REQUEST_ID),
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
            meta = scheduler.build_connector_meta(object())  # pyright: ignore[reportArgumentType]
            worker.bind_connector_metadata(meta)
            worker.start_load_kv(None)  # pyright: ignore[reportArgumentType]
            _wait_producer_send_slot(worker, mint_transfer_id(REQUEST_ID))

            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            scheduler.request_finished(request, SOURCE_BLOCK_IDS)  # pyright: ignore[reportArgumentType]
            meta = scheduler.build_connector_meta(object())  # pyright: ignore[reportArgumentType]
            worker.bind_connector_metadata(meta)
            worker.start_load_kv(None)  # pyright: ignore[reportArgumentType]
            _wait_producer_send_ready(worker, mint_transfer_id(REQUEST_ID))

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
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    od_config = OmniDiffusionConfig.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        max_model_len=64,
        kv_transfer_config=_kv_transfer_config(engine_id=DIT_ENGINE_ID, kv_role="kv_consumer"),
    )
    scheduler: BaseScheduler | None = None
    dit_kv: torch.Tensor | None = None
    try:
        vllm_config = create_diffusion_vllm_config(device, od_config)
        _patch_model_config_for_mooncake(vllm_config.model_config)
        with set_current_vllm_config(vllm_config):
            kv_cache_config = _make_kv_cache_config()
            _init_single_gpu_parallel(dist_port)
            init_worker_kv_connector_v1(vllm_config, kv_cache_config=kv_cache_config)
            dit_kv = _stub_6102_allocate_and_register_pages(device)

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
                diffusion_kv_requests=(
                    DiffusionKVRequest(
                        f"{REQUEST_ID}/diffusion-kv/0",
                        sequence_id=0,
                        prefix_len=PREFIX_LEN,
                        target_len=TARGET_LEN,
                        seq_len=SEQ_LEN,
                    ),
                ),
                kv_transfer_params={
                    "transfer_id": mint_transfer_id(REQUEST_ID),
                    "do_remote_prefill": True,
                    "do_remote_decode": False,
                    "remote_engine_id": AR_ENGINE_ID,
                    "remote_bootstrap_addr": bootstrap_addr,
                },
            )
            scheduler.add_request(request)
            sched_output = scheduler.schedule()
            assert sched_output.kv_connector_metadata is not None
            assert sched_output.scheduled_new_reqs
            metadata = sched_output.scheduled_new_reqs[0].diffusion_kv_metadata
            assert metadata is not None
            target_block_ids = metadata.sequences[0].block_ids[0]
            assert len(target_block_ids) == len(SOURCE_BLOCK_IDS)

            runner = object.__new__(DiffusionModelRunner)
            runner.od_config = od_config
            runner._maybe_start_remote_kv_load(sched_output)

            deadline = time.monotonic() + RECV_TIMEOUT_S
            finished = False
            while time.monotonic() < deadline:
                if has_kv_transfer_group():
                    # Upstream contract: (finished_sending, finished_recving).
                    _finished_sending, finished_recving = get_kv_transfer_group().get_finished(set())
                    if finished_recving and REQUEST_ID in finished_recving:
                        finished = True
                        break
                    seq_id = f"{REQUEST_ID}/diffusion-kv/0"
                    if finished_recving and seq_id in finished_recving:
                        finished = True
                        break
                time.sleep(0.2)
            assert finished, "DiT Mooncake consumer did not report finished_recving in time"

            torch.accelerator.synchronize()
            for dest_block, source_block in zip(target_block_ids, SOURCE_BLOCK_IDS, strict=True):
                got = dit_kv[dest_block].float().mean().item()
                assert got == pytest.approx(_expected_page_value(source_block), abs=1e-2), (
                    f"landed KV mismatch dest_block={dest_block} source_block={source_block} got={got}"
                )
    finally:
        if scheduler is not None:
            shutdown_kv_connector_v1(scheduler_connector=scheduler.kv_connector_v1)
        elif has_kv_transfer_group():
            ensure_kv_transfer_shutdown()
        _destroy_parallel()
