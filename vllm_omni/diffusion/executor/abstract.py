from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode, is_scheduler_paged_kv_mode
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorOutput

    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.utils import BaseRunnerOutput


class DiffusionExecutor(ABC):
    """Abstract base class for Diffusion executors."""

    uses_multiproc: bool = False

    @staticmethod
    def get_class(od_config: OmniDiffusionConfig) -> type[DiffusionExecutor]:
        executor_class: type[DiffusionExecutor]
        distributed_executor_backend = od_config.distributed_executor_backend
        # Mirror vLLM's `world_size == 1 -> "uni"` default
        # (vllm/config/parallel.py). A single-GPU diffusion deployment has
        # nothing to distribute: spawning a worker only adds MessageQueues,
        # /dev/shm output segments, and a second model load. Explicit "mp"
        # keeps process isolation and RPC timeouts. Multi-GPU still defaults
        # to "mp".
        if distributed_executor_backend is None:
            num_gpus = od_config.num_gpus or 1
            distributed_executor_backend = "uni" if num_gpus == 1 else "mp"

        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, DiffusionExecutor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"DiffusionExecutor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            raise NotImplementedError("ray backend is not yet supported.")
        elif distributed_executor_backend == "mp":
            from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor

            executor_class = MultiprocDiffusionExecutor
        elif distributed_executor_backend == "uni":
            from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor

            executor_class = UniProcDiffusionExecutor
        elif distributed_executor_backend == "external_launcher":
            raise NotImplementedError("external_launcher backend is not yet supported.")
        elif isinstance(distributed_executor_backend, str):
            try:
                executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            except (ImportError, ValueError) as e:
                raise ValueError(
                    f"Failed to load executor backend '{distributed_executor_backend}'. "
                    f"Ensure it is a valid python path. Error: {e}"
                ) from e

            if not issubclass(executor_class, DiffusionExecutor):
                raise TypeError(
                    f"distributed_executor_backend must be a subclass of DiffusionExecutor. Got {executor_class}."
                )
        else:
            raise ValueError(f"Unknown distributed executor backend: {distributed_executor_backend}")
        return executor_class

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        self.od_config = od_config
        self._kv_output_aggregator: KVOutputAggregator | None = None
        self._init_executor()

    def init_kv_output_aggregator(self, connector: KVConnectorBase_V1) -> None:
        self._kv_output_aggregator = KVOutputAggregator.from_connector(
            connector,
            int(self.od_config.num_gpus or 1),
        )

    def _prepare_kv_for_forward(self, scheduler_output: DiffusionSchedulerOutput) -> KVConnectorOutput | None:
        connector_metadata = scheduler_output.kv_connector_metadata
        expected_ids = scheduler_output.kv_connector_request_ids
        if connector_metadata is None and expected_ids:
            raise RuntimeError("KV Connector sequence IDs were emitted without metadata")
        if connector_metadata is not None and not expected_ids:
            raise RuntimeError("KV Connector metadata was emitted without expected sequence IDs")
        paged_mode = is_scheduler_paged_kv_mode(
            getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
        )
        has_paged_work = paged_mode and bool(scheduler_output.scheduled_new_reqs)
        has_paged_work = has_paged_work or any(
            new_req.diffusion_kv_metadata is not None for new_req in scheduler_output.scheduled_new_reqs
        )
        if not has_paged_work and connector_metadata is None:
            return None

        timeout_s = float(getattr(self.od_config, "dist_timeout", None) or 300)
        try:
            aggregator = self._kv_output_aggregator if connector_metadata is not None else None
            if connector_metadata is not None and aggregator is None:
                raise RuntimeError("KV Connector receive was scheduled without KVOutputAggregator")
            output = self.collective_rpc(
                "prepare_kv_for_forward",
                timeout=timeout_s + 5.0,
                args=(scheduler_output, timeout_s),
                kv_output_aggregator=aggregator,
            )
            if connector_metadata is None:
                return None
            if not isinstance(output, ModelRunnerOutput) or output.kv_connector_output is None:
                raise RuntimeError("KV Connector completion aggregation produced no output")
            kv_output = output.kv_connector_output
            if kv_output.invalid_block_ids:
                raise RuntimeError(
                    f"KV Connector completion contains invalid blocks: {sorted(kv_output.invalid_block_ids)}"
                )
            finished = kv_output.finished_recving or set()
            if not expected_ids.issubset(finished):
                raise RuntimeError(
                    "KV Connector all-rank completion is incomplete; missing sequence IDs: "
                    f"{sorted(expected_ids - finished)}"
                )
            return kv_output
        except Exception as exc:
            if connector_metadata is None:
                raise
            # Quiesce every Worker before target pages can be recycled.
            try:
                self.collective_rpc(
                    "shutdown_kv_connector",
                    timeout=10.0,
                    unique_reply_rank=0,
                    exec_all_ranks=True,
                )
            except Exception:
                pass
            self.shutdown()
            from vllm_omni.diffusion.diffusion_kv.kv_connector import KVConnectorFatalError

            if isinstance(exc, KVConnectorFatalError):
                raise
            raise KVConnectorFatalError(f"KV Connector prepare failed closed: {exc}") from exc

    @abstractmethod
    def _init_executor(self) -> None:
        """Initialize the executor (e.g., launch workers, setup IPC)."""
        pass

    @property
    @abstractmethod
    def is_dead(self) -> bool:
        """Whether the executor is shut down or has failed fatally."""
        pass

    @abstractmethod
    def execute_request(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Execute request-mode work from a scheduler output."""
        pass

    @abstractmethod
    def execute_batch(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Execute request-mode work through the request-batch path."""
        pass

    @abstractmethod
    def execute_step(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Execute step-mode work from a scheduler output."""
        pass

    @abstractmethod
    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
        exec_all_ranks: bool = False,
        kv_output_aggregator: KVOutputAggregator | None = None,
    ) -> Any:
        """Execute a method on workers."""
        pass

    @abstractmethod
    def check_health(self) -> None:
        """Check if the executor and workers are healthy."""
        pass

    def register_failure_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked when the executor fatally fails.

        Executors without a background failure monitor can keep the default
        no-op implementation.
        """
        return None

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        """Collect rank-local native specs after every Worker loads its model."""

        result = self.collective_rpc(
            "get_kv_cache_specs",
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        if not isinstance(result, list):
            raise TypeError(f"get_kv_cache_specs returned {type(result).__name__}, expected list")
        return result

    def determine_available_kv_memory(self, profile_requests: list[OmniDiffusionRequest]) -> list[int]:
        """Profile and collect the KV memory budget on every Worker rank."""

        result = self.collective_rpc(
            "determine_available_kv_memory",
            args=(profile_requests,),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        if not isinstance(result, list) or not all(isinstance(value, int) for value in result):
            raise TypeError("determine_available_kv_memory must return list[int]")
        return result

    def set_kv_cache_configs(self, kv_cache_configs: list[KVCacheConfig], resolved_max_model_len: int) -> None:
        """Send rank-local configs and the resolved model length to all Workers."""

        # The default control-plane RPC mode executes on every rank and has
        # rank 0 return the gathered rank statuses, so failures on nonzero
        # ranks are not silently dropped.
        self.collective_rpc("set_kv_cache_configs", args=(kv_cache_configs, resolved_max_model_len))

    def remove_diffusion_kv_requests(self, request_ids: list[str]) -> None:
        """Clear request rows on every Worker after Scheduler retirement."""

        unique_request_ids = list(dict.fromkeys(request_ids))
        if not unique_request_ids:
            return
        self.collective_rpc(
            "remove_diffusion_kv_requests",
            args=(unique_request_ids,),
        )

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the executor and release resources."""
        pass
