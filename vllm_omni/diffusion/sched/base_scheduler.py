# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from dataclasses import fields
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.kv_connector import (
    create_scheduler_kv_connector_v1,
    shutdown_kv_connector_v1,
)
from vllm_omni.diffusion.diffusion_kv.manager import (
    DiffusionKVAdmissionError,
    DiffusionKVCacheManager,
    DiffusionKVReservationState,
)
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.diffusion_kv.request import prepare_kv_connector_request
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    KVPrefetchJob,
    NewRequestData,
    RequestBatchSamplingParamsKey,
    SchedulerRequestState,
    StepBatchSamplingParamsKey,
    _AdmissionWaitDecision,
)
from vllm_omni.diffusion.worker.utils import RunnerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

logger = init_logger(__name__)

BatchSamplingParamsKey = StepBatchSamplingParamsKey | RequestBatchSamplingParamsKey

# LoRA identity is derived from `sampling.lora_request`, not a same-named field
# on sampling params, so it must be resolved separately from the bulk lookup.
_STEP_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES = frozenset(field.name for field in fields(StepBatchSamplingParamsKey)) - {
    "lora_int_id"
}


class BaseScheduler(ABC):
    """Shared queue/state bookkeeping for diffusion schedulers."""

    def __init__(self) -> None:
        self.od_config: OmniDiffusionConfig | None = None
        self._request_states: dict[str, SchedulerRequestState] = {}
        self._step_id: int = 0
        self._waiting: deque[str] = deque()
        self._running: list[str] = []
        self._running_sampling_params_key: BatchSamplingParamsKey | None = None
        self._finished_req_ids: set[str] = set()
        self.max_num_running_reqs: int = 1
        self._prefetch_enabled: bool = False
        self.kv_cache_manager: DiffusionKVCacheManager | None = None
        self.connector: KVConnectorBase_V1 | None = None

    def initialize(
        self,
        od_config: OmniDiffusionConfig,
        *,
        kv_cache_config: KVCacheConfig | None = None,
        scheduler_block_size: int | None = None,
        hash_block_size: int | None = None,
        kv_vllm_config: VllmConfig | None = None,
    ) -> None:
        self.od_config = od_config
        self._request_states.clear()
        self._step_id = 0
        self._waiting.clear()
        self._running.clear()
        self._running_sampling_params_key = None
        self._finished_req_ids.clear()
        max_num_seqs = getattr(od_config, "max_num_seqs", 1)
        try:
            self.max_num_running_reqs = max(1, int(max_num_seqs))
        except (TypeError, ValueError):
            self.max_num_running_reqs = 1

        omni_kv = getattr(od_config, "omni_kv_config", None) or {}
        self._prefetch_enabled = bool(omni_kv.get("enable_kv_async_prefetch", False))

        diffusion_kv_enabled = (
            getattr(od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
            is DiffusionKVCacheMode.PAGED_SCHEDULER
        )
        if diffusion_kv_enabled:
            if kv_cache_config is None:
                raise ValueError("paged_scheduler Diffusion KV requires a native Scheduler KVCacheConfig")
            if scheduler_block_size is None or hash_block_size is None:
                raise ValueError("paged_scheduler Diffusion KV requires native scheduler/hash block sizes")
            if kv_vllm_config is None:
                raise ValueError("paged_scheduler Diffusion KV requires the native VllmConfig used for cache sizing")
            self.kv_cache_manager = DiffusionKVCacheManager(
                kv_cache_config,
                max_model_len=kv_vllm_config.model_config.max_model_len,
                scheduler_block_size=scheduler_block_size,
                hash_block_size=hash_block_size,
                max_in_flight_tokens=kv_vllm_config.max_in_flight_tokens,
            )
        else:
            if any(
                value is not None
                for value in (
                    kv_cache_config,
                    scheduler_block_size,
                    hash_block_size,
                    kv_vllm_config,
                )
            ):
                raise ValueError("dense_legacy Scheduler received unexpected Diffusion KV cache initialization state")
            self.kv_cache_manager = None
        self.connector = create_scheduler_kv_connector_v1(
            od_config,
            kv_cache_config=kv_cache_config,
            vllm_config=kv_vllm_config,
        )
        self._reset_scheduler_state()

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        """Return the Scheduler-owned KV Connector, if configured."""

        return self.connector

    def add_request(self, request: OmniDiffusionRequest) -> str:
        return self._add_request_with_request_id(request.request_id, request)

    def _add_request_with_request_id(self, request_id: str, request: OmniDiffusionRequest) -> str:
        if request_id in self._request_states:
            raise ValueError(f"request_id {request_id!r} is already active.")
        state = self._make_request_state(request_id, request)
        self._request_states[request_id] = state
        self._waiting.append(request_id)
        logger.debug("%s add_request: %s (waiting=%d)", self.__class__.__name__, request_id, len(self._waiting))
        return request_id

    def schedule(self) -> DiffusionSchedulerOutput:
        scheduled_new_reqs: list[NewRequestData] = []
        scheduled_cached_request_ids: list[str] = []
        kv_connector_request_ids: set[str] = set()

        # First, schedule the RUNNING request(s)
        for request_id in self._running:
            state = self._request_states.get(request_id)
            if state is not None:
                scheduled_cached_request_ids.append(request_id)

        # Second, schedule WAITING requests while capacity remains.
        while self._waiting and len(self._running) < self.max_num_running_reqs:
            request_id = self._waiting[0]
            state = self._request_states.get(request_id)
            if state is None:
                self._waiting.popleft()
                continue
            if not self._can_schedule_waiting(state):
                break

            diffusion_kv_metadata: DiffusionKVMetadata | None = None
            if self.kv_cache_manager is not None:
                try:
                    allocation = self._allocate_kv_cache(state)
                except Exception as exc:
                    if not isinstance(exc, DiffusionKVAdmissionError):
                        logger.exception("Unexpected Diffusion KV admission failure for request %s", request_id)
                    self._finish_requests(
                        {request_id: DiffusionRequestStatus.FINISHED_ERROR},
                        {request_id: str(exc)},
                    )
                    continue
                if allocation is None:
                    break
                diffusion_kv_metadata, expected_ids = allocation
                kv_connector_request_ids.update(expected_ids)

            self._waiting.popleft()
            was_new_request = state.status == DiffusionRequestStatus.WAITING
            if not self._running:
                self._running_sampling_params_key = state.sampling_params_key
            state.status = DiffusionRequestStatus.RUNNING
            self._running.append(request_id)
            if was_new_request:
                scheduled_new_reqs.append(
                    NewRequestData.from_state(
                        state,
                        diffusion_kv_metadata=diffusion_kv_metadata,
                    )
                )
            else:
                scheduled_cached_request_ids.append(request_id)

        # Expose the next waiting request (serial mode) so the runner can
        # prefetch its KV during this forward.  Skip a request without
        # kv_sender_info (would target the wrong sender under multi-replica) or
        # one already finished/aborted (would consume its sender buffer for
        # nothing).
        kv_prefetch_job: KVPrefetchJob | None = None
        if self._prefetch_enabled and self._waiting:
            nxt = self._request_states.get(self._waiting[0])
            if nxt is not None and not nxt.is_finished():
                sender_info = getattr(nxt.req, "kv_sender_info", None)
                if sender_info:
                    kv_prefetch_job = {
                        "request_id": nxt.request_id,
                        "kv_sender_info": sender_info,
                    }

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(request_ids=scheduled_cached_request_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
            kv_prefetch_job=kv_prefetch_job,
            kv_connector_metadata=None,
            kv_connector_request_ids=frozenset(kv_connector_request_ids),
        )
        if kv_connector_request_ids:
            try:
                if self.connector is None:
                    raise RuntimeError("KV Connector was closed before metadata construction")
                scheduler_output.kv_connector_metadata = self._build_kv_connector_meta(
                    self.connector,
                    scheduler_output,
                )
            except Exception as exc:
                logger.exception("Fatal KV connector metadata construction failure")
                loading_request_ids = self._loading_public_request_ids(
                    new_req.request_id for new_req in scheduled_new_reqs
                )
                self._finish_requests(
                    {request_id: DiffusionRequestStatus.FINISHED_ERROR for request_id in loading_request_ids},
                    {request_id: str(exc) for request_id in loading_request_ids},
                )
                scheduler_output.scheduled_new_reqs = [
                    new_req
                    for new_req in scheduler_output.scheduled_new_reqs
                    if new_req.request_id not in loading_request_ids
                ]
                scheduler_output.finished_req_ids.update(loading_request_ids)
                scheduler_output.kv_connector_request_ids = frozenset()
                scheduler_output.num_running_reqs = len(self._running)

        # update after schedule
        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def _allocate_kv_cache(
        self,
        state: SchedulerRequestState,
    ) -> tuple[DiffusionKVMetadata, frozenset[str]] | None:
        manager = self.kv_cache_manager
        assert manager is not None
        if manager.has_request(state.request_id):
            return manager.get_metadata(state.request_id), frozenset()

        connector = self.connector
        connector_admission = []
        if connector is None:
            if self.od_config is not None and getattr(self.od_config, "kv_transfer_config", None) is not None:
                raise RuntimeError("Diffusion KV Connector is fail-closed")
            if any(request.kv_transfer_params is not None for request in state.diffusion_kv_requests):
                raise RuntimeError("Diffusion KV request requires a configured KV Connector")
        else:
            # Preflight every CFG sequence before mutating connector state.
            for request in state.diffusion_kv_requests:
                computed_blocks, local_tokens, _ = manager.get_computed_blocks(request)
                external_tokens, load_async = connector.get_num_new_matched_tokens(
                    request,  # type: ignore[arg-type]
                    local_tokens,
                )
                if external_tokens is None:
                    return None
                connector_admission.append((computed_blocks, local_tokens, external_tokens, bool(load_async)))

        allocation = manager.reserve_request(
            state.request_id,
            state.diffusion_kv_requests,
            reservation_inputs=[entry[:3] for entry in connector_admission] or None,
        )
        if allocation is None:
            return None

        try:
            expected_ids: set[str] = set()
            if connector is not None:
                # Commit only after the complete public request is reserved.
                for request, (_, _, external_tokens, load_async) in zip(
                    state.diffusion_kv_requests,
                    connector_admission,
                    strict=True,
                ):
                    connector.update_state_after_alloc(
                        request,  # type: ignore[arg-type]
                        manager.get_external_blocks(
                            manager.get_blocks(request.request_id),
                            external_tokens,
                        ),
                        external_tokens,
                    )
                    if external_tokens > 0 or load_async:
                        expected_ids.add(request.request_id)
            frozen_expected_ids = frozenset(expected_ids)
            if frozen_expected_ids:
                manager.mark_loading(state.request_id, frozen_expected_ids)
            else:
                manager.mark_resident_without_load(state.request_id)
        except Exception:
            self._fail_close_kv_connector()
            manager.free_request(state.request_id)
            raise
        return allocation, frozen_expected_ids

    def _build_kv_connector_meta(
        self,
        connector: KVConnectorBase_V1,
        scheduler_output: DiffusionSchedulerOutput,
    ):
        return connector.build_connector_meta(scheduler_output)  # type: ignore[arg-type]

    @abstractmethod
    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        pass

    def has_requests(self) -> bool:
        return bool(self._waiting or self._running)

    def num_waiting_requests(self) -> int:
        return len(self._waiting)

    def num_running_requests(self) -> int:
        return len(self._running)

    def get_admission_wait_decision(
        self,
        *,
        now: float,
        dp_concurrent: bool = False,
    ) -> _AdmissionWaitDecision:
        """Return the admission-delay policy for the next scheduling wave."""
        del now, dp_concurrent
        return _AdmissionWaitDecision(should_wait=False)

    def should_end_admission_wait(
        self,
        decision: _AdmissionWaitDecision,
        *,
        now: float,
        stable_since: float,
    ) -> bool:
        """Return whether an active admission delay should end."""
        del decision, now, stable_since
        return True

    def get_request_state(self, request_id: str) -> SchedulerRequestState | None:
        return self._request_states.get(request_id)

    def pop_request_state(self, request_id: str) -> SchedulerRequestState | None:
        if self.kv_cache_manager is not None:
            if request_id in self._loading_public_request_ids((request_id,)):
                self._fail_close_kv_connector()
            self.kv_cache_manager.free_request(request_id)
        self._pop_extra_request_state(request_id)
        return self._request_states.pop(request_id, None)

    def preempt_request(self, request_id: str) -> bool:
        if request_id not in self._request_states:
            return False
        if request_id in self._running:
            self._running.remove(request_id)
            if not self._running:
                self._running_sampling_params_key = None
            self._waiting.appendleft(request_id)
            self._request_states[request_id].status = DiffusionRequestStatus.PREEMPTED
            return True
        return False

    def finish_requests(self, request_ids: str | list[str], status: DiffusionRequestStatus) -> None:
        assert DiffusionRequestStatus.is_finished(status)
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        self._finish_requests({request_id: status for request_id in request_ids})

    def close(self) -> None:
        shutdown_kv_connector_v1(scheduler_connector=self.connector)
        self.connector = None
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.close()
            self.kv_cache_manager = None
        self._request_states.clear()
        self._waiting.clear()
        self._running.clear()
        self._running_sampling_params_key = None
        self._finished_req_ids.clear()
        self._reset_scheduler_state()

    def _finish_requests(
        self,
        statuses: dict[str, DiffusionRequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        if not statuses:
            return set()

        # Loading allocations may still be DMA targets.
        if self._loading_public_request_ids(statuses):
            self._fail_close_kv_connector()

        finished_req_ids: set[str] = set()
        running_to_remove: set[str] = set()
        waiting_to_remove: set[str] = set()

        for request_id, status in statuses.items():
            assert DiffusionRequestStatus.is_finished(status)
            state = self._request_states.get(request_id)
            if state is None or state.is_finished():
                continue

            finished_req_ids.add(request_id)
            if request_id in self._running:
                running_to_remove.add(request_id)
            if request_id in self._waiting:
                waiting_to_remove.add(request_id)

        if running_to_remove:
            self._running = [request_id for request_id in self._running if request_id not in running_to_remove]
            if not self._running:
                self._running_sampling_params_key = None
        if waiting_to_remove:
            self._waiting = deque(request_id for request_id in self._waiting if request_id not in waiting_to_remove)

        for request_id in finished_req_ids:
            if self.kv_cache_manager is not None:
                self.kv_cache_manager.free_request(request_id)
            state = self._request_states[request_id]
            status = statuses[request_id]
            state.status = status
            if status == DiffusionRequestStatus.FINISHED_ERROR:
                state.error = None if errors is None else errors.get(request_id)
            else:
                state.error = None

        self._finished_req_ids |= finished_req_ids
        return finished_req_ids

    def _loading_public_request_ids(self, request_ids: Iterable[str]) -> set[str]:
        manager = self.kv_cache_manager
        if manager is None:
            return set()

        return {
            request_id
            for request_id in request_ids
            if manager.has_request(request_id)
            and manager.get_request_state(request_id) is DiffusionKVReservationState.LOADING
        }

    def _finalize_update_from_output(
        self,
        sched_output: DiffusionSchedulerOutput,
        output: RunnerOutput,
        statuses: dict[str, DiffusionRequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        # A scheduled request may be aborted after schedule() but before
        # update_from_output() processes the runner output. It is already
        # marked finished at that point, but we still need to surface its id
        # in this update so the engine can observe the terminal state.
        # Also surface admission failures recorded while schedule() built this
        # output. Older finished ids retained only for Worker cleanup have
        # already been popped by the Engine and are deliberately ignored.
        kv_output = getattr(output, "kv_connector_output", None)
        if kv_output is not None:
            self._update_from_kv_xfer_finished(sched_output.scheduled_request_ids, kv_output)

        finished_req_ids = {
            request_id for request_id in sched_output.finished_req_ids if request_id in self._request_states
        }
        finished_req_ids |= {
            request_id for request_id in sched_output.scheduled_request_ids if request_id in self._finished_req_ids
        }
        finished_req_ids |= self._finish_requests(statuses, errors)
        return finished_req_ids

    def _update_from_kv_xfer_finished(
        self,
        request_ids: Iterable[str],
        kv_connector_output: KVConnectorOutput,
    ) -> None:
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)
        if self.kv_cache_manager is None:
            return

        finished_recving = kv_connector_output.finished_recving or set()
        for request_id in request_ids:
            if (
                self.kv_cache_manager.has_request(request_id)
                and self.kv_cache_manager.get_request_state(request_id) is DiffusionKVReservationState.LOADING
            ):
                self.kv_cache_manager.mark_resident(request_id, finished_recving)

    def _fail_close_kv_connector(self) -> None:
        connector = self.connector
        self.connector = None
        if connector is not None:
            shutdown_kv_connector_v1(scheduler_connector=connector)

    def _reset_scheduler_state(self) -> None:
        """Reset subclass-owned state during initialize()/close()."""

    def _pop_extra_request_state(self, request_id: str) -> None:
        """Remove subclass-owned per-request state before popping request state."""

    def _make_request_state(self, request_id: str, request: OmniDiffusionRequest) -> SchedulerRequestState:
        kv_requests = request.diffusion_kv_requests or ()
        if self.kv_cache_manager is not None:
            self._reject_legacy_dense_kv(request)
            if not kv_requests:
                raise ValueError("paged_scheduler request preprocessing did not produce DiffusionKVRequest state")
            for kv_request in kv_requests:
                prepare_kv_connector_request(kv_request, request.kv_transfer_params)
            if request.kv_transfer_params is not None:
                request.kv_sender_info = None
        elif kv_requests:
            raise ValueError("dense_legacy request unexpectedly contains Scheduler Diffusion KV requests")

        # DiffusionKVRequest objects are mutable Scheduler/native-KVCacheManager
        # state and must never ride the normal request payload to a Worker.
        request.diffusion_kv_requests = None
        return SchedulerRequestState(
            request_id=request_id,
            req=request,
            sampling_params_key=self._build_sampling_params_key(request),
            diffusion_kv_requests=kv_requests,
        )

    @staticmethod
    def _reject_legacy_dense_kv(request: OmniDiffusionRequest) -> None:
        """Keep dense injected KV out of the Scheduler-owned paged path."""

        populated_fields: list[str] = []
        for owner_name, owner in (
            ("request", request),
            ("sampling_params", request.sampling_params),
        ):
            for field_name, value in vars(owner).items():
                if value is not None and (field_name == "past_key_values" or field_name.endswith("_past_key_values")):
                    populated_fields.append(f"{owner_name}.{field_name}")
        if populated_fields:
            fields_text = ", ".join(sorted(populated_fields))
            raise ValueError(
                "paged_scheduler Diffusion KV does not accept legacy dense KV payloads; "
                f"clear these fields before admission: {fields_text}"
            )

    def _can_schedule_waiting(self, state: SchedulerRequestState) -> bool:
        if not self._running:
            return True

        current_key = self._current_sampling_params_key()
        return current_key is not None and current_key == state.sampling_params_key

    def _current_sampling_params_key(self) -> BatchSamplingParamsKey | None:
        if self._running_sampling_params_key is not None or not self._running:
            return self._running_sampling_params_key
        state = self._request_states.get(self._running[0])
        self._running_sampling_params_key = None if state is None else state.sampling_params_key
        return self._running_sampling_params_key

    def _build_sampling_params_key(
        self, request: OmniDiffusionRequest
    ) -> StepBatchSamplingParamsKey | RequestBatchSamplingParamsKey:  # return type loosened for subclassing
        """Build a step-batch compatibility key from sampling parameters."""
        sampling = request.sampling_params
        # LoRA identity is optional on sampling params (and on test stubs).
        lora_request = getattr(sampling, "lora_request", None)
        return StepBatchSamplingParamsKey(
            lora_int_id=lora_request.lora_int_id if lora_request is not None else None,
            **{name: getattr(sampling, name) for name in _STEP_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES},
        )


class SchedulerInterface(BaseScheduler):
    """Deprecated compatibility base for custom scheduler injection.

    Prefer subclassing :class:`BaseScheduler` directly. Subclassing this name
    still works but emits a :class:`DeprecationWarning`.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        import warnings

        warnings.warn(
            "SchedulerInterface is deprecated; subclass BaseScheduler instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)
