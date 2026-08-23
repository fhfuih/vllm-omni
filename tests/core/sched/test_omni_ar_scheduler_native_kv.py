# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm.v1.request import RequestStatus

from vllm_omni.config.omni_config import KVTransferBackend
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler() -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._kv_transfer_backend = KVTransferBackend.KV_CONNECTOR
    sched._omits_kv_transfer_cache = {}
    sched._inflight_prefills = set()
    sched.encoder_cache_manager = SimpleNamespace(free=lambda _req: None)
    sched.finished_req_ids = set()
    sched.finished_req_ids_dict = None
    sched._new_prompt_len_snapshot = {}
    sched.waiting_for_transfer_free = set()
    sched.transfer_triggered_requests = set()
    sched.active_kv_transfers = set()
    sched.requests_needing_kv_transfer = {}
    sched.chunk_transfer_adapter = None
    sched.freed = []
    sched._free_blocks = lambda request: sched.freed.append(request.request_id)
    sched._free_input_coordinator_request = lambda _req_id: None
    return sched


def _request(*, status: RequestStatus, native: bool):
    class _Req:
        request_id = "req-native"
        client_index = 0
        additional_information = None

        def __init__(self):
            self.status = status
            self.num_computed_tokens = 1232
            self.num_output_placeholders = 2
            self.kv_transfer_params = (
                {"transfer_id": "xfer-req-native", "do_remote_decode": True, "do_remote_prefill": False}
                if native
                else None
            )

        def is_finished(self) -> bool:
            return RequestStatus.is_finished(self.status)

        def __hash__(self) -> int:
            return hash(self.request_id)

        def __eq__(self, other) -> bool:
            return isinstance(other, _Req) and other.request_id == self.request_id

    return _Req()


def test_native_eos_shims_to_length_capped_and_uses_delay_free_only() -> None:
    sched = _scheduler()
    request = _request(status=RequestStatus.FINISHED_STOPPED, native=True)

    def _connector_finished(req):
        assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
        return True, {"transfer_id": "xfer-req-native", "remote_engine_id": "ar-engine"}

    sched._connector_finished = _connector_finished

    kv_params, extra = sched._free_request(request)

    assert kv_params == {
        "transfer_id": "xfer-req-native",
        "remote_engine_id": "ar-engine",
        "num_transfer_tokens": 1230,
    }
    assert extra is None
    assert sched.freed == []
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_native_error_does_not_shim_and_may_free() -> None:
    sched = _scheduler()
    request = _request(status=RequestStatus.FINISHED_ERROR, native=True)
    sched._connector_finished = lambda _req: (False, None)

    sched._free_request(request)

    assert request.status == RequestStatus.FINISHED_ERROR
    assert sched.freed == ["req-native"]
