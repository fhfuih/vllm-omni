# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm import SamplingParams

from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _orchestrator_with_native_diffusion_stage() -> Orchestrator:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [
        SimpleNamespace(stage_type="llm", stage_vllm_config=SimpleNamespace(kv_transfer_config=None)),
        SimpleNamespace(
            stage_type="diffusion",
            stage_vllm_config=SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnector")),
        ),
    ]
    return orchestrator


def test_native_ticket_is_added_to_upstream_params_and_forwarded_unchanged() -> None:
    orchestrator = _orchestrator_with_native_diffusion_stage()
    req_state = OrchestratorRequestState(
        request_id="req-native-0",
        sampling_params_list=[SamplingParams(max_tokens=4)],
        final_stage_id=1,
    )

    orchestrator._maybe_attach_native_kv_transfer_params(req_state)

    assert req_state.native_kv_transfer_id == "xfer-req-native-0"
    assert req_state.sampling_params_list[0].extra_args["kv_transfer_params"] == {"transfer_id": "xfer-req-native-0"}
    orchestrator._build_kv_sender_info = lambda *args, **kwargs: None
    submit_kwargs = orchestrator._diffusion_submit_kwargs(
        "req-native-0",
        0,
        SimpleNamespace(engine_input_source=None),
        req_state,
    )
    assert submit_kwargs["kv_transfer_params"] == {"transfer_id": "xfer-req-native-0"}


def test_native_ticket_is_not_created_for_legacy_diffusion_stage() -> None:
    orchestrator = _orchestrator_with_native_diffusion_stage()
    orchestrator.stage_pools[1].stage_vllm_config.kv_transfer_config = None
    req_state = OrchestratorRequestState(
        request_id="req-legacy-0",
        sampling_params_list=[SamplingParams(max_tokens=4)],
        final_stage_id=1,
    )

    orchestrator._maybe_attach_native_kv_transfer_params(req_state)

    assert req_state.native_kv_transfer_id is None
    assert req_state.sampling_params_list[0].extra_args is None
