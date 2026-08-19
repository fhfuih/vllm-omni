# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.config import KVTransferConfig

from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyARStage:
    stage_type = "llm"
    final_output = False
    od_config = None

    def get_kv_sender_info(self):
        return {"host": "10.0.0.1", "zmq_port": 50151}


class _DummyDiffusionStage:
    stage_type = "diffusion"
    final_output = True
    custom_process_input_func = None
    engine_input_source = [0]

    def __init__(self, kv_transfer_config):
        self.od_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
        self.calls = []

    async def add_request_async(
        self,
        request_id,
        prompt,
        sampling_params,
        kv_sender_info=None,
        kv_transfer_params=None,
    ):
        self.calls.append(
            {
                "request_id": request_id,
                "kv_sender_info": kv_sender_info,
                "kv_transfer_params": kv_transfer_params,
            }
        )


def test_orchestrator_mints_matching_transfer_id_on_ar_and_dit() -> None:
    ar_cfg = KVTransferConfig(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
        engine_id="ar-engine",
        kv_connector_extra_config={"bootstrap_addr": "ar-host:9991"},
    )
    dit_cfg = KVTransferConfig(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        engine_id="dit-engine",
        kv_connector_extra_config={"bootstrap_addr": "dit-host:9992"},
    )
    ar_pool = StagePool(
        0,
        _DummyARStage(),
        output_processor=object(),
        stage_vllm_config=SimpleNamespace(kv_transfer_config=ar_cfg),
    )
    dit_stage = _DummyDiffusionStage(dit_cfg)
    dit_pool = StagePool(1, dit_stage, stage_vllm_config=SimpleNamespace(kv_transfer_config=dit_cfg))

    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [ar_pool, dit_pool]

    req_state = OrchestratorRequestState(
        request_id="req-hy",
        prompt={"prompt": "hello"},
        sampling_params_list=[SamplingParams(max_tokens=4), OmniDiffusionSamplingParams()],
        final_stage_id=1,
    )

    orchestrator._maybe_attach_native_kv_handshake(req_state)

    assert req_state.native_kv_transfer_id == "xfer-req-hy"
    ar_params = req_state.sampling_params_list[0].extra_args["kv_transfer_params"]
    dit_params = req_state.native_kv_target_params
    assert ar_params["transfer_id"] == dit_params["transfer_id"] == "xfer-req-hy"
    assert ar_params["do_remote_decode"] is True
    assert dit_params["do_remote_prefill"] is True
    assert ar_params["remote_engine_id"] == "dit-engine"
    assert dit_params["remote_engine_id"] == "ar-engine"
