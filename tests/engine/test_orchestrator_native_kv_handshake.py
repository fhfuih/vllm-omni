# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.config import KVTransferConfig

from vllm_omni.engine.kv_transfer_backend import KVTransferBackendManager
from vllm_omni.engine.orchestrator import OrchestratorRequestState
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyARStage:
    stage_type = "llm"
    final_output = False
    od_config = None

    def __init__(self, kv_transfer_config):
        self.vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)


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
        self.calls.append(kv_transfer_params)


def _config(role: str, engine_id: str, port: int) -> KVTransferConfig:
    return KVTransferConfig(
        kv_connector="MooncakeConnector",
        kv_role=role,
        engine_id=engine_id,
        kv_connector_extra_config={"bootstrap_addr": f"host:{port}"},
    )


def _request_state(request_id: str) -> OrchestratorRequestState:
    return OrchestratorRequestState(
        request_id=request_id,
        prompt={"prompt": "hello"},
        sampling_params_list=[SamplingParams(max_tokens=4), OmniDiffusionSamplingParams()],
        final_stage_id=1,
    )


def test_bound_ar_metadata_and_dynamic_dit_routing_share_one_control_path() -> None:
    ar_config = _config("kv_producer", "ar-0", 9000)
    dit_configs = [_config("kv_consumer", f"dit-{replica_id}", 9100 + replica_id) for replica_id in range(2)]
    ar_pool = StagePool(
        0,
        _DummyARStage(ar_config),
        output_processor=object(),
        stage_vllm_config=SimpleNamespace(kv_transfer_config=ar_config),
    )
    dit_stages = [_DummyDiffusionStage(config) for config in dit_configs]
    dit_pool = StagePool(1, dit_stages)
    manager = KVTransferBackendManager([ar_pool, dit_pool])

    async def forward(request_id: str, source_output=None):
        state = _request_state(request_id)
        plan = manager.create_plan(request_id, source_stage_id=0, final_stage_id=1)
        assert plan is not None
        source_request = SimpleNamespace(sampling_params=state.sampling_params_list[0].clone())
        manager.prepare_source(
            plan,
            state.sampling_params_list[0],
            source_request=source_request,
        )
        assert dit_pool.get_bound_replica_id(request_id) is None
        assert ar_pool.select_replica_id(request_id) == 0
        submit_kwargs = manager.build_target_submit_kwargs(
            plan,
            source_params=state.sampling_params_list[0],
            source_output=source_output,
        )
        replica_id = await dit_pool.submit_initial(
            request_id,
            state,
            state.prompt,
            submit_kwargs=submit_kwargs,
        )
        source_params = state.sampling_params_list[0].extra_args["kv_transfer_params"]
        return source_request, source_params, submit_kwargs["kv_transfer_params"], replica_id

    first = asyncio.run(
        forward(
            "req-0",
            SimpleNamespace(kv_transfer_params={"transfer_id": "connector-ticket", "num_transfer_tokens": 1232}),
        )
    )
    second = asyncio.run(forward("req-1"))

    first_source_request, first_source, first_target, first_replica = first
    _, second_source, second_target, second_replica = second

    assert first_source_request.sampling_params.extra_args["kv_transfer_params"] == first_source
    assert first_source["do_remote_decode"] is True
    assert "remote_engine_id" not in first_source
    assert first_target["transfer_id"] == "connector-ticket"
    assert first_target["do_remote_prefill"] is True
    assert first_target["remote_engine_id"] == "ar-0"
    assert first_target["remote_bootstrap_addr"] == "host:9000"
    assert first_target["num_transfer_tokens"] == 1232
    assert first_source["transfer_id"] != second_source["transfer_id"]
    assert first_replica == dit_pool.get_bound_replica_id("req-0") == 0
    assert second_replica == dit_pool.get_bound_replica_id("req-1") == 1
    assert dit_stages[0].calls == [first_target]
    assert dit_stages[1].calls == [second_target]
