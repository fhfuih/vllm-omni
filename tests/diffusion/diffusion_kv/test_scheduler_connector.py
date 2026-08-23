# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from unittest.mock import Mock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

from tests.diffusion.diffusion_kv.helper import (
    FAKE_TRANSFER_PARAMS,
    ConcreteScheduler,
    initialize_paged_scheduler,
    make_omni_diffusion_request,
)
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _fake_connector(*, ext_tokens: int | None = 8, load_async: bool = True) -> Mock:
    connector = Mock(spec=KVConnectorBase_V1)
    connector.get_num_new_matched_tokens.return_value = (ext_tokens, load_async)
    connector.build_connector_meta.return_value = object()
    return connector


def test_no_params_skips_connector_metadata() -> None:
    scheduler = ConcreteScheduler()
    initialize_paged_scheduler(scheduler)
    connector = _fake_connector(ext_tokens=0, load_async=False)
    scheduler.connector = connector
    request = make_omni_diffusion_request("req-0")

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.kv_connector_metadata is None
    assert scheduler.get_request_state("req-0").status is DiffusionRequestStatus.RUNNING
    connector.build_connector_meta.assert_not_called()
    assert connector.get_num_new_matched_tokens.call_count == 1
    connector.update_state_after_alloc.assert_called_once()


def test_cfg_sequences_each_call_connector_and_build_meta_once() -> None:
    scheduler = ConcreteScheduler()
    initialize_paged_scheduler(scheduler)
    connector = _fake_connector(ext_tokens=8, load_async=True)
    expected_meta = connector.build_connector_meta.return_value
    scheduler.connector = connector
    request = make_omni_diffusion_request(
        "req-cfg",
        num_sequences=2,
        kv_transfer_params=dict(FAKE_TRANSFER_PARAMS),
    )

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.kv_connector_metadata is expected_meta
    assert scheduler.get_request_state("req-cfg").status is DiffusionRequestStatus.RUNNING
    assert connector.get_num_new_matched_tokens.call_count == 2
    assert connector.update_state_after_alloc.call_count == 2
    connector.build_connector_meta.assert_called_once_with(output)
    seq_ids = [call.args[0].request_id for call in connector.get_num_new_matched_tokens.call_args_list]
    assert seq_ids == ["req-cfg/diffusion-kv/0", "req-cfg/diffusion-kv/1"]
    for call in connector.get_num_new_matched_tokens.call_args_list:
        assert call.args[1] == 0


def test_source_token_count_limits_connector_prefix() -> None:
    scheduler = ConcreteScheduler()
    initialize_paged_scheduler(scheduler)
    public_params = {**FAKE_TRANSFER_PARAMS, "num_transfer_tokens": 7}
    request = make_omni_diffusion_request(
        "req-cfg",
        num_sequences=2,
        seq_len=20,
        prefix_len=12,
        kv_transfer_params=public_params,
    )

    scheduler.add_request(request)

    sequences = scheduler.get_request_state("req-cfg").diffusion_kv_requests
    assert [len(sequence.prompt_token_ids) for sequence in sequences] == [7, 7]


def test_ext_tokens_none_frees_and_stays_waiting() -> None:
    scheduler = ConcreteScheduler()
    initialize_paged_scheduler(scheduler)
    connector = _fake_connector(ext_tokens=None, load_async=False)
    scheduler.connector = connector
    request = make_omni_diffusion_request("req-0", kv_transfer_params=dict(FAKE_TRANSFER_PARAMS))

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.scheduled_new_reqs == []
    assert output.kv_connector_metadata is None
    assert scheduler.get_request_state("req-0").status is DiffusionRequestStatus.WAITING
    assert not scheduler.kv_cache_manager.has_request("req-0")
    connector.build_connector_meta.assert_not_called()
