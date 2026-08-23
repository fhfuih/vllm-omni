# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm.config import KVTransferConfig
from vllm.logger import init_logger

from vllm_omni.config.omni_config import KVTransferBackend
from vllm_omni.engine.stage_pool import StagePool

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class KVTransferPlan:
    request_id: str
    source_stage_id: int
    target_stage_id: int
    attempt: int
    backend: KVTransferBackend


class KVTransferBackendManager:
    def __init__(self, stage_pools: list[StagePool]) -> None:
        self._stage_pools = stage_pools

    @staticmethod
    def _config(pool: StagePool, client: Any | None = None) -> KVTransferConfig | None:
        client = client or pool.stage_client
        for config in (
            getattr(client, "od_config", None),
            getattr(client, "vllm_config", None),
            pool.stage_vllm_config,
        ):
            kv_transfer_config = getattr(config, "kv_transfer_config", None)
            if kv_transfer_config is not None:
                return kv_transfer_config
        return None

    @classmethod
    def _backend(cls, pool: StagePool, client: Any | None = None) -> KVTransferBackend:
        client = client or pool.stage_client
        od_config = getattr(client, "od_config", None)
        vllm_config = getattr(client, "vllm_config", None) or pool.stage_vllm_config
        omni_kv_config = getattr(od_config, "omni_kv_config", None)
        if omni_kv_config is None:
            omni_kv_config = getattr(client, "_omni_kv_config", None)
        if omni_kv_config is None:
            omni_kv_config = getattr(getattr(vllm_config, "model_config", None), "omni_kv_config", None)
        return KVTransferBackend.resolve(
            kv_transfer_config=cls._config(pool, client),
            omni_kv_config=omni_kv_config,
        )

    def create_plan(
        self,
        request_id: str,
        *,
        source_stage_id: int,
        final_stage_id: int,
        attempt: int = 0,
    ) -> KVTransferPlan | None:
        targets = [
            stage_id
            for stage_id in range(source_stage_id + 1, final_stage_id + 1)
            if self._stage_pools[stage_id].stage_type == "diffusion"
        ]
        if not targets:
            return None
        enabled_targets = [
            stage_id
            for stage_id in targets
            if self._backend(self._stage_pools[stage_id]) is not KVTransferBackend.DISABLED
        ]
        if not enabled_targets:
            return None
        if len(enabled_targets) != 1:
            raise ValueError(
                "KV transfer supports exactly one logical Diffusion target stage; "
                f"configured target stages: {enabled_targets}"
            )

        target_stage_id = enabled_targets[0]
        backend = self._backend(self._stage_pools[target_stage_id])
        source_backend = self._backend(self._stage_pools[source_stage_id])
        if source_backend is not backend:
            raise ValueError(
                "Source and Diffusion stages must use the same KV transfer backend; "
                f"got {source_backend.value} -> {backend.value}"
            )
        return KVTransferPlan(
            request_id=request_id,
            source_stage_id=source_stage_id,
            target_stage_id=target_stage_id,
            attempt=attempt,
            backend=backend,
        )

    @staticmethod
    def prepare_source(
        plan: KVTransferPlan,
        source_params: Any,
        *,
        source_request: Any,
    ) -> None:
        if plan.backend is not KVTransferBackend.KV_CONNECTOR:
            return

        submitted_params = getattr(source_request, "sampling_params", None)
        if submitted_params is None:
            raise RuntimeError("KV Connector source request has no sampling parameters")
        kv_transfer_params = KVTransferBackendManager._build_source_params(plan)
        for params in (source_params, submitted_params):
            extra_args = params.extra_args
            if extra_args is None:
                extra_args = {}
                params.extra_args = extra_args
            extra_args["kv_transfer_params"] = dict(kv_transfer_params)

    def build_target_submit_kwargs(
        self,
        plan: KVTransferPlan | None,
        *,
        source_params: Any | None = None,
        source_output: Any | None = None,
    ) -> dict[str, Any]:
        if plan is None or plan.backend is KVTransferBackend.DISABLED:
            return {}
        source_pool = self._stage_pools[plan.source_stage_id]
        source_client = source_pool.get_bound_client(plan.request_id)
        if source_client is None:
            raise RuntimeError("KV transfer source request is not bound to a replica")

        if plan.backend is KVTransferBackend.KV_CONNECTOR:
            return {
                "kv_transfer_params": self._build_target_params(
                    self._config(source_pool, source_client),
                    (getattr(source_params, "extra_args", None) or {}).get("kv_transfer_params"),
                    getattr(source_output, "kv_transfer_params", None),
                )
            }
        if plan.backend is KVTransferBackend.OMNI_KV_TRANSFER:
            sender_info = self._sender_info(source_client, plan.source_stage_id)
            return {"kv_sender_info": {plan.source_stage_id: sender_info} if sender_info else None}
        raise AssertionError(f"Unhandled KV transfer backend: {plan.backend}")

    @staticmethod
    def _build_source_params(plan: KVTransferPlan) -> dict[str, object]:
        if not plan.request_id:
            raise ValueError("request_id must be non-empty")
        if min(plan.source_stage_id, plan.target_stage_id, plan.attempt) < 0:
            raise ValueError("source_stage_id, target_stage_id, and attempt must be non-negative")
        return {
            "transfer_id": (f"xfer-{plan.request_id}-s{plan.source_stage_id}-t{plan.target_stage_id}-a{plan.attempt}"),
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }

    @classmethod
    def _build_target_params(
        cls,
        source_config: KVTransferConfig | None,
        source_params: Mapping[str, Any] | None,
        connector_output_params: Mapping[str, Any] | None,
    ) -> dict[str, object]:
        if source_config is None:
            raise RuntimeError("KV Connector source replica has no KVTransferConfig")
        if source_config.kv_connector != "MooncakeConnector" or source_config.kv_role != "kv_producer":
            raise RuntimeError("KV Connector source replica must be a Mooncake producer")

        params = dict(source_params or {})
        params.update(connector_output_params or {})
        transfer_id = params.get("transfer_id")
        remote_engine_id = source_config.engine_id
        remote_bootstrap_addr = cls._bootstrap_addr(source_config)
        cls._require_nonempty("transfer_id", transfer_id)
        cls._require_nonempty("remote_engine_id", remote_engine_id)
        cls._require_nonempty("remote_bootstrap_addr", remote_bootstrap_addr)
        target_params: dict[str, object] = {
            "transfer_id": transfer_id,
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_engine_id": remote_engine_id,
            "remote_bootstrap_addr": remote_bootstrap_addr,
        }
        if "num_transfer_tokens" in params:
            target_params["num_transfer_tokens"] = params["num_transfer_tokens"]
        return target_params

    @staticmethod
    def _bootstrap_addr(config: KVTransferConfig) -> str | None:
        extra = config.kv_connector_extra_config or {}
        for key in ("bootstrap_addr", "prefill_bootstrap_addr", "mooncake_master"):
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if (
            isinstance(config.kv_ip, str)
            and config.kv_ip.strip()
            and type(config.kv_port) is int
            and config.kv_port > 0
        ):
            return f"http://{config.kv_ip}:{config.kv_port}"
        return None

    @staticmethod
    def _require_nonempty(name: str, value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Mooncake {name} must be a non-empty string")

    @staticmethod
    def _sender_info(source_client: Any, source_stage_id: int) -> dict[str, Any] | None:
        get_sender_info = getattr(source_client, "get_kv_sender_info", None)
        if not callable(get_sender_info):
            return None
        sender_info = get_sender_info()
        if not sender_info:
            logger.warning("Stage-%s has no KV sender info available", source_stage_id)
            return None
        return sender_info
