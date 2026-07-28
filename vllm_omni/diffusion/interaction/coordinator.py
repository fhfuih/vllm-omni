# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interaction coordinator to be co-owned by a pipeline and a runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_omni.diffusion.interaction.modality_handlers.prompt import PromptInteractionHandler
from vllm_omni.diffusion.interaction.registry import STRUCTURED_HANDLER_REGISTRY
from vllm_omni.diffusion.interaction.types import (
    InteractionEventArrival,
    InteractionPayload,
)
from vllm_omni.diffusion.models.interface import supports_interaction_apply
from vllm_omni.diffusion.worker.utils import StepRequestState

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler


class InteractionCoordinator:
    """Handles mid-way interaction inputs for one loaded pipeline.

    Request-local session state lives on ``StepRequestState.extra``; this object
    is request-agnostic.
    """

    def __init__(
        self,
        handlers: dict[str, InteractionHandler],
        *,
        model_class_name: str | None = None,
    ) -> None:
        self._handlers = handlers
        self.model_class_name = model_class_name

    @classmethod
    def build(cls, pipeline: object, od_config: OmniDiffusionConfig) -> InteractionCoordinator:
        model_class_name = od_config.model_class_name
        handlers: dict[str, InteractionHandler] = {}

        if supports_interaction_apply(pipeline):
            handlers["prompt"] = PromptInteractionHandler.from_pipeline(pipeline)

        for modality, handler_cls in STRUCTURED_HANDLER_REGISTRY.get(model_class_name or "", {}).items():
            handlers[modality] = handler_cls()

        return cls(handlers, model_class_name=model_class_name)

    def has_modality(self, modality: str) -> bool:
        return modality in self._handlers

    def get_handler(self, modality: str) -> InteractionHandler:
        handler = self._handlers.get(modality)
        if handler is None:
            raise ValueError(
                f"interaction modality {modality!r} is not supported by pipeline {self.model_class_name!r}"
            )
        return handler

    def handlers_in_apply_order(self) -> list[InteractionHandler]:
        """Return handlers in a stable apply order (prompt first when present)."""
        ordered: list[InteractionHandler] = []
        if "prompt" in self._handlers:
            ordered.append(self._handlers["prompt"])
        for modality, handler in self._handlers.items():
            if modality == "prompt":
                continue
            ordered.append(handler)
        return ordered

    def enqueue(
        self,
        state: StepRequestState,
        *,
        modality: str,
        event_arrival: InteractionEventArrival,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        handler = self.get_handler(modality)
        handler.enqueue(
            state,
            event_arrival=event_arrival,
            payload=payload,
            transition_chunks=transition_chunks,
        )
