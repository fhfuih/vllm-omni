# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-local prompt update state for diffusion step execution."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.worker.utils import DiffusionRequestState

PROMPT_UPDATE_VERSION_KEY = "prompt_update_version"
PROMPT_UPDATE_STATE_KEY = "prompt_update_state"


@dataclass
class PromptUpdatePayload:
    """Control-plane payload for a midway prompt update."""

    prompt: str
    transition_duration_chunks: int = 1

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        if self.transition_duration_chunks < 0:
            raise ValueError("transition_duration_chunks must be >= 0")


@dataclass
class PromptUpdateState:
    """Per-request prompt interpolation state stored in ``state.extra``."""

    source_prompt_embeds: torch.Tensor
    target_prompt_embeds: torch.Tensor
    transition_duration_chunks: int
    elapsed_transition_chunks: int = 0
    pending_target_prompt_embeds: torch.Tensor | None = None
    pending_transition_duration_chunks: int | None = None

    def current_alpha(self) -> float:
        if self.transition_duration_chunks <= 0:
            return 1.0
        return min(1.0, self.elapsed_transition_chunks / self.transition_duration_chunks)

    def blended_prompt_embeds(self) -> torch.Tensor:
        alpha = self.current_alpha()
        if alpha >= 1.0:
            return self.target_prompt_embeds
        return (1.0 - alpha) * self.source_prompt_embeds + alpha * self.target_prompt_embeds

    def advance_transition(self) -> None:
        if self.transition_duration_chunks <= 0:
            self.source_prompt_embeds = self.target_prompt_embeds
            return
        self.elapsed_transition_chunks += 1
        if self.elapsed_transition_chunks >= self.transition_duration_chunks:
            self.source_prompt_embeds = self.target_prompt_embeds
            self.elapsed_transition_chunks = self.transition_duration_chunks

    def queue_pending(
        self,
        target_prompt_embeds: torch.Tensor,
        transition_duration_chunks: int,
    ) -> None:
        self.pending_target_prompt_embeds = target_prompt_embeds
        self.pending_transition_duration_chunks = transition_duration_chunks

    def activate_pending_if_any(self) -> bool:
        if self.pending_target_prompt_embeds is None:
            return False
        current = self.blended_prompt_embeds()
        self.source_prompt_embeds = current
        self.target_prompt_embeds = self.pending_target_prompt_embeds
        self.transition_duration_chunks = int(self.pending_transition_duration_chunks or 0)
        self.elapsed_transition_chunks = 0
        self.pending_target_prompt_embeds = None
        self.pending_transition_duration_chunks = None
        return True


def get_prompt_update_state(state: DiffusionRequestState) -> PromptUpdateState | None:
    value = state.extra.get(PROMPT_UPDATE_STATE_KEY)
    return value if isinstance(value, PromptUpdateState) else None


def set_prompt_update_state(state: DiffusionRequestState, update_state: PromptUpdateState | None) -> None:
    if update_state is None:
        state.extra.pop(PROMPT_UPDATE_STATE_KEY, None)
    else:
        state.extra[PROMPT_UPDATE_STATE_KEY] = update_state


def bump_prompt_update_version(state: DiffusionRequestState) -> int:
    version = int(state.extra.get(PROMPT_UPDATE_VERSION_KEY, 0)) + 1
    state.extra[PROMPT_UPDATE_VERSION_KEY] = version
    return version


def prompt_update_versions(states: list[DiffusionRequestState]) -> tuple[int, ...]:
    return tuple(int(state.extra.get(PROMPT_UPDATE_VERSION_KEY, 0)) for state in states)


def payload_from_dict(data: dict[str, Any]) -> PromptUpdatePayload:
    if "negative_prompt" in data and data["negative_prompt"] is not None:
        raise ValueError("negative_prompt is not supported for prompt_update in this release")
    return PromptUpdatePayload(
        prompt=str(data["prompt"]),
        transition_duration_chunks=int(data.get("transition_duration_chunks", 1)),
    )
