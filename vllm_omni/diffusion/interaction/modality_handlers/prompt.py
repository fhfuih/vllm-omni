# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt-track interaction handler (midway prompt updates)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, ClassVar, TypedDict, cast, override

import torch

from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler
from vllm_omni.diffusion.interaction.types import (
    InteractionChunkMetadata,
    InteractionPayload,
)
from vllm_omni.diffusion.worker.utils import StepRequestState

DEFAULT_TRANSITION_CHUNKS = 3

logger = logging.getLogger(__name__)

# Pipelines return (prompt_embeds, ...) with optional negative embeds / masks.
EncodePromptFn = Callable[..., tuple[torch.Tensor, ...]]


class PromptUpdateExtra(TypedDict, total=False):
    """A protocol for ``StepRequestState.extra`` keys used by prompt updates."""

    pending_prompt_update: _PendingPromptUpdate
    prompt_update_state: _PromptUpdateState
    prompt_update_version: int
    prompt_update_lock: Lock


def prompt_update_versions(states: Sequence[StepRequestState]) -> tuple[int, ...]:
    """Return per-request prompt-update versions for batch cache comparison."""
    return tuple(int(cast(PromptUpdateExtra, state.extra).get("prompt_update_version", 0)) for state in states)


def _prompt_update_lock(extra: PromptUpdateExtra) -> Lock:
    return extra.setdefault("prompt_update_lock", Lock())


class PromptInteractionHandler(InteractionHandler):
    """Encode-and-queue prompt updates; lerp embeds at chunk boundaries.

    Mirrors the former ``PromptUpdateMixin`` behavior. Encoding happens at
    enqueue time via an injected ``encode_prompt`` callable.
    """

    modality: ClassVar[str] = "prompt"

    def __init__(
        self,
        *,
        encode_prompt: EncodePromptFn,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self._encode_prompt = encode_prompt
        self._device = device
        self._dtype = dtype

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> PromptInteractionHandler:
        """Build a handler from a diffusion pipeline that exposes ``encode_prompt``."""
        return cls(
            encode_prompt=pipeline.encode_prompt,
            device=pipeline.device,
            dtype=pipeline.transformer.dtype,
        )

    @override
    def enqueue(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        received_at: float,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        """Prompt updates are last-write-win and unbuffered at chunk boundary."""
        del received_at  # Prompt updates are chunk-based and ignore arrival time.
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be non-empty")
        if not event_id:
            raise ValueError("event_id must be non-empty")
        if state.prompt_embeds is None:
            raise ValueError(
                f"prompt_update is not allowed before initial generation has started (request_id={state.request_id!r})"
            )
        duration = DEFAULT_TRANSITION_CHUNKS if transition_chunks is None else transition_chunks
        if duration < 0:
            raise ValueError("transition_chunks must be >= 0")

        target_prompt_embeds, _ = self._encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=state.sampling.num_outputs_per_prompt,
            max_sequence_length=state.sampling.max_sequence_length,
            device=self._device,
            dtype=self._dtype,
        )
        extra = cast(PromptUpdateExtra, state.extra)
        with _prompt_update_lock(extra):
            extra["pending_prompt_update"] = {
                "event_id": event_id,
                "prompt": prompt,
                "target_prompt_embeds": target_prompt_embeds,
                "transition_chunks": duration,
            }

    @override
    def apply_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        chunk_index: int,
        num_frames: int,
        fps: float,
        boundary_at: float,
    ) -> InteractionChunkMetadata | None:
        """Advance or start prompt interpolation before the next chunk."""
        del chunk_index, num_frames, fps, boundary_at  # Prompt lerp is chunk-based and last-write-wins
        extra = cast(PromptUpdateExtra, state.extra)
        update_state: _PromptUpdateState | None = extra.get("prompt_update_state")
        with _prompt_update_lock(extra):
            pending = extra.pop("pending_prompt_update", None)
        embeds_changed = False
        next_chunk_index = state.chunk_index
        started_event_ids: list[str] = []
        active_event_ids: list[str] = []
        completed_event_ids: list[str] = []

        # If current transition is not complete, advance it.
        # After completion, leave prompt_update_state in place (so a later pending
        # update can abort/overwrite), but do not keep bumping the version.
        if update_state is not None:
            in_transition = (
                update_state.transition_chunks > 0
                and update_state.elapsed_transition_chunks < update_state.transition_chunks
            )
            if in_transition:
                update_state.advance_transition()
                state.prompt_embeds = update_state.blended_prompt_embeds()
                embeds_changed = True
                active_event_ids.append(update_state.event_id)
                if update_state.elapsed_transition_chunks >= update_state.transition_chunks:
                    state.prompt_embeds = update_state.target_prompt_embeds
                    update_state.source_prompt_embeds = update_state.target_prompt_embeds
                    completed_event_ids.append(update_state.event_id)
                    if update_state.target_prompt is not None:
                        logger.debug(
                            "prompt_update transition complete request_id=%s next_chunk_index=%d prompt=%.20s...",
                            state.request_id,
                            next_chunk_index,
                            update_state.target_prompt,
                        )

        # If a new prompt update is pending, start a new transition.
        if pending is not None:
            if state.prompt_embeds is None:
                raise RuntimeError(
                    "internal error: trying to apply a pending prompt update but "
                    f"current prompt_embeds is None (request_id={state.request_id!r})"
                )
            source = state.prompt_embeds.detach().clone()
            target = pending["target_prompt_embeds"]
            duration = int(pending["transition_chunks"])
            prompt = str(pending.get("prompt", ""))
            event_id = pending.get("event_id")
            update_state = _PromptUpdateState(
                source_prompt_embeds=source,
                target_prompt_embeds=target,
                transition_chunks=duration,
                event_id=event_id,
                target_prompt=prompt,
            )
            extra["prompt_update_state"] = update_state
            started_event_ids.append(event_id)
            active_event_ids.append(event_id)
            if duration <= 0:
                # A hard/immediate transition
                state.prompt_embeds = target
                completed_event_ids.append(event_id)
                logger.debug(
                    "prompt_update sharp transition request_id=%s next_chunk_index=%d prompt=%.20s...",
                    state.request_id,
                    next_chunk_index,
                    prompt,
                )
            else:
                # A transition that really takes some time
                update_state.advance_transition()
                state.prompt_embeds = update_state.blended_prompt_embeds()
                if update_state.elapsed_transition_chunks >= update_state.transition_chunks:
                    state.prompt_embeds = update_state.target_prompt_embeds
                    update_state.source_prompt_embeds = update_state.target_prompt_embeds
                    completed_event_ids.append(event_id)
                logger.debug(
                    "prompt_update transition start request_id=%s next_chunk_index=%d prompt=%.20s...",
                    state.request_id,
                    next_chunk_index,
                    prompt,
                )
            embeds_changed = True

        # Indicate that prompt embeddings have changed---clear input batch cache.
        if embeds_changed:
            new_version = int(extra.get("prompt_update_version", 0)) + 1
            extra["prompt_update_version"] = new_version

        metadata = InteractionChunkMetadata(
            started_event_ids=started_event_ids,
            active_event_ids=active_event_ids,
            completed_event_ids=completed_event_ids,
        )
        return metadata


@dataclass
class _PromptUpdateState:
    """Per-request prompt interpolation stored in ``state.extra["prompt_update_state"]``."""

    source_prompt_embeds: torch.Tensor
    target_prompt_embeds: torch.Tensor
    transition_chunks: int
    event_id: str
    elapsed_transition_chunks: int = 0
    target_prompt: str | None = None

    def blended_prompt_embeds(self) -> torch.Tensor:
        alpha = self._current_alpha()
        if alpha >= 1.0:
            return self.target_prompt_embeds
        return (1.0 - alpha) * self.source_prompt_embeds + alpha * self.target_prompt_embeds

    def advance_transition(self) -> None:
        if self.transition_chunks <= 0:
            self.source_prompt_embeds = self.target_prompt_embeds
            return
        self.elapsed_transition_chunks += 1
        if self.elapsed_transition_chunks >= self.transition_chunks:
            self.source_prompt_embeds = self.target_prompt_embeds
            self.elapsed_transition_chunks = self.transition_chunks

    def _current_alpha(self) -> float:
        if self.transition_chunks <= 0:
            return 1.0
        return min(1.0, self.elapsed_transition_chunks / self.transition_chunks)


class _PendingPromptUpdate(TypedDict):
    event_id: str
    prompt: str
    target_prompt_embeds: torch.Tensor
    transition_chunks: int
