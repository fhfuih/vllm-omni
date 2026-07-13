# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Midway prompt updates for chunked diffusion step execution.

``PromptUpdateMixin`` exposes
- a pipeline interface for the external to queue a prompt update,
- a pipeline internal method to apply prompt transitions at chunk boundaries.

``prompt_update_versions``: a helper for ``InputBatch`` cache invalidation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, TypedDict, cast

import torch

from vllm_omni.diffusion.worker.utils import DiffusionRequestState

DEFAULT_TRANSITION_DURATION_CHUNKS = 3

logger = logging.getLogger(__name__)


class PromptUpdateExtra(TypedDict, total=False):
    """A "protocol" for ``DiffusionRequestState.extra`` keys used by the prompt-update feature."""

    pending_prompt_update: _PendingPromptUpdate
    prompt_update_state: _PromptUpdateState
    prompt_update_version: int  # Bumped when prompt_embeds change; invalidates InputBatch cache.


def prompt_update_versions(states: Sequence[DiffusionRequestState]) -> tuple[int, ...]:
    """Return per-request prompt-update versions for batch cache comparison.

    To be used by ``InputBatch`` or other external to determine if the cached batch is still valid.
    """
    return tuple(int(cast(PromptUpdateExtra, state.extra).get("prompt_update_version", 0)) for state in states)


class PromptUpdateMixin:
    """Mixin for chunked streaming pipelines that support midway prompt updates.

    Implements the behavior expected by
    :class:`~vllm_omni.diffusion.models.interface.SupportsPromptUpdate`.
    All per-request transition state lives on ``DiffusionRequestState.extra``;
    this mixin is stateless and safe to use on a shared pipeline instance.
    """

    supports_prompt_update: ClassVar[bool] = True

    def prepare_prompt_update(
        self,
        state: DiffusionRequestState,
        prompt: str,
        transition_duration_chunks: int | None = None,
    ) -> None:
        """Encode and queue a prompt update to apply at the next chunk boundary.
        It does not actually apply the prompt update.

        This is an extended interface for the pipeline, to be called by the runner.
        """
        if not prompt:
            raise ValueError("prompt must be non-empty")
        duration = (
            DEFAULT_TRANSITION_DURATION_CHUNKS if transition_duration_chunks is None else transition_duration_chunks
        )
        if duration < 0:
            raise ValueError("transition_duration_chunks must be >= 0")

        target_prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=state.sampling.num_outputs_per_prompt,
            max_sequence_length=state.sampling.max_sequence_length,
            device=self.device,
            dtype=self.transformer.dtype,
        )
        extra = cast(PromptUpdateExtra, state.extra)
        extra["pending_prompt_update"] = {
            "prompt": prompt,
            "target_prompt_embeds": target_prompt_embeds,
            "transition_duration_chunks": duration,
        }

    def _apply_prompt_update_at_chunk_boundary(self, state: DiffusionRequestState) -> None:
        """Advance or start prompt interpolation before the next chunk."""
        extra = cast(PromptUpdateExtra, state.extra)
        update_state: _PromptUpdateState | None = extra.get("prompt_update_state")
        pending = extra.pop("pending_prompt_update", None)
        embeds_changed = False
        next_chunk_index = state.chunk_index

        # If current transition is not complete, advance it.
        # After completion, leave prompt_update_state in place (so a later pending
        # update can abort/overwrite), but do not keep bumping the version.
        if update_state is not None:
            in_transition = (
                update_state.transition_duration_chunks > 0
                and update_state.elapsed_transition_chunks < update_state.transition_duration_chunks
            )
            if in_transition:
                update_state.advance_transition()
                state.prompt_embeds = update_state.blended_prompt_embeds()
                embeds_changed = True
                if update_state.elapsed_transition_chunks >= update_state.transition_duration_chunks:
                    state.prompt_embeds = update_state.target_prompt_embeds
                    update_state.source_prompt_embeds = update_state.target_prompt_embeds
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
                return
            source = state.prompt_embeds.detach().clone()
            target = pending["target_prompt_embeds"]
            duration = int(pending["transition_duration_chunks"])
            prompt = str(pending.get("prompt", ""))
            update_state = _PromptUpdateState(
                source_prompt_embeds=source,
                target_prompt_embeds=target,
                transition_duration_chunks=duration,
                target_prompt=prompt,
            )
            extra["prompt_update_state"] = update_state
            if duration <= 0:
                state.prompt_embeds = target
                logger.debug(
                    "prompt_update sharp transition request_id=%s next_chunk_index=%d prompt=%.20s...",
                    state.request_id,
                    next_chunk_index,
                    prompt,
                )
            else:
                state.prompt_embeds = update_state.blended_prompt_embeds()
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


@dataclass
class _PromptUpdateState:
    """Per-request prompt interpolation stored in ``state.extra["prompt_update_state"]``."""

    source_prompt_embeds: torch.Tensor
    target_prompt_embeds: torch.Tensor
    transition_duration_chunks: int
    elapsed_transition_chunks: int = 0
    target_prompt: str | None = None

    def blended_prompt_embeds(self) -> torch.Tensor:
        alpha = self._current_alpha()
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

    def _current_alpha(self) -> float:
        if self.transition_duration_chunks <= 0:
            return 1.0
        return min(1.0, self.elapsed_transition_chunks / self.transition_duration_chunks)


class _PendingPromptUpdate(TypedDict):
    prompt: str
    target_prompt_embeds: torch.Tensor
    transition_duration_chunks: int
