# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary weightless realtime-video pipeline for local integration testing.

Do not ship: remove this package and its registry entry before merge.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.interaction.mixin import InteractionMixin
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

# Frames emitted per streaming chunk (last chunk may be shorter).
FRAMES_PER_CHUNK = 8
# One scheduler step per chunk; sleep covers the full chunk playback duration.
STEPS_PER_CHUNK = 1
DEFAULT_FPS = 16
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 384


class _DummyTransformer(nn.Module):
    """Parameterless module exposing the device and dtype expected by handlers."""

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("_device_marker", torch.empty(0, dtype=dtype), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_marker.device

    @property
    def dtype(self) -> torch.dtype:
        return self._device_marker.dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class DummyRealtimeVideoPipeline(nn.Module, InteractionMixin):
    """Temporary stub: synthetic video chunks, no weights. Remove before merge.

    Uses regular sampling params ``height``, ``width``, ``fps``, and ``num_frames``.
    Chunk size is ``FRAMES_PER_CHUNK``; per-step sleep matches chunk duration at ``fps``.
    """

    supports_step_execution: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        # Keep the default loader path while declaring that there are no
        # checkpoint components to discover or download.
        self.weights_sources: tuple[Any, ...] = ()
        self.transformer = _DummyTransformer(dtype=getattr(od_config, "dtype", torch.float32))

    @property
    def device(self) -> torch.device:
        return self.transformer.device

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Accept the standard loader contract without reading weight files."""
        del weights
        return set()

    @staticmethod
    def _extract_prompt_text(prompt: Any) -> str:
        if isinstance(prompt, str) and prompt:
            return prompt
        if isinstance(prompt, Mapping):
            value = prompt.get("prompt")
            if isinstance(value, str) and value:
                return value
        raise ValueError("DummyRealtimeVideoPipeline requires a non-empty text prompt.")

    @staticmethod
    def _resolve_fps(state: StepRequestState) -> float:
        return float(state.sampling.fps or DEFAULT_FPS)

    @staticmethod
    def _chunk_frames(state: StepRequestState) -> int:
        total_frames = max(int(state.sampling.num_frames or 1), 1)
        first_frame = state.chunk_index * FRAMES_PER_CHUNK
        return min(FRAMES_PER_CHUNK, max(total_frames - first_frame, 0))

    @staticmethod
    def _chunk_delay_seconds(*, num_frames: int, fps: float) -> float:
        return float(num_frames) / max(fps, 1e-6)

    def encode_prompt(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        do_classifier_free_guidance: bool = False,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode text as a deterministic embedding for prompt-interaction plumbing."""
        del max_sequence_length, kwargs
        target_device = self.device if device is None else torch.device(device)
        target_dtype = self.transformer.dtype if dtype is None else dtype

        def _encode(text: str) -> torch.Tensor:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            color = torch.tensor(
                [digest[0], digest[1], digest[2]],
                device=target_device,
                dtype=torch.float32,
            )
            color = (color / 255.0).to(dtype=target_dtype)
            return color.view(1, 1, 3).repeat(num_videos_per_prompt, 1, 1)

        prompt_embeds = _encode(prompt)
        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt_embeds = _encode(negative_prompt or "")
        return prompt_embeds, negative_prompt_embeds

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        """Initialize prompt, latent, and chunk state from regular sampling params."""
        del kwargs
        sampling = state.sampling
        if sampling.num_outputs_per_prompt != 1:
            raise ValueError("Dummy realtime video streaming supports exactly one output per prompt.")

        # Normalize omitted request fields onto the shared sampling object.
        sampling.num_frames = max(int(sampling.num_frames or 1), 1)
        sampling.fps = int(sampling.fps or DEFAULT_FPS)
        sampling.width = max(int(sampling.width or DEFAULT_WIDTH), 1)
        sampling.height = max(int(sampling.height or DEFAULT_HEIGHT), 1)

        prompt = self._extract_prompt_text(state.prompt)
        state.prompt_embeds, state.negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=sampling.max_sequence_length,
            device=self.device,
            dtype=self.transformer.dtype,
        )

        window_num_frames = min(FRAMES_PER_CHUNK, sampling.num_frames)
        state.latents = torch.zeros((1, 1), device=self.device, dtype=self.transformer.dtype)
        state.timesteps = torch.arange(STEPS_PER_CHUNK - 1, -1, -1, device=self.device)
        state.step_index = 0
        state.step_in_chunk = 0
        state.chunk_index = 0
        state.chunk_num_steps = STEPS_PER_CHUNK
        state.total_chunks = math.ceil(sampling.num_frames / FRAMES_PER_CHUNK)
        # InteractionMixin defaults read chunk media length from state.extra.
        state.extra["window_num_frames"] = window_num_frames
        return state

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Sleep for one chunk of playback time, then return a synthetic prediction."""
        del kwargs
        if not states:
            raise ValueError("Dummy denoise_step requires at least one request state.")
        delay = max(
            self._chunk_delay_seconds(
                num_frames=self._chunk_frames(state),
                fps=self._resolve_fps(state),
            )
            for state in states
        )
        if delay:
            time.sleep(delay)
        return torch.zeros_like(input_batch.latents)

    def step_scheduler(
        self,
        state: StepRequestState,
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Advance the normal step/chunk state without numerical diffusion."""
        del noise_pred, kwargs
        state.step_in_chunk += 1
        state.step_index = state.step_in_chunk

    @staticmethod
    def _format_event_lines(interaction_metadata: Mapping[str, Any]) -> list[str]:
        return [
            f"started: {', '.join(interaction_metadata.get('started_event_ids', [])) or '-'}",
            f"active: {', '.join(interaction_metadata.get('active_event_ids', [])) or '-'}",
            f"completed: {', '.join(interaction_metadata.get('completed_event_ids', [])) or '-'}",
        ]

    @staticmethod
    def _render_status_frame(
        *,
        height: int,
        width: int,
        lines: Sequence[str],
    ) -> np.ndarray:
        image = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        y = max(height // 20, 8)
        left = max(width // 40, 8)
        line_gap = max(height // 16, 14)
        for line in lines:
            draw.text((left, y), line, fill=(0, 0, 0), font=font)
            y += line_gap
        return np.asarray(image, dtype=np.uint8)

    def _render_frames(
        self,
        state: StepRequestState,
        *,
        num_frames: int,
        interaction_metadata: Mapping[str, Any],
    ) -> np.ndarray:
        sampling = state.sampling
        frame = self._render_status_frame(
            height=int(sampling.height or DEFAULT_HEIGHT),
            width=int(sampling.width or DEFAULT_WIDTH),
            lines=self._format_event_lines(interaction_metadata),
        )
        return np.repeat(frame[None, ...], num_frames, axis=0)

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        """Render one video chunk and advance interaction state at its boundary."""
        del kwargs
        completed_chunk_index = state.chunk_index
        num_frames = self._chunk_frames(state)
        fps = self._resolve_fps(state)
        interaction_metadata = state.extra.pop("interaction_chunk_metadata", {})
        frames = self._render_frames(
            state,
            num_frames=num_frames,
            interaction_metadata=interaction_metadata,
        )

        state.chunk_index += 1
        finished = state.request_denoise_completed
        if not finished:
            next_num_frames = self._chunk_frames(state)
            state.extra["window_num_frames"] = next_num_frames
            self.apply_interaction_at_chunk_boundary(
                state,
                chunk_index=state.chunk_index,
                num_frames=next_num_frames,
                fps=fps,
            )
            state.step_index = 0
            state.step_in_chunk = 0

        return DiffusionOutput(
            output={
                "payload": {"video": frames},
                "metadata": {"video": {"fps": fps}},
            },
            finished=finished,
            chunk_index=completed_chunk_index,
            total_chunks=state.total_chunks,
            started_event_ids=list(interaction_metadata.get("started_event_ids", [])),
            active_event_ids=list(interaction_metadata.get("active_event_ids", [])),
            completed_event_ids=list(interaction_metadata.get("completed_event_ids", [])),
        )
