# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `/v1/videos/stream` WebSocket video output streaming."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI, WebSocket
from pytest_mock import MockerFixture
from starlette.testclient import TestClient

from vllm_omni.entrypoints.openai.serving_video_output_stream import OmniStreamingVideoOutputHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fake_video_frames(num_frames: int = 2) -> list[np.ndarray]:
    return [np.full((4, 4, 3), fill_value=i, dtype=np.uint8) for i in range(num_frames)]


def _build_test_app(
    *,
    mocker: MockerFixture,
    streaming_chunks: list[tuple[bytes, bool]],
    mock_generate: Callable[..., AsyncGenerator[OmniRequestOutput, None]],
    final_chunk: bytes = b"",
) -> tuple[FastAPI, OmniStreamingVideoOutputHandler, MagicMock]:
    engine_client = mocker.MagicMock()
    engine_client.abort = mocker.AsyncMock()
    engine_client.default_sampling_params_list = [OmniDiffusionSamplingParams()]
    engine_client.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    engine_client.generate = mock_generate

    class FakeStreamingVideoEncoder:
        encode_calls: list[int] = []

        def encode(self, video):
            del video
            idx = len(self.encode_calls)
            self.encode_calls.append(idx)
            return streaming_chunks[idx][0]

        def close(self):
            return final_chunk

    def _make_encoder(*, output_format, fps, video_codec_options=None):
        del output_format, fps, video_codec_options
        return FakeStreamingVideoEncoder()

    encoder_factory = mocker.MagicMock(side_effect=_make_encoder)
    engine_client.streaming_encoder_factory = encoder_factory
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video_output_stream.create_streaming_video_encoder",
        encoder_factory,
    )
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video_output_stream.get_stage_type",
        return_value="diffusion",
    )
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video_output_stream.build_stage_sampling_params_list",
        return_value=[OmniDiffusionSamplingParams()],
    )
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video_output_stream.get_default_sampling_params_list",
        return_value=[OmniDiffusionSamplingParams()],
    )

    handler = OmniStreamingVideoOutputHandler(
        engine_client=engine_client,
        model_name="test-model",
        stage_configs=engine_client.stage_configs,
        idle_timeout=5.0,
        start_timeout=5.0,
    )
    app = FastAPI()

    @app.websocket("/v1/videos/stream")
    async def ws_endpoint(websocket: WebSocket):
        await handler.handle_session(websocket)

    return app, handler, engine_client


class TestStreamingVideoOutputWebSocket:
    """WebSocket protocol tests for streaming generated video output."""

    def test_streaming_session_emits_video_start_binary_chunks_and_done(self, mocker: MockerFixture):
        """A full session delivers video.start, binary chunks, then session.done."""

        async def mock_generate(*_args, **_kwargs):
            for frames, finished in [
                (_fake_video_frames(2), False),
                (_fake_video_frames(3), True),
            ]:
                yield OmniRequestOutput.from_diffusion(
                    request_id="req-ws",
                    images=[frames],
                    final_output_type="image",
                    finished=finished,
                )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", False), (b"mp4-chunk-1", True)],
            mock_generate=mock_generate,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/videos/stream") as ws:
                ws.send_json({"type": "session.start", "prompt": "A cat walking in the rain"})

                start = ws.receive_json()
                assert start["type"] == "video.start"
                assert start["format"] == "m4s"
                assert "format" not in start["config"]
                assert "request_id" in start

                assert ws.receive_bytes() == b"mp4-chunk-0"
                assert ws.receive_bytes() == b"mp4-chunk-1"

                done = ws.receive_json()
                assert done["type"] == "session.done"
                assert done["chunks"] == 2
                assert done["stopped"] is False

        engine_client.abort.assert_not_called()
        engine_client.streaming_encoder_factory.assert_called_once()
        assert engine_client.streaming_encoder_factory.call_args.kwargs["output_format"] == "m4s"

    def test_streaming_session_rejects_unknown_format(self, mocker: MockerFixture):
        async def mock_generate(*_args, **_kwargs):
            if False:
                yield None

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[],
            mock_generate=mock_generate,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/videos/stream") as ws:
                ws.send_json({"type": "session.start", "prompt": "hello", "format": "webm"})
                err = ws.receive_json()
                assert err["type"] == "error"
                assert "format" in err["message"]

        engine_client.abort.assert_not_called()

    def test_streaming_session_emits_final_encoder_delta_before_done(self, mocker: MockerFixture):
        """A non-empty encoder close delta is delivered as the final binary chunk."""

        async def mock_generate(*_args, **_kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", True)],
            mock_generate=mock_generate,
            final_chunk=b"mp4-trailer",
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/videos/stream") as ws:
                ws.send_json({"type": "session.start", "prompt": "A cat walking in the rain"})
                assert ws.receive_json()["type"] == "video.start"
                assert ws.receive_bytes() == b"mp4-chunk-0"
                assert ws.receive_bytes() == b"mp4-trailer"

                done = ws.receive_json()
                assert done["type"] == "session.done"
                assert done["chunks"] == 2
                assert done["stopped"] is False

        engine_client.abort.assert_not_called()

    def test_generation_error_output_returns_websocket_error(self, mocker: MockerFixture):
        """An erroneous OmniRequestOutput from generate is sent to the client as a WebSocket error."""
        error_message = "diffusion engine exploded"

        async def mock_generate(*_args, **_kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=False,
            )
            yield OmniRequestOutput.from_error("req-ws", error_message)

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", False)],
            mock_generate=mock_generate,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/videos/stream") as ws:
                ws.send_json({"type": "session.start", "prompt": "hello"})
                assert ws.receive_json()["type"] == "video.start"
                assert ws.receive_bytes() == b"mp4-chunk-0"
                err = ws.receive_json()
                assert err["type"] == "error"
                assert error_message in err["message"]

        engine_client.abort.assert_not_called()
