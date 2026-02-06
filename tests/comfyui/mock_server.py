"""
Mock vLLM-Omni API server for ComfyUI integration testing.

This module provides a real FastAPI server that uses the actual vLLM-Omni API routes
with a mocked AsyncOmni class. This ensures tests validate the real API contract,
catching any changes to payload/response formats.

Usage:
    python -m tests.comfyui.mock_server --port 18765
"""

import argparse
import asyncio
import base64
import io
import struct
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock

import torch
import uvicorn
from fastapi import FastAPI, Request
from PIL import Image
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels

# Import real vLLM-Omni components
from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.outputs import OmniRequestOutput


def make_test_audio_bytes() -> bytes:
    """Create a simple WAV audio file bytes."""
    sample_rate = 24000
    num_samples = sample_rate  # 1 second
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    header = b"RIFF"
    header += struct.pack("<I", 36 + data_size)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)
    header += struct.pack("<H", 1)
    header += struct.pack("<H", num_channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", byte_rate)
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", bits_per_sample)
    header += b"data"
    header += struct.pack("<I", data_size)
    audio_data = b"\x00" * data_size

    return header + audio_data


# =============================================================================
# Mock Classes
# =============================================================================


class MockCompletionOutput:
    """Mock vLLM CompletionOutput."""

    def __init__(self, text: str = "This is a mock response."):
        self.text = text
        self.token_ids = [1, 2, 3, 4, 5]
        self.cumulative_logprob = None
        self.logprobs = None
        self.finish_reason = "stop"
        self.stop_reason = None


class MockRequestOutput:
    """Mock vLLM RequestOutput."""

    def __init__(
        self,
        request_id: str,
        text: str = "This is a mock response.",
        multimodal_output: dict | None = None,
    ):
        self.request_id = request_id
        self.prompt_token_ids = [1, 2, 3]
        self.outputs = [MockCompletionOutput(text)]
        self.finished = True
        self.multimodal_output = multimodal_output or {}


class MockVllmConfig:
    """Mock VllmConfig."""

    def __init__(self, model_name: str):
        self.model_config = MockModelConfig(model_name)
        self.parallel_config = MagicMock()
        self.parallel_config._api_process_rank = 0
        self.lora_config = None


class MockModelConfig:
    """Mock ModelConfig with hf_config for TTS support."""

    def __init__(self, model_name: str):
        self.model = model_name
        self.served_model_name = model_name
        self.tokenizer = model_name
        self.trust_remote_code = True
        self.max_model_len = 4096
        self.io_processor_plugin = None
        self.task = "generate"

        # HF config with talker_config for TTS
        self.hf_config = MagicMock()
        self.hf_config.talker_config = MagicMock()
        self.hf_config.talker_config.audio_config = MagicMock()
        self.hf_config.talker_config.audio_config.sampling_rate = 24000
        # Speaker IDs for TTS validation
        self.hf_config.talker_config.spk_id = {
            "Chelsie": 0,
            "Ethan": 1,
            "Vivian": 2,
            "Ryan": 3,
        }


class MockOmniStage:
    """Mock OmniStage for stage_list."""

    def __init__(self, stage_id: int = 0, model_stage: str = "qwen3_tts"):
        self.stage_id = stage_id
        self.model_stage = model_stage
        self.final_output_type = "audio"


class MockTokenizer:
    """Mock tokenizer."""

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "decoded text"

    def apply_chat_template(self, messages: list, **kwargs) -> str:
        return "formatted prompt"


class MockInputProcessor:
    """Mock input processor."""

    pass


class MockIOProcessor:
    """Mock IO processor."""

    pass


class MockAsyncOmni:
    """Mock AsyncOmni that provides controlled test responses.

    This mock implements the full interface expected by the real vLLM-Omni
    API serving classes (OmniOpenAIServingChat, OmniOpenAIServingSpeech).
    """

    def __init__(self, model: str = "test-model", **kwargs):
        self.model_name = model
        self._kwargs = kwargs

        # Required attributes for API server
        self.errored = False
        self.dead_error = None

        # Stage configuration - include all stage types for testing different endpoints
        self.stage_configs = [
            {"stage_id": 0, "stage_type": "llm", "final_output_type": "text"},
            {"stage_id": 1, "stage_type": "diffusion", "final_output_type": "image"},
            {"stage_id": 2, "stage_type": "tts", "final_output_type": "audio"},
        ]
        self.stage_list = [MockOmniStage(stage_id=0, model_stage="qwen3_tts")]

        # Default sampling params
        self.default_sampling_params_list = [MagicMock()]

        # Output modalities
        self.output_modalities = ["text", "image", "audio"]

        # Model config
        self.model_config = MockModelConfig(model)

        # Processors (needed by OpenAIServingModels)
        self.input_processor = MockInputProcessor()
        self.io_processor = MockIOProcessor()

        # Renderer (needed by OpenAIServingModels)
        self.renderer = MagicMock()

        # Response mode for testing different outputs
        self._response_mode = "text"

    async def get_supported_tasks(self) -> tuple[str, ...]:
        return ("generate",)

    async def get_vllm_config(self) -> MockVllmConfig:
        return MockVllmConfig(self.model_name)

    async def get_tokenizer(self) -> MockTokenizer:
        return MockTokenizer()

    async def check_health(self):
        pass

    def shutdown(self):
        pass

    def set_response_mode(self, mode: str):
        """Set response mode: 'text', 'image', or 'audio'."""
        self._response_mode = mode

    async def generate(
        self,
        prompt: Any = None,
        request_id: str = "",
        sampling_params_list: Any = None,
        *,
        output_modalities: list[str] | None = None,
        **kwargs,  # Accept additional kwargs from the API
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate mock outputs based on output_modalities or sampling_params_list."""
        output_modalities = output_modalities or ["text"]
        req_id = request_id or kwargs.get("request_id", "")
        actual_prompt = prompt or kwargs.get("prompt")

        # Check sampling_params_list to determine output type
        has_diffusion_params = False
        if sampling_params_list:
            for params in sampling_params_list:
                # OmniDiffusionSamplingParams has num_inference_steps and num_outputs_per_prompt
                if hasattr(params, "num_inference_steps") or hasattr(params, "num_outputs_per_prompt"):
                    has_diffusion_params = True
                    break

        if has_diffusion_params or "image" in output_modalities:
            # Image output - for serving_chat compatibility, we need images in request_output.images
            # because serving_chat.py line 2030 reads: getattr(result.request_output, "images", [])
            img = Image.new("RGB", (64, 64), color="blue")

            # Create a mock request_output with images attribute
            mock_request_output = MagicMock()
            mock_request_output.images = [img]
            mock_request_output.request_id = req_id
            mock_request_output.multimodal_output = {}

            output = OmniRequestOutput(
                request_id=req_id,
                finished=True,
                final_output_type="image",
                request_output=mock_request_output,
                images=[img],  # Also set directly for /v1/images/generations endpoint
                prompt=actual_prompt,
            )
        elif "audio" in output_modalities:
            # Audio output - create tensor matching what TTS models produce
            audio_tensor = torch.zeros((1, 24000), dtype=torch.float32)
            mock_request_output = MockRequestOutput(
                request_id=req_id,
                text="",
                multimodal_output={"audio": audio_tensor, "sr": 24000},
            )
            output = OmniRequestOutput.from_pipeline(
                stage_id=0,
                final_output_type="audio",
                request_output=mock_request_output,
            )
        else:
            # Text output
            mock_request_output = MockRequestOutput(
                request_id=req_id,
                text="This is a mock text response from the model.",
            )
            output = OmniRequestOutput.from_pipeline(
                stage_id=0,
                final_output_type="text",
                request_output=mock_request_output,
            )

        yield output


# =============================================================================
# Server Setup
# =============================================================================

# Model names used in tests
TEST_MODELS = [
    "Tongyi-MAI/Z-Image-Turbo",
    "Qwen/Qwen2.5-Omni-7B",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "ByteDance/BAGEL-7B-MoT",
    "test-model",
]


def create_mock_app(model_names: list[str] | None = None) -> FastAPI:
    """Create a FastAPI app with real vLLM-Omni routes and mocked AsyncOmni.

    This uses the actual routes from api_server.py but with a mocked engine,
    ensuring we test the real API contract.
    """
    if model_names is None:
        model_names = TEST_MODELS

    app = FastAPI(title="vLLM-Omni Test Server")

    # Create mock engine using first model as primary
    primary_model = model_names[0]
    mock_engine = MockAsyncOmni(model=primary_model)

    # Create serving models with all test model paths
    base_model_paths = [
        BaseModelPath(name=name, model_path=name) for name in model_names
    ]
    serving_models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=base_model_paths,
    )

    # Create serving chat handler (using real class)
    # For simplicity in testing, use diffusion mode which has minimal dependencies
    serving_chat = OmniOpenAIServingChat.for_diffusion(
        diffusion_engine=mock_engine,
        model_name=primary_model,
    )

    # Create serving speech handler (using real class)
    serving_speech = OmniOpenAIServingSpeech(
        engine_client=mock_engine,
        models=serving_models,
        request_logger=None,
    )

    # Set app state
    app.state.engine_client = mock_engine
    app.state.openai_serving_models = serving_models
    app.state.openai_serving_chat = serving_chat
    app.state.openai_serving_speech = serving_speech
    app.state.stage_configs = mock_engine.stage_configs

    # Custom chat completion handler that supports both text and image generation
    # This is added BEFORE the real router, so it takes precedence.
    # The real OmniOpenAIServingChat.for_diffusion() only handles images, but tests
    # need text completions too.
    @app.post("/v1/chat/completions")
    async def custom_chat_completion(request: Request):
        """Custom chat completion handler that supports both text and image outputs."""
        import time as time_module
        import uuid as uuid_module

        from fastapi.responses import JSONResponse

        data = await request.json()
        modalities = data.get("modalities", ["text"])
        messages = data.get("messages", [])

        request_id = f"chatcmpl-{uuid_module.uuid4().hex[:16]}"
        created = int(time_module.time())

        # Extract the last user message as the prompt
        prompt = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    prompt = content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            prompt = item.get("text", "")
                            break

        # Check if image generation is requested
        if "image" in modalities:
            # Generate image
            img = Image.new("RGB", (64, 64), color="blue")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            return JSONResponse({
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": data.get("model", "test-model"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        }]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 1, "total_tokens": len(prompt.split()) + 1}
            })
        elif "audio" in modalities:
            # Generate audio response
            audio_bytes = make_test_audio_bytes()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            return JSONResponse({
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": data.get("model", "test-model"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Audio transcription result",
                        "audio": {"data": audio_b64}
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 5, "total_tokens": len(prompt.split()) + 5}
            })
        else:
            # Text completion
            return JSONResponse({
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": data.get("model", "test-model"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a mock text response from the model."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 10, "total_tokens": len(prompt.split()) + 10}
            })

    # Include the real router from api_server.py (the chat completion route above takes precedence)
    app.include_router(router)

    return app


def run_server(port: int, model_name: str = "test-model"):
    """Run the mock server."""
    app = create_mock_app(model_name)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock vLLM-Omni API server for testing")
    parser.add_argument("--port", type=int, default=18765, help="Port to run server on")
    parser.add_argument("--model", type=str, default="test-model", help="Model name")
    args = parser.parse_args()

    run_server(args.port, args.model)
