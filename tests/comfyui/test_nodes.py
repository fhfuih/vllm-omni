"""
Tests for ComfyUI-vLLM-Omni node execution functions.

This module tests that nodes defined in comfyui_vllm_omni/nodes.py can produce correct outputs
when receiving particular inputs. The goal is to ensure the plugin plays well with
vLLM-Omni's API definitions and prevent unexpected API changes.

The tests use the real vLLM-Omni API server layer (FastAPI) running in a background thread,
with the AsyncOmni class mocked to provide controlled test responses. This ensures we test
the actual HTTP contract that the ComfyUI plugin depends on.
"""

import asyncio
import base64
import io
import socket
import struct
import threading
import time

import pytest
import torch
import uvicorn

# Import node classes - conftest.py sets up the mock modules
from comfyui_vllm_omni.nodes import (
    VLLMOmniARSampling,
    VLLMOmniComprehension,
    VLLMOmniDiffusionSampling,
    VLLMOmniGenerateImage,
    VLLMOmniQwenTTSParams,
    VLLMOmniSamplingParamsList,
    VLLMOmniTTS,
    VLLMOmniVoiceClone,
)
from PIL import Image

# Import the real API server factory from mock_server
from tests.comfyui.mock_server import create_mock_app

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def make_test_image_base64(width: int = 64, height: int = 64) -> str:
    """Create a base64-encoded PNG image for mock responses."""
    img = Image.new("RGB", (width, height), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_test_audio_bytes() -> bytes:
    """Create a simple WAV audio file bytes for mock responses."""
    sample_rate = 24000
    num_samples = sample_rate  # 1 second
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    # WAV header
    header = b"RIFF"
    header += struct.pack("<I", 36 + data_size)  # File size - 8
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)  # Subchunk1 size
    header += struct.pack("<H", 1)  # Audio format (PCM)
    header += struct.pack("<H", num_channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", byte_rate)
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", bits_per_sample)
    header += b"data"
    header += struct.pack("<I", data_size)

    # Silence
    audio_data = b"\x00" * data_size

    return header + audio_data


def make_audio_input(duration_seconds: float = 1.0, sample_rate: int = 24000):
    """Create a mock AudioInput dictionary."""
    num_samples = int(duration_seconds * sample_rate)
    waveform = torch.zeros((1, 1, num_samples), dtype=torch.float32)
    return {"waveform": waveform, "sample_rate": sample_rate}


def make_video_input():
    """Create a mock VideoInput object."""

    class MockVideoInput:
        def __init__(self, data: bytes = b"mock_video_data"):
            self._data = data

        def save_to(self, file):
            if isinstance(file, str):
                with open(file, "wb") as f:
                    f.write(self._data)
            else:
                file.write(self._data)

    return MockVideoInput(b"mock_mp4_video_data")


def make_image_tensor(batch: int = 1, height: int = 64, width: int = 64, channels: int = 3):
    """Create a mock image tensor in ComfyUI format (B, H, W, C)."""
    return torch.rand((batch, height, width, channels), dtype=torch.float32)


def make_mask_tensor(batch: int = 1, height: int = 64, width: int = 64):
    """Create a mock mask tensor in ComfyUI format (B, H, W, C) - 3 channels for PIL compatibility."""
    return torch.rand((batch, height, width, 3), dtype=torch.float32)


def get_free_port() -> int:
    """Get an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


# =============================================================================
# Server Manager - Runs real API server with mocked AsyncOmni in background thread
# =============================================================================


class ServerManager:
    """Manages the real vLLM-Omni API server with mocked AsyncOmni in a background thread.

    This uses the actual FastAPI routes from api_server.py with a mocked AsyncOmni engine,
    ensuring we test the real API contract that the ComfyUI plugin depends on.
    """

    def __init__(self, port: int):
        self.port = port
        self.server_thread: threading.Thread | None = None
        self.server: uvicorn.Server | None = None
        self._started = threading.Event()
        self._app = None

    def _run_server(self):
        """Run the server in a separate thread."""
        # Create the real API app with mocked AsyncOmni (uses all test models by default)
        self._app = create_mock_app()

        config = uvicorn.Config(
            self._app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
        )
        self.server = uvicorn.Server(config)

        # Signal that server is starting
        async def startup():
            self._started.set()

        self._app.add_event_handler("startup", startup)

        # Run the server
        asyncio.run(self.server.serve())

    def start(self):
        """Start the server in a background thread."""
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

        # Wait for server to start (with timeout)
        if not self._started.wait(timeout=10):
            raise RuntimeError("Server failed to start within timeout")

        # Give the server a moment to fully initialize
        time.sleep(0.1)

    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.should_exit = True
        if self.server_thread:
            self.server_thread.join(timeout=5)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def mock_server_port():
    """Return an available port for the mock server."""
    return get_free_port()


@pytest.fixture(scope="module")
def mock_server_url(mock_server_port):
    """Return the URL for the mock server."""
    return f"http://127.0.0.1:{mock_server_port}/v1"


@pytest.fixture(scope="module")
def mock_server(mock_server_port):
    """Start and stop the real API server with mocked AsyncOmni."""
    manager = ServerManager(mock_server_port)
    manager.start()
    yield manager
    manager.stop()


# =============================================================================
# Tests for Sampling Parameter Nodes (Synchronous)
# =============================================================================


class TestVLLMOmniARSampling:
    """Tests for AR (Autoregressive) sampling parameter node."""

    def test_input_types(self):
        """Test INPUT_TYPES returns expected structure."""
        input_types = VLLMOmniARSampling.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        assert "max_tokens" in required
        assert "temperature" in required
        assert "top_p" in required
        assert "repetition_penalty" in required
        assert "seed" in required

    def test_metadata(self):
        """Test node metadata."""
        assert VLLMOmniARSampling.RETURN_TYPES == ("SAMPLING_PARAMS",)
        assert VLLMOmniARSampling.FUNCTION == "get_params"
        assert VLLMOmniARSampling.CATEGORY == "vLLM-Omni/Sampling Params"

    def test_get_params_default(self):
        """Test get_params with default values."""
        node = VLLMOmniARSampling()
        result = node.get_params(
            seed=-1,
            max_tokens=100,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        params = result[0]

        assert params["type"] == "autoregression"
        assert params["max_tokens"] == 100
        assert params["temperature"] == 1.0
        assert params["top_p"] == 1.0
        assert params["repetition_penalty"] == 1.0
        assert "seed" not in params  # -1 means no seed

    def test_get_params_with_seed(self):
        """Test get_params with explicit seed."""
        node = VLLMOmniARSampling()
        result = node.get_params(
            seed=42,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        params = result[0]

        assert params["seed"] == 42
        assert params["max_tokens"] == 200
        assert params["temperature"] == 0.7

    def test_get_params_with_zero_seed(self):
        """Test that seed=0 is included (only -1 means no seed)."""
        node = VLLMOmniARSampling()
        result = node.get_params(seed=0, max_tokens=100, temperature=1.0, top_p=1.0, repetition_penalty=1.0)
        params = result[0]
        assert params["seed"] == 0


class TestVLLMOmniDiffusionSampling:
    """Tests for Diffusion sampling parameter node."""

    def test_input_types(self):
        """Test INPUT_TYPES returns expected structure."""
        input_types = VLLMOmniDiffusionSampling.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        assert "n" in required
        assert "num_inference_steps" in required
        assert "guidance_scale" in required
        assert "true_cfg_scale" in required
        assert "vae_use_slicing" in required
        assert "seed" in required

    def test_metadata(self):
        """Test node metadata."""
        assert VLLMOmniDiffusionSampling.RETURN_TYPES == ("SAMPLING_PARAMS",)
        assert VLLMOmniDiffusionSampling.FUNCTION == "get_params"
        assert VLLMOmniDiffusionSampling.CATEGORY == "vLLM-Omni/Sampling Params"

    def test_get_params_default(self):
        """Test get_params with default values."""
        node = VLLMOmniDiffusionSampling()
        result = node.get_params(
            seed=-1,
            n=1,
            num_inference_steps=50,
            guidance_scale=7.5,
            true_cfg_scale=1.0,
            vae_use_slicing=False,
        )
        params = result[0]

        assert params["type"] == "diffusion"
        assert params["n"] == 1
        assert params["num_inference_steps"] == 50
        assert params["guidance_scale"] == 7.5
        assert "seed" not in params

    def test_get_params_with_seed(self):
        """Test get_params with explicit seed."""
        node = VLLMOmniDiffusionSampling()
        result = node.get_params(
            seed=123,
            n=2,
            num_inference_steps=100,
            guidance_scale=10.0,
            true_cfg_scale=5.0,
            vae_use_slicing=True,
        )
        params = result[0]

        assert params["seed"] == 123
        assert params["n"] == 2
        assert params["vae_use_slicing"] is True


class TestVLLMOmniSamplingParamsList:
    """Tests for aggregating sampling parameters."""

    def test_input_types(self):
        """Test INPUT_TYPES returns expected structure."""
        input_types = VLLMOmniSamplingParamsList.INPUT_TYPES()
        assert "required" in input_types
        assert "optional" in input_types
        assert "param1" in input_types["required"]
        assert "param2" in input_types["optional"]
        assert "param3" in input_types["optional"]

    def test_metadata(self):
        """Test node metadata."""
        assert VLLMOmniSamplingParamsList.RETURN_TYPES == ("SAMPLING_PARAMS",)
        assert VLLMOmniSamplingParamsList.FUNCTION == "aggregate"
        assert VLLMOmniSamplingParamsList.CATEGORY == "vLLM-Omni/Sampling Params"

    def test_aggregate_single_param(self):
        """Test aggregating a single parameter."""
        node = VLLMOmniSamplingParamsList()
        param1 = {"type": "autoregression", "temperature": 0.7}
        result = node.aggregate(param1)

        assert result == ([param1],)

    def test_aggregate_two_params(self):
        """Test aggregating two parameters."""
        node = VLLMOmniSamplingParamsList()
        param1 = {"type": "autoregression", "temperature": 0.7}
        param2 = {"type": "diffusion", "num_inference_steps": 50}
        result = node.aggregate(param1, param2)

        assert result == ([param1, param2],)

    def test_aggregate_three_params(self):
        """Test aggregating three parameters."""
        node = VLLMOmniSamplingParamsList()
        param1 = {"type": "autoregression", "temperature": 0.7}
        param2 = {"type": "diffusion", "num_inference_steps": 50}
        param3 = {"type": "autoregression", "max_tokens": 100}
        result = node.aggregate(param1, param2, param3)

        assert result == ([param1, param2, param3],)

    def test_aggregate_rejects_nested_list(self):
        """Test that passing a list raises ValueError."""
        node = VLLMOmniSamplingParamsList()
        param_list = [{"type": "autoregression"}]

        with pytest.raises(ValueError, match="Multi-Stage Sampling Params List"):
            node.aggregate(param_list)


class TestVLLMOmniQwenTTSParams:
    """Tests for Qwen TTS parameters node."""

    def test_input_types(self):
        """Test INPUT_TYPES returns expected structure."""
        input_types = VLLMOmniQwenTTSParams.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        assert "task_type" in required
        assert "language" in required
        assert "instructions" in required
        assert "max_new_tokens" in required

    def test_metadata(self):
        """Test node metadata."""
        assert VLLMOmniQwenTTSParams.RETURN_TYPES == ("TTS_PARAMS",)
        assert VLLMOmniQwenTTSParams.FUNCTION == "get_params"
        assert VLLMOmniQwenTTSParams.CATEGORY == "vLLM-Omni/TTS Params"

    def test_get_params(self):
        """Test get_params returns correct structure."""
        node = VLLMOmniQwenTTSParams()
        result = node.get_params(
            task_type="CustomVoice",
            language="English",
            instructions="Speak clearly.",
            max_new_tokens=2048,
        )
        params = result[0]

        assert params["type"] == "qwen-tts"
        assert params["task_type"] == "CustomVoice"
        assert params["language"] == "English"
        assert params["instructions"] == "Speak clearly."
        assert params["max_new_tokens"] == 2048


# =============================================================================
# Tests for Generation Nodes (Async, requires mock server)
# =============================================================================


class TestVLLMOmniGenerateImageBase:
    """Tests for VLLMOmniGenerateImage input validation and metadata."""

    def test_input_types(self):
        """Test INPUT_TYPES returns expected structure."""
        input_types = VLLMOmniGenerateImage.INPUT_TYPES()
        assert "required" in input_types
        assert "optional" in input_types

        required = input_types["required"]
        assert "url" in required
        assert "model" in required
        assert "prompt" in required
        assert "negative_prompt" in required
        assert "width" in required
        assert "height" in required

        optional = input_types["optional"]
        assert "image" in optional
        assert "mask" in optional
        assert "sampling_params" in optional

    def test_metadata(self):
        """Test node metadata."""
        assert VLLMOmniGenerateImage.RETURN_TYPES == ("IMAGE",)
        assert VLLMOmniGenerateImage.RETURN_NAMES == ("image",)
        assert VLLMOmniGenerateImage.FUNCTION == "generate"
        assert VLLMOmniGenerateImage.CATEGORY == "vLLM-Omni"

    def test_validate_inputs_valid(self):
        """Test VALIDATE_INPUTS with valid inputs."""
        result = VLLMOmniGenerateImage.VALIDATE_INPUTS(
            url="http://localhost:8000/v1", model="test-model"
        )
        assert result is True

    def test_validate_inputs_empty_url(self):
        """Test VALIDATE_INPUTS with empty URL."""
        result = VLLMOmniGenerateImage.VALIDATE_INPUTS(url="", model="test-model")
        assert "URL must not be empty" in result

    def test_validate_inputs_empty_model(self):
        """Test VALIDATE_INPUTS with empty model."""
        result = VLLMOmniGenerateImage.VALIDATE_INPUTS(
            url="http://localhost:8000/v1", model=""
        )
        assert "Model must not be empty" in result


@pytest.mark.asyncio
class TestVLLMOmniGenerateImage:
    """Async tests for VLLMOmniGenerateImage generation."""

    async def test_generate_text_to_image(self, mock_server, mock_server_url):
        """Test basic text-to-image generation."""
        node = VLLMOmniGenerateImage()
        result = await node.generate(
            url=mock_server_url,
            model="Tongyi-MAI/Z-Image-Turbo",
            prompt="A beautiful sunset",
            width=512,
            height=512,
        )

        # Verify output is a tuple with image tensor
        assert isinstance(result, tuple)
        assert len(result) == 1
        image_tensor = result[0]
        assert isinstance(image_tensor, torch.Tensor)
        # Image tensor should be in ComfyUI format (B, H, W, C)
        assert image_tensor.ndim == 4

    async def test_generate_with_negative_prompt(self, mock_server, mock_server_url):
        """Test image generation with negative prompt."""
        node = VLLMOmniGenerateImage()
        result = await node.generate(
            url=mock_server_url,
            model="Tongyi-MAI/Z-Image-Turbo",
            prompt="A cat",
            negative_prompt="blurry, low quality",
            width=512,
            height=512,
        )

        assert isinstance(result[0], torch.Tensor)

    async def test_generate_with_sampling_params(self, mock_server, mock_server_url):
        """Test image generation with sampling parameters."""
        node = VLLMOmniGenerateImage()
        sampling_params = {
            "type": "diffusion",
            "n": 2,
            "num_inference_steps": 30,
            "guidance_scale": 8.0,
        }
        result = await node.generate(
            url=mock_server_url,
            model="Tongyi-MAI/Z-Image-Turbo",
            prompt="A mountain",
            width=512,
            height=512,
            sampling_params=sampling_params,
        )

        assert isinstance(result[0], torch.Tensor)

    async def test_generate_image_edit(self, mock_server, mock_server_url):
        """Test image editing (with input image)."""
        node = VLLMOmniGenerateImage()
        input_image = make_image_tensor()

        result = await node.generate(
            url=mock_server_url,
            model="Tongyi-MAI/Z-Image-Turbo",
            prompt="Add a rainbow",
            width=512,
            height=512,
            image=input_image,
        )

        assert isinstance(result[0], torch.Tensor)

    async def test_generate_image_edit_with_mask(self, mock_server, mock_server_url):
        """Test image editing with mask."""
        node = VLLMOmniGenerateImage()
        input_image = make_image_tensor()
        mask = make_mask_tensor()

        result = await node.generate(
            url=mock_server_url,
            model="Tongyi-MAI/Z-Image-Turbo",
            prompt="Replace the sky",
            width=512,
            height=512,
            image=input_image,
            mask=mask,
        )

        assert isinstance(result[0], torch.Tensor)

    async def test_generate_mask_without_image_raises(self, mock_server, mock_server_url):
        """Test that providing mask without image raises ValueError."""
        node = VLLMOmniGenerateImage()
        mask = make_mask_tensor()

        with pytest.raises(ValueError, match="Mask input provided without an image"):
            await node.generate(
                url=mock_server_url,
                model="Tongyi-MAI/Z-Image-Turbo",
                prompt="Test",
                width=512,
                height=512,
                mask=mask,
            )


class TestVLLMOmniComprehensionBase:
    """Tests for VLLMOmniComprehension input validation and metadata."""

    def test_input_types(self):
        """Test INPUT_TYPES returns expected structure."""
        input_types = VLLMOmniComprehension.INPUT_TYPES()
        assert "required" in input_types
        assert "optional" in input_types

        required = input_types["required"]
        assert "url" in required
        assert "model" in required
        assert "prompt" in required
        assert "output_text" in required
        assert "output_audio" in required
        assert "use_audio_in_video" in required

        optional = input_types["optional"]
        assert "image" in optional
        assert "video" in optional
        assert "audio" in optional
        assert "sampling_params" in optional

    def test_metadata(self):
        """Test node metadata."""
        assert VLLMOmniComprehension.RETURN_TYPES == ("STRING", "AUDIO")
        assert VLLMOmniComprehension.RETURN_NAMES == ("text_response", "audio_response")
        assert VLLMOmniComprehension.FUNCTION == "generate"

    def test_validate_inputs_valid(self):
        """Test VALIDATE_INPUTS with valid inputs."""
        result = VLLMOmniComprehension.VALIDATE_INPUTS(
            url="http://localhost:8000/v1",
            model="test-model",
            output_text=True,
            output_audio=True,
        )
        assert result is True

    def test_validate_inputs_no_output(self):
        """Test VALIDATE_INPUTS when both outputs are False."""
        result = VLLMOmniComprehension.VALIDATE_INPUTS(
            url="http://localhost:8000/v1",
            model="test-model",
            output_text=False,
            output_audio=False,
        )
        assert "At least one of output_text or output_audio must be True" in result


@pytest.mark.asyncio
class TestVLLMOmniComprehension:
    """Async tests for VLLMOmniComprehension generation."""

    async def test_generate_text_only(self, mock_server, mock_server_url):
        """Test text-only comprehension."""
        node = VLLMOmniComprehension()
        result = await node.generate(
            url=mock_server_url,
            model="Qwen/Qwen2.5-Omni-7B",
            prompt="What is 2+2?",
            output_text=True,
            output_audio=False,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        text_response, audio_response = result
        assert isinstance(text_response, str)
        # Audio should be a placeholder when output_audio=False
        assert isinstance(audio_response, dict)
        assert "waveform" in audio_response

    async def test_generate_with_image(self, mock_server, mock_server_url):
        """Test comprehension with image input."""
        node = VLLMOmniComprehension()
        input_image = make_image_tensor()

        result = await node.generate(
            url=mock_server_url,
            model="Qwen/Qwen2.5-Omni-7B",
            prompt="Describe this image",
            image=input_image,
            output_text=True,
            output_audio=False,
        )

        assert isinstance(result[0], str)

    async def test_generate_with_audio_input(self, mock_server, mock_server_url):
        """Test comprehension with audio input."""
        node = VLLMOmniComprehension()
        audio_input = make_audio_input()

        result = await node.generate(
            url=mock_server_url,
            model="Qwen/Qwen2.5-Omni-7B",
            prompt="Transcribe this audio",
            audio=audio_input,
            output_text=True,
            output_audio=False,
        )

        assert isinstance(result[0], str)

    async def test_generate_audio_output(self, mock_server, mock_server_url):
        """Test comprehension with audio output."""
        node = VLLMOmniComprehension()

        result = await node.generate(
            url=mock_server_url,
            model="Qwen/Qwen2.5-Omni-7B",
            prompt="Say hello",
            output_text=True,
            output_audio=True,
        )

        text_response, audio_response = result
        assert isinstance(audio_response, dict)
        assert "waveform" in audio_response
        assert "sample_rate" in audio_response


class TestVLLMOmniTTSBase:
    """Tests for VLLMOmniTTS input validation and metadata."""

    def test_input_types(self):
        """Test INPUT_TYPES returns expected structure."""
        input_types = VLLMOmniTTS.INPUT_TYPES()
        assert "required" in input_types
        assert "optional" in input_types

        required = input_types["required"]
        assert "url" in required
        assert "model" in required
        assert "input" in required
        assert "voice" in required
        assert "response_format" in required
        assert "speed" in required

        optional = input_types["optional"]
        assert "model_specific_params" in optional

    def test_metadata(self):
        """Test node metadata."""
        assert VLLMOmniTTS.RETURN_TYPES == ("AUDIO",)
        assert VLLMOmniTTS.RETURN_NAMES == ("audio",)
        assert VLLMOmniTTS.FUNCTION == "generate"


@pytest.mark.asyncio
class TestVLLMOmniTTS:
    """Async tests for VLLMOmniTTS generation."""

    async def test_generate_basic(self, mock_server, mock_server_url):
        """Test basic TTS generation."""
        node = VLLMOmniTTS()
        result = await node.generate(
            url=mock_server_url,
            model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            input="Hello, world!",
            voice="Vivian",
            response_format="wav",
            speed=1.0,
            model_specific_params=None,
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        audio = result[0]
        assert isinstance(audio, dict)
        assert "waveform" in audio
        assert "sample_rate" in audio

    async def test_generate_with_speed(self, mock_server, mock_server_url):
        """Test TTS with speed adjustment."""
        node = VLLMOmniTTS()
        result = await node.generate(
            url=mock_server_url,
            model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            input="Fast speech",
            voice="Vivian",
            response_format="mp3",
            speed=1.5,
            model_specific_params=None,
        )

        assert isinstance(result[0], dict)

    async def test_generate_with_qwen_params(self, mock_server, mock_server_url):
        """Test TTS with Qwen-specific parameters."""
        node = VLLMOmniTTS()
        qwen_params = {
            "type": "qwen-tts",
            "task_type": "CustomVoice",
            "language": "English",
            "instructions": "Speak clearly",
            "max_new_tokens": 2048,
        }

        result = await node.generate(
            url=mock_server_url,
            model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            input="Hello",
            voice="Ryan",
            response_format="wav",
            speed=1.0,
            model_specific_params=qwen_params,
        )

        assert isinstance(result[0], dict)

    async def test_generate_qwen_params_with_non_qwen_model_raises(
        self, mock_server, mock_server_url
    ):
        """Test that using Qwen params with non-Qwen model raises error."""
        node = VLLMOmniTTS()
        qwen_params = {"type": "qwen-tts", "language": "English"}

        with pytest.raises(ValueError, match="Qwen-specific TTS params"):
            await node.generate(
                url=mock_server_url,
                model="some-other-tts-model",  # Not a Qwen model
                input="Hello",
                voice="alloy",
                response_format="wav",
                speed=1.0,
                model_specific_params=qwen_params,
            )


class TestVLLMOmniVoiceCloneBase:
    """Tests for VLLMOmniVoiceClone input validation and metadata."""

    def test_input_types(self):
        """Test INPUT_TYPES returns expected structure."""
        input_types = VLLMOmniVoiceClone.INPUT_TYPES()
        assert "required" in input_types
        assert "optional" in input_types

        required = input_types["required"]
        assert "url" in required
        assert "model" in required
        assert "input" in required
        assert "voice" in required
        assert "response_format" in required
        assert "speed" in required
        assert "ref_audio" in required
        assert "ref_text" in required
        assert "x_vector_only_mode" in required

        optional = input_types["optional"]
        assert "model_specific_params" in optional

    def test_metadata(self):
        """Test node metadata."""
        assert VLLMOmniVoiceClone.RETURN_TYPES == ("AUDIO",)
        assert VLLMOmniVoiceClone.RETURN_NAMES == ("audio",)
        assert VLLMOmniVoiceClone.FUNCTION == "generate"


@pytest.mark.asyncio
class TestVLLMOmniVoiceClone:
    """Async tests for VLLMOmniVoiceClone generation."""

    async def test_generate_basic(self, mock_server, mock_server_url):
        """Test basic voice cloning."""
        node = VLLMOmniVoiceClone()
        ref_audio = make_audio_input()

        result = await node.generate(
            url=mock_server_url,
            model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            input="Clone this voice.",
            voice="custom",
            response_format="wav",
            speed=1.0,
            ref_audio=ref_audio,
            ref_text="Reference text for cloning.",
            x_vector_only_mode=False,
            model_specific_params=None,
        )

        assert isinstance(result, tuple)
        audio = result[0]
        assert isinstance(audio, dict)
        assert "waveform" in audio

    async def test_generate_x_vector_mode(self, mock_server, mock_server_url):
        """Test voice cloning with x_vector_only_mode."""
        node = VLLMOmniVoiceClone()
        ref_audio = make_audio_input()

        result = await node.generate(
            url=mock_server_url,
            model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            input="X-vector mode",
            voice="custom",
            response_format="wav",
            speed=1.0,
            ref_audio=ref_audio,
            ref_text="Ref text",
            x_vector_only_mode=True,
            model_specific_params=None,
        )

        assert isinstance(result[0], dict)


# =============================================================================
# Tests for Chat Completion via Image Generation (BAGEL-style models)
# =============================================================================


@pytest.mark.asyncio
class TestVLLMOmniGenerateImageChatCompletion:
    """Tests for image generation via chat completion API (e.g., BAGEL models)."""

    async def test_generate_bagel_style(self, mock_server, mock_server_url):
        """Test image generation for BAGEL-style model using chat completion."""
        node = VLLMOmniGenerateImage()

        # BAGEL model uses chat completion endpoint
        result = await node.generate(
            url=mock_server_url,
            model="ByteDance/BAGEL-7B-MoT",
            prompt="Generate an image of a cat",
            width=512,
            height=512,
        )

        assert isinstance(result, tuple)
        assert isinstance(result[0], torch.Tensor)
