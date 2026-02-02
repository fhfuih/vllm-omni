from typing import Literal, cast

# vllm_omni/nodes.py
import torch
import numpy as np
from comfy_api.input import AudioInput, VideoInput

from .utils.api_client import VLLMOmniClient
from .utils.validators import (
    add_sampling_parameters_to_stage,
    validate_sampling_params_types,
)
from .utils.models import MODEL_PIPELINE_SPECS


class _VLLMOmniGenerateBase:
    """Base class for vLLM-Omni generation nodes with shared functionality."""

    CATEGORY = "vLLM-Omni/Generate"

    def __init__(self):
        self._client = None
        self._last_url = None

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs) -> Literal[True] | str:
        if not kwargs.get("model", ""):
            return "Model name must not be empty."
        if "image" not in kwargs and "mask" in kwargs:
            return "Mask input provided without an image input."
        if kwargs.get("sampling_params", None) is not None:
            try:
                validate_sampling_params_types(
                    kwargs["model"], kwargs["sampling_params"]
                )
            except Exception as e:
                return str(e)
        return True

    def _get_client(self, url):
        """Lazy-initialize OpenAI client, reusing if URL hasn't changed."""
        if self._client is None or self._last_url != url:
            self._client = VLLMOmniClient(url)
            self._last_url = url
        return self._client


class VLLMOmniGenerateImage(_VLLMOmniGenerateBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "http://localhost:8000/v1"}),
                "model": ("STRING", {"default": "Tongyi-MAI/Z-Image-Turbo"}),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                # "video": ("VIDEO",),
                # "audio": ("AUDIO",),
                "sampling_params": ("SAMPLING_PARAMS",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text_response")
    FUNCTION = "generate"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        if output := super().VALIDATE_INPUTS(**kwargs):
            return output
        return True

    async def generate(
        self,
        url: str,
        model: str,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: str | None = None,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        audio: AudioInput | None = None,  # Hidden & unused
        video: VideoInput | None = None,  # Hidden & unused
        sampling_params: dict | list[dict] | None = None,
        **kwargs,
    ):
        print("DEBUG: Uncaught kwargs:", kwargs)
        print("DEBUG: Got sampling params", sampling_params)

        client = self._get_client(url)

        # Try use DALL-E compatible API first
        if (
            model not in MODEL_PIPELINE_SPECS
            or MODEL_PIPELINE_SPECS["model"]["stages"] == "diffusion"
            or MODEL_PIPELINE_SPECS["model"]["stages"] == ["diffusion"]
        ):
            sampling_params = cast(dict | None, sampling_params)
            if audio is None and image is None and video is None:
                # No multimodal input --- use DALL-E image generation
                print("DEBUG: Using DALL-E image generation endpoint")
                output = await client.generate_image(
                    model, prompt, width, height, negative_prompt, sampling_params
                )
                return (output,)
            elif image is not None and audio is None and video is None:
                # Image and text input --- use DALL-E image edit
                print("DEBUG: Using DALL-E image edit endpoint")
                output = await client.edit_image(
                    model,
                    prompt,
                    image,
                    width,
                    height,
                    negative_prompt,
                    mask,
                    sampling_params,
                )
                return (output,)

        print("DEBUG: Using chat completion endpoint")
        sampling_params = add_sampling_parameters_to_stage(
            model, sampling_params, "diffusion", width=width, height=height
        )
        print("DEBUG: Edited sampling params", sampling_params)

        output = await client.generate_image_chat_completion(
            model, prompt, negative_prompt, image, audio, video, sampling_params
        )

        return (output,)


class VLLMOmniComprehension(_VLLMOmniGenerateBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "http://localhost:8000/v1"}),
                "model": ("STRING", {"default": "Qwen/Qwen2.5-Omni-7B"}),
                "prompt": ("STRING", {"multiline": True}),
                "output_text": ("BOOLEAN", {"default": True}),
                "output_audio": ("BOOLEAN", {"default": True}),
                "use_audio_in_video": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
                "sampling_params": ("SAMPLING_PARAMS",),
            },
        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("text_response", "audio_response")
    FUNCTION = "generate"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        if output := super().VALIDATE_INPUTS(**kwargs):
            return output
        if not kwargs["output_text"] and not kwargs["output_audio"]:
            return "At least one of output_text or output_audio must be True."
        return True

    async def generate(
        self,
        url: str,
        model: str,
        prompt: str,
        image: torch.Tensor | None = None,
        audio: AudioInput | None = None,
        video: VideoInput | None = None,
        sampling_params: dict | list[dict] | None = None,
        output_text: bool = True,
        output_audio: bool = True,
        use_audio_in_video: bool = True,
        **kwargs,
    ):
        print("DEBUG: Uncaught kwargs:", kwargs)
        print("DEBUG: Got sampling params", sampling_params)

        client = self._get_client(url)

        modalities = []
        if output_text:
            modalities.append("text")
        if output_audio:
            modalities.append("audio")

        (
            text_response,
            audio_tensor,
        ) = await client.generate_comprehension_chat_completion(
            model,
            prompt,
            image,
            audio,
            video,
            sampling_params,
            modalities,
            mm_processor_kwargs={"use_audio_in_video": use_audio_in_video},
        )

        return (text_response, audio_tensor)


class VLLMOmniGenerateVideo(_VLLMOmniGenerateBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "http://localhost:8000/v1"}),
                "model": ("STRING", {"default": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"}),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "num_frames": ("INT", {"default": 16, "min": 1, "max": 100}),
                "fps": ("INT", {"default": 15, "min": 1, "max": 60}),
                "sampling_params": ("SAMPLING_PARAMS",),
            },
            "optional": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "text_response")
    FUNCTION = "generate"

    def generate(
        self,
        url,
        model,
        prompt,
        negative_prompt,
        width,
        height,
        num_frames,
        fps,
        sampling_params,
        image=None,
        audio=None,
    ):
        raise NotImplementedError()
        client = self._get_client(url)

        payload = self._validate_and_prepare_chat_completion_payload(
            model,
            prompt,
            negative_prompt,
            sampling_params,
            image=image,
            audio=audio,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
        )

        response = client.generate_video(payload)
        video_data = response.get("video", None)
        text_response = response.get("text", "")

        if video_data is None:
            raise RuntimeError("Failed to generate video")

        # 转换为 ComfyUI 视频格式
        video_tensor = torch.from_numpy(
            np.array(video_data).astype(np.float32) / 255.0
        ).unsqueeze(0)
        return (video_tensor, text_response)


class VLLMOmniGenerateAudio(_VLLMOmniGenerateBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "http://localhost:8000/v1"}),
                "model": ("STRING", {"default": "stabilityai/stable-audio-open-1.0"}),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "sampling_params": ("SAMPLING_PARAMS",),
            },
            "optional": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "text_response")
    FUNCTION = "generate"

    def generate(
        self,
        url,
        model,
        prompt,
        negative_prompt,
        sampling_params,
        image=None,
        audio=None,
        video=None,
    ):
        raise NotImplementedError()
        client = self._get_client(url)

        payload = self._validate_and_prepare_chat_completion_payload(
            model,
            prompt,
            negative_prompt,
            sampling_params,
            image=image,
            audio=audio,
            video=video,
        )

        response = client.generate_audio(payload)
        audio_data = response.get("audio", None)
        text_response = response.get("text", "")

        if audio_data is None:
            raise RuntimeError("Failed to generate audio")

        # 转换为 ComfyUI 音频格式
        audio_tensor = torch.from_numpy(np.array(audio_data).astype(np.float32))
        return (audio_tensor, text_response)


class VLLMOmniARSampling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_tokens": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("SAMPLING_PARAMS",)
    FUNCTION = "get_params"
    CATEGORY = "vLLM-Omni/Sampling Params"

    def get_params(self, **kwargs):
        return ({"type": "llm", **kwargs},)


class VLLMOmniDiffusionSampling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "n": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of images to generate",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Number of denoising steps (higher = better quality, slower).",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 7.5,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance scale (higher = more prompt adherence).",
                    },
                ),
                "true_cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "True CFG scale for advanced control (model-specific).",
                    },
                ),
                "vae_use_slicing": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable VAE slicing for reduced memory usage (slight quality trade-off)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SAMPLING_PARAMS",)
    FUNCTION = "get_params"
    CATEGORY = "vLLM-Omni/Sampling Params"

    def get_params(self, **kwargs):
        print("DEBUG: in sampling parameter node, got", kwargs)
        return ({"type": "diffusion", **kwargs},)


class VLLMOmniSamplingParamsList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "param1": ("SAMPLING_PARAMS",),
            },
            "optional": {
                "param2": ("SAMPLING_PARAMS",),
                "param3": ("SAMPLING_PARAMS",),
            },
        }

    RETURN_TYPES = ("SAMPLING_PARAMS",)
    FUNCTION = "aggregate"
    CATEGORY = "vLLM-Omni/Sampling Params"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs) -> Literal[True] | str:
        for k, v in kwargs.items():
            if isinstance(v, list):
                return f"Input {k} is a Multi-Stage Sampling Params List. Expected a single sampling parameters node (either AR or Diffusion)."
        return True

    def aggregate(
        self, param1: dict, param2: dict | None = None, param3: dict | None = None
    ):
        params = [param1]
        if param2 is not None:
            params.append(param2)
        if param3 is not None:
            params.append(param3)
        return (params,)
