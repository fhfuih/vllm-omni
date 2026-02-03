from typing import cast

import numpy as np
import torch
from comfy_api.input import AudioInput, VideoInput

from .utils.api_client import VLLMOmniClient
from .utils.models import lookup_model_spec
from .utils.validators import (
    add_sampling_parameters_to_stage,
    validate_model_and_sampling_params_types,
)


class _VLLMOmniGenerateBase:
    """Base class for vLLM-Omni generation nodes with shared functionality."""

    CATEGORY = "vLLM-Omni"

    @classmethod
    def VALIDATE_INPUTS(cls, url, model):
        """
        Can only validate this model's own input. Cannot check inputs from other nodes.
        See: https://docs.comfy.org/custom-nodes/backend/server_overview#validate_inputs
        """
        if not url:
            return "URL must not be empty"
        if not model:
            return "Model must not be empty"
        return True


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

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"

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
        validate_model_and_sampling_params_types(model, sampling_params)
        if image is None and mask is not None:
            raise ValueError("Mask input provided without an image input.")

        client = VLLMOmniClient(url)

        spec, pattern = lookup_model_spec(model)
        is_bagel = pattern is not None and "bagel" in pattern.lower()

        # Prefer DALL-E compatible API for simple (one-stage) diffusion models
        if (spec is None or spec["stages"] == ["diffusion"]) and not is_bagel:
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
    def VALIDATE_INPUTS(cls, url, model, output_text, output_audio):
        super().VALIDATE_INPUTS(url, model)
        if not output_text and not output_audio:
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
    ) -> tuple[str, AudioInput]:
        print("DEBUG: Uncaught kwargs:", kwargs)
        print("DEBUG: Got sampling params", sampling_params)
        validate_model_and_sampling_params_types(model, sampling_params)

        client = VLLMOmniClient(url)
        spec, pattern = lookup_model_spec(model)
        is_bagel = pattern is not None and "bagel" in pattern.lower()

        if is_bagel:
            # A lot of special handlings here...
            if output_audio:
                raise ValueError("BAGEL models do not support audio output.")
            if audio is not None or video is not None:
                raise ValueError("BAGEL models do not support audio or video input.")
            (
                text_response,
                _,
            ) = await client.generate_comprehension_chat_completion(
                model,
                prompt,
                image,
                None,
                None,
                sampling_params,
                ["text"],
            )
        else:
            modalities = []
            if output_text:
                modalities.append("text")
            if output_audio:
                modalities.append("audio")

            if use_audio_in_video and video is not None:
                use_audio_in_video = True
            else:
                use_audio_in_video = False

            (
                text_response,
                audio,
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

        if text_response is None:
            text_response = ""
        if audio is None:
            channels = 1
            duration = 1
            sample_rate = 44100
            num_samples = int(round(duration * sample_rate))
            waveform = torch.zeros((1, channels, num_samples), dtype=torch.float32)
            audio = {"waveform": waveform, "sample_rate": sample_rate}

        return (text_response, audio)


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
        client = VLLMOmniClient(url)

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
        client = VLLMOmniClient(url)

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
    RETURN_NAMES = ("AR sampling params",)
    FUNCTION = "get_params"
    CATEGORY = "vLLM-Omni/Sampling Params"

    def get_params(self, **kwargs):
        return ({"type": "autoregression", **kwargs},)


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
    RETURN_NAMES = ("diffusion sampling params",)
    FUNCTION = "get_params"
    CATEGORY = "vLLM-Omni/Sampling Params"

    def get_params(self, **kwargs):
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
    RETURN_NAMES = ("param list",)
    FUNCTION = "aggregate"
    CATEGORY = "vLLM-Omni/Sampling Params"

    def aggregate(
        self, param1: dict, param2: dict | None = None, param3: dict | None = None
    ):
        for i, p in enumerate((param1, param2, param3)):
            print(f"Param {i} is {p}")
            if isinstance(p, list):
                raise ValueError(
                    f"Input {i} is a Multi-Stage Sampling Params List. Expected a single sampling parameters node (either AR or Diffusion)."
                )

        params = [param1]
        if param2 is not None:
            params.append(param2)
        if param3 is not None:
            params.append(param3)
        return (params,)
