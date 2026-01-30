"""Image/tensor format helpers.

The image generation part is derived from dougbtv/comfyui-vllm-omni by Doug (@dougbtv).
Original source at https://github.com/dougbtv/comfyui-vllm-omni, distributed under the MIT License.
"""

import aiohttp
import openai
from openai.types.chat import ChatCompletionMessageParam
from typing import Any, Iterable, Literal, Optional

import torch

from .models import MODEL_PIPELINE_SPECS

from .format import base64_to_image_tensor, image_tensor_to_base64, image_tensor_to_png_bytes


class VLLMOmniClient:
    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.client = openai.OpenAI(base_url=base_url, api_key="fake-key")

    async def generate_image(
        self,
        model: str,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: str | None = None,
        sampling_params: dict | None = None,
    ) -> torch.Tensor:
        """Run text-to-image generatation via DALLE API"""
        if not self._check_model_exist(model):
            raise ValueError(f"Model {model} does not exist.")
        size = f"{width}x{height}"
        payload = {
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if sampling_params is not None:
            for k in ("n", "num_inference_steps", "guidance_scale", "true_cfg_scale", "vae_use_slicing"):
                if k in sampling_params:
                    payload[k] = sampling_params[k]
            if sampling_params.get("seed", 0) != 0:
                payload["seed"] = sampling_params["seed"]
        print("DEBUG: img gen payload", payload)

        url = self.base_url + "/images/generations"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")
                    if "data" not in data:
                        raise RuntimeError("API response missing 'data' field - expected OpenAI DALL-E format")
                    if not data["data"]:
                        raise RuntimeError("API returned empty data array")

                    image_tensors = []
                    for idx, img in enumerate(data["data"]):
                        if "b64_json" not in img:
                            raise RuntimeError(f"API returned image #{idx} without 'b64_json' field")
                        base64_str = img["b64_json"]
                        tensor = base64_to_image_tensor(base64_str)
                        image_tensors.append(tensor)
                        print(f"DEBUG: Image #{idx} has shape {tensor.shape}")
                        # from torchvision.utils import save_image

                        # save_image(tensor, f"~/{idx}.png")

                    batch_tensor = torch.stack(image_tensors, dim=0)
                    print("DEBUG: batch_tensor output has shape", {batch_tensor.shape})
                    return batch_tensor

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {url}: {e}")

    async def edit_image(
        self,
        model: str,
        prompt: str,
        image: torch.Tensor,
        width: int,
        height: int,
        negative_prompt: str | None = None,
        mask: torch.Tensor | None = None,
        sampling_params: dict | None = None,
    ) -> torch.Tensor:
        """Run image editing via DALLE API"""
        if not self._check_model_exist(model):
            raise ValueError(f"Model {model} does not exist.")
        size = f"{width}x{height}"
        image_filename = "image.png"  # Required for multipart form
        form = aiohttp.FormData()
        form.add_field("image", image_tensor_to_png_bytes(image, image_filename), filename=image_filename, content_type="image/png")
        form.add_field("prompt", prompt)
        form.add_field("size", size)
        if negative_prompt:
            form.add_field("negative_prompt", negative_prompt)
        if sampling_params is not None:
            for k in ("n", "num_inference_steps", "guidance_scale", "true_cfg_scale"):
                if k in sampling_params:
                    form.add_field(k, str(sampling_params[k]))
            if sampling_params.get("seed", 0) != 0:
                form.add_field("seed", str(sampling_params["seed"]))
        if mask is not None:
            mask_filename = "mask.png"
            form.add_field("mask", image_tensor_to_png_bytes(mask, mask_filename), filename=mask_filename, content_type="image/png")

        url = self.base_url + "/images/edits"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(url, data=form) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")

                    if "data" not in data:
                        raise RuntimeError("API response missing 'data' field - expected OpenAI DALL-E format")
                    if not data["data"]:
                        raise RuntimeError("API returned empty data array")

                    image_tensors = []
                    for idx, img in enumerate(data["data"]):
                        if "b64_json" not in img:
                            raise RuntimeError(f"API returned image #{idx} without 'b64_json' field")
                        base64_str = img["b64_json"]
                        tensor = base64_to_image_tensor(base64_str)
                        image_tensors.append(tensor)

                    return torch.stack(image_tensors, dim=0)

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {url}: {e}")

    async def generate_image_chat_completion(
        self,
        model: str,
        prompt: str,
        negative_prompt: str | None = None,
        image: torch.Tensor | None = None,
        audio: str | None = None,
        video: str | None = None,
        sampling_params: dict | list[dict] | None = None,
    ) -> torch.Tensor:
        if not self._check_model_exist(model):
            raise ValueError(f"Model {model} does not exist.")

        payload = VLLMOmniClient._prepare_chat_completion_messages(model, prompt, negative_prompt, image, audio, video, sampling_params)

        url = self.base_url + "/chat/completions"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")

                    if "choices" not in data:
                        raise RuntimeError("API response missing 'choices' field - expected OpenAI Chat Completion format")
                    if not data["choices"]:
                        raise RuntimeError("API returned empty choices array")
                    choice = data["choices"][0]

                    image_tensors = []
                    for idx, img_content in enumerate(choice["message"]["content"]):
                        base64_str = img_content.get("image_url", {}).get("url", "")
                        if not base64_str:
                            raise RuntimeError(f"API returned image #{idx} without image url")
                        tensor = base64_to_image_tensor(base64_str)
                        image_tensors.append(tensor)

                    return torch.stack(image_tensors, dim=0)

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {self.base_url}: {e}")

    def generate_video(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        modalities: Optional[list[Literal["text", "audio"]]] | openai.Omit = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not self._check_model_exist(model):
            raise ValueError(f"Model {model} does not exist.")

        try:
            response = self.client.chat.completions.create(model=model, messages=messages, modalities=modalities, extra_body=extra_body)
            return {
                "video": response.choices[0].message.content,
                "text": response.choices[0].message.content,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to generate video: {str(e)}")

    def generate_audio(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        modalities: Optional[list[Literal["text", "audio"]]] | openai.Omit = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not self._check_model_exist(model):
            raise ValueError(f"Model {model} does not exist.")

        try:
            response = self.client.chat.completions.create(model=model, messages=messages, modalities=modalities, extra_body=extra_body)
            return {
                "audio": response.choices[0].message.audio,
                "text": response.choices[0].message.content,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to generate audio: {str(e)}")

    def _check_model_exist(self, model: str):
        model_list = self.client.models.list().data
        return next((True for m in model_list if m.id == model), False)

    @staticmethod
    def _prepare_chat_completion_messages(
        model: str,
        prompt: str,
        negative_prompt: str | None,
        image: torch.Tensor | None = None,
        audio: Any | None = None,
        video: Any | None = None,
        sampling_params: dict | list[dict] | None = None,
    ):
        message_content: list[dict] = [{"type": "text", "text": prompt}]
        message_content.append({"type": "text", "text": prompt})
        if image is not None:
            message_content.append({"type": "image_url", "image_url": {"url": image_tensor_to_base64(image)}})
        if audio is not None:
            message_content.append({"type": "audio_url", "audio_url": {"url": audio}})
        if video is not None:
            message_content.append({"type": "video_url", "video_url": {"url": video}})
        messages = [{"role": "user", "content": message_content}]

        extra_body: dict[str, Any] = {}
        if sampling_params is not None:
            if (not model in MODEL_PIPELINE_SPECS or MODEL_PIPELINE_SPECS["model"]["stages"] == ["diffusion"]) and (
                isinstance(sampling_params, dict) or len(sampling_params) == 1
            ):
                # If model is not registered, default to regular diffusion models and diffusion format: extra_body directly contains sampling params
                sampling_params = sampling_params if isinstance(sampling_params, dict) else sampling_params[0]
                extra_body: dict[str, Any] = {**sampling_params}
                if "n" in extra_body:
                    extra_body["num_outputs_per_prompt"] = extra_body["n"]
                    del extra_body["n"]
            else:
                # Use AR style payload, extra_body has a sampling_params_list
                extra_body: dict[str, Any] = {"sampling_params_list": sampling_params}

        if negative_prompt:
            extra_body["negative_prompt"] = negative_prompt

        payload: dict[str, Any] = {"messages": messages}
        if extra_body:
            payload["extra_body"] = extra_body

        # TODO: Elegant way to handle Qwen Omni mm_processor_kwargs
        if False:
            extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}

        return payload
