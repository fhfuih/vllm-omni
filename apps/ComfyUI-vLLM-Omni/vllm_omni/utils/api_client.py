"""Image/tensor format helpers.

The image generation part is derived from dougbtv/comfyui-vllm-omni by Doug (@dougbtv).
Original source at https://github.com/dougbtv/comfyui-vllm-omni, distributed under the MIT License.
"""

import aiohttp
import openai
from openai.types.chat import ChatCompletionMessageParam
from typing import Any, Iterable, Literal, Optional
from comfy_api.input import AudioInput, VideoInput

import torch


from .models import lookup_model_spec

from .format import (
    audio_to_base64,
    base64_to_audio,
    base64_to_image_tensor,
    image_tensor_to_base64,
    image_tensor_to_png_bytes,
    video_to_base64,
)

from .log import pretty_printer


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
            # Only select specific sampling params
            for k in (
                "n",
                "num_inference_steps",
                "guidance_scale",
                "true_cfg_scale",
                "vae_use_slicing",
            ):
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
                        raise RuntimeError(
                            "API response missing 'data' field - expected OpenAI DALL-E format"
                        )
                    if not data["data"]:
                        raise RuntimeError("API returned empty data array")

                    image_tensors = []
                    for idx, img in enumerate(data["data"]):
                        if "b64_json" not in img:
                            raise RuntimeError(
                                f"API returned image #{idx} without 'b64_json' field"
                            )
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
                raise RuntimeError(
                    f"Network error connecting to vLLM-Omni at {url}: {e}"
                )

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
        form.add_field("model", model)
        form.add_field(
            "image",
            image_tensor_to_png_bytes(image, image_filename),
            filename=image_filename,
            content_type="image/png",
        )
        form.add_field("prompt", prompt)
        form.add_field("size", size)
        if negative_prompt:
            form.add_field("negative_prompt", negative_prompt)
        if sampling_params is not None:
            # Only select specific sampling params
            for k in ("n", "num_inference_steps", "guidance_scale", "true_cfg_scale"):
                if k in sampling_params:
                    form.add_field(k, str(sampling_params[k]))
            if sampling_params.get("seed", 0) != 0:
                form.add_field("seed", str(sampling_params["seed"]))
        if mask is not None:
            mask_filename = "mask.png"
            form.add_field(
                "mask",
                image_tensor_to_png_bytes(mask, mask_filename),
                filename=mask_filename,
                content_type="image/png",
            )

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
                        raise RuntimeError(
                            "API response missing 'data' field - expected OpenAI DALL-E format"
                        )
                    if not data["data"]:
                        raise RuntimeError("API returned empty data array")

                    image_tensors = []
                    for idx, img in enumerate(data["data"]):
                        if "b64_json" not in img:
                            raise RuntimeError(
                                f"API returned image #{idx} without 'b64_json' field"
                            )
                        base64_str = img["b64_json"]
                        tensor = base64_to_image_tensor(base64_str)
                        image_tensors.append(tensor)

                    return torch.stack(image_tensors, dim=0)

            except aiohttp.ClientError as e:
                raise RuntimeError(
                    f"Network error connecting to vLLM-Omni at {url}: {e}"
                )

    async def generate_image_chat_completion(
        self,
        model: str,
        prompt: str,
        negative_prompt: str | None = None,
        image: torch.Tensor | None = None,
        audio: AudioInput | None = None,
        video: VideoInput | None = None,
        sampling_params: dict | list[dict] | None = None,
    ) -> torch.Tensor:
        payload = VLLMOmniClient._prepare_chat_completion_messages(
            model,
            prompt,
            negative_prompt,
            image,
            audio,
            video,
            sampling_params,
            ["image"],
        )
        choices = await self._generate_base_chat_completion(model, payload)

        image_tensors = []
        for idx, img_content in enumerate(choices[0]["message"]["content"]):
            base64_str = img_content.get("image_url", {}).get("url", "")
            if not base64_str:
                raise RuntimeError(f"API returned image #{idx} without image url")
            tensor = base64_to_image_tensor(base64_str)
            image_tensors.append(tensor)

        return torch.stack(image_tensors, dim=0)

    async def generate_comprehension_chat_completion(
        self,
        model: str,
        prompt: str,
        image: torch.Tensor | None = None,
        audio: AudioInput | None = None,
        video: VideoInput | None = None,
        sampling_params: dict | list[dict] | None = None,
        modalities: list[str] = ["text", "audio"],
        **extra_body,
    ) -> tuple[str | None, AudioInput | None]:
        # Response may contain two choices: one with text, one with audio
        payload = VLLMOmniClient._prepare_chat_completion_messages(
            model,
            prompt,
            None,
            image,
            audio,
            video,
            sampling_params,
            modalities,
            **extra_body,
        )

        choices = await self._generate_base_chat_completion(model, payload)
        text_response = None
        audio_base64 = None
        for choice in choices:
            try:
                text_response = choice["message"]["content"]
            except (KeyError, TypeError):
                pass
            try:
                audio_base64 = choice["message"]["audio"]["data"]
            except (KeyError, TypeError):
                pass
        if audio_base64 is None and text_response is None:
            raise RuntimeError(
                "API response missing both '.message.audio' and 'message.content' fields."
                f"The choices object is {choices}"
            )
        if audio_base64 is not None:
            audio = base64_to_audio(audio_base64)
            print(
                f"!!!DEBUG: audio sample rate {audio['sample_rate']}, audio shape {audio['waveform'].shape}, duration in second {audio['waveform'].shape[2] / audio['sample_rate']}"
            )
        else:
            audio = None
        return text_response, audio

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
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                modalities=modalities,
                extra_body=extra_body,
            )
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
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                modalities=modalities,
                extra_body=extra_body,
            )
            return {
                "audio": response.choices[0].message.audio,
                "text": response.choices[0].message.content,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to generate audio: {str(e)}")

    async def _generate_base_chat_completion(
        self, model: str, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        print("!!!DEBUG: Omni payload", pretty_printer.pformat(payload))

        if not self._check_model_exist(model):
            raise ValueError(f"Model {model} does not exist.")

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

                    print(
                        "!!!DEBUG: chat completion response",
                        pretty_printer.pformat(data),
                    )

                    try:
                        return data["choices"]
                    except (KeyError, TypeError):
                        raise RuntimeError(
                            "Invalid JSON response from vLLM-Omni: missing 'choices' field"
                        )

            except aiohttp.ClientError as e:
                raise RuntimeError(
                    f"Network error connecting to vLLM-Omni at {self.base_url}: {e}"
                )

    def _check_model_exist(self, model: str):
        model_list = self.client.models.list().data
        print("!!!", model_list)
        return next((True for m in model_list if m.id == model), False)

    @staticmethod
    def _prepare_chat_completion_messages(
        model: str,
        prompt: str,
        negative_prompt: str | None,
        image: torch.Tensor | None = None,
        audio: AudioInput | None = None,
        video: VideoInput | None = None,
        sampling_params: dict | list[dict] | None = None,
        modalities: list[str] | None = None,  # diffusion don't have this field
        **extra_body,
    ):
        message_content: list[dict] = [{"type": "text", "text": prompt}]
        if image is not None:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_tensor_to_base64(image)},
                }
            )
        if audio is not None:
            message_content.append(
                {"type": "audio_url", "audio_url": {"url": audio_to_base64(audio)}}
            )
        if video is not None:
            message_content.append(
                {"type": "video_url", "video_url": {"url": video_to_base64(video)}}
            )
        messages = [{"role": "user", "content": message_content}]

        combined_extra_body: dict[str, Any] = {}
        if sampling_params is not None:
            spec, _ = lookup_model_spec(model)
            is_single_sampling_param = (
                isinstance(sampling_params, dict) or len(sampling_params) == 1
            )

            # Exclude internal key
            if isinstance(sampling_params, dict):
                sampling_params = {
                    k: v for k, v in sampling_params.items() if k != "type"
                }
            else:
                sampling_params = [
                    {k: v for k, v in sp.items() if k != "type"}
                    for sp in sampling_params
                ]

            if (spec is None and is_single_sampling_param) or (
                spec is not None and spec["stages"] == ["diffusion"]
            ):
                # Diffusion format: extra_body directly contains sampling params.
                # Validation should have taken care of matching sampling params' types.
                # * Use this mode if the model is a simple one-stage diffusion model.
                # * Fallback to this mode if model is not registered and a single sampling param is provided.
                sampling_params = (
                    sampling_params
                    if isinstance(sampling_params, dict)
                    else sampling_params[0]
                )
                combined_extra_body: dict[str, Any] = {**sampling_params}
                if "n" in combined_extra_body:
                    combined_extra_body["num_outputs_per_prompt"] = combined_extra_body[
                        "n"
                    ]
                    del combined_extra_body["n"]
            else:
                # Use AR style payload, extra_body has a sampling_params_list field
                combined_extra_body: dict[str, Any] = {
                    "sampling_params_list": sampling_params
                }

        if negative_prompt:
            combined_extra_body["negative_prompt"] = negative_prompt

        if extra_body:
            combined_extra_body.update(extra_body)

        payload: dict[str, Any] = {"messages": messages}
        if combined_extra_body:
            payload["extra_body"] = combined_extra_body
        if modalities:
            payload["modalities"] = modalities

        spec, _ = lookup_model_spec(model)
        if spec:
            preprocessor = spec.get("payload_preprocessor", None)
            if preprocessor is not None:
                payload = preprocessor(payload)

        return payload
