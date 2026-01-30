"""Image/tensor format helpers.

The image generation part is derived from dougbtv/comfyui-vllm-omni by Doug (@dougbtv).
Original source at https://github.com/dougbtv/comfyui-vllm-omni, distributed under the MIT License.
"""

import base64
from io import BytesIO
import mimetypes

import numpy as np
import torch
from PIL import Image


def base64_to_image_tensor(base64_str: str, mode: str = "RGB") -> torch.Tensor:
    """
    Convert base64-encoded image to ComfyUI image tensor.

    Args:
        base64_str: Base64-encoded image string
        mode: PIL image mode (default RGB for transparency support)

    Returns:
        torch.Tensor with shape (1, H, W, C) in float32 [0, 1] range

    Raises:
        ValueError: If base64 string is invalid or image cannot be decoded
    """
    if base64_str.startswith("data:image"):
        _, base64_str = base64_str.split(",", 1)

    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {e}")

    # Create BytesIO object for PIL
    image_bytesio = BytesIO(image_bytes)

    # Open with PIL and convert to desired mode
    try:
        pil_image = Image.open(image_bytesio)
        pil_image = pil_image.convert(mode)
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {e}")

    image_array = np.asarray(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)
    return image_tensor


def image_tensor_to_png_bytes(tensor: torch.Tensor, filename: str = "image.png") -> BytesIO:
    """
    Convert ComfyUI image tensor to PNG BytesIO for multipart upload.

    This function converts a ComfyUI IMAGE tensor to a PNG-encoded BytesIO object
    suitable for multipart/form-data upload. The BytesIO object has its .name
    attribute set, which is required by aiohttp for file uploads.

    Args:
        tensor: ComfyUI IMAGE tensor with shape (B, H, W, C), dtype float32, range [0, 1]
        filename: Name attribute to set on BytesIO (default: "image.png")

    Returns:
        BytesIO object containing PNG-encoded image with .name attribute set

    Raises:
        ValueError: If tensor format is invalid (not 4D, wrong dtype, etc.)
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor with shape (B, H, W, C), got {tensor.ndim}D tensor")

    image_tensor = tensor[0]  # Shape: (H, W, C)
    image_np = (image_tensor.cpu().numpy() * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(image_np)

    # Save to BytesIO as image file
    img_bytes = BytesIO()
    # Set name attribute (required for multipart upload and mimetype detection)
    img_bytes.name = filename
    try:
        pil_image.save(img_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to save image as file: {e}")

    # Reset position to beginning
    img_bytes.seek(0)

    return img_bytes


def image_tensor_to_base64(tensor: torch.Tensor, filename: str = "image.png") -> str:
    """
    Convert ComfyUI image tensor to base64-encoded image string.

    Args:
        tensor: ComfyUI IMAGE tensor with shape (B, H, W, C), dtype float32, range [0, 1]
        filename: Name attribute to set on BytesIO (default: "image.png")
        format: File format of the output image file buffer (default: "PNG")

    Returns:
        Base64-encoded image string

    Raises:
        ValueError: If tensor format is invalid (not 4D, wrong dtype, etc.)
    """
    img_bytes = image_tensor_to_png_bytes(tensor, filename)
    img_bytes.seek(0)
    byte_data = img_bytes.read()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return f"data:{mime_type};base64,{base64_str}"
