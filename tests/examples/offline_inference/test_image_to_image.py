"""
Offline inference tests: image-to-image.

Covers:
  - Offline image editing (single-image, multi-image, layered)
  - Offline BAGEL 2-stage img2img
"""

import pytest

from tests.examples.conftest import (
    I2I_SCRIPT,
    assert_image_valid,
    run_script_with_successful_return,
)

pytestmark = [pytest.mark.advanced_model, pytest.mark.example]

# ---------------------------------------------------------------------------
# Doc-linked tests: docs/user_guide/examples/offline_inference/image_to_image.md
# ---------------------------------------------------------------------------

# --- ### Single Image Editing ---


def test_single_image_editing_1(model_prefix, output_dir, bear):
    """CLI snippet: single-image editing with Qwen/Qwen-Image-Edit."""
    out = output_dir / "doc-i2i-single_image_editing_1.png"
    run_script_with_successful_return(
        I2I_SCRIPT,
        "--model", f"{model_prefix}Qwen/Qwen-Image-Edit",
        "--image", bear,
        "--prompt", "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'",
        "--output", str(out),
        "--num-inference-steps", "50",
        "--cfg-scale", "4.0",
    )  # fmt: skip
    assert_image_valid(out)


# --- ### Multiple Image Editing (Qwen-Image-Edit-2509) ---


def test_multiple_image_editing_qwen_image_edit_2509_1(model_prefix, output_dir, bear, horse):
    """CLI snippet: multi-image editing with Qwen/Qwen-Image-Edit-2509."""
    out = output_dir / "doc-i2i-multiple_image_editing_qwen_image_edit_2509_1.png"
    run_script_with_successful_return(
        I2I_SCRIPT,
        "--model", f"{model_prefix}Qwen/Qwen-Image-Edit-2509",
        "--image", bear, horse,
        "--prompt", "Combine these images into a single scene",
        "--output", str(out),
        "--num-inference-steps", "50",
        "--cfg-scale", "4.0",
        "--guidance-scale", "1.0",
    )  # fmt: skip
    assert_image_valid(out)
