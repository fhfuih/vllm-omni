"""
Offline inference tests: image-to-video.

Covers:
  - Offline image-to-video (Wan2.2 TI2V-5B)

Doc-linked tests (test_<subsection>_<id>) mirror every executable snippet in
docs/user_guide/examples/offline_inference/image_to_video.md.
"""

from tests.examples.conftest import I2V_SCRIPT, assert_video_valid, run_script

# ---------------------------------------------------------------------------
# Doc-linked tests: docs/user_guide/examples/offline_inference/image_to_video.md
# ---------------------------------------------------------------------------

# --- ### Wan2.2-I2V-A14B-Diffusers (MoE) ---


def test_wan2_2_i2v_a14b_diffusers_moe_1(model_prefix, output_dir, horse):
    """CLI snippet: Wan2.2-I2V-A14B-Diffusers MoE model."""
    out = output_dir / "doc-i2v-wan2_2_i2v_a14b_diffusers_moe_1.mp4"
    run_script(
        I2V_SCRIPT,
        "--model", f"{model_prefix}Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "--image", horse,
        "--prompt", "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion",
        "--negative-prompt", "",
        "--height", "480",
        "--width", "832",
        "--num-frames", "48",
        "--guidance-scale", "5.0",
        "--guidance-scale-high", "6.0",
        "--num-inference-steps", "40",
        "--boundary-ratio", "0.875",
        "--flow-shift", "12.0",
        "--fps", "16",
        "--output", str(out),
    )  # fmt: skip
    assert_video_valid(out, width=832, height=480, num_frames=48)


# --- ### Wan2.2-TI2V-5B-Diffusers (Unified) ---


def test_wan2_2_ti2v_5b_diffusers_unified_1(model_prefix, output_dir, horse):
    """CLI snippet: Wan2.2-TI2V-5B-Diffusers unified model."""
    out = output_dir / "doc-i2v-wan2_2_ti2v_5b_diffusers_unified_1.mp4"
    run_script(
        I2V_SCRIPT,
        "--model", f"{model_prefix}Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "--image", horse,
        "--prompt", "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion",
        "--negative-prompt", "",
        "--height", "480",
        "--width", "832",
        "--num-frames", "48",
        "--guidance-scale", "4.0",
        "--num-inference-steps", "40",
        "--flow-shift", "12.0",
        "--fps", "16",
        "--output", str(out),
    )  # fmt: skip
    assert_video_valid(out, width=832, height=480, num_frames=48)
