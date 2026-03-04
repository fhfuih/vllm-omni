"""
Offline inference tests: text-to-video.

Covers:
  - Offline text-to-video  (Wan2.2 T2V-A14B)

Doc-linked tests (test_<subsection>_<id>) mirror every executable snippet in
docs/user_guide/examples/offline_inference/text_to_video.md.
"""

from tests.example.conftest import T2V_SCRIPT, assert_video_valid, run_script

# ---------------------------------------------------------------------------
# Doc-linked tests: docs/user_guide/examples/offline_inference/text_to_video.md
# ---------------------------------------------------------------------------

# --- ## Local CLI Usage ---


def test_local_cli_usage_1(model_prefix, output_dir):
    """CLI snippet from doc: Wan2.2-T2V-A14B-Diffusers, width=832, num-frames=33."""
    out = output_dir / "doc-t2v-local_cli_usage_1.mp4"
    run_script(
        T2V_SCRIPT,
        "--model", f"{model_prefix}Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "--prompt", "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
        "--negative-prompt", "",
        "--height", "480",
        "--width", "832",
        "--num-frames", "33",
        "--guidance-scale", "4.0",
        "--guidance-scale-high", "3.0",
        "--flow-shift", "12.0",
        "--num-inference-steps", "40",
        "--fps", "16",
        "--output", str(out),
    )  # fmt: skip
    assert_video_valid(out, width=832, height=480, num_frames=33)
