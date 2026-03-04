"""
Offline inference tests: text-to-audio.

Covers:
  - Offline text-to-audio  (Stable Audio Open 1.0)

Doc-linked tests (test_<subsection>_<id>) mirror every executable snippet in
docs/user_guide/examples/offline_inference/text_to_audio.md.
"""

from tests.example.conftest import T2A_SCRIPT, assert_audio_valid, run_script

# ---------------------------------------------------------------------------
# Doc-linked tests: docs/user_guide/examples/offline_inference/text_to_audio.md
# ---------------------------------------------------------------------------

# --- ## Local CLI Usage ---


def test_local_cli_usage_1(model_prefix, output_dir):
    """CLI snippet from doc: stable-audio-open-1.0."""
    out = output_dir / "doc-t2a-local_cli_usage_1.wav"
    run_script(
        T2A_SCRIPT,
        "--model", f"{model_prefix}stabilityai/stable-audio-open-1.0",
        "--prompt", "The sound of a hammer hitting a wooden surface",
        "--negative-prompt", "Low quality",
        "--seed", "42",
        "--guidance-scale", "7.0",
        "--audio-length", "10.0",
        "--num-inference-steps", "100",
        "--output", str(out),
    )  # fmt: skip
    assert_audio_valid(out, sample_rate=44100, channels=2, duration_s=10.0)
