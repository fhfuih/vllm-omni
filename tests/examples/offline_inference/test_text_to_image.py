"""
Offline inference tests: text-to-image.

Covers:
  - Offline text-to-image  (multiple models, parallelism variants, cache backends)
  - Offline BAGEL 2-stage text2img

Doc-linked tests (test_<subsection>_<id>) mirror every executable snippet in
docs/user_guide/examples/offline_inference/text_to_image.md.
"""

from tests.examples.conftest import (
    T2I_SCRIPT,
    assert_image_valid,
    run_script,
)

# ---------------------------------------------------------------------------
# Doc-linked tests: docs/user_guide/examples/offline_inference/text_to_image.md
# ---------------------------------------------------------------------------

# --- ## Basic Usage ---


def test_basic_usage_1(model_prefix, output_dir):
    """Snippet 1: single string prompt via Omni API."""
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=f"{model_prefix}Qwen/Qwen-Image")
    prompt = "a cup of coffee on the table"
    outputs = omni.generate(prompt)
    images = outputs[0].request_output[0].images  # pyright: ignore[reportIndexIssue,reportOptionalSubscript]
    out = output_dir / "doc-t2i-basic_usage_1.png"
    images[0].save(str(out))
    assert_image_valid(out)


def test_basic_usage_2(model_prefix, output_dir):
    """Snippet 2: list of prompts via Omni API."""
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=f"{model_prefix}Qwen/Qwen-Image")
    prompts = [
        "a cup of coffee on a table",
        "a toy dinosaur on a sandy beach",
        "a fox waking up in bed and yawning",
    ]
    outputs = omni.generate(prompts)
    for i, output in enumerate(outputs):
        out = output_dir / f"doc-t2i-basic_usage_2-{i}.jpg"
        output.request_output[0].images[0].save(str(out))  # pyright: ignore[reportIndexIssue,reportOptionalSubscript]
        assert_image_valid(out)


def test_basic_usage_3(model_prefix, output_dir):
    """Snippet 3: dict prompts with negative_prompt via Omni API."""
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=f"{model_prefix}Qwen/Qwen-Image")
    outputs = omni.generate(
        [
            {"prompt": "a cup of coffee on a table", "negative_prompt": "low resolution"},
            {"prompt": "a toy dinosaur on a sandy beach", "negative_prompt": "cinematic, realistic"},
        ]
    )
    for i, output in enumerate(outputs):
        out = output_dir / f"doc-t2i-basic_usage_3-{i}.jpg"
        output.request_output[0].images[0].save(str(out))  # pyright: ignore[reportIndexIssue,reportOptionalSubscript]
        assert_image_valid(out)


# --- ### Qwen/Tongyi Models ---


def test_qwen_tongyi_models_1(model_prefix, output_dir):
    """CLI snippet for Tongyi-MAI/Z-Image-Turbo."""
    out = output_dir / "doc-t2i-qwen_tongyi_models_1.png"
    run_script(
        T2I_SCRIPT,
        "--model", f"{model_prefix}Tongyi-MAI/Z-Image-Turbo",
        "--prompt", "a cup of coffee on the table",
        "--seed", "42",
        "--cfg-scale", "4.0",
        "--num-images-per-prompt", "1",
        "--num-inference-steps", "50",
        "--height", "1024",
        "--width", "1024",
        "--output", str(out),
    )  # fmt: skip
    assert_image_valid(out, width=1024, height=1024)


# --- ### NextStep Models ---


def test_nextstep_models_1(model_prefix, output_dir):
    """CLI snippet for stepfun-ai/NextStep-1.1 with its specific arguments."""
    out = output_dir / "doc-t2i-nextstep_models_1.png"
    run_script(
        T2I_SCRIPT,
        "--model", f"{model_prefix}stepfun-ai/NextStep-1.1",
        "--prompt", "A baby panda wearing an Iron Man mask, holding a board with 'NextStep-1' written on it",
        "--height", "512",
        "--width", "512",
        "--num-inference-steps", "28",
        "--guidance-scale", "7.5",
        "--guidance-scale-2", "1.0",
        "--cfg-schedule", "constant",
        "--output", str(out),
        "--seed", "42",
    )  # fmt: skip
    assert_image_valid(out, width=512, height=512)
