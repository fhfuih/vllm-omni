import base64
import gc
import io
from pathlib import Path

import pytest
import requests
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import export_to_video  # pyright: ignore[reportPrivateImportUsage]
from PIL import Image

from benchmarks.accuracy.common import pil_to_base64
from tests.e2e.accuracy.helpers import assert_similarity, model_output_dir
from tests.e2e.accuracy.wan22_i2v.test_wan22_i2v_video_similarity import (
    _parse_psnr_score,
    _parse_ssim_score,
    _run_ffmpeg_similarity,
)
from tests.helpers.env import run_post_test_cleanup, run_pre_test_cleanup
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OpenAIClientHandler

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


PROMPT = "A photo of a cat sitting on a laptop keyboard, digital art style."
NEGATIVE_PROMPT = "blurry, low quality"
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 20
TRUE_CFG_SCALE = 4.0
SEED = 42
SSIM_THRESHOLD = 0.97
PSNR_THRESHOLD = 30.0

VIDEO_PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
VIDEO_WIDTH = 832
VIDEO_HEIGHT = 480
VIDEO_NUM_INFERENCE_STEPS = 2
NUM_FRAMES = 8
FPS = 8
GUIDANCE_SCALE = 4.0
GUIDANCE_SCALE_2 = 1.0
VIDEO_SSIM_THRESHOLD = 0.92
VIDEO_PSNR_THRESHOLD = 26.0


def _run_vllm_omni_wan22_i2v(
    *,
    model: str,
    output_path: Path,
    conditioning_image: Image.Image,
) -> Path:
    server_args = [
        "--num-gpus",
        "1",
        "--stage-init-timeout",
        "400",
        "--init-timeout",
        "900",
        "--diffusion-load-format",
        "diffusers",
    ]
    form_data = {
        "prompt": VIDEO_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": VIDEO_HEIGHT,
        "width": VIDEO_WIDTH,
        "num_inference_steps": VIDEO_NUM_INFERENCE_STEPS,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_scale_2": GUIDANCE_SCALE_2,
        "seed": SEED,
    }
    with OmniServer(model, server_args, use_omni=True) as omni_server:
        client = OpenAIClientHandler(
            host=omni_server.host,
            port=omni_server.port,
            run_level="full_model",
        )
        request_config = {
            "model": omni_server.model,
            "form_data": form_data,
            "image_reference": f"data:image/png;base64,{pil_to_base64(conditioning_image, 'png')}",
        }
        result = client.send_video_diffusion_request(request_config)[0]
        video_bytes = result.videos[0]  # pyright: ignore[reportOptionalSubscript] # Guaranteed not None
        output_path.write_bytes(video_bytes)
        return output_path


def _run_diffusers_wan22_i2v(*, model: str, output_path: Path, conditioning_image: Image.Image) -> Path:
    from diffusers import WanImageToVideoPipeline  # pyright: ignore[reportPrivateImportUsage]

    run_pre_test_cleanup(enable_force=True)
    pipe: WanImageToVideoPipeline | None = None
    try:
        pipe = WanImageToVideoPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        pipe.to("cuda")

        generator = torch.Generator(device="cuda").manual_seed(SEED)
        result = pipe(
            image=conditioning_image,
            prompt=VIDEO_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=VIDEO_HEIGHT,
            width=VIDEO_WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_frames=NUM_FRAMES,
            guidance_scale=GUIDANCE_SCALE,
            guidance_scale_2=GUIDANCE_SCALE_2,
            generator=generator,
        )
        frames = result.frames[0]  # pyright: ignore[reportAttributeAccessIssue]
        export_to_video(frames, str(output_path), fps=FPS)  # pyright: ignore[reportArgumentType]
        return output_path
    finally:
        if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_post_test_cleanup(enable_force=True)


def _run_vllm_omni_qwen_image(*, model: str, output_path: Path) -> Image.Image:
    server_args = [
        "--num-gpus",
        "1",
        "--stage-init-timeout",
        "400",
        "--init-timeout",
        "900",
        "--diffusion-load-format",
        "diffusers",
    ]
    with OmniServer(model, server_args, use_omni=True) as omni_server:
        response = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/images/generations",
            json={
                "model": omni_server.model,
                "prompt": PROMPT,
                "size": f"{WIDTH}x{HEIGHT}",
                "n": 1,
                "response_format": "b64_json",
                "negative_prompt": NEGATIVE_PROMPT,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "true_cfg_scale": TRUE_CFG_SCALE,
                "seed": SEED,
            },
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        assert len(payload["data"]) == 1
        image_bytes = base64.b64decode(payload["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
        image.save(output_path)
        return image


def _run_diffusers_qwen_image(*, model: str, output_path: Path) -> Image.Image:
    run_pre_test_cleanup(enable_force=True)
    pipe: DiffusionPipeline | None = None
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        result = pipe(  # pyright: ignore[reportCallIssue]
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            generator=generator,
        )
        output_image = result.images[0].convert("RGB")
        output_image.save(output_path)
        return output_image
    finally:
        if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_post_test_cleanup(enable_force=True)


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen-Image"])
def test_diffusers_backend_matches_diffusers(model_id: str, accuracy_artifact_root: Path) -> None:
    output_dir = model_output_dir(accuracy_artifact_root, model_id + "-diffusers-backend")

    vllm_output = _run_vllm_omni_qwen_image(model=model_id, output_path=output_dir / "vllm_omni.png")
    diffusers_output = _run_diffusers_qwen_image(model=model_id, output_path=output_dir / "diffusers.png")

    assert_similarity(
        model_name=model_id,
        vllm_image=vllm_output,
        diffusers_image=diffusers_output,
        width=WIDTH,
        height=HEIGHT,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("model_id", ["Wan-AI/Wan2.2-I2V-A14B-Diffusers"])
def test_diffusers_backend_wan22_i2v_matches_diffusers(
    model_id: str,
    accuracy_artifact_root: Path,
    qwen_bear_image: Image.Image,
) -> None:
    output_dir = model_output_dir(accuracy_artifact_root, model_id + "-diffusers-backend")

    resized_image = qwen_bear_image.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)

    vllm_path = _run_vllm_omni_wan22_i2v(
        model=model_id,
        output_path=output_dir / "vllm_omni.mp4",
        conditioning_image=resized_image,
    )
    diffusers_path = _run_diffusers_wan22_i2v(
        model=model_id,
        output_path=output_dir / "diffusers.mp4",
        conditioning_image=resized_image,
    )

    ssim_output = _run_ffmpeg_similarity("ssim", vllm_path, diffusers_path)
    psnr_output = _run_ffmpeg_similarity("psnr", vllm_path, diffusers_path)
    ssim_score = _parse_ssim_score(ssim_output)
    psnr_score = _parse_psnr_score(psnr_output)
    print(f"{model_id} similarity metrics:")
    print(f"  SSIM: value={ssim_score:.6f}, threshold>={VIDEO_SSIM_THRESHOLD:.6f}, range=[-1, 1], higher_is_better")
    print(
        f"  PSNR: value={psnr_score:.6f} dB, threshold>={VIDEO_PSNR_THRESHOLD:.6f} dB, range=[0, +inf), higher_is_better"
    )

    assert ssim_score >= VIDEO_SSIM_THRESHOLD, (
        f"SSIM below threshold for {model_id}: got {ssim_score:.6f}, expected >= {VIDEO_SSIM_THRESHOLD:.6f}."
    )
    assert psnr_score >= VIDEO_PSNR_THRESHOLD, (
        f"PSNR below threshold for {model_id}: got {psnr_score:.6f}, expected >= {VIDEO_PSNR_THRESHOLD:.6f}."
    )
