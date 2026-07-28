import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import build_image_to_image_prompt, get_model_class_name
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Qwen-Image-Edit-2511 torch profile.")
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--cfg-parallel-size", type=int, default=1)
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1)
    parser.add_argument("--vae-use-tiling", action="store_true")
    return parser.parse_args()


def make_sampling_params(steps: int) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(0),
        true_cfg_scale=4.0,
        guidance_scale=1.0,
        num_inference_steps=steps,
        height=1536,
        width=1536,
    )


def main() -> None:
    args = parse_args()
    profile_dir = Path(args.profile_dir)
    summary_file = Path(args.summary_file)
    profile_dir.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    image_paths = [
        "/tmp/diffusion_benchmark_random_image_0.png",
        "/tmp/diffusion_benchmark_random_image_1.png",
    ]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    prompt = "Random prompt 0 for benchmarking diffusion models"
    negative_prompt = "Negative prompt 0 for benchmarking diffusion models"

    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(profile_dir),
        "torch_profiler_record_shapes": False,
        "torch_profiler_with_stack": True,
        "torch_profiler_with_memory": False,
        "torch_profiler_use_gzip": True,
    }

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
    )
    omni = Omni(
        model="Qwen/Qwen-Image-Edit-2511",
        parallel_config=parallel_config,
        vae_use_tiling=args.vae_use_tiling,
        enable_diffusion_pipeline_profiler=True,
        profiler_config=profiler_config,
    )
    model_class_name = get_model_class_name(omni)
    prompt_dict = build_image_to_image_prompt(
        model_class_name=model_class_name,
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=images,
        height=1536,
        width=1536,
    )

    warmup_start = time.perf_counter()
    omni.generate(prompt_dict, sampling_params_list=[make_sampling_params(2)])
    warmup_ms = (time.perf_counter() - warmup_start) * 1000

    omni.start_profile()
    profiled_start = time.perf_counter()
    omni.generate(prompt_dict, sampling_params_list=[make_sampling_params(35)])
    profiled_request_ms = (time.perf_counter() - profiled_start) * 1000
    profile_results = omni.stop_profile()

    summary = {
        "model": "Qwen/Qwen-Image-Edit-2511",
        "width": 1536,
        "height": 1536,
        "num_inference_steps": 35,
        "num_input_images": 2,
        "ulysses_degree": args.ulysses_degree,
        "cfg_parallel_size": args.cfg_parallel_size,
        "vae_patch_parallel_size": args.vae_patch_parallel_size,
        "vae_use_tiling": args.vae_use_tiling,
        "warmup_ms": warmup_ms,
        "profiled_request_ms": profiled_request_ms,
        "profile_results": profile_results,
    }
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
