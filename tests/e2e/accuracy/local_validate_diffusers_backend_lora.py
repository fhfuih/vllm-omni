#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local validation for static LoRA on Qwen-Image (native vs diffusers backend).

Comparisons (same prompt/seed/size/steps):
  1. Native backend: base vs LoRA  (LoRA must change the image)
  2. Diffusers backend: base vs LoRA  (LoRA must apply in the adapter path)
  3. Native+LoRA vs Diffusers+LoRA  (backends should match closely)

Usage (from repo root, with .venv activated)::

    gpu run --nonblock -- python tests/e2e/accuracy/local_validate_diffusers_backend_lora.py

Outputs and a short REPORT.md are written under ``./local_lora_validation_outputs/``.
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LORA = REPO_ROOT / "flymy_realism"
DEFAULT_OUTPUT = REPO_ROOT / "local_lora_validation_outputs"

MODEL = "Qwen/Qwen-Image"
# Prompt includes the LoRA trigger word from the adapter card.
PROMPT = (
    "Super Realism, a photo of a cat sitting on a laptop keyboard, digital art style."
)
NEGATIVE_PROMPT = "blurry, low quality"
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 20
TRUE_CFG_SCALE = 4.0
SEED = 42
LORA_SCALE = 1.0

# Base vs LoRA should differ; backends with the same LoRA should match closely.
BASE_VS_LORA_SSIM_MAX = 0.98  # expect LoRA to move the image
NATIVE_VS_DIFFUSERS_SSIM_MIN = 0.80
NATIVE_VS_DIFFUSERS_PSNR_MIN = 22.0


@dataclass
class RunResult:
    label: str
    image: Image.Image
    path: Path
    latency_s: float


def _ffmpeg_similarity(filter_name: str, first: Path, second: Path) -> str:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(first),
            "-i",
            str(second),
            "-lavfi",
            f"[0:v][1:v]{filter_name}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stderr


def _parse_ssim(output: str) -> float:
    import re

    match = re.search(r"All:(?P<score>[0-9.]+)", output)
    if match is None:
        raise ValueError(f"Could not parse SSIM from ffmpeg output:\n{output}")
    return float(match.group("score"))


def _parse_psnr(output: str) -> float:
    import re

    match = re.search(r"average:(?P<score>[0-9.]+)", output)
    if match is None:
        raise ValueError(f"Could not parse PSNR from ffmpeg output:\n{output}")
    return float(match.group("score"))


def compare_images(a: Path, b: Path) -> tuple[float, float]:
    ssim = _parse_ssim(_ffmpeg_similarity("ssim", a, b))
    psnr = _parse_psnr(_ffmpeg_similarity("psnr", a, b))
    return ssim, psnr


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.accelerator.empty_cache()


def _run_offline(
    *,
    label: str,
    output_path: Path,
    diffusion_load_format: str | None,
    lora_path: Path | None,
) -> RunResult:
    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.lora.request import LoRARequest
    from vllm_omni.lora.utils import stable_lora_int_id
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.platforms import current_omni_platform

    kwargs: dict = {
        "model": MODEL,
    }
    if diffusion_load_format is not None:
        kwargs["diffusion_load_format"] = diffusion_load_format
    if lora_path is not None:
        kwargs["lora_path"] = str(lora_path)
        kwargs["lora_scale"] = LORA_SCALE

    print(f"\n=== [{label}] loading Omni({kwargs}) ===", flush=True)
    omni = Omni(**kwargs)
    try:
        lora_request = None
        lora_scale = 1.0
        # Diffusers backend applies static LoRA at load time and rejects request LoRA.
        # Native backend needs an explicit request to keep the adapter active.
        if lora_path is not None and diffusion_load_format != "diffusers":
            lora_request = LoRARequest(
                lora_name=lora_path.name,
                lora_int_id=stable_lora_int_id(str(lora_path)),
                lora_path=str(lora_path),
            )
            lora_scale = LORA_SCALE

        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED)
        sampling = OmniDiffusionSamplingParams(
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            seed=SEED,
            generator=generator,
            num_outputs_per_prompt=1,
            lora_request=lora_request,
            lora_scale=lora_scale,
        )
        sampling.extra_args["negative_prompt"] = NEGATIVE_PROMPT

        start = time.perf_counter()
        outputs = omni.generate(
            {"prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT},
            sampling,
        )
        latency = time.perf_counter() - start

        first = outputs[0]
        assert isinstance(first, OmniRequestOutput)
        req_out = first.request_output
        assert isinstance(req_out, OmniRequestOutput) and req_out.images
        image = req_out.images[0].convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"=== [{label}] done in {latency:.2f}s -> {output_path} ===", flush=True)
        return RunResult(label=label, image=image, path=output_path, latency_s=latency)
    finally:
        del omni
        _cleanup_cuda()


def _run_online(
    *,
    label: str,
    output_path: Path,
    diffusion_load_format: str | None,
    lora_path: Path | None,
) -> RunResult:
    """Online path via OmniServer (same request surface as CI)."""
    # Import locally so offline-only runs do not require the full test harness early.
    sys.path.insert(0, str(REPO_ROOT))
    from tests.helpers.runtime import OmniServer

    server_args = [
        "--num-gpus",
        "1",
        "--stage-init-timeout",
        "400",
        "--init-timeout",
        "900",
    ]
    if diffusion_load_format is not None:
        server_args.extend(["--diffusion-load-format", diffusion_load_format])
    if lora_path is not None:
        server_args.extend(["--lora-path", str(lora_path), "--lora-scale", str(LORA_SCALE)])

    print(f"\n=== [{label}] starting server args={server_args} ===", flush=True)
    with OmniServer(MODEL, server_args, use_omni=True) as omni_server:
        url = f"http://{omni_server.host}:{omni_server.port}/v1/images/generations"
        body: dict = {
            "model": omni_server.model,
            "prompt": PROMPT,
            "size": f"{WIDTH}x{HEIGHT}",
            "n": 1,
            "response_format": "b64_json",
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "true_cfg_scale": TRUE_CFG_SCALE,
            "seed": SEED,
        }
        # Native: request-level LoRA keeps the preloaded adapter active.
        # Diffusers: static LoRA only; do not send request LoRA.
        if lora_path is not None and diffusion_load_format != "diffusers":
            body["lora"] = {
                "name": lora_path.name,
                "local_path": str(lora_path.resolve()),
                "scale": LORA_SCALE,
            }

        start = time.perf_counter()
        response = requests.post(url, json=body, timeout=900)
        latency = time.perf_counter() - start
        response.raise_for_status()
        payload = response.json()
        image_bytes = base64.b64decode(payload["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"=== [{label}] done in {latency:.2f}s -> {output_path} ===", flush=True)
        return RunResult(label=label, image=image, path=output_path, latency_s=latency)


def _validate_lora_dir(lora_path: Path) -> None:
    if not (lora_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(
            f"Missing PEFT weights at {lora_path / 'adapter_model.safetensors'}. "
            "Convert the Diffusers LoRA first (adapter_model.safetensors + adapter_config.json)."
        )
    if not (lora_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing {lora_path / 'adapter_config.json'}")


def write_report(
    output_dir: Path,
    *,
    metrics: dict[str, dict[str, float]],
    latencies: dict[str, float],
    checks: list[tuple[str, bool, str]],
    mode: str,
) -> Path:
    report_path = output_dir / "REPORT.md"
    lines = [
        "## Local validation: Diffusers-backend static LoRA (Qwen-Image)",
        "",
        f"- Model: `{MODEL}`",
        f"- LoRA: `flymy_realism` (PEFT: `adapter_model.safetensors` + `adapter_config.json`)",
        f"- Prompt: `{PROMPT}`",
        f"- Size: {WIDTH}x{HEIGHT}, steps={NUM_INFERENCE_STEPS}, seed={SEED}, "
        f"true_cfg_scale={TRUE_CFG_SCALE}, lora_scale={LORA_SCALE}",
        f"- Runner mode: `{mode}`",
        "",
        "### Images",
        "",
        "| Case | File |",
        "| --- | --- |",
    ]
    for name in (
        "native_base.png",
        "native_lora.png",
        "diffusers_base.png",
        "diffusers_lora.png",
    ):
        path = output_dir / name
        lines.append(f"| `{name}` | `{path if path.exists() else 'MISSING'}` |")

    lines.extend(
        [
            "",
            "### Similarity (ffmpeg SSIM / PSNR)",
            "",
            "| Comparison | SSIM | PSNR (dB) |",
            "| --- | ---: | ---: |",
        ]
    )
    for key, vals in metrics.items():
        lines.append(f"| {key} | {vals['ssim']:.6f} | {vals['psnr']:.4f} |")

    lines.extend(
        [
            "",
            "### Latency",
            "",
            "| Case | Seconds |",
            "| --- | ---: |",
        ]
    )
    for key, value in latencies.items():
        lines.append(f"| {key} | {value:.2f} |")

    lines.extend(["", "### Checks", ""])
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        lines.append(f"- **{status}** `{name}`: {detail}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("offline", "online"),
        default="offline",
        help="offline uses Omni.generate; online uses OmniServer + /v1/images/generations",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=(
            "native_base",
            "native_lora",
            "diffusers_base",
            "diffusers_lora",
            "all",
        ),
        default=["all"],
        help="Subset of generations to run (default: all).",
    )
    args = parser.parse_args()

    lora_path = args.lora_path.resolve()
    _validate_lora_dir(lora_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run = _run_offline if args.mode == "offline" else _run_online
    wanted = set(args.cases)
    if "all" in wanted:
        wanted = {"native_base", "native_lora", "diffusers_base", "diffusers_lora"}

    results: dict[str, RunResult] = {}
    plan = [
        ("native_base", None, None),
        ("native_lora", None, lora_path),
        ("diffusers_base", "diffusers", None),
        ("diffusers_lora", "diffusers", lora_path),
    ]

    state_path = output_dir / "partial_state.json"
    if state_path.exists():
        try:
            prior = json.loads(state_path.read_text())
            print(f"Resuming with prior state keys={list(prior.get('completed', []))}", flush=True)
        except Exception:
            prior = {}
    else:
        prior = {}

    completed = set(prior.get("completed", []))
    for label, load_format, lora in plan:
        if label not in wanted:
            continue
        out_path = output_dir / f"{label}.png"
        if label in completed and out_path.exists():
            print(f"Skipping completed {label}", flush=True)
            image = Image.open(out_path).convert("RGB")
            results[label] = RunResult(label=label, image=image, path=out_path, latency_s=0.0)
            continue
        try:
            results[label] = run(
                label=label,
                output_path=out_path,
                diffusion_load_format=load_format,
                lora_path=lora,
            )
            completed.add(label)
            state_path.write_text(json.dumps({"completed": sorted(completed)}, indent=2) + "\n")
        except Exception as exc:
            print(f"ERROR while running {label}: {exc}", flush=True)
            state_path.write_text(
                json.dumps(
                    {
                        "completed": sorted(completed),
                        "failed": label,
                        "error": str(exc),
                        "remaining": sorted(wanted - completed),
                    },
                    indent=2,
                )
                + "\n"
            )
            raise

    metrics: dict[str, dict[str, float]] = {}
    checks: list[tuple[str, bool, str]] = []
    # Always include any on-disk images for similarity comparisons, even when
    # only a subset of cases was regenerated in this invocation.
    for label, _, _ in plan:
        out_path = output_dir / f"{label}.png"
        if label not in results and out_path.exists():
            results[label] = RunResult(
                label=label,
                image=Image.open(out_path).convert("RGB"),
                path=out_path,
                latency_s=0.0,
            )
    latencies = {k: v.latency_s for k, v in results.items()}

    def _pair(name: str, left: str, right: str) -> None:
        if left not in results or right not in results:
            checks.append((name, False, f"missing outputs ({left}/{right})"))
            return
        ssim, psnr = compare_images(results[left].path, results[right].path)
        metrics[name] = {"ssim": ssim, "psnr": psnr}
        print(f"{name}: SSIM={ssim:.6f} PSNR={psnr:.4f} dB", flush=True)

    _pair("native_base_vs_lora", "native_base", "native_lora")
    _pair("diffusers_base_vs_lora", "diffusers_base", "diffusers_lora")
    _pair("native_lora_vs_diffusers_lora", "native_lora", "diffusers_lora")

    if "native_base_vs_lora" in metrics:
        ssim = metrics["native_base_vs_lora"]["ssim"]
        ok = ssim < BASE_VS_LORA_SSIM_MAX
        checks.append(
            (
                "native LoRA is effective",
                ok,
                f"SSIM(base,lora)={ssim:.6f} (expect < {BASE_VS_LORA_SSIM_MAX})",
            )
        )
    if "diffusers_base_vs_lora" in metrics:
        ssim = metrics["diffusers_base_vs_lora"]["ssim"]
        ok = ssim < BASE_VS_LORA_SSIM_MAX
        checks.append(
            (
                "diffusers-backend LoRA is effective",
                ok,
                f"SSIM(base,lora)={ssim:.6f} (expect < {BASE_VS_LORA_SSIM_MAX})",
            )
        )
    if "native_lora_vs_diffusers_lora" in metrics:
        ssim = metrics["native_lora_vs_diffusers_lora"]["ssim"]
        psnr = metrics["native_lora_vs_diffusers_lora"]["psnr"]
        ok = ssim >= NATIVE_VS_DIFFUSERS_SSIM_MIN and psnr >= NATIVE_VS_DIFFUSERS_PSNR_MIN
        checks.append(
            (
                "native vs diffusers LoRA similarity",
                ok,
                f"SSIM={ssim:.6f} (>= {NATIVE_VS_DIFFUSERS_SSIM_MIN}), "
                f"PSNR={psnr:.4f} (>= {NATIVE_VS_DIFFUSERS_PSNR_MIN})",
            )
        )

    report = write_report(
        output_dir,
        metrics=metrics,
        latencies=latencies,
        checks=checks,
        mode=args.mode,
    )
    print(f"\nWrote report: {report}", flush=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "latencies": latencies, "checks": checks}, indent=2) + "\n"
    )

    if any(not ok for _, ok, _ in checks):
        return 1
    return 0


if __name__ == "__main__":
    # Ensure repo imports resolve when launched outside pytest.
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
