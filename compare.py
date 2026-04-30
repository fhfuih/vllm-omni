#!/usr/bin/env python3
"""
Compare QWEN_EDIT_DEBUG_DIR artifacts dumped by:
- vllm_omni debug hooks (prefix: vllm_, vllm_api_)
- diffusers debug hooks (prefix: diffusers_)

Design goals:
1) Compare fields in pipeline-flow order.
2) Cover all debug fields added in the instrumentation.
3) Flag missing, mismatched, and unexpected artifacts.

Usage:
  python compare_qwen_edit_debug.py --debug-dir /path/to/QWEN_EDIT_DEBUG_DIR
  python compare_qwen_edit_debug.py --debug-dir /path --atol 1e-3 --rtol 1e-3
"""

from __future__ import annotations

import argparse
import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

Side = Literal["both", "vllm", "diffusers"]
Kind = Literal["tensor", "structured", "both"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    side: Side
    kind: Kind
    wildcard: bool = False
    note: str = ""


# Ordered by pipeline/debug flow.
FLOW_SPECS: list[FieldSpec] = [
    # ---- API ingress (vLLM-only) ----
    FieldSpec("edit_request_inputs", "vllm", "structured"),
    FieldSpec("edit_request_model_capabilities", "vllm", "structured"),
    FieldSpec("edit_request_loaded_images", "vllm", "structured"),
    # ---- Pipeline init / call entry ----
    FieldSpec("transformer_config", "vllm", "structured"),
    FieldSpec("call_inputs", "diffusers", "structured"),
    # ---- Preprocess ----
    FieldSpec("preprocess_raw_images", "vllm", "structured"),
    FieldSpec("preprocess_raw_image_summaries", "vllm", "structured"),
    FieldSpec("preprocess_outputs", "both", "structured"),
    FieldSpec("preprocess_vae_images", "both", "tensor"),
    FieldSpec("forward_preprocessed_inputs", "vllm", "structured"),
    # ---- Prompt encode (positive) ----
    FieldSpec("prompt_text_inputs", "both", "structured"),
    FieldSpec("prompt_input_ids", "both", "tensor"),
    FieldSpec("prompt_attention_mask", "both", "tensor"),
    FieldSpec("prompt_pixel_values", "both", "tensor"),
    FieldSpec("prompt_image_grid_thw", "both", "tensor"),
    FieldSpec("prompt_embeds", "both", "tensor"),
    FieldSpec("prompt_embeds_mask", "both", "tensor"),
    # ---- Prompt encode (negative; present when CFG/negative prompt enabled) ----
    FieldSpec("negative_prompt_text_inputs", "both", "structured"),
    FieldSpec("negative_prompt_input_ids", "both", "tensor"),
    FieldSpec("negative_prompt_attention_mask", "both", "tensor"),
    FieldSpec("negative_prompt_pixel_values", "both", "tensor"),
    FieldSpec("negative_prompt_image_grid_thw", "both", "tensor"),
    FieldSpec("negative_prompt_embeds", "both", "tensor"),
    FieldSpec("negative_prompt_embeds_mask", "both", "tensor"),
    # ---- Latent prep ----
    FieldSpec("prepare_latents_image_", "both", "tensor", wildcard=True),
    FieldSpec("prepare_latents_noise", "both", "tensor"),
    FieldSpec("prepare_latents_image_concat", "both", "tensor"),
    # ---- Forward before denoise ----
    FieldSpec("forward_denoise_inputs", "both", "structured"),
    FieldSpec("forward_prompt_embeds", "both", "tensor"),
    FieldSpec("forward_negative_prompt_embeds", "both", "tensor"),
    FieldSpec("forward_latents_initial", "both", "tensor"),
    FieldSpec("forward_image_latents", "both", "tensor"),
    # ---- Denoise step 0 ----
    FieldSpec("diffuse_step0_inputs", "both", "structured"),
    FieldSpec("diffuse_step0_latent_model_input", "both", "tensor"),
    FieldSpec("diffuse_step0_positive_noise_pred", "both", "tensor"),
    FieldSpec("diffuse_step0_negative_noise_pred", "both", "tensor"),
    FieldSpec("diffuse_step0_cfg_noise_pred", "diffusers", "tensor"),
    FieldSpec("diffuse_step0_noise_pred", "vllm", "tensor"),  # combined/noise_pred in vLLM mixin
    FieldSpec("diffuse_step0_latents_after_scheduler", "both", "tensor"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--debug-dir", type=Path, required=True)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--strict-structured", action="store_true", help="Fail on any structured mismatch.")
    p.add_argument("--allow-missing-negative", action="store_true", default=True)
    return p.parse_args()


def load_txt(path: Path) -> Any:
    text = path.read_text().strip()
    if text == "":
        return text
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    t = t.detach().float().cpu()
    if t.numel() == 0:
        return {"shape": tuple(t.shape), "dtype": str(t.dtype), "numel": 0}
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "numel": int(t.numel()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
    }


def compare_tensors(a: Any, b: Any, atol: float, rtol: float) -> tuple[bool, str]:
    # Supports Tensor OR list[Tensor]
    if torch.is_tensor(a) and torch.is_tensor(b):
        ta = a.detach().float().cpu()
        tb = b.detach().float().cpu()
        if ta.shape != tb.shape:
            return False, f"shape mismatch {tuple(ta.shape)} vs {tuple(tb.shape)}"
        diff = (ta - tb).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
        ok = torch.allclose(ta, tb, atol=atol, rtol=rtol)
        return ok, f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}"

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, f"list length mismatch {len(a)} vs {len(b)}"
        for i, (ai, bi) in enumerate(zip(a, b)):
            if not (torch.is_tensor(ai) and torch.is_tensor(bi)):
                return False, f"list[{i}] is non-tensor"
            ok, msg = compare_tensors(ai, bi, atol=atol, rtol=rtol)
            if not ok:
                return False, f"list[{i}] {msg}"
        return True, f"list[{len(a)}] allclose"

    return False, f"type mismatch {type(a).__name__} vs {type(b).__name__}"


def deep_equal(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(deep_equal(a[k], b[k]) for k in a.keys())
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return a == b


def collect_names(debug_dir: Path) -> dict[str, set[str]]:
    names = {"vllm": set(), "diffusers": set(), "vllm_api": set()}
    for p in debug_dir.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        if stem.startswith("vllm_api_"):
            names["vllm_api"].add(stem[len("vllm_api_") :])
        elif stem.startswith("vllm_"):
            names["vllm"].add(stem[len("vllm_") :])
        elif stem.startswith("diffusers_"):
            names["diffusers"].add(stem[len("diffusers_") :])
    return names


def read_artifact(debug_dir: Path, prefix: str, name: str, kind: Kind) -> Any | None:
    pt = debug_dir / f"{prefix}_{name}.pt"
    txt = debug_dir / f"{prefix}_{name}.txt"

    if kind in ("tensor", "both") and pt.exists():
        return torch.load(pt, map_location="cpu")
    if kind in ("structured", "both") and txt.exists():
        return load_txt(txt)
    if pt.exists():
        return torch.load(pt, map_location="cpu")
    if txt.exists():
        return load_txt(txt)
    return None


def expand_wildcard_names(spec: FieldSpec, names: dict[str, set[str]]) -> list[str]:
    assert spec.wildcard
    if spec.side == "both":
        s = sorted(n for n in (names["vllm"] | names["diffusers"]) if n.startswith(spec.name))
        return s
    if spec.side == "vllm":
        key = "vllm_api" if spec.name.startswith("edit_request_") else "vllm"
        return sorted(n for n in names[key] if n.startswith(spec.name))
    return sorted(n for n in names["diffusers"] if n.startswith(spec.name))


def main() -> None:
    args = parse_args()
    debug_dir: Path = args.debug_dir
    if not debug_dir.exists():
        raise SystemExit(f"Debug dir not found: {debug_dir}")

    names = collect_names(debug_dir)

    print(f"[info] debug_dir={debug_dir}")
    print(
        f"[info] found vllm={len(names['vllm'])},diffusers={len(names['diffusers'])},vllm_api={len(names['vllm_api'])}"
    )

    covered = {"vllm": set(), "diffusers": set(), "vllm_api": set()}
    mismatches = 0
    missing = 0

    def mark(side_key: str, n: str) -> None:
        covered[side_key].add(n)

    print("\n=== Ordered Flow Comparison ===")
    for spec in FLOW_SPECS:
        field_names = expand_wildcard_names(spec, names) if spec.wildcard else [spec.name]

        for fname in field_names:
            if spec.side == "vllm":
                key = "vllm_api" if fname.startswith("edit_request_") else "vllm"
                v = read_artifact(debug_dir, "vllm_api" if key == "vllm_api" else "vllm", fname, spec.kind)
                mark(key, fname)
                if v is None:
                    # negative prompt artifacts can be absent depending on config
                    if args.allow_missing_negative and fname.startswith("negative_prompt_"):
                        print(f"[skip] {fname}: vllm missing (allowed negative-prompt optional)")
                    else:
                        print(f"[MISS] {fname}: vllm missing")
                        missing += 1
                else:
                    print(f"[ok]   {fname}: vllm-only present")
                continue

            if spec.side == "diffusers":
                d = read_artifact(debug_dir, "diffusers", fname, spec.kind)
                mark("diffusers", fname)
                if d is None:
                    print(f"[MISS] {fname}: diffusers missing")
                    missing += 1
                else:
                    print(f"[ok]   {fname}: diffusers-only present")
                continue

            # both
            v = read_artifact(debug_dir, "vllm", fname, spec.kind)
            d = read_artifact(debug_dir, "diffusers", fname, spec.kind)
            mark("vllm", fname)
            mark("diffusers", fname)

            if v is None or d is None:
                # some negative artifacts may be optional
                if args.allow_missing_negative and fname.startswith("negative_prompt_"):
                    print(f"[skip] {fname}: missing on one side (allowed negative-prompt optional)")
                else:
                    print(
                        f"[MISS] {fname}: "
                        f"{'vllm missing' if v is None else ''} "
                        f"{'diffusers missing' if d is None else ''}".strip()
                    )
                    missing += 1
                continue

            if spec.kind == "tensor":
                ok, msg = compare_tensors(v, d, args.atol, args.rtol)
                if ok:
                    print(f"[ok]   {fname}: {msg}")
                else:
                    print(f"[DIFF] {fname}: {msg}")
                    mismatches += 1
            else:
                eq = deep_equal(v, d)
                if eq:
                    print(f"[ok]   {fname}: structured equal")
                else:
                    print(f"[DIFF] {fname}: structured mismatch")
                    if args.strict_structured:
                        mismatches += 1

    # Coverage / unexpected check
    all_seen_vllm = names["vllm"]
    all_seen_diffusers = names["diffusers"]
    all_seen_vllm_api = names["vllm_api"]

    extra_vllm = sorted(all_seen_vllm - covered["vllm"])
    extra_diffusers = sorted(all_seen_diffusers - covered["diffusers"])
    extra_vllm_api = sorted(all_seen_vllm_api - covered["vllm_api"])

    print("\n=== Coverage Check ===")
    if not extra_vllm and not extra_diffusers and not extra_vllm_api:
        print("[ok]   All discovered artifacts are covered by FLOW_SPECS.")
    else:
        if extra_vllm:
            print(f"[WARN] Uncovered vllm artifacts: {extra_vllm}")
        if extra_diffusers:
            print(f"[WARN] Uncovered diffusers artifacts: {extra_diffusers}")
        if extra_vllm_api:
            print(f"[WARN] Uncovered vllm_api artifacts: {extra_vllm_api}")

    print("\n=== Summary ===")
    print(f"missing={missing}, mismatches={mismatches}")
    if missing > 0 or mismatches > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
