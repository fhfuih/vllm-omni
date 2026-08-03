# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for preparing PEFT LoRA adapters used by accuracy e2e tests."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def convert_diffusers_lora_safetensors_to_peft(
    src_safetensors: Path,
    out_dir: Path,
    *,
    lora_alpha: int | None = None,
) -> Path:
    """Convert a Diffusers single-file LoRA (``*.lora.down/up.weight``) to PEFT.

    Writes ``adapter_model.safetensors`` and ``adapter_config.json`` under ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict = {}
    targets: set[str] = set()
    rank: int | None = None

    with safe_open(str(src_safetensors), framework="pt") as handle:
        for key in handle.keys():
            match = re.match(r"^(?P<mod>.+)\.lora\.(?P<which>down|up)\.weight$", key)
            if match is None:
                raise ValueError(f"Unexpected Diffusers LoRA key (expected *.lora.down/up.weight): {key}")
            mod = match.group("mod")
            which = match.group("which")
            ab = "lora_A" if which == "down" else "lora_B"
            tensor = handle.get_tensor(key)
            tensors[f"base_model.model.{mod}.{ab}.weight"] = tensor.contiguous()
            leaf = mod.rsplit(".", 1)[-1]
            if leaf == "0":
                leaf = ".".join(mod.rsplit(".", 2)[-2:])
            targets.add(leaf)
            if which == "down":
                rank = int(tensor.shape[0])

    if rank is None:
        raise ValueError(f"No LoRA down weights found in {src_safetensors}")

    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    alpha = int(lora_alpha) if lora_alpha is not None else rank
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": sorted(targets),
        "bias": "none",
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return out_dir


def resolve_qwen_image_lora_path(
    *,
    repo_root: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Resolve a PEFT LoRA directory for Qwen-Image accuracy tests.

    Search order:
    1. ``QWEN_IMAGE_LORA_PATH`` env (must already be PEFT)
    2. ``<repo>/flymy_realism`` if it already contains PEFT files
    3. Convert ``<repo>/flymy_realism/flymy_realism.safetensors`` into a cache dir
    4. Download Diffusers LoRA from HuggingFace (``QWEN_IMAGE_LORA_REPO`` /
       default ``flymy-ai/qwen-image-realism-lora``) and convert
    """
    env_path = os.environ.get("QWEN_IMAGE_LORA_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        _require_peft(path)
        return path

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    local_dir = repo_root / "flymy_realism"
    if (local_dir / "adapter_model.safetensors").is_file() and (local_dir / "adapter_config.json").is_file():
        return local_dir.resolve()

    if cache_dir is None:
        cache_dir = Path(os.environ.get("QWEN_IMAGE_LORA_CACHE", repo_root / ".lora_cache" / "flymy_realism_peft"))
    cache_dir = cache_dir.resolve()
    if (cache_dir / "adapter_model.safetensors").is_file() and (cache_dir / "adapter_config.json").is_file():
        return cache_dir

    local_src = local_dir / "flymy_realism.safetensors"
    if local_src.is_file():
        return convert_diffusers_lora_safetensors_to_peft(local_src, cache_dir)

    repo_id = os.environ.get("QWEN_IMAGE_LORA_REPO", "flymy-ai/qwen-image-realism-lora")
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:  # pragma: no cover
        raise FileNotFoundError(
            "Neither a local PEFT LoRA nor huggingface_hub is available to prepare Qwen-Image LoRA."
        ) from exc

    files = list_repo_files(repo_id)
    peft_ready = "adapter_model.safetensors" in files and "adapter_config.json" in files
    if peft_ready:
        snapshot_dir = cache_dir / "hub_peft"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            hf_hub_download(repo_id=repo_id, filename=name, local_dir=str(snapshot_dir))
        _require_peft(snapshot_dir)
        return snapshot_dir

    # Prefer common Diffusers single-file names.
    candidates = [
        name
        for name in files
        if name.endswith(".safetensors") and ("lora" in name.lower() or name == "pytorch_lora_weights.safetensors")
    ]
    if not candidates:
        candidates = [name for name in files if name.endswith(".safetensors")]
    if not candidates:
        raise FileNotFoundError(f"No LoRA safetensors found in HuggingFace repo {repo_id}")

    weight_name = sorted(candidates, key=lambda n: (0 if "flymy" in n.lower() else 1, len(n)))[0]
    downloaded = Path(hf_hub_download(repo_id=repo_id, filename=weight_name))
    return convert_diffusers_lora_safetensors_to_peft(downloaded, cache_dir)


def _require_peft(path: Path) -> None:
    if not (path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing PEFT weights: {path / 'adapter_model.safetensors'}")
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing PEFT config: {path / 'adapter_config.json'}")
