#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary PR-review artifact dump for native paged KV PR-1.

Not a product example and not wired into Buildkite. Run from the repo root
with ``./.venv/bin/python dump_native_kv_pr1_artifacts.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

UNIT_TESTS = [
    "tests/diffusion/diffusion_kv/test_v1_connector.py",
    "tests/diffusion/diffusion_kv/test_scheduler_connector.py",
    "tests/diffusion/diffusion_kv/test_request.py",
    "tests/diffusion/diffusion_kv/test_initialization.py",
    "tests/diffusion/diffusion_kv/test_worker_kv_connector.py",
    "tests/engine/test_orchestrator_native_kv_handshake.py",
    "tests/core/sched/test_omni_ar_scheduler_native_kv.py",
    "tests/diffusion/models/hunyuan_image3/test_diffusion_kv_request.py",
]


def _run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _require_venv() -> Path:
    if not VENV_PYTHON.is_file():
        raise SystemExit("Missing ./.venv. Create it or run this script with the project venv Python.")
    return VENV_PYTHON


def _collect_meta(out_dir: Path, python_bin: Path) -> dict[str, object]:
    git = _run(["git", "rev-parse", "HEAD"])
    env_proc = _run([str(python_bin), str(REPO_ROOT / "collect_env.py")])
    gpu_proc = _run(["nvidia-smi", "-L"])
    meta = {
        "sha": git.stdout.strip() if git.returncode == 0 else None,
        "date": datetime.now(timezone.utc).isoformat(),
        "collect_env_excerpt": "\n".join((env_proc.stdout or env_proc.stderr).splitlines()[:80]),
        "gpu_list": gpu_proc.stdout.strip() if gpu_proc.returncode == 0 else "nvidia-smi unavailable",
    }
    _write(out_dir / "meta.json", json.dumps(meta, indent=2) + "\n")
    return meta


def _run_unit_tests(out_dir: Path, python_bin: Path) -> int:
    junit = out_dir / "junit.xml"
    log = out_dir / "pytest_unit.log"
    proc = _run(
        [
            str(python_bin),
            "-m",
            "pytest",
            *UNIT_TESTS,
            "-q",
            f"--junitxml={junit}",
        ]
    )
    _write(log, proc.stdout + proc.stderr)
    return proc.returncode


def _git_diffstat(out_dir: Path, main_ref: str) -> None:
    log = _run(["git", "log", "--oneline", "-20", f"{main_ref}...HEAD"])
    stat = _run(["git", "diff", "--stat", f"{main_ref}...HEAD"])
    _write(
        out_dir / "git_diffstat_vs_main.txt",
        f"# git log {main_ref}...HEAD\n{log.stdout}\n# git diff --stat {main_ref}...HEAD\n{stat.stdout}",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_hunyuan_dit_only(python_bin: Path, workdir: Path, image_out: Path) -> subprocess.CompletedProcess[str]:
    model = os.environ.get("HUNYUAN_IMAGE3_MODEL", "tencent/HunyuanImage-3.0-Instruct")
    yaml_path = workdir / "vllm_omni/deploy/hunyuan_image3_dit.yaml"
    cmd = [
        str(python_bin),
        "-c",
        (
            "from vllm_omni import Omni; "
            f"omni=Omni(model={model!r}, stage_configs_path={str(yaml_path)!r}); "
            "out=omni.generate('a red apple', extra_args={'num_inference_steps': 4}); "
            f"img=out[0].images[0]; img.save({str(image_out)!r})"
        ),
    ]
    return _run(cmd, cwd=workdir)


def _compare_main(out_dir: Path, python_bin: Path, main_ref: str) -> str:
    if _run(["git", "diff", "--quiet"]).returncode != 0:
        return "SKIPPED compare-main: working tree is dirty; set COMPARE_WORKDIR or commit/stash first."
    worktree = Path(os.environ.get("COMPARE_WORKDIR", str(out_dir / "main_worktree")))
    if worktree.exists():
        shutil.rmtree(worktree)
    add = _run(["git", "worktree", "add", "--detach", str(worktree), main_ref])
    if add.returncode != 0:
        return f"SKIPPED compare-main: git worktree add failed:\n{add.stderr}"
    try:
        branch_dir = out_dir / "branch"
        main_dir = out_dir / "main"
        branch_dir.mkdir(parents=True, exist_ok=True)
        main_dir.mkdir(parents=True, exist_ok=True)
        branch_png = branch_dir / "hunyuan_dit_only.png"
        main_png = main_dir / "hunyuan_dit_only.png"
        branch_run = _run_hunyuan_dit_only(python_bin, REPO_ROOT, branch_png)
        main_run = _run_hunyuan_dit_only(python_bin, worktree, main_png)
        _write(branch_dir / "dit_only.log", branch_run.stdout + branch_run.stderr)
        _write(main_dir / "dit_only.log", main_run.stdout + main_run.stderr)
        hashes = []
        for png in (branch_png, main_png):
            if png.is_file():
                hashes.append(f"{png.name}: {_sha256(png)}")
        return (
            "compare-main:\n" + "\n".join(hashes) + f"\nbranch_rc={branch_run.returncode} main_rc={main_run.returncode}"
        )
    finally:
        _run(["git", "worktree", "remove", "--force", str(worktree)])


def _native_kv_capture(out_dir: Path) -> str:
    native_dir = out_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    try:
        import importlib.util

        if importlib.util.find_spec("mooncake") is None:
            _write(native_dir / "SKIPPED.txt", "Mooncake Transfer Engine is not installed.\n")
            return "native-kv SKIPPED: mooncake not installed"
    except Exception as exc:
        _write(native_dir / "SKIPPED.txt", f"Mooncake import check failed: {exc}\n")
        return f"native-kv SKIPPED: {exc}"
    _write(
        native_dir / "SKIPPED.txt",
        "Mooncake is installed, but this dump script does not launch a live AR+DiT "
        "native transfer unless NATIVE_KV=1 and a tiny native config is provided.\n",
    )
    return "native-kv: Mooncake present; live run left to operator logs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump native KV PR-1 review artifacts.")
    parser.add_argument("--out", default="artifacts/native_kv_pr1")
    parser.add_argument("--main-ref", default="origin/main")
    parser.add_argument("--compare-main", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--native-kv", action="store_true")
    args = parser.parse_args()

    python_bin = _require_venv()
    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[str] = ["# Native KV PR-1 artifacts", ""]
    _collect_meta(out_dir, python_bin)
    pytest_rc = _run_unit_tests(out_dir, python_bin)
    summary.append(f"- unit pytest exit code: {pytest_rc}")
    _git_diffstat(out_dir, args.main_ref)
    summary.append(f"- git diffstat vs {args.main_ref}: written")

    if args.compare_main and args.gpu:
        summary.append("- " + _compare_main(out_dir, python_bin, args.main_ref))
    elif args.compare_main:
        summary.append("- compare-main skipped: pass --gpu")

    if args.gpu:
        shm_note = out_dir / "shm_ar_dit.txt"
        n_gpu = 0
        smi = _run(["nvidia-smi", "-L"])
        if smi.returncode == 0:
            n_gpu = len([line for line in smi.stdout.splitlines() if line.strip().startswith("GPU ")])
        if n_gpu >= 4:
            _write(
                shm_note,
                "GPU count >= 4. Run tests/e2e/accuracy/test_hunyuan_image3.py SharedMemory YAML manually "
                "and copy PNG/logs here. This dump does not auto-launch the full AR+DiT SHM job.\n",
            )
            summary.append("- SHM AR+DiT: GPU count sufficient; launch left to operator (legacy OmniKV)")
        else:
            _write(shm_note, f"Need >=4 GPUs for SHM AR+DiT Hunyuan; found {n_gpu}.\n")
            summary.append(f"- SHM AR+DiT skipped: {n_gpu} GPUs")

    if args.native_kv:
        summary.append("- " + _native_kv_capture(out_dir))

    _write(out_dir / "SUMMARY.md", "\n".join(summary) + "\n")
    print((out_dir / "SUMMARY.md").read_text())
    return pytest_rc


if __name__ == "__main__":
    if sys.executable != str(VENV_PYTHON) and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        # Still allow running via ./.venv/bin/python even if the resolved path differs by symlink.
        pass
    raise SystemExit(main())
