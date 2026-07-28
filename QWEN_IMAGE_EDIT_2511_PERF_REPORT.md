# Qwen-Image-Edit-2511 Performance Analysis

## Scope

- Model: `Qwen/Qwen-Image-Edit-2511`
- Workload: `1536x1536`, two input images, positive prompt `Random prompt 0/1/2 for benchmarking diffusion models`, negative prompt `Negative prompt 0/1/2 for benchmarking diffusion models`, 35 inference steps, concurrency 1.
- Benchmark protocol: one 2-step warmup request, then 3 measured requests through `tests/dfx/perf/scripts/run_diffusion_benchmark.py`.
- Torch profile protocol: one 2-step warmup outside profiler, then one 35-step request with torch profiler enabled. Profiler timings are diagnostic only.
- Hardware observed by benchmark filename: `L20X`.

## Commands

Baseline benchmark:

```bash
source ./.venv/bin/activate
export DIFFUSION_BENCHMARK_DIR=perf_results/qwen_image_edit_2511_head
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
export CACHE_DIT_VERSION=1.3.0
pytest -s -v tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file qwen_image_edit_2511_perf_analysis.json
```

VAE patch ablation:

```bash
source ./.venv/bin/activate
export DIFFUSION_BENCHMARK_DIR=perf_results/qwen_image_edit_2511_vae_patch_ablation_head
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
export CACHE_DIT_VERSION=1.3.0
pytest -s -v tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file qwen_image_edit_2511_vae_patch_ablation.json
```

Torch profile helper:

```bash
source ./.venv/bin/activate
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
export CACHE_DIT_VERSION=1.3.0

python qwen_image_edit_2511_profile_once.py \
  --profile-dir perf_results/qwen_image_edit_2511_head/profiles/single_offline \
  --summary-file perf_results/qwen_image_edit_2511_head/profile_requests/single_offline/summary.json

python qwen_image_edit_2511_profile_once.py \
  --profile-dir perf_results/qwen_image_edit_2511_head/profiles/ulysses2_cfg2_vae_patch4_offline \
  --summary-file perf_results/qwen_image_edit_2511_head/profile_requests/ulysses2_cfg2_vae_patch4_offline/summary.json \
  --ulysses-degree 2 \
  --cfg-parallel-size 2 \
  --vae-patch-parallel-size 4 \
  --vae-use-tiling
```

Trace analyzer:

```bash
python .claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py \
  perf_results/qwen_image_edit_2511_head/profiles/single_offline/20260729-013915_stage_0_rep_0_diffusion_1785260355/trace_rank0.json.gz \
  --min-gap-ms 5 --topn 20

python .claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py \
  perf_results/qwen_image_edit_2511_head/profiles/ulysses2_cfg2_vae_patch4_offline/20260729-014524_stage_0_rep_0_diffusion_1785260724/trace_rank0.json.gz \
  perf_results/qwen_image_edit_2511_head/profiles/ulysses2_cfg2_vae_patch4_offline/20260729-014524_stage_0_rep_0_diffusion_1785260724/trace_rank1.json.gz \
  perf_results/qwen_image_edit_2511_head/profiles/ulysses2_cfg2_vae_patch4_offline/20260729-014524_stage_0_rep_0_diffusion_1785260724/trace_rank2.json.gz \
  perf_results/qwen_image_edit_2511_head/profiles/ulysses2_cfg2_vae_patch4_offline/20260729-014524_stage_0_rep_0_diffusion_1785260724/trace_rank3.json.gz \
  --min-gap-ms 5 --topn 20
```

## E2E Benchmark Results

All latency values are milliseconds.

| Config | Mean latency | Diffuse | Text encoder | VAE encode | VAE decode | Stage gen | Peak memory MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single card | 56,709.9 | 55,091.5 | 123.1 | 85.9 | 162.8 | 55,701.7 | 67,792 |
| CFG parallel 2 only | 29,120.1 | 27,658.6 | 129.4 | 85.8 | 162.0 | 28,300.3 | 67,868 |
| Ulysses 2 only | 33,098.6 | 31,462.4 | 122.7 | 86.0 | 163.2 | 32,089.0 | 67,868 |
| CFG 2 + Ulysses 2 + VAE patch 4 | 18,317.3 | 15,758.4 | 189.8 | 382.7 | 622.4 | 17,635.1 | 56,362 |
| CFG 2 + Ulysses 2, no VAE patch | 17,409.4 | 15,749.5 | 192.0 | 85.9 | 161.0 | 16,585.2 | 67,910 |

## Findings

Single-card pipeline overhead has near-zero practical room. The request is dominated by DiT diffuse: `55,091.5 ms` of `56,709.9 ms` mean latency. Text encoder plus VAE encode/decode is only `371.8 ms`, or about `0.66%` of e2e latency. Even deleting all VAE/text overhead would not move a 56.7s request meaningfully.

CFG parallel is the best measured 2-card strategy for this workload. `CFG=2` reduces mean latency to `29,120.1 ms`, while `Ulysses=2` gives `33,098.6 ms`. This implies the CFG branch split is more valuable than sequence parallel alone for the 1536 image-edit token shape.

The full 4-card strategy is useful, but VAE patch parallel is a net loss for this image workload. With `CFG=2 + Ulysses=2 + VAE patch=4`, mean latency is `18,317.3 ms`. Disabling VAE patch parallel while keeping `CFG=2 + Ulysses=2` improves mean latency to `17,409.4 ms`, a `907.9 ms` improvement (`4.96%` relative). The diffuse stage is unchanged (`15,758.4 ms` vs `15,749.5 ms`), while VAE encode/decode drops from `1,005.1 ms` to `246.9 ms`.

Torch trace supports the same conclusion. Single-card rank 0 has `2.21%` GPU idle, `54,882 ms` busy time out of a `56,124 ms` GPU span, and the largest GPU operators are FlashAttention (`24,475 ms`) and GEMM (`13,509 ms` + `9,501 ms`). There is no large host/runtime bubble to remove.

For 4-card `CFG=2 + Ulysses=2 + VAE patch=4`, all ranks show about `15.75%` to `16.36%` idle. NCCL all-to-all device work is about `1,238 ms` to `1,369 ms` per rank, and the VAE/tile all-gather region appears as about `421 ms` user annotation with `368 ms` to `385 ms` GPU annotation. The all-to-all cost is the expected Ulysses communication tax; the VAE all-gather/broadcast cost is avoidable by disabling VAE patch parallel for this image workload.

## Optimization Outcome

Applied optimization attempt: configuration-level change, do not use `--vae-patch-parallel-size 4 --vae-use-tiling` for this workload. This is not a source-code patch because the trace shows the code-level candidate would be a risky distributed VAE contract change for less than 1 second of e2e upside, while the config ablation achieves the gain immediately.

Result: `18,317.3 ms` to `17,409.4 ms`, a `907.9 ms` mean latency improvement (`4.96%`) for the 4-card parallel setting.

Recommended serving config for 4 cards:

```bash
--cfg-parallel-size 2 --ulysses-degree 2
```

Avoid for this workload unless memory pressure requires it:

```bash
--vae-patch-parallel-size 4 --vae-use-tiling
```

## Why No Code Patch

The remaining large costs are model math and intended communication:

- Single-card FlashAttention plus GEMM accounts for about `47,485 ms` of GPU kernel time in the trace, while GPU idle is only `1,242 ms`.
- VAE/text work is sub-`400 ms` on single card and sub-`250 ms` in the no-VAE-patch 4-card config.
- RMSNorm/custom CUDA changes were explicitly out of scope and are not the bottleneck relative to attention/GEMM.
- A distributed VAE change to avoid final broadcast/all-gather would need output ownership changes across ranks. The ablation proves the safer answer is not to invoke VAE patch parallel for this image shape.

## Artifacts

- Baseline result JSON: `perf_results/qwen_image_edit_2511_head/diffusion_result_qwen_image_edit_2511_perf_analysis_L20X_20260729-011347.json`
- VAE patch ablation result JSON: `perf_results/qwen_image_edit_2511_vae_patch_ablation_head/diffusion_result_qwen_image_edit_2511_vae_patch_ablation_L20X_20260729-015345.json`
- Single-card trace analyzer: `perf_results/qwen_image_edit_2511_head/analysis/single_rank0_trace_analyzer.txt`
- 4-card trace analyzer: `perf_results/qwen_image_edit_2511_head/analysis/ulysses2_cfg2_vae_patch4_all_ranks_trace_analyzer.txt`
- Profile summaries:
  - `perf_results/qwen_image_edit_2511_head/profile_requests/single_offline/summary.json`
  - `perf_results/qwen_image_edit_2511_head/profile_requests/ulysses2_cfg2_vae_patch4_offline/summary.json`

