# Qwen-Image-Edit-2511 performance audit

## Measurement protocol

- Hardware: 4× NVIDIA L20X (143,771 MiB each, NV18 between GPUs); runs were restricted to GPU 0 or GPUs 0–1.
- Environment: repository `./.venv`, PyTorch 2.11.0+cu130, vLLM 0.25.0.
- Workload: `Qwen/Qwen-Image-Edit-2511`, random dataset, two 512×512 white input images, 1536×1536 output, 20 steps, positive prompt `Random prompt N for benchmarking diffusion models`, negative prompt `Negative prompt N for benchmarking diffusion models`, concurrency 1.
- Performance benchmark: one two-step warmup followed by three measured requests. `benchmark_2gpu.json` is authoritative.
- Diagnostic profiles: one exact-shape warm request followed by one profiled request with operator shapes enabled and stack collection disabled. A separate single-card host-stack trace was captured after the optimization. Profile latency is diagnostic only.
- Quality check: the same workload with seed 42, one measured output before and after the change. `run_quality.sh` is authoritative for correctness.

The pytest runner only recognizes a fixed hardware-mark enum, so the local config uses an H100 collection mark even though result metadata and `nvidia-smi` correctly identify the machine as L20X.

## Baseline

All durations below are milliseconds.

| Configuration | Client mean | Server generation | Text encoder | VAE encode | Diffuse | VAE decode | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single card | 33,795.9 | 32,664.2 | 140.4 | 85.9 | 32,020.7 | 161.5 | 67,790 |
| Ulysses 2 + VAE PP 2 | 20,864.4 | 19,862.0 | 129.0 | 386.8 | 18,433.8 | 642.5 | 56,174 |
| CFG 2 + VAE PP 2 | **18,533.8** | **17,466.9** | 122.3 | 370.2 | **16,062.9** | 620.2 | 57,178 |

CFG parallel is the best of the two fixed dual-card choices: 11.2% lower client latency than Ulysses and 45.2% lower than single-card. VAE patch parallel costs roughly 740–900 ms across encode/decode versus single-card, but the DiT speedup dominates.

## Diagnostic findings

- The single-card trace had 32,183.8 ms of summed CUDA kernel time:
  - FlashAttention: 14,317.9 ms (44.5%).
  - Two dominant GEMM families: 13,451.1 ms combined (41.8%).
  - RoPE application: 808.7 ms (2.5%).
- Ulysses reduced attention/GEMM work per rank but added 9,600 all-to-all kernels:
  - rank 0: 1,241 ms NCCL send/recv;
  - rank 1: 1,883 ms NCCL send/recv;
  - GPU idle was 13.5%/13.3%, versus 3.7% on single-card.
- CFG parallel used only one DiT branch per rank. Rank 1 spent 465 ms in output all-gather kernels while waiting for the slower branch; rank 0's corresponding kernels were about 1 ms. This is primarily branch synchronization, not bandwidth saturation.
- No anomalous `cudaMalloc` block appeared. Profiled client latency was 0.6% above baseline for single-card, 4.9% for Ulysses, and 7.0% for CFG, so no rerun was required.
- Host-stack tracing attributes the two in-request 17–21 ms GPU gaps to Qwen-VL processor/tokenization work. Larger 263–386 ms gaps belong to profiler RPC status/dequeue orchestration, not the model pipeline.

Perfetto SQL summaries are in `perfetto_summary.json`; timeline gap reports are under each `profile_*` directory.

## Implemented optimization

`ModulateIndexPrepare` rebuilt a 17,408-element Python list, converted it to a CPU tensor, and copied it from pageable memory to GPU on every DiT branch invocation. The value depends only on image shapes and device.

The change adds a bounded 16-entry on-device cache keyed by shape and device. It does not modify RMSNorm or any model arithmetic.

Perfetto confirms the optimized single-card trace removed exactly 40 calls each to `aten::lift_fresh`, `aten::_to_copy`, and `aten::copy_`. Inclusive `_to_copy` time fell from 13,930.9 ms to 7,075.4 ms; most of that inclusive time is waiting on earlier stream work rather than copy-kernel execution.

## A/B result

| Configuration | Baseline mean | Cached mean | Delta | Delta % | Baseline diffuse | Cached diffuse |
|---|---:|---:|---:|---:|---:|---:|
| Single card | 33,795.9 | 33,646.6 | -149.3 | -0.44% | 32,020.7 | 32,043.5 |
| Ulysses 2 + VAE PP 2 | 20,864.4 | 20,838.4 | -26.0 | -0.12% | 18,433.8 | 18,400.1 |
| CFG 2 + VAE PP 2 | 18,533.8 | **18,167.3** | **-366.5** | **-1.98%** | 16,062.9 | 16,022.0 |

The win is clearest for the target CFG configuration. Its server generation time improved by 102.1 ms (0.58%); the larger client-side change includes frontend/response variance. Single-card and Ulysses gains are marginal at this sample size.

Peak memory was unchanged within sampling noise.

## Quality gate

The uncached and cached seed-42 outputs are byte-identical:

- SHA-256 (both): `c610d11084173808fae0559a60f8c34e1bceea011decce1aaacda2f3877fb09f`
- max absolute pixel error: 0
- MAE: 0
- MSE: 0
- PSNR: infinity

## Seed-idea assessment

| Idea | Finding |
|---|---|
| Repeated resizing | Applicable but low ROI. Preprocess resizes condition images to 384×384, then Qwen-VL aligns them to 392×392. Removing an interpolation changes pixels and needs a visual gate; processor work is only tens of milliseconds. |
| Move preprocess images to GPU | The request preprocessor runs outside the model worker. Creating frontend CUDA tensors would add a CUDA context and IPC/lifetime complexity. The model worker already transfers tensors to its local GPU; not recommended. |
| Keep text encoder/VAE on GPU | Already done: both are moved to `self.device` during pipeline initialization. |
| Load weights directly onto GPU | Potential startup-only optimization. Current initialization loads then calls `.to(self.device)`. It does not affect warm request latency and needs a separate startup-memory/time study across multi-worker loading. |

## Next candidates

| Priority | Layer | Candidate | Evidence | Expected benefit | Risk / validation |
|---|---|---|---|---|---|
| P0 | Measurement | Interleaved CFG A/B with 5–10 repeats | Client gain exceeds server-stage gain | Confirm the ~2% result | Low; same fixed command and seed |
| P1 | Text/media | Reuse Qwen-VL image preprocessing or visual features across positive/negative prompts | Two 17–21 ms processor gaps; identical images are encoded twice | Small, likely <1% E2E | Medium; multimodal placeholder and output-equivalence tests |
| P1 | Parallel communication | Investigate CFG branch imbalance before changing collectives | Rank 1 waits ~465 ms in all-gather | Up to a few percent if imbalance is removable | Medium; all-rank trace and exact output |
| P1 | VAE | Reduce VAE patch split/gather/tiling overhead | Dual-card VAE adds 740–900 ms | Potentially material | Medium; image integrity and memory checks |
| P2 | Attention | Backend/layout or approximate attention work | FlashAttention is 44.5% of single-card CUDA time | Potentially large | High; strict quality gate required |

RMSNorm is intentionally excluded from all candidates.

## Commands

```bash
source ./.venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 \
DIFFUSION_BENCHMARK_DIR=$PWD/perf_qwen_image_edit_2511/results \
pytest -s tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file perf_qwen_image_edit_2511/benchmark_2gpu.json -m diffusion

perf_qwen_image_edit_2511/run_profile.sh single shapes
perf_qwen_image_edit_2511/run_profile.sh ulysses2 shapes
perf_qwen_image_edit_2511/run_profile.sh cfg2 shapes
perf_qwen_image_edit_2511/run_profile.sh single stacks

perf_qwen_image_edit_2511/run_quality.sh quality_check

pytest -q tests/diffusion/models/qwen_image/test_qwen_image_modulate_index.py
pre-commit run --files \
  vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py \
  tests/diffusion/models/qwen_image/test_qwen_image_modulate_index.py
```
