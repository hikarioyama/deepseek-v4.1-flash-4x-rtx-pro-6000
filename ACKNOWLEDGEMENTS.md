# Acknowledgements

This recipe is a deployment and measurement layer around an inference engine that
other people wrote. Everyone below is credited for the part they actually
contributed. The exact modifications this repository makes to that engine are listed
separately in [ENGINE-CHANGES.md](ENGINE-CHANGES.md); pinned revisions are in
[README §8 Provenance](README.md#8-provenance) and [STUDY-PROTOCOL.md](STUDY-PROTOCOL.md).

## Authorship of this recipe

- The recipe, the launcher, the storage adapter and every measurement in this
  repository were written by the upstream authors of
  [sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000)
  (committed as `seroxdesign` / `0xSero`). `LICENSE`, `LICENSE.sglang` and the
  published results are theirs, and their repository remains the source of truth
  for the recipe.
- This fork is **documentation-only** at publication time: `ACKNOWLEDGEMENTS.md`,
  `ENGINE-CHANGES.md`, `NOTICE` and README §8. It does not alter, relicense or
  re-measure upstream code. Where this file says "we", it means the recipe's
  authors as recorded in that upstream history.

## The inference engine

- **[SGLang](https://github.com/sgl-project/sglang)** (the `lmsysorg/sglang` image,
  digest `sha256:c4ca6511…`) — the scheduler, the DSpark speculative verification
  path, the `dsv4` attention backend, the decode CUDA-graph runner, the RadixAttention
  pool and the OpenAI-compatible server. Every number in this repository was produced
  by SGLang. `runtime/flash_mla_sm120.py` is derived from SGLang's own SM120 FlashMLA
  module and stays Apache-2.0 ([LICENSE.sglang](LICENSE.sglang)). Credit for the SM120
  Triton decode path, the FP8 paged-KV footer layout and the FlashInfer fallback in
  that file belongs to the SGLang contributors, not to us.
- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — the SM120 sparse-MLA
  paged-attention kernel this recipe's sparse prefill calls, and the `flashinfer_mxfp4`
  MoE runner the launcher selects. Without it there is no sparse decode on this GPU.
- **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)** — the grouped FP8/FP4 GEMMs
  for MoE and the SM120 split-128 planner initialization. Our adapter explicitly
  preserves DeepGEMM's planner path for ratio-1/2 indexers rather than routing around
  it.
- **[vLLM](https://github.com/vllm-project/vllm)** — the FP8 block-quantization
  lineage: SGLang's `fp8.py` carries an "Adapted from vllm-project/vllm" notice, and
  that code is on the hot path of this recipe. We never ran vLLM here, which is
  exactly why the credit has to be explicit.
- **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**
  (revision `1bae0e1088993b9b82de08f59de5be5a34e42170`) — the model, the native FP8
  Engram tables, and the upstream inference code SGLang executes for it (`engram.py`,
  `deepseek_v4.py`, `dspark_*`). Weights are downloaded at runtime under their own
  license and are **not** redistributed here.
- **PyTorch** and the **NVIDIA CUDA** toolchain, driver, container runtime and the
  RTX PRO 6000 Blackwell (SM120) target — everything below the engine, including the
  `cudaLaunchHostFunc` contract that makes the Engram callback graph-safe.

## Local Inference Lab

[Local Inference Lab](https://github.com/local-inference-lab) published the study
discipline and the storage-reader design this work is measured against. Thank you.

- **[b12x](https://github.com/local-inference-lab/b12x)** (pinned `01ac763`) — its
  `AGENTS.md` shaped the working rules here: correctness before timings, exercise the
  real production path, keep the workspace fixed, name the exact hardware and
  artifact, bound every experiment. Its PLE reader
  ([`b12x/loader/_ple_reader.c`](https://github.com/local-inference-lab/b12x/blob/01ac763bef7503d934c4c2874bfd005799ca24a1/b12x/loader/_ple_reader.c))
  is the reference model for the native 256-byte FP8 row plus 8-byte E8M0 scale
  layout our storage adapter is verified against. Repository license: Apache-2.0.
  **No B12x source is redistributed in this repository** and its attention kernels
  are not in the default image: `adapter/row_store.cpp` is an independent `pread`
  implementation, and the io_uring reader bridge is a bounded candidate whose
  receipts are `results/b12x-uring-*.json`. Those are our measurements of that
  candidate, not B12x results, and not an endorsement of them.
- **[local-inference-lab/sglang](https://github.com/local-inference-lab/sglang)**
  (pinned `c662a9fd`) — the performance, benchmarking and profiler skills that guided
  this study, retained under Apache-2.0. Skills distributed in a fork are not authored
  exclusively by the fork owner, so credit for them belongs to the SGLang community as
  well. The skills are not redistributed here.
- **[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)**
  (pinned `ccd9ad8`, MIT) — the measurement discipline behind the README: separate
  sustained, burst and end-to-end throughput; verify the concurrency actually
  achieved; inspect repetition; compare quality in pairs. The distinction between
  allocated and populated KV capacity, and between a shared decode window and
  per-request rates, comes from following that guidance.
- **[rtx6kpro](https://github.com/local-inference-lab/rtx6kpro)** (pinned `b0fd406`)
  — its KLD evaluation skill was reviewed and deliberately **not** applied: it is
  Qwen3.5-specific and its thresholds do not transfer to DeepSeek. Listing it is a
  thank-you for publishing it, not a claim that we used it.

## Other recipes on the same road

- **[DeepSpark](https://github.com/brandonmmusic-max/deepspark)** — related work in
  the same single-machine, memory-constrained space. We credit the direction; none of
  its kernels are integrated into the default image.
- **[deepseek-v4-flash-0731-sm120](https://github.com/jacklarmer/deepseek-v4-flash-0731-sm120)**
  — an earlier DeepSeek SM120 recipe that showed this class of deployment was
  reachable before we measured it ourselves. Our configuration, numbers and failures
  are our own.

## What this repository says about itself

Every change we make to the engine is itemized in
[ENGINE-CHANGES.md](ENGINE-CHANGES.md), including which upstream file it touches and
why. If that ledger and the code ever disagree, the ledger is wrong and we want to
know.

## What we still owe back

- The SM120 sparse-decode page-splitting fix in `runtime/flash_mla_sm120.py` should go
  to SGLang as a patch instead of staying a fork-only edit.
- The `PagedIndexerMetadata` guard fix in `adapter/sitecustomize.py` is a one-line
  upstream bug workaround; it belongs in SGLang.
- The storage parity harness in `tests/` exists because the B12x reader design made
  the required checks obvious. If it is useful beyond this rig, it belongs upstream
  too.

## Independence

This repository is not affiliated with, sponsored by, or endorsed by SGLang, vLLM,
FlashInfer, DeepGEMM, PyTorch, NVIDIA, DeepSeek, Local Inference Lab, DeepSpark or the
earlier DS4 SM120 recipe, and this fork is not endorsed by the upstream recipe's
author either. It records one operator's measured attempt on one machine. Naming
these projects is a citation, not a claim of approval, and no upstream maintainer is
responsible for any number or limitation published here.
