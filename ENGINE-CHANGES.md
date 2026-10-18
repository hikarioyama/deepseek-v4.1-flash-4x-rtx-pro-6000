# What this repository changes in the engine

This recipe does not fork SGLang and does not ship a patched image. Everything it
changes is either one replaced source file or an import-time hook, and it is listed
here file by file so the engine's authors can see exactly what is theirs and what is
ours. Credits are in [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

This ledger describes the recipe as published by its upstream authors (see
[Authorship](ACKNOWLEDGEMENTS.md#authorship-of-this-recipe)); this fork adds the
document and does not change the code it describes.

The upstream baseline is the pinned image
`lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5`;
SGLang-derived files here remain Apache-2.0 ([LICENSE.sglang](LICENSE.sglang)).

## 1. Replaced engine file — one file only

| Ours | Replaces (inside the image) | License |
|---|---|---|
| `runtime/flash_mla_sm120.py` | `/sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py` | Apache-2.0, same terms (see [NOTICE](NOTICE)) |

`Dockerfile` copies this file over the image's copy at the same module path. The rest
of `/sgl-workspace/sglang` is untouched on disk.

### What we changed in it

The module's SM120 decode path, FP8 paged-KV footer layout, Triton kernel and
FlashInfer fallback come from SGLang. Our additions are the sparse **prefill**
page-split:

- `_split_kv_pages_to_64()` (and the `_page_split_kernel` / `_page_mark_kernel`
  helpers) converts the 256-token / 128-token footer-format pages that the V4.1
  ratio-1/2 sources use into the 64-token pages the SM120 prefill specialization
  requires, for the primary (SWA) source and the extra source independently.
- Only **touched** source pages are copied: a per-step mark kernel flags pages that
  actually contain a referenced token, so the conversion no longer rewrites the whole
  KV pool on every step.
- The destination buffer is persistent and reused, and each source gets its own
  scratch buffer (`buffer_tag`) so converting the extra source cannot overwrite the
  SWA source's data.
- `_flash_mla_sm120_prefill()` calls the split only when the source page size differs
  from 64; a 64-token source passes through unchanged.

Why it exists: without the split, prefill on a 256-token-page SWA pool either fails
the specialization's layout assumption or pays a full-pool rewrite per step. This is
the change we most want to send upstream rather than maintain here.

## 2. Import-time hooks — no file patched on disk

`adapter/sitecustomize.py` is installed via `PYTHONPATH` and does nothing unless
`DSV41_SOURCE` is set (the launcher always sets it; a plain `sglang.launch_server`
never sees the adapter).

| Hooked module | Change |
|---|---|
| `sglang.srt.layers.engram` | `EngramEmbedding.__init__` and `EngramEmbedding._owned_rows` are replaced by the file-backed exact reader in `adapter/engram_backend.py` (see §3). Hash, gating, projections, TP all-reduce and the model around it are unchanged. |
| `sglang.srt.layers.attention.dsv4.metadata` | `PagedIndexerMetadata.__post_init__` additionally sets `force_deep_gemm_metadata = True` when the engine is on SM120 and `compress_ratio in (1, 2)`. The upstream guard misses the V4.1 ratio-1/2 indexers, which always call the FP4 DeepGEMM kernel and need its split-128 planner on SM120. This is a one-line workaround for an upstream condition, not a new kernel. |

Both hooks are additive wrappers: the upstream module is imported and executed
normally first, then patched. Nothing is monkey-patched when the adapter is disabled.

## 3. Engram storage adapter — new code, not modified upstream code

| Ours | Upstream behaviour it stands in for |
|---|---|
| `adapter/engram_backend.py` | `EngramEmbedding`'s owned-row gather of the native FP8 Engram tables |
| `adapter/row_store.cpp` | the same gather, but from NVMe |

Both are new files (MIT, see [LICENSE](LICENSE)) — they replace a call, not a file:

- The tables are **not** materialised in memory: the engine's loader sees parameters
  with zero elements and a validating `weight_loader`, so checkpoint loading never
  copies the full 189 GiB of rows.
- On a cache miss the adapter reads the original 256-byte FP8 row and 8-byte E8M0
  scale bytes directly from the shard with `pread`, into a bounded direct-mapped DDR5
  cache (budget from `DSV41_CACHE_GIB`).
- The fetch runs in a `cudaLaunchHostFunc` host node, so lookups execute inside
  decode CUDA graphs. The callback makes no CUDA API calls, which is what keeps it
  graph-safe; D2H ids and H2D rows are copied on the stream around it.
- Model semantics are preserved on purpose: hashing, gating, projection and the TP
  all-reduce are the engine's, and the returned rows are the checkpoint's original
  bytes.

## 4. Launcher and harness — no engine change

- `boot.py` only passes CLI flags to `sglang.launch_server` and verifies the
  checkpoint. Nothing in it modifies engine behaviour beyond configuration. The flags
  that matter for reproducing the measurements are in
  [README §3](README.md#3-configuration); the non-default ones are
  `--min-free-slots-delay 1` (keeps the eighth slot usable),
  `--cuda-graph-max-bs-decode` pinned to the running-request limit,
  `--speculative-dspark-block-size 5`, `--enable-decoder-swa-bounded-replay`,
  `--attention-backend dsv4` and `--moe-runner-backend flashinfer_mxfp4`.
- `benchmarks/`, `tests/`, `compose*.yaml` and `.github/workflows/` are measurement,
  parity and delivery code. They do not enter the engine's address space.

## 5. What we deliberately did not change

Model weights and the published mixed precision (FP4 routed experts, FP8/BF16
components, FP8 Engram); KV cache dtype (FP8 throughout); the DSpark algorithm
itself; SGLang's scheduler, RadixAttention or sampling; and the FlashInfer,
DeepGEMM and Triton kernels the engine calls. No kernel from B12x, DeepSpark or any
other recipe is integrated into the default image.

## 6. How the changes are verified

- `tests/test_row_store.py` plus [.github/workflows/cpu-tests.yml](.github/workflows/cpu-tests.yml)
  — NVMe direct-I/O path, cache evictions, concurrent callers, locked-RAM byte
  parity, `-Werror` compile.
- `results/*.json` — receipts for the storage byte parity, the actual-checkpoint row
  reads and the graph replay, each labelled with the scope it does **not** prove.
- [README §4–5](README.md#4-current-local-speed-sweep) — matched sweeps with the
  limitation of each configuration stated next to the number.
