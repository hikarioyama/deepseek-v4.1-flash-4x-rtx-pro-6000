# Measurements on this rig

These are my own measurements on the same four GPUs, taken on 2026-10-18 while
maintaining this fork. The workload is different from the sweep recorded in
[README §4](README.md#4-current-local-speed-sweep), so these numbers are their own
result set: they do not replace that table and are not a comparison with it.

Measured by the maintainer of this fork (hikarioyama). The recipe, the launcher and
the adapter are the upstream author's; the numbers below are mine.

## Conditions

| Item | Value |
|---|---|
| GPUs | 4 × RTX PRO 6000 Blackwell Max-Q, devices 0–3, **300 W cap each** (1,200 W total), clocks unlocked, PCIe without NVLink |
| Image | `sha256:e10a916ee577b60e94a9ffbc097fd9d2c902cf00a243b8ba5b243cfb41b72d31` |
| Parallelism | TP4 / EP4 |
| Server settings | KV `fp8_e4m3`, DSpark block 5, `dsv4` attention, `flashinfer_mxfp4` MoE, context 524,288, `max_total_tokens` 4,200,000, 8 running requests, `mem_fraction_static` 0.95, chunked prefill 2,048, deterministic inference off |
| Stack | 14 accepted levers: `fp8-tune-exact`, `engram-parallel-io`, `nccl-p2p-sys`, `hc-stats-verify-draft`, `engram-prefetch-draft`, `fp8-tune-splitk`, `verify-candidate-graphs`, `autotune-stable-cache`, `engram-chunk-prefetch`, `engram-scale-cache`, `dspark-compact-verify`, `pf-idx-bounded`, `decode-exact-bundle`, `decode-exact-bundle-2` (`@300a8add72c5`) |
| Workload | Real prompts, code and prose lanes, `prompts_sha256` `1496e8fe…`; temperature 0, thinking off, **1,024 output tokens** per request |
| Repetitions | 3 per lane (9 samples on the single-request decode lanes) |
| Harness | `loop/harness.py`, `harness_sha256` `60c372…` |
| Run | `20261018T121231Z-A-for-pf-exact-bundle`, elapsed 340.2 s, no errors |

## Decode

`tok/s` for the C=1 lanes is the single stream's rate. For the C=4 and C=8 lanes,
`tok/s` is the total delivered in the shared decode window and `per request` is the
median request rate inside that same window; the product of the two need not equal
the total.

| Lane | C | tok/s | Range (3 reps) | Per request | Tokens/step | TTFT s | Power W | % of 1,200 W | J/token |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| code | 1 | **339.2** | 288–364 | – | 4.41 | 0.106 | 857.0 | 71.4% | 2.6007 |
| prose | 1 | **158.3** | 153–196 | – | 2.02 | 0.097 | 866.3 | 72.2% | 5.1593 |
| code | 4 | **750.9** | 737–776 | 183.5 | 4.10 | – | 983.9 | 82.0% | 1.3039 |
| code | 8 | **1,172.7** | 1,157–1,187 | 144.8 | 4.03 | – | 1,049.4 | 87.5% | 0.8951 |
| prose | 4 | **451.0** | 444–453 | 109.5 | 2.02 | – | 945.0 | 78.8% | 2.1027 |
| prose | 8 | **669.6** | 668–672 | 80.5 | 1.85 | – | 986.5 | 82.2% | 1.4727 |

## Prefill

| Lane | Input tokens | Prefill tok/s | Range (3 reps) | TTFT s | Power W | % of 1,200 W | J/token |
|---|---:|---:|---|---:|---:|---:|---:|
| code | 8,192 | **9,044.2** | 8,961–9,181 | 0.91 | 818.0 | 68.2% | 0.0903 |
| code | 32,768 | **10,834.3** | 10,788–10,932 | 3.02 | 947.2 | 78.9% | 0.0873 |
| code | 131,072 | **10,238.2** | 10,220–10,265 | 12.80 | 957.0 | 79.8% | 0.0935 |
| prose | 8,192 | **8,427.6** | 8,174–8,437 | 0.97 | 772.5 | 64.4% | 0.0926 |
| prose | 32,768 | **10,850.8** | 10,719–10,945 | 3.02 | 934.8 | 77.9% | 0.0863 |
| prose | 131,072 | **10,212.5** | 10,195–10,320 | 12.83 | 942.6 | 78.5% | 0.0920 |

Ranges are min–max across the three repetitions. Power is the measured total across
the four GPUs; percentages are of the 1,200 W cap. Nothing here was measured on a
second machine.

## Evidence

The raw records are not published: `loop/runs/` is excluded from git. What identifies
this run:

| File | SHA-256 |
|---|---|
| `summary.json` | `2394963ae7a6ce10ee35d9db2676a9b97fc6d553be939fffc4f00a6644fee1b9` |
| `manifest.json` | `0cdeadc0cce3de3961d39461c96cfba65ec47fb208d41646b558c659b890a998` |
| `parity.json` | `898f4a364d5565897518ec9cacbdccec7837b9289a3532388e8f4e58ea860b12` |

Each run directory also holds `raw.jsonl` (per-sample timings) and `energy.csv`
(power over time). A later candidate run exists but was not accepted into the
14-lever stack, so its numbers are deliberately not listed here.

## Scope

These are throughput and latency measurements with 1,024-token outputs on real
prompts under the settings above. They say nothing about answer quality,
long-context quality, video or soak reliability, and they are not an SLA.
