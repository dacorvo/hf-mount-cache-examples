# torch.compile + hf-mount Integration Test

Shares `torch.compile`'s on-disk Inductor cache across machines through
an HF Bucket mounted with `hf-mount`. Consumers mount the bucket with
`--overlay` so they read shared artifacts but keep any new compilations
local.

## What it does

`TORCHINDUCTOR_CACHE_DIR` redirects every Inductor artifact (Triton
kernels, FX graphs, etc.) to a chosen directory. Pointing it inside an
`hf-mount` mount makes the cache shared.

| Phase    | Mount    | Action                                              | Bucket effect            |
|----------|----------|-----------------------------------------------------|--------------------------|
| warmup   | rw       | Compile two warmup shapes                           | Artifacts uploaded       |
| consume  | overlay  | Rerun warmup shapes (cache hit) + compile new shape | Unchanged (local writes) |

Overlay matters even for cache hits: Inductor rewrites a few metadata
files on every compile call (autotuning `.best_config`, codegen `.py`)
even on a perfect on-disk hit. Overlay absorbs those writes locally so
the bucket stays pristine.

## Running

Prerequisites: [Homebrew](https://brew.sh) (used by `setup.sh` to install
`hf-mount` — available on Linux and macOS) and, on Debian/Ubuntu,
`sudo` for the apt-get step in `setup.sh`. `HF_TOKEN` must be exported
and have write access to the bucket.

```bash
./setup.sh                       # one-time: install hf-mount + venv + torch
source ../.venv/bin/activate
./run.sh clear-bucket            # optional clean slate
./run.sh run-all                 # warmup + consume
```

Individual commands: `warmup`, `consume`, `teardown`, `clear-bucket`.

Each phase writes a JSON report (`results-warmup.json`,
`results-consume.json`) under `logs/` with per-shape first-call and
second-call timings plus `cache_files_added` — a real cache hit writes
only a handful of metadata files (autotuning `.best_config`, codegen
`.py`); a miss writes hundreds of Triton kernels, FX graphs, etc.

## Configuration

| Variable             | Default                              |
|----------------------|--------------------------------------|
| `MODEL`              | `google/gemma-4-E4B-it`              |
| `DTYPE`              | `bfloat16`                           |
| `BUCKET`             | `dacorvo/torch-compile-cache`        |

Shape sets (`SHAPES_WARMUP`, `SHAPES_RECOMPILE`) and the mount/cache
paths (under `/tmp`) are at the top of `run.sh`.

## Caveats

This example demonstrates the mechanism end-to-end (overlay semantics,
bucket invariance, recompile isolation), but the Inductor JIT cache's
file profile does not flatter the bucket-sync path. Two effects to
know about before generalizing the pattern to other workloads:

### 1. Compile mode dramatically changes file count

The example uses transformers' default compile mode. Switching to
`max-autotune-no-cudagraphs` or `max-autotune` makes Inductor emit
many additional autotuning variants — measured cache for
`gemma-4-E4B-it` with `max-autotune-no-cudagraphs` + chunked prefill +
one warmup shape: **5,520 files, median 8 KB**. Most are debug/hygiene
IR siblings (`.source / .ttir / .ttgir / .llir / .ptx / .cubin`) plus
per-op `.json` and `.best_config`.

No Inductor flag (`bundle_triton_into_fx_graph_cache`,
`bundled_autotune_remote_cache`, `fx_graph_remote_cache`,
`autotune_remote_cache`) reduces the per-kernel file count that hits
the file-based cache dir. The remote-cache path is a separate channel
that routes to Redis in OSS and bypasses `TORCHINDUCTOR_CACHE_DIR`
entirely.

### 2. Per-file overhead is the bottleneck

Bucket access cost is dominated by the per-file metadata roundtrip:

| Access path                                                       | Rate                  |
|-------------------------------------------------------------------|-----------------------|
| Producer-through-mount upload (`--advanced-writes`)               | ~7 files/sec          |
| Direct API upload (`HfApi.sync_bucket`)                           | ~51 files/sec         |
| Direct API download (`HfApi.sync_bucket`, cold xet)               | ~16 files/sec (noisy) |

For a workload that emits thousands of small files, the producer-mount
path takes long enough that cold consumers can be barely faster than a
fresh compile (Llama-3.2-3B: 192 s fresh compile vs 166 s cold-xet
fetch on the same payload), and cold downloads have a long tail of
multi-minute stalls.

### When this pattern earns its keep

The bucket + overlay model wins when **each cached artifact is large
and expensive to produce** — per-file overhead becomes negligible
against per-artifact compute. Examples: AWS Neuron NEFFs, TensorRT
engines, quantized model weights (GPTQ / AWQ), AOTInductor `.so`
bundles. The torch.compile JIT cache (small files, modest per-file
compute) is the wrong shape.

## Files

```
torch.compile/
  setup.sh           # install torch + transformers into the shared venv
  run.sh             # phase orchestrator (warmup / consume / teardown)
  compile_run.py     # load + torch.compile + time generate across shapes
  README.md          # this file
  logs/              # JSON results, bucket snapshots, hf-mount logs (gitignored)
```
