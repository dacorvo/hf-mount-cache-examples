# torch.compile + hf-mount Integration Test

Shares `torch.compile`'s on-disk Inductor cache across machines through
an HF Bucket mounted with `hf-mount`. Consumers mount the bucket with
`--overlay` so they read shared artifacts but keep any new compilations
local.

## What it does

`TORCHINDUCTOR_CACHE_DIR` redirects every Inductor artifact (Triton
kernels, FX graphs, etc.) to a chosen directory. Pointing it inside an
`hf-mount` mount makes the cache shared.

| Phase    | Mount    | Action                                                 | Bucket effect            |
|----------|----------|--------------------------------------------------------|--------------------------|
| vanilla  | —        | Run each shape with a fresh local Inductor cache       | None                     |
| warmup   | rw       | Compile each shape in `SHAPES`                         | Artifacts uploaded       |
| consume  | overlay  | Run each shape; cache hit if the bucket holds artifacts | Unchanged (local writes) |

The benchmark the example reports is **vanilla** (no hf-mount, cold
compile from scratch) versus **consume** after **warmup** (hf-mount
overlay, cache hit from the bucket). `run-all` runs all three phases
and prints the per-shape `first_call_s` comparison.

`run-all` never deletes bucket content. A pre-existing populated bucket
makes the comparison more realistic — warmup just re-asserts what's
already there. Use the standalone `clear-bucket` command when you
explicitly want to start from an empty subtree.

Overlay matters even on a cache hit: Inductor rewrites a few metadata
files on every compile call (autotuning `.best_config`, codegen `.py`)
even when the on-disk cache matches. Overlay absorbs those writes
locally so the bucket stays pristine.

## Running

Prerequisites: [Homebrew](https://brew.sh) (`setup.sh` uses it to
install `hf-mount` — bottled for both Linux and macOS) and, on
Debian/Ubuntu, `sudo` for the apt-get step. `HF_TOKEN` must be
exported and have write access to the bucket. Python dependencies are
declared inline in `compile_run.py` (PEP 723) and resolved by `uv run`
on first invocation — no manual venv to activate.

```bash
./setup.sh        # one-time: install hf-mount + uv
./run.sh run-all  # vanilla + warmup + consume, prints summary
```

`run-all` is the headline command: it runs `vanilla` for the no-cache
baseline, then `warmup` to populate the bucket, then `consume` to
read it back through an overlay mount. It prints a per-shape
`vanilla_s` vs `cached_s` table.

Individual commands also work: `vanilla`, `warmup`, `consume`,
`teardown`, `clear-bucket`. Each phase writes a JSON report under
`logs/` (`results-vanilla.json`, `results-warmup.json`,
`results-consume.json`) with per-shape first-call and second-call
timings plus `cache_files_added` — a real cache hit writes only a
handful of metadata files (autotuning `.best_config`, codegen `.py`);
a miss writes hundreds of Triton kernels, FX graphs, etc.

## Configuration

| Variable             | Default                              |
|----------------------|--------------------------------------|
| `MODEL`              | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| `BUCKET`             | `dacorvo/torch-compile-cache`        |

The `SHAPES` list and the mount path (under `/tmp`) are at the top of
`run.sh`.

## Caveats

This example exercises the warmup → consume cache-reuse flow, but the
Inductor JIT cache's file profile does not flatter the bucket-sync
path. Two effects to know about before generalizing the pattern to
other workloads:

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
the file-based cache dir.

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
