# torch.compile + hf-mount Integration Test

Verifies that `torch.compile`'s on-disk Inductor cache can be shared
across machines through an HF Bucket mounted with `hf-mount`, and that
the **overlay** mode lets consumers reuse the shared cache while keeping
new compilations local.

## What it does

The `TORCHINDUCTOR_CACHE_DIR` environment variable redirects all
`torch.compile` artifacts (Triton kernels, FX graphs, etc.) to a
specific directory. Pointing it inside an `hf-mount`-mounted bucket
makes those artifacts shared.

Three phases:

| Phase    | Mount    | Action                                             | Bucket effect            |
|----------|----------|----------------------------------------------------|--------------------------|
| warmup   | rw       | Compile shapes `1x16`, `1x32`                      | Artifacts uploaded       |
| consume  | overlay  | Rerun `1x16`, `1x32` (cache hit) + compile `1x64`  | Unchanged (local writes) |
| verify   | —        | Diff the bucket, check cache-hit timings           | —                        |

> **Note on the second mount.** The user-facing description says
> "remount RW", but to verify "the bucket has not been updated and
> compilation artifacts are stored locally" the second mount must be
> overlay — RW would upload the new shape's artifacts back to the
> bucket. The test uses overlay for the consume phase.

## Prerequisites

NVIDIA GPU with CUDA (or CPU for a slow but functional smoke test),
Rust toolchain, Python 3.10+. From the repo root:

```bash
./setup.sh                       # build hf-mount-nfs (claude/rebase-overlay-on-main)
source .venv/bin/activate
cd torch.compile
./setup.sh                       # install torch + transformers
```

`HF_TOKEN` must be exported (or in `~/.cache/huggingface/token`) and
must have write access to the target bucket.

## Quick start

```bash
source .venv/bin/activate
cd torch.compile

# Optional: clear the bucket before a fresh run
./test-compile.sh clear-bucket

# Run all phases end-to-end
./test-compile.sh run-all

# Or step through manually
./test-compile.sh warmup
./test-compile.sh consume
./test-compile.sh verify
```

## How verify decides "cache hit"

The primary signal is **`cache_files_added`** for each shape — the
number of new files that appeared in `TORCHINDUCTOR_CACHE_DIR` during
that shape's compile call:

- A real Inductor cache hit writes **zero** new files.
- A miss / recompile writes ~30+ new files per shape (`.cpp`, `.so`,
  FX graph, AOTAutograd entry, etc).

So in `consume`:

- **HIT** for warmup shapes when `cache_files_added == 0`.
- **RECOMPILE** for the new shape when `cache_files_added > 0`.

First-call latency is shown for context but is **not** used as the
verdict, because under overlay + NFS the first call also pays for
lazy-fetching all the cached `.so` / `.cpp` bytes from the bucket.
On small models that fetch can take roughly as long as a fresh CPU
recompile, so the latency signal is noisy. The file-count signal is
unambiguous.

Bucket invariance is checked by listing the bucket via the HF API
before and after the consume phase and diffing the file lists.

Overlay-local artifacts are captured by listing files under the mount
point **after** unmount — overlay mode persists local writes at the
mount point itself, so once the NFS daemon is gone the directory holds
only the locally-written files (recompile output for the new shape).

## Configuration

| Variable             | Default                                     |
|----------------------|---------------------------------------------|
| `MODEL`              | `unsloth/Llama-3.2-1B-Instruct`             |

For a fast macOS / CPU-only smoke test, override
`MODEL=HuggingFaceTB/SmolLM2-135M-Instruct`. End-to-end time on Apple
Silicon CPU is roughly 90 s warmup + 90 s consume.
| `BUCKET`             | `dacorvo/torch-compile-cache`               |
| `MOUNT_POINT`        | `/tmp/hf-mount-torch-compile`               |
| `HF_MOUNT_CACHE_DIR` | `/tmp/hf-mount-cache-torch-compile`         |
| `HF_MOUNT_BIN`       | `../hf-mount/target/release/hf-mount-nfs`   |
| `LOG_DIR`            | `torch.compile/logs`                        |

To change the input shapes, edit `SHAPES_WARMUP` and `SHAPES_RECOMPILE`
near the top of `test-compile.sh`.

## Files

```
torch.compile/
  setup.sh           # install torch + transformers in the shared venv
  test-compile.sh    # phase orchestrator (warmup / consume / verify / teardown)
  compile_run.py     # load + torch.compile + time forward passes across shapes
  README.md          # this file
  logs/              # JSON results, bucket snapshots, hf-mount logs (gitignored)
```
