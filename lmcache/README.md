# LMCache + vLLM + hf-mount Integration Test

Integration test for hf-mount with vLLM and LMCache. Compares KV cache
performance across local disk, read-write bucket mounts, and overlay
mounts. Supports multiple models and tensor parallelism configurations
via profiles.

## Prerequisites

NVIDIA GPU(s) with CUDA 12.9, Rust toolchain, Python 3.10+.

### Setup

From the repo root:

```bash
# Shared setup: submodule, build hf-mount, venv with vLLM, opencode
./setup.sh
source .venv/bin/activate

# LMCache-specific
cd lmcache && ./setup.sh
```

### HuggingFace token

```bash
export HF_TOKEN=hf_...
# or: huggingface-cli login
```

## Quick start

```bash
source .venv/bin/activate
cd lmcache

# Run all 7 phases with the default profile (gemma4-e4b-tp1):
./test-cache.sh run-all

# Run all phases for all profiles:
./test-cache.sh run-suite

# Run a specific phase with a specific profile:
./test-cache.sh --profile gemma4-e4b-tp1 bucket-overlay

# Show results:
./test-cache.sh status           # current profile
./test-cache.sh status --all     # all profiles
```

## Profiles

Each profile defines a model, tensor parallelism degree, and resource
limits. Profiles live in `profiles/`:

| Profile | Model | TP | GPUs |
|---------|-------|----|------|
| `gemma4-e4b-tp1` (default) | Gemma-4-E4B-it | 1 | 1x A10 |
| `qwen2.5-7b-tp1` | Qwen2.5-Coder-7B-Instruct | 1 | 1x A10 |
| `qwen3-coder-30b-fp8-tp2` | Qwen3-Coder-30B-A3B-Instruct-FP8 | 2 | 2x A10 |

Select via `--profile <name>` or `PROFILE=<name>`.

## Phases

Each phase runs 6 multi-turn opencode conversations. Each conversation
grows the context through code generation prompts until it reaches 90%
of the model's max context length (triggering opencode's compaction),
then runs 3 more turns to verify cache hits survive compaction.
Warmup and consume use different topics to avoid biased cache hit rates.

| Phase | Mount | Cache state | Purpose |
|-------|-------|-------------|---------|
| `baseline` | none | CPU only | Pure prefix cache reference |
| `local-cold` | none | empty disk | Cold start with disk writes |
| `local-warmup` | none | → warm | Populate local disk cache |
| `local-warm` | none | warm | Warm local cache baseline |
| `bucket-warmup` | read-write | → warm | Populate bucket cache |
| `bucket-rw` | read-write | warm | Shared cache via RW mount |
| `bucket-overlay` | overlay | warm | Shared cache via overlay |

## Results

(pending — rerunning with gemma4-e4b-tp1 profile)

## Directory structure

```
lmcache/
  test-cache.sh           # main CLI
  setup.sh                # LMCache-specific deps
  lib/
    helpers.sh            # logging, metrics, summaries
    vllm.sh               # vLLM lifecycle (TP support)
    hf-mount.sh           # mount lifecycle
    conversations.sh      # conversation runner + topic definitions
  profiles/
    gemma4-e4b-tp1.sh             # Gemma-4-E4B-it, TP=1 (default)
    qwen2.5-7b-tp1.sh             # Qwen2.5-Coder-7B, TP=1
    qwen3-coder-30b-fp8-tp2.sh    # Qwen3-Coder-30B-FP8, TP=2
  logs/<profile>/         # per-profile logs and summaries (gitignored)
  opencode.json           # generated at runtime (gitignored)
```

## Model compatibility

LMCache uses vLLM's `--kv-transfer-config` to intercept KV cache
operations. This **disables the hybrid KV cache manager**, which means
models using hybrid attention architectures (e.g. GatedDeltaNet +
standard attention) are incompatible.

| Architecture | Example models | LMCache compatible |
|-------------|---------------|-------------------|
| Standard attention (GQA/MHA) | Gemma-4, Qwen2.5-Coder, Qwen3-Coder-30B | Yes |
| Hybrid (GatedDeltaNet) | Qwen3.5, Qwen3-Coder-Next | **No** |

See [LMCache#2845](https://github.com/LMCache/LMCache/issues/2845) and
[vllm#36771](https://github.com/vllm-project/vllm/issues/36771) for
details.

Additionally, models must support tool calling via vLLM's
`--tool-call-parser` for opencode to read files and interact with the
codebase. Gemma4 uses `gemma4` parser natively (with
`--reasoning-parser gemma4` and the bundled chat template);
Qwen2.5-Coder requires a
[custom parser plugin](https://github.com/hanXen/vllm-qwen2.5-coder-tool-parser);
Qwen3-Coder uses `qwen3_xml` natively.

## Configuration

Environment variable overrides:

| Variable         | Default                             | Description                    |
|------------------|-------------------------------------|--------------------------------|
| `PROFILE`        | `gemma4-e4b-tp1`                    | Profile name                   |
| `BUCKET`         | `dacorvo/lm-cache`                  | HuggingFace bucket ID          |
| `MOUNT_POINT`    | `/tmp/hf-mount-lmcache`             | Mount directory                |
| `VLLM_PORT`      | `8000`                              | vLLM API port                  |
| `LOG_DIR`        | `lmcache/logs/<profile>`            | Log files directory            |
| `HF_MOUNT_BIN`   | `hf-mount/target/release/hf-mount-nfs` | Path to hf-mount-nfs binary |

Model-specific settings (`MODEL`, `TP_SIZE`, `MAX_MODEL_LEN`, etc.) are
set by the profile and should not normally be overridden.
