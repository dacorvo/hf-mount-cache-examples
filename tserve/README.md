# tserve — research vehicle for cache-aware serving

Wraps the patched `transformers serve` (in the `transformers/`
submodule, branch `advanced_serve_cache`) with a config that points
OpenCode at a local Gemma-4 endpoint and captures per-request
metadata for downstream cache analysis.

## Quick start

1. **Editable-install the patched transformers fork** into the project venv:
   ```bash
   uv pip install --python ./.venv/bin/python -e ./transformers
   uv pip install --python ./.venv/bin/python accelerate kernels sentencepiece typer uvicorn fastapi safetensors tqdm
   ```

2. **Start the server** (one tmux window):
   ```bash
   TSERVE_TRACE_DIR=runs/$(date +%Y-%m-%d-%H%M)-smoke \
   PYTORCH_ALLOC_CONF=expandable_segments:True \
   ./.venv/bin/transformers serve \
     --device balanced --dtype bfloat16 \
     --task text-generation \
     --host 0.0.0.0 --port 8000
   ```

   - `--task text-generation` forces `AutoModelForCausalLM` instead
     of the default `AutoModelForMultimodalLM`. On Gemma-4 E4B this
     skips the vision encoder (saves ~7 GB) and lets the LM fit
     across 4 A10Gs comfortably even at 12k+ context. For longer
     prompts on tighter setups, add `--quantization bnb-4bit`.
   - `--device balanced` distributes weights across all visible
     GPUs.
   - `TSERVE_TRACE_DIR` enables per-request trace dump (see below).

3. **Drive the agent** (other window):
   ```bash
   cd tserve   # so OpenCode picks up opencode.json
   opencode run --format json "your task here"
   ```

## What gets captured under `TSERVE_TRACE_DIR`

For each chat-completion request:

| file | contents |
|---|---|
| `<request_id>.request.json` | raw OpenAI-format request body (messages, tools, params) |
| `<request_id>.manifest.json` | section breakdown of the rendered prompt: per-message tok_range, tool injection delta, agent_build_id |
| `<request_id>.response.json` | rendered prompt text, generated text, input/output token counts (non-streaming only — streaming traces still TODO) |

The manifest is the write-time hand-off to a future cache layer:

```json
{
  "agent_build_id": "8687ae2485f84162",
  "n_messages": 4,
  "n_tools": 10,
  "sections": [
    {"id": "msg-0-system",    "role": "system",    "tok_range": [0, 10489],
     "tokens": 10489, "tokens_without_tools": 2906, "tools_injection_tokens": 7583, "stable": true},
    {"id": "msg-1-user",      "role": "user",      "tok_range": [10489, 10522], ...},
    {"id": "msg-2-assistant", "role": "assistant", "tok_range": [10522, 10552], ...},
    {"id": "msg-3-tool",      "role": "tool",      "tok_range": [10552, 11642], ...}
  ]
}
```

`agent_build_id` is `sha256(system_prompt + tools_json)` truncated;
stable across all sessions of the same OpenCode/Hermes config + tool
set, and the natural disk-cache key for the long shared prefix.

## Known limitations / TODOs

- **Streaming response trace** is not yet captured (only the
  non-streaming path writes `response.json`). OpenCode requests
  use `stream:true`, so currently we only get request + manifest.
- **Section identification** is structural at the role level. Sub-
  message sections (`agents-md`, `skills`, `<system-reminder>`
  injections) need agent-framework cooperation or text-pattern
  heuristics — phase 2 work.
- **Multi-GPU + Gemma-4 hybrid attention** can hit cross-device
  errors with `--device balanced` on small GPU counts (2 GPUs hit
  this; 4 GPUs work). Gemma-4 E2B trips this earlier than E4B.

## See also

- `../PLAN.md` — multi-phase plan this is phase 1 of.
- `../reagent/` — splice-tolerance experiments that motivated the
  cache work; phase 4 reuses them on data this server captures.
