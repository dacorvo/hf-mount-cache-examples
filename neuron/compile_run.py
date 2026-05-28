#!/usr/bin/env python3
"""
Load a causal LM on a single Neuron core, compile it via the transformers
static-cache path, and run `model.generate()` for each requested shape.
Reports per-shape wall-clock time as JSON.

The on-disk NEFF cache is controlled by TORCH_NEURONX_NEFF_CACHE_DIR. When
that directory points inside an hf-mount, compiled NEFFs are shared through
the bucket transparently.

Each --shape is BxC (batch x prefill_chunk_size); varying the chunk size
produces distinct compiled prefill graphs.

Requires transformers >= 5.6.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

MAX_NEW_TOKENS = 8
PROMPT = "The future of AI is"


def parse_shape(s: str) -> tuple[int, int]:
    b, c = s.split("x")
    return int(b), int(c)


def fmt_dir_size(path: Path) -> str:
    if not path.exists():
        return "0B"
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    for unit in ("B", "K", "M", "G"):
        if total < 1024 or unit == "G":
            return f"{total:.1f}{unit}"
        total /= 1024


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--shape", action="append", required=True,
                    help="Input shape as BxC (batch x prefill_chunk_size). Repeat to test multiple.")
    ap.add_argument("--output", required=True, help="JSON results file.")
    ap.add_argument("--phase", default="", help="Free-form phase label for the report.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from transformers.generation.configuration_utils import CompileConfig

    cache_dir = os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR", "<unset>")
    print(f"[compile_run] TORCH_NEURONX_NEFF_CACHE_DIR = {cache_dir}", flush=True)
    print(f"[compile_run] model={args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[compile_run] Loading model to Neuron...", flush=True)
    t_load0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
    ).to("neuron")
    model.eval()
    t_load = time.perf_counter() - t_load0
    print(f"[compile_run] Model loaded in {t_load:.2f}s", flush=True)

    enc = tok(PROMPT, return_tensors="pt").to("neuron")

    compile_config = CompileConfig(backend="neuron", fullgraph=False, dynamic=False)

    cache_path = Path(cache_dir) if cache_dir != "<unset>" else None
    files_before = count_files(cache_path) if cache_path else 0

    results = []
    for shape_str in args.shape:
        b, chunk_size = parse_shape(shape_str)

        gen_config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            cache_implementation="static",
            prefill_chunk_size=chunk_size,
            compile_config=compile_config,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

        # First call: triggers compile (or loads from on-disk NEFF cache).
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
                generation_config=gen_config,
            )
        t_first = time.perf_counter() - t0

        # Second call: hot path — NEFFs already resident.
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
                generation_config=gen_config,
            )
        t_second = time.perf_counter() - t0

        files_after = count_files(cache_path) if cache_path else 0
        files_added = files_after - files_before
        files_before = files_after

        print(f"[compile_run] shape={b}x{chunk_size} "
              f"first={t_first:.2f}s second={t_second:.3f}s cache_files+={files_added}",
              flush=True)
        results.append({
            "shape": f"{b}x{chunk_size}",
            "prefill_chunk_size": chunk_size,
            "first_call_s": round(t_first, 3),
            "second_call_s": round(t_second, 3),
            "cache_files_added": files_added,
        })

    cache_size = fmt_dir_size(cache_path) if cache_path else "n/a"
    cache_files = count_files(cache_path) if cache_path else 0

    report = {
        "phase": args.phase,
        "model": args.model,
        "cache_dir": cache_dir,
        "cache_files_total": cache_files,
        "cache_size_total": cache_size,
        "model_load_s": round(t_load, 3),
        "shapes": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[compile_run] Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
