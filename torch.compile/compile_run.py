#!/usr/bin/env python3
"""
Load a causal LM, compile it via the transformers static-cache path, and run
`model.generate()` for each requested shape. Reports per-shape wall-clock time
as JSON.

The Inductor on-disk cache is controlled by TORCHINDUCTOR_CACHE_DIR. When that
directory points inside an hf-mount, compiled artifacts are shared through the
bucket transparently.

Shape semantics
---------------
Each --shape is BxC where B is batch size and C is prefill_chunk_size. Varying
the prefill chunk size produces distinct compiled prefill kernels (one per
chunk shape); decode kernels are shape-flexible across cache_len, so we hold
max_new_tokens fixed and only vary the parameter that actually drives compile.

Input prompt length is fixed (--input-len, default 512) so that chunking
actually fires (input_len must be larger than the largest chunk size).

Usage
-----
  compile_run.py --model google/gemma-4-E4B-it \
      --shape 1x64 --shape 1x128 --shape 1x256 \
      --output /tmp/results.json --phase warmup
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig


# Fixed max_new_tokens — decode kernels are shape-flexible across cache_len,
# so this only affects how long the (already-compiled) decode loop runs.
MAX_NEW_TOKENS = 8


def parse_shape(s: str):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--shape", action="append", required=True,
                    help="Input shape as BxC (batch x prefill_chunk_size). Repeat to test multiple.")
    ap.add_argument("--output", required=True, help="JSON results file.")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default=None, help="cuda / cpu (auto-detect if unset)")
    ap.add_argument("--phase", default="", help="Free-form phase label for the report.")
    ap.add_argument("--input-len", type=int, default=512,
                    help="Fixed prompt length (token count). Must be > largest chunk size for chunking to fire.")
    args = ap.parse_args()

    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "<unset>")
    print(f"[compile_run] TORCHINDUCTOR_CACHE_DIR = {cache_dir}", flush=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32
        print("[compile_run] CPU detected — falling back to float32", flush=True)

    print(f"[compile_run] device={device} dtype={dtype} model={args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print("[compile_run] Loading model...", flush=True)
    t_load0 = time.perf_counter()
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    t_load = time.perf_counter() - t_load0
    print(f"[compile_run] Model loaded in {t_load:.2f}s", flush=True)

    # Compile-friendly path: static KV cache triggers transformers' automatic
    # compile in generate. Default compile mode is "reduce-overhead" which uses
    # CUDA Graphs — that fails with chunked prefill ("accessing tensor output of
    # CUDAGraphs that has been overwritten by a subsequent run") because chunks
    # share a single output buffer. We override with max-autotune-no-cudagraphs
    # to keep the heavy autotuning work (substantial cache content) without
    # CUDA Graphs' lifetime constraint.
    model.generation_config.cache_implementation = "static"
    model.generation_config.compile_config = CompileConfig(mode="max-autotune-no-cudagraphs")

    # Prepare a fixed-content prompt of args.input_len tokens. Content doesn't
    # matter; shape does.
    prompt_ids = torch.full((1, args.input_len), tok.bos_token_id or 0,
                            dtype=torch.long, device="cuda:0" if device == "cuda" else device)

    cache_path = Path(cache_dir) if cache_dir != "<unset>" else None
    files_before = count_files(cache_path) if cache_path else 0

    results = []
    for shape_str in args.shape:
        b, chunk_size = parse_shape(shape_str)
        if chunk_size > args.input_len:
            raise ValueError(f"chunk_size={chunk_size} > input_len={args.input_len}; chunking won't fire")
        # Replicate prompt for the requested batch size.
        input_ids = prompt_ids.expand(b, -1).contiguous()

        # Configure chunked prefill for this shape. Varying chunk_size forces
        # distinct compiled prefill kernels per shape.
        model.generation_config.prefill_chunk_size = chunk_size

        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            cache_implementation="static",
        )

        # First call: triggers compile (or loads from on-disk cache).
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(input_ids, **gen_kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        t_first = time.perf_counter() - t0

        # Second call: hot path.
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(input_ids, **gen_kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        t_second = time.perf_counter() - t0

        files_after = count_files(cache_path) if cache_path else 0
        files_added = files_after - files_before
        files_before = files_after

        print(f"[compile_run] shape={b}x{chunk_size} (chunks={args.input_len // chunk_size}) "
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
        "device": device,
        "dtype": str(dtype),
        "cache_dir": cache_dir,
        "cache_files_total": cache_files,
        "cache_size_total": cache_size,
        "model_load_s": round(t_load, 3),
        "input_len": args.input_len,
        "shapes": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[compile_run] Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
