#!/usr/bin/env python3
"""
Load a small causal LM, torch.compile it, and run a forward pass for each
requested input shape. Reports per-shape wall-clock time as JSON on stdout.

The Inductor on-disk cache is controlled by TORCHINDUCTOR_CACHE_DIR.
When that directory points inside an hf-mount, compiled artifacts are
shared through the bucket transparently.

Usage:
  compile_run.py --model meta-llama/Llama-3.2-1B \
      --shape 1x16 --shape 1x32 --shape 1x64 \
      --output /tmp/results.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_shape(s: str):
    b, n = s.split("x")
    return int(b), int(n)


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
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--shape", action="append", required=True,
                    help="Input shape as BxN (batch x seq_len). Repeat to test multiple.")
    ap.add_argument("--output", required=True, help="JSON results file.")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default=None, help="cuda / cpu (auto-detect if unset)")
    ap.add_argument("--phase", default="", help="Free-form phase label for the report.")
    args = ap.parse_args()

    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "<unset>")
    print(f"[compile_run] TORCHINDUCTOR_CACHE_DIR = {cache_dir}", flush=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32  # fp16 on CPU is unsupported / very slow
        print("[compile_run] CPU detected — falling back to float32", flush=True)

    print(f"[compile_run] device={device} dtype={dtype} model={args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print("[compile_run] Loading model...", flush=True)
    t_load0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()
    t_load = time.perf_counter() - t_load0
    print(f"[compile_run] Model loaded in {t_load:.2f}s", flush=True)

    # Force a fresh dynamo state per process so each shape triggers a real
    # compile lookup against the on-disk cache.
    torch._dynamo.reset()
    compiled = torch.compile(model, dynamic=False, mode="default", fullgraph=False)

    cache_path = Path(cache_dir) if cache_dir != "<unset>" else None
    files_before = count_files(cache_path) if cache_path else 0

    results = []
    for shape_str in args.shape:
        b, n = parse_shape(shape_str)
        input_ids = torch.randint(0, tok.vocab_size, (b, n), device=device)
        attn_mask = torch.ones_like(input_ids)

        # First call: forward + compile (or load from on-disk cache).
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = compiled(input_ids=input_ids, attention_mask=attn_mask)
        if device == "cuda":
            torch.cuda.synchronize()
        t_first = time.perf_counter() - t0

        # Second call: pure execution, no compile. Used as a sanity baseline.
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = compiled(input_ids=input_ids, attention_mask=attn_mask)
        if device == "cuda":
            torch.cuda.synchronize()
        t_second = time.perf_counter() - t0

        files_after = count_files(cache_path) if cache_path else 0
        files_added = files_after - files_before
        files_before = files_after

        print(f"[compile_run] shape={b}x{n} first={t_first:.2f}s "
              f"second={t_second:.3f}s cache_files+={files_added}", flush=True)
        results.append({
            "shape": f"{b}x{n}",
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
        "shapes": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[compile_run] Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
