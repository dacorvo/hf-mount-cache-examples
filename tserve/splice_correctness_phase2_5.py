#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4,<2.11",
#   "accelerate>=1.0",
#   "transformers>=5.5",
#   "kernels>=0.5",
#   "sentencepiece",
#   "sentence-transformers>=3.0",
#   "bitsandbytes>=0.46.1",
# ]
# ///
"""Phase 4 measurement: splice correctness on phase 2.5 body-match
candidates.

Takes the matches from runs/<phase2_5-trace>/match_density.json and
for each (A, B) pair with at least one body-splice candidate:

  1. Load Gemma-4 E4B (LM-only, sdpa, no chunking needed for short
     reagent measurements since seq is already capped at the rendered
     prompt length).
  2. Render A and B's request bodies to token ids.
  3. Snapshot A's K/V via reagent's prefill_and_snapshot.
  4. Run reagent's multi_splice_b_forward on B with the matches
     (chunked prefill of B's gaps + injection of A's K/V at matched
     spans, RoPE-shifted by the position delta).
  5. Run fresh_forward on B for comparison.
  6. Compute sim(fresh_text, reused_text), KL, top1 agreement.

Writes <trace_dir>/splice_correctness.json with per-pair results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Reagent imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reagent"))
from measure_multi_splice import prefill_and_snapshot  # noqa: E402
from measure_multi_splice_b import multi_splice_b_forward  # noqa: E402
from measure_multi_splice import fresh_forward  # noqa: E402
from similarity import load_embedder_and_cos_sim  # noqa: E402


def _render_ids(processor, body: dict) -> list[int]:
    out = processor.apply_chat_template(
        body.get("messages", []),
        tools=body.get("tools"),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    ids = out["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace_dir", type=Path)
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument(
        "--quantize-4bit",
        action="store_true",
        help="Load with bitsandbytes nf4 to save weights memory; "
        "useful when measuring multiple pairs back-to-back since "
        "the bf16 model leaves no GPU headroom on Gemma-4 4×A10G.",
    )
    p.add_argument(
        "--max-longest-match",
        type=int,
        default=5000,
        help="Skip pairs whose longest-match is above this size; "
        "those are typically OpenCode session-replay artefacts, "
        "not real cross-session tool-result reuse.",
    )
    args = p.parse_args()

    density_path = args.trace_dir / "match_density.json"
    if not density_path.exists():
        sys.exit(f"missing {density_path}; run measure_phase2_5.py first")
    density = json.loads(density_path.read_text())
    pairs = [
        p for p in density["pairs"] if p["longest_match"] < args.max_longest_match
    ]
    print(f"[info] {len(pairs)} body-match pairs (after filtering longest>{args.max_longest_match})")
    if not pairs:
        sys.exit("no pairs to measure")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[info] loading model {args.model} (lm-only, sdpa, balanced)")
    tok = AutoTokenizer.from_pretrained(args.model)
    load_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="balanced",
        attn_implementation="sdpa",
    )
    if args.quantize_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # device_map=balanced incompatible with quantization-load on
        # this transformers version; auto picks better.
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    print("[info] model loaded")

    cos_sim = load_embedder_and_cos_sim("BAAI/bge-small-en-v1.5")

    # Stop ids: include the model's eos
    stop_ids = set()
    eos = getattr(tok, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple)):
            stop_ids.update(int(x) for x in eos)
        else:
            stop_ids.add(int(eos))

    results = []
    for pi, p in enumerate(pairs):
        print(f"\n[{pi + 1}/{len(pairs)}] pair: {p['a_sid'][:50]} ↔ {p['b_sid'][:50]}")
        kv_a = None
        try:
            a_body = json.loads(Path(p["a_request_path"]).read_text())
            b_body = json.loads(Path(p["b_request_path"]).read_text())
            a_ids = _render_ids(tok, a_body)
            b_ids = _render_ids(tok, b_body)
            a_tensor = torch.tensor(a_ids, dtype=torch.long)
            b_tensor = torch.tensor(b_ids, dtype=torch.long)

            # Snapshot A's K/V each pair — Gemma-4 multi-GPU is
            # memory-tight, holding multiple A snapshots in CPU
            # memory plus residual GPU fragments OOMs.
            print(f"  prefill A ({len(a_ids)} tokens)")
            kv_a = prefill_and_snapshot(model, a_tensor, offload="cpu")

            # Fresh forward on B
            print(f"  fresh forward B ({len(b_ids)} tokens)")
            fresh_log, fresh_gen, _ = fresh_forward(
                model, b_tensor, args.gen_tokens, stop_ids
            )
            fresh_text = tok.decode(fresh_gen)

            # Splice-B forward on B with matches
            matches = [tuple(m) for m in p["match_spans"]]
            print(f"  splice forward B with {len(matches)} match(es)")
            reused_log, reused_gen = multi_splice_b_forward(
                model,
                b_tensor,
                kv_a,
                matches,
                args.gen_tokens,
                stop_ids,
            )
            reused_text = tok.decode(reused_gen)

            import torch.nn.functional as F

            log_p = F.log_softmax(fresh_log, dim=-1)
            log_q = F.log_softmax(reused_log, dim=-1)
            kl = float((log_p.exp() * (log_p - log_q)).sum().item())
            top1_fresh = int(fresh_log.argmax().item())
            top1_reused = int(reused_log.argmax().item())
            sim = cos_sim(fresh_text, reused_text)

            rec = {
                **p,
                "fresh_text": fresh_text[:300],
                "reused_text": reused_text[:300],
                "kl": kl,
                "top1_fresh": top1_fresh,
                "top1_reused": top1_reused,
                "agree": int(top1_fresh == top1_reused),
                "sim_fresh_reused": float(sim) if sim == sim else None,
                "ok": True,
            }
            print(
                f"  → sim={rec['sim_fresh_reused']}  kl={kl:.4f}  agree={rec['agree']}"
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            rec = {**p, "ok": False, "error": f"{type(e).__name__}: {e}"}
        results.append(rec)
        # Free per-pair tensors and force a sync — Gemma-4 multi-GPU is
        # memory-tight; without this the next pair OOMs as fragments
        # accumulate on the most-loaded GPU.
        import gc

        del kv_a
        gc.collect()
        if torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                torch.cuda.synchronize(d)
            torch.cuda.empty_cache()

    out_path = args.trace_dir / "splice_correctness.json"
    out_path.write_text(json.dumps({"results": results}, indent=2, default=str))
    print(f"\n[info] wrote {out_path}")

    ok = [r for r in results if r.get("ok")]
    if ok:
        sims = [r["sim_fresh_reused"] for r in ok if r.get("sim_fresh_reused") is not None]
        agrees = sum(r["agree"] for r in ok)
        print(f"\n=== summary ({len(ok)} pairs measured) ===")
        if sims:
            import statistics
            print(
                f"  sim: mean={statistics.mean(sims):.3f}  "
                f"min={min(sims):.3f}  max={max(sims):.3f}"
            )
        print(f"  top1 agree: {agrees}/{len(ok)}")


if __name__ == "__main__":
    main()
