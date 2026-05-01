#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4,<2.11",
#   "transformers>=5.5",
#   "accelerate>=1.0",
#   "kernels>=0.5",
#   "sentencepiece",
#   "sentence-transformers>=3.0",
# ]
# ///
"""Phase 2.5 measurement: cross-session body-splice density and
correctness on real piloted-agent traces.

For each captured request body in the trace dir:
  1. Render through Gemma-4's chat template to get token ids + per-token roles.
  2. Compute first-turn end (last leading non-assistant message).

Across all session pairs (one canonical request per session — the
deepest one, i.e. with most messages — chosen as that session's
"trajectory snapshot"):

  3. Use reagent's find_matches with role + first-turn-skip filters to
     get post-prefix byte-exact ≥min-match-length matches.
  4. Report per-pair: matches, total covered tokens, longest match.
  5. Headline: % of pairs with ≥1 match, distribution of match sizes.

If matches exist:
  6. Build a fake "session" view (iid + token list) per OpenCode session
     and feed pairs to reagent's measure_multi_splice_b for splice
     correctness — but on a small subset (long enough chunks only) to
     keep runtime sane.

Usage:
    uv run --script tserve/measure_phase2_5.py runs/phase2_5-2026-05-01-1430
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Reagent's match finder + filters
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reagent"))
from measure_multi_splice import find_matches  # noqa: E402


def _load_request(path: Path) -> dict:
    return json.loads(path.read_text())


def _session_id(req: dict) -> str:
    """Group requests by their first user message — that's stable across
    all turns in one OpenCode session."""
    for m in req.get("messages") or []:
        if m.get("role") == "user":
            return str(m.get("content", ""))[:200]
    return ""


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


def _per_token_roles(processor, body: dict) -> list[str]:
    msgs = body.get("messages") or []
    tools = body.get("tools")
    roles: list[str] = []
    prev_len = 0
    for i in range(len(msgs)):
        out = processor.apply_chat_template(
            msgs[: i + 1], tools=tools, tokenize=True, return_dict=True
        )
        ids = out["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        cur_len = len(ids)
        roles.extend([msgs[i].get("role", "?")] * (cur_len - prev_len))
        prev_len = cur_len
    return roles


def _first_turn_end(processor, body: dict) -> int:
    msgs = body.get("messages") or []
    n_pre = 0
    for m in msgs:
        if m.get("role") == "assistant":
            break
        n_pre += 1
    if n_pre == 0:
        return 0
    out = processor.apply_chat_template(
        msgs[:n_pre], tools=body.get("tools"), tokenize=True, return_dict=True
    )
    ids = out["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace_dir", type=Path)
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--min-match", type=int, default=128)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    request_files = sorted(args.trace_dir.glob("*.request.json"))
    print(f"[info] {len(request_files)} request bodies in {args.trace_dir}")

    # Group requests by session anchor; pick the deepest as canonical
    sessions: dict[str, list[Path]] = defaultdict(list)
    for rf in request_files:
        try:
            req = _load_request(rf)
        except Exception:
            continue
        sid = _session_id(req)
        if sid:
            sessions[sid].append(rf)

    print(f"[info] {len(sessions)} sessions found")

    canonical = {}
    for sid, files in sessions.items():
        # Deepest = most messages
        deepest = max(files, key=lambda f: len(_load_request(f).get("messages", [])))
        canonical[sid] = deepest

    # Render canonical request per session
    session_data: dict[str, dict] = {}
    for sid, path in canonical.items():
        try:
            req = _load_request(path)
            ids = _render_ids(tok, req)
            roles = _per_token_roles(tok, req)
            while len(roles) < len(ids):
                roles.append("template")
            ft_end = _first_turn_end(tok, req)
            n_msgs = len(req.get("messages", []))
            n_tools = len(req.get("tools") or [])
            session_data[sid] = {
                "ids": ids,
                "roles": roles,
                "first_turn_end": ft_end,
                "n_msgs": n_msgs,
                "n_tools": n_tools,
                "request_path": str(path),
            }
        except Exception as e:
            print(f"  [skip {sid[:30]!r}: {e}]")
            continue

    print(f"\n=== per-session summary ===")
    print(f"  {'session':38s} {'tok':>6} {'msgs':>4} {'tools':>5} {'1st-turn':>8}")
    for sid, d in session_data.items():
        print(
            f"  {sid[:38]!r:38s} {len(d['ids']):>6} {d['n_msgs']:>4} {d['n_tools']:>5} {d['first_turn_end']:>8}"
        )

    # Cross-session pairs
    sids = list(session_data.keys())
    n_pairs = len(sids) * (len(sids) - 1) // 2
    print(f"\n=== body-splice match density ===")
    print(f"  comparing {len(sids)} sessions ({n_pairs} ordered pairs, B>A)")

    pair_records: list[dict] = []
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            a = session_data[sids[i]]
            b = session_data[sids[j]]
            matches = find_matches(
                b["ids"],
                a["ids"],
                args.min_match,
                target_roles=b["roles"],
                prior_roles=a["roles"],
                target_first_turn_end=b["first_turn_end"],
                prior_first_turn_end=a["first_turn_end"],
            )
            if not matches:
                continue
            covered = sum(be - bs for bs, be, _, _ in matches)
            longest = max(be - bs for bs, be, _, _ in matches)
            pair_records.append(
                {
                    "a_sid": sids[i],
                    "b_sid": sids[j],
                    "n_matches": len(matches),
                    "covered_tokens": covered,
                    "coverage_frac": covered / max(1, len(b["ids"])),
                    "longest_match": longest,
                    "match_spans": [list(m) for m in matches],
                    "a_request_path": session_data[sids[i]]["request_path"],
                    "b_request_path": session_data[sids[j]]["request_path"],
                }
            )

    print(f"\n  pairs with >=1 body match: {len(pair_records)}/{n_pairs}")
    if pair_records:
        sizes_longest = [r["longest_match"] for r in pair_records]
        sizes_covered = [r["covered_tokens"] for r in pair_records]
        print(
            f"  longest-match  min={min(sizes_longest)}  median={statistics.median(sizes_longest):.0f}  max={max(sizes_longest)}"
        )
        print(
            f"  covered-tokens min={min(sizes_covered)}  median={statistics.median(sizes_covered):.0f}  max={max(sizes_covered)}"
        )
        # Top 10 most-covering pairs
        top = sorted(pair_records, key=lambda r: -r["covered_tokens"])[:10]
        print(f"\n  top 10 by covered tokens:")
        for r in top:
            print(
                f"    {r['a_sid'][:30]:30s} ↔ {r['b_sid'][:30]:30s} "
                f"n={r['n_matches']} cov={r['covered_tokens']} longest={r['longest_match']}"
            )

    out = {
        "trace_dir": str(args.trace_dir),
        "model": args.model,
        "min_match": args.min_match,
        "n_sessions": len(sids),
        "n_pairs_total": n_pairs,
        "n_pairs_with_match": len(pair_records),
        "pairs": pair_records,
    }
    if args.output:
        args.output.write_text(json.dumps(out, indent=2))
        print(f"\n[info] wrote {args.output}")


if __name__ == "__main__":
    main()
