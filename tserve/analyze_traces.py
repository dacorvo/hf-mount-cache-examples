#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4,<2.11",
#   "transformers>=5.5",
#   "sentencepiece",
# ]
# ///
"""Canonical trace-dir analyzer.

Walks a TSERVE_TRACE_DIR, filters out OpenCode session-machinery
(auto-title, no-tools session-replay), and on the remaining real
agent sessions reports:

  - per-session: tokens, n_msgs, n_tools, n_tool_calls, first_turn_end
  - cross-session: byte-exact body matches past the first turn,
    decoded content snippets per match, classification heuristic
    (file-list, grep-result, code, mixed), per-B "popularity"
    (how many other sessions match this one)

Default model (Gemma-4 E4B) for re-rendering. Override via --model.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Reagent's match finder + filters
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reagent"))
from measure_multi_splice import find_matches  # noqa: E402


_FILE_LIST_PAT = re.compile(r"(?:/[^\s/]+){2,}\.\w+(?:\n|$)", re.M)
_GREP_PAT = re.compile(r":[\d]+(:|\n)", re.M)


def _classify_content(text: str) -> str:
    """Best-effort categorisation of a match's content."""
    if _FILE_LIST_PAT.search(text):
        n_paths = len(_FILE_LIST_PAT.findall(text))
        if n_paths >= 3:
            return "file-list"
    if _GREP_PAT.search(text):
        return "grep-result"
    if "def " in text or "class " in text or "import " in text:
        return "code"
    return "other"


def _load_request(path: Path) -> dict:
    return json.loads(path.read_text())


def _session_id(req: dict) -> str:
    """Group requests by their first user message (stable across turns)."""
    for m in req.get("messages") or []:
        if m.get("role") == "user":
            return str(m.get("content", ""))[:200]
    return ""


def _is_real_agent_session(req: dict) -> bool:
    """Filter: drop OpenCode auto-title (no tools) and session-replay
    requests (we identify these as having n_msgs > 4 with n_tools=0)."""
    n_tools = len(req.get("tools") or [])
    if n_tools == 0:
        return False
    msgs = req.get("messages") or []
    if len(msgs) < 2:
        return False
    return True


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
        cur = len(ids)
        roles.extend([msgs[i].get("role", "?")] * (cur - prev_len))
        prev_len = cur
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
    p.add_argument(
        "--max-longest-match",
        type=int,
        default=5000,
        help="Pairs whose longest match exceeds this are flagged as "
        "session-machinery artefacts and excluded from headline stats.",
    )
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    request_files = sorted(args.trace_dir.glob("*.request.json"))
    print(f"[info] {len(request_files)} request bodies in {args.trace_dir}")

    # Group requests by session anchor; pick deepest as canonical
    sessions: dict[str, list[Path]] = defaultdict(list)
    for rf in request_files:
        try:
            req = _load_request(rf)
        except Exception:
            continue
        sid = _session_id(req)
        if sid:
            sessions[sid].append(rf)

    # Filter to real agent sessions
    real_sessions: dict[str, Path] = {}
    skipped: list[tuple[str, str]] = []
    for sid, files in sessions.items():
        deepest = max(files, key=lambda f: len(_load_request(f).get("messages", [])))
        req = _load_request(deepest)
        if not _is_real_agent_session(req):
            skipped.append((sid[:60], "no tools / too few msgs"))
            continue
        real_sessions[sid] = deepest

    print(f"[info] {len(real_sessions)} real agent sessions")
    print(f"[info] skipped {len(skipped)} non-agent sessions:")
    for s, why in skipped:
        print(f"  - {s[:55]!r:55s}  ({why})")

    # Render canonical request per real session
    session_data: dict[str, dict] = {}
    for sid, path in real_sessions.items():
        req = _load_request(path)
        try:
            ids = _render_ids(tok, req)
            roles = _per_token_roles(tok, req)
            while len(roles) < len(ids):
                roles.append("template")
            ft_end = _first_turn_end(tok, req)
        except Exception as e:
            print(f"  [render failed for {sid[:30]!r}: {e}]")
            continue
        n_msgs = len(req.get("messages", []))
        n_tools = len(req.get("tools") or [])
        # Count assistant turns with tool calls in the original
        # request to estimate trajectory depth.
        n_tool_calls = 0
        for m in req.get("messages", []):
            tc = m.get("tool_calls") if isinstance(m.get("tool_calls"), list) else []
            n_tool_calls += len(tc) if tc else 0
        session_data[sid] = {
            "ids": ids,
            "roles": roles,
            "first_turn_end": ft_end,
            "n_msgs": n_msgs,
            "n_tools": n_tools,
            "n_tool_calls": n_tool_calls,
            "request_path": str(path),
        }

    print(f"\n=== per-session summary ===")
    print(
        f"  {'session':52s} {'tok':>6} {'msgs':>4} {'tcalls':>6} {'1st-turn':>8}"
    )
    for sid, d in sorted(session_data.items()):
        print(
            f"  {sid[:52]!r:52s} {len(d['ids']):>6} {d['n_msgs']:>4} "
            f"{d['n_tool_calls']:>6} {d['first_turn_end']:>8}"
        )

    # Cross-session match density
    sids = list(session_data.keys())
    n_pairs = len(sids) * (len(sids) - 1) // 2
    print(f"\n=== body-splice match density ===")
    print(f"  comparing {len(sids)} sessions ({n_pairs} ordered pairs, B>A)")

    pair_records: list[dict] = []
    per_b_match_count: dict[str, int] = defaultdict(int)
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
            # Decode each match's content + classify
            decoded_matches = []
            for bs, be, as_, ae in matches:
                snippet = tok.decode(b["ids"][bs : bs + min(120, be - bs)])
                cat = _classify_content(snippet)
                decoded_matches.append(
                    {
                        "b_start": bs,
                        "b_end": be,
                        "a_start": as_,
                        "a_end": ae,
                        "shift": bs - as_,
                        "category": cat,
                        "preview": snippet[:160],
                    }
                )
            pair_records.append(
                {
                    "a_sid": sids[i],
                    "b_sid": sids[j],
                    "n_matches": len(matches),
                    "covered_tokens": covered,
                    "coverage_frac": covered / max(1, len(b["ids"])),
                    "longest_match": longest,
                    "match_spans": [list(m) for m in matches],
                    "decoded_matches": decoded_matches,
                    "a_request_path": session_data[sids[i]]["request_path"],
                    "b_request_path": session_data[sids[j]]["request_path"],
                    "is_machinery_outlier": longest > args.max_longest_match,
                }
            )
            per_b_match_count[sids[j]] += 1
            per_b_match_count[sids[i]] += 1

    real_pairs = [r for r in pair_records if not r["is_machinery_outlier"]]

    print(f"\n  total pairs with >=1 body match: {len(pair_records)}/{n_pairs}")
    print(
        f"  after dropping session-machinery outliers (longest > {args.max_longest_match}): {len(real_pairs)}"
    )

    if real_pairs:
        sizes_longest = [r["longest_match"] for r in real_pairs]
        sizes_covered = [r["covered_tokens"] for r in real_pairs]
        print(
            f"\n  longest-match  min={min(sizes_longest)}  median={statistics.median(sizes_longest):.0f}  max={max(sizes_longest)}"
        )
        print(
            f"  covered-tokens min={min(sizes_covered)}  median={statistics.median(sizes_covered):.0f}  max={max(sizes_covered)}"
        )
        # Category mix
        cat_counts = defaultdict(int)
        for r in real_pairs:
            for m in r["decoded_matches"]:
                cat_counts[m["category"]] += 1
        print(f"\n  match content categories:")
        for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {c:>14s}: {n}")
        # Top by coverage
        top = sorted(real_pairs, key=lambda r: -r["covered_tokens"])[:10]
        print(f"\n  top {len(top)} pairs by covered tokens:")
        for r in top:
            print(
                f"    {r['a_sid'][:40]:40s} ↔ {r['b_sid'][:40]:40s} "
                f"n={r['n_matches']} cov={r['covered_tokens']} longest={r['longest_match']}"
            )

    # Per-session "popularity" — how many other sessions share at least one body match
    if per_b_match_count:
        print(f"\n  most-matched sessions (≥1 body match with N other sessions):")
        for sid, n in sorted(per_b_match_count.items(), key=lambda x: -x[1])[:10]:
            print(f"    n={n:>3}  {sid[:60]!r}")

    out = {
        "trace_dir": str(args.trace_dir),
        "model": args.model,
        "min_match": args.min_match,
        "n_sessions_real": len(sids),
        "n_pairs_total": n_pairs,
        "n_pairs_with_match_real": len(real_pairs),
        "n_pairs_machinery_outlier": len(pair_records) - len(real_pairs),
        "pairs": pair_records,
    }
    if args.output:
        args.output.write_text(json.dumps(out, indent=2))
        print(f"\n[info] wrote {args.output}")


if __name__ == "__main__":
    main()
