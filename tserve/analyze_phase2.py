#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4,<2.11",
#   "transformers>=5.5",
#   "sentencepiece",
# ]
# ///
"""Phase 2 trace-dir analyzer.

Walks a TSERVE_TRACE_DIR populated by `run_phase2.sh` and reports:

  1. Per-session summary: number of chat-completion requests in the
     session (= one OpenCode "session" → one chain of requests as
     the agent reasons turn-by-turn), peak input length, agent_build_id.
  2. Cross-session manifest stability: do all sessions of the same
     agent build hash to the same `agent_build_id`? Are the
     prefix-section token ranges byte-stable?
  3. Body-splice match density: for every pair of sessions, find
     byte-exact ≥128-token matches in the rendered prompts past the
     first turn (using reagent's find_matches with role + first-turn
     filters). Reports n_pairs with matches, mean coverage, longest
     match. This is the "do real piloted-agent traces have body
     splice candidates" question.

Usage:
    ./tserve/analyze_phase2.py runs/phase2-2026-05-01-1113/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Reagent's match finder (role + first-turn filters live here too).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reagent"))
from measure_multi_splice import find_matches, _is_input_role  # noqa: E402


def _load_request(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _request_session_id(req: dict) -> str:
    """OpenCode embeds its session ID in the system prompt header
    (or as a header). For now use the first user message's content
    hash + n_messages as a coarse session-grouping key. Replace with
    a real header read once we have one.
    """
    msgs = req.get("messages") or []
    # Use first user message's content as a stable session anchor —
    # OpenCode fires multiple requests per session, all sharing the
    # same first user message.
    for m in msgs:
        if m.get("role") == "user":
            return str(m.get("content", ""))[:200]
    return ""


def _render_token_ids(processor, body: dict) -> list[int]:
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
    """Render incrementally to label each token with its message role.
    Tokens added by tools= injection get the role of the message they
    appear in (usually 'system').
    """
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
    """Token index where the first turn ends (last leading
    non-assistant message before the first assistant turn)."""
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
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    request_files = sorted(args.trace_dir.glob("*.request.json"))
    manifest_files = sorted(args.trace_dir.glob("*.manifest.json"))
    print(f"[info] {len(request_files)} requests, {len(manifest_files)} manifests")

    # Group requests by session anchor (first user message content)
    sessions: dict[str, list[Path]] = defaultdict(list)
    for rf in request_files:
        try:
            req = _load_request(rf)
        except Exception:
            continue
        sid = _request_session_id(req)
        sessions[sid].append(rf)

    print(f"\n=== sessions ===")
    print(f"  unique sessions: {len(sessions)}")
    for sid, files in sessions.items():
        # Pick the LAST request as canonical (longest msg list)
        last = max(files, key=lambda f: len(_load_request(f).get("messages", [])))
        try:
            req = _load_request(last)
            mf = last.with_name(last.name.replace(".request.json", ".manifest.json"))
            mani = _load_manifest(mf) if mf.exists() else {}
        except Exception:
            continue
        n_msgs = len(req.get("messages", []))
        n_tools = len(req.get("tools") or [])
        agent_id = mani.get("agent_build_id", "?")
        print(
            f"  session {sid[:60]!r:62s} requests={len(files):2d}  "
            f"max_msgs={n_msgs:2d}  tools={n_tools}  agent={agent_id[:10]}"
        )

    # Agent-build-id stability check
    print(f"\n=== agent_build_id distribution ===")
    build_ids: dict[str, int] = defaultdict(int)
    for mf in manifest_files:
        try:
            m = _load_manifest(mf)
        except Exception:
            continue
        bid = m.get("agent_build_id")
        if bid:
            build_ids[bid] += 1
    for bid, n in sorted(build_ids.items(), key=lambda x: -x[1]):
        print(f"  {bid}: {n} requests")

    # Cross-session body-splice match density
    print(f"\n=== body-splice match density across sessions ===")
    # Pick the LAST request from each session for cross-session comparison
    # (deepest trajectory, most tool results = most cacheable body content).
    session_tokens: dict[str, dict] = {}
    for sid, files in sessions.items():
        last = max(files, key=lambda f: len(_load_request(f).get("messages", [])))
        req = _load_request(last)
        try:
            ids = _render_token_ids(tok, req)
            roles = _per_token_roles(tok, req)
            # Pad roles to ids length
            while len(roles) < len(ids):
                roles.append("template")
            ft_end = _first_turn_end(tok, req)
        except Exception as e:
            print(f"  [skip {sid[:30]}: {e}]")
            continue
        session_tokens[sid] = {
            "ids": ids,
            "roles": roles,
            "first_turn_end": ft_end,
            "n_msgs": len(req.get("messages", [])),
        }

    sids = list(session_tokens.keys())
    print(f"  comparing {len(sids)} sessions ({len(sids) * (len(sids) - 1) // 2} ordered pairs)")
    pairs_with_match = 0
    total_matched_tokens = 0
    longest_match_overall = 0
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            a = session_tokens[sids[i]]
            b = session_tokens[sids[j]]
            matches = find_matches(
                b["ids"],
                a["ids"],
                args.min_match,
                target_roles=b["roles"],
                prior_roles=a["roles"],
                target_first_turn_end=b["first_turn_end"],
                prior_first_turn_end=a["first_turn_end"],
            )
            if matches:
                pairs_with_match += 1
                covered = sum(be - bs for bs, be, _, _ in matches)
                longest = max(be - bs for bs, be, _, _ in matches)
                total_matched_tokens += covered
                longest_match_overall = max(longest_match_overall, longest)
                print(
                    f"  {sids[i][:30]:30} ↔ {sids[j][:30]:30}  "
                    f"n_matches={len(matches)} cov={covered} longest={longest}"
                )
    n_pairs = len(sids) * (len(sids) - 1) // 2
    if n_pairs > 0:
        print(
            f"\n  pairs with ≥1 body-match: {pairs_with_match}/{n_pairs} "
            f"({pairs_with_match / n_pairs:.0%})"
        )
        if pairs_with_match:
            print(
                f"  total matched tokens across all pairs: {total_matched_tokens:,}\n"
                f"  longest single body-match: {longest_match_overall} tokens"
            )


if __name__ == "__main__":
    main()
