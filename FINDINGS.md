# Findings: post-prefix splice on real piloted-agent traces

What we measured: does llama.cpp-style post-prefix K/V splice
(byte-exact match anywhere past the first turn, RoPE-shifted to
new position) preserve model behaviour when applied to **real
agent trajectories** that we generated under controlled conditions
on Gemma-4?

This document summarises the answer from phases 1-4 of [PLAN.md](PLAN.md).
The infrastructure, scripts, and per-pair measurement files are in
[tserve/](tserve/) and the trace dirs under [runs/](runs/).

## Setup

- **Model:** Gemma-4 E4B (LM-only via `--task text-generation`),
  bf16, balanced across 4 × A10G. Forked
  [transformers](transformers/) on `advanced_serve_cache` branch
  with three patches:
  - `--task` flag for the LM-only loader (saves the vision encoder).
  - `TSERVE_TRACE_DIR` per-request body / response / section
    manifest dump.
  - **Query-dim chunking in SDPA** to dodge the math-attention
    fallback's seq² × heads × fp32 attention-scores tensor —
    Gemma-4's hybrid sliding-window mask blocks both FA and
    mem-efficient backends, so without chunking a 21k-token
    prefill OOMs at 28 GiB on a single GPU. Default chunk = 4096,
    override via `TSERVE_SDPA_CHUNK_Q`. At chunk=1024 the
    21k-token Gemma-4 prefill peaks at 16.7 GiB / GPU.
- **Driver:** [OpenCode](https://opencode.ai/) 1.4.3 in `run`
  (non-interactive) mode, all 15 sessions on the lmcache-src
  codebase with realistic engineering questions (no contrived
  scaffolding).
- **Measurement:** [reagent](reagent/)'s `find_matches` (with role
  + first-turn-skip filters) for body-match density;
  `multi_splice_b_forward` + `fresh_forward` for splice correctness.

## Phase 2.5 — body-match density on real agent trajectories

15 OpenCode sessions on lmcache-src, all natural prompts. Tool-use
breakdown: 13/15 sessions used tools at all (median 2 tool calls
per session, max 4). Two prompts ("implement a backend",
"instrument cache hit rate") let the model give a generic
hallucinated answer without reading the codebase — that's a model
behaviour, not a setup issue.

Cross-session body-match density past the first turn (after
applying the role and first-turn-skip filters):

| metric | value |
|---|---|
| ordered task-pair combinations | 105 (15 × 14 / 2) |
| pairs with ≥1 byte-exact body match (≥128 tok) | **3 / 105 ≈ 3%** |
| match sizes (real cross-task pairs) | 284 / 491 / 524 tokens; longest 497 |
| total covered tokens per pair (max) | 852 |
| match content type | deterministic tool outputs (glob file lists, grep results) |

The 22k-token "outlier" match in the raw analysis turned out to be
OpenCode session-state replay machinery, not real cross-session
reuse of agent input. Filtered out for the headline.

So real cross-session body-match opportunities exist on
piloted-agent traces over the same codebase, but they're **rare
(~3% of pairs)** and **small (median ~500 tokens)** — well below
the typical disk-cache load break-even (~1k tokens).

## Phase 4 — splice correctness on the candidates

For each of the 3 real body-match pairs, prefill A's K/V, run
reagent's chunked-prefill-with-injection forward of B at the
matched spans (RoPE-shifted to B's positions), compare logits and
greedy 64-tok continuation against a fresh forward of B.

| pair | matches | covered | sim(fresh, reused) | KL | top-1 agree |
|---|---|---|---|---|---|
| concurrent-access ↔ Redis-serialize | 1 × 524 | 524 | **0.85** | 11.8 | ✗ |
| add-metric ↔ vLLM-entry | 3 × ≤497 | 852 | **1.00** | 0.0002 | ✓ |
| vLLM-handoff ↔ vLLM-entry | 2 × ≤284 | 491 | **1.00** | 0.0002 | ✓ |

**2/3 splices bit-exact, 1/3 catastrophic.** Sample size is too
small for a rate estimate, but the qualitative pattern is
consistent with what reagent's earlier corpus measurements on
SWE-smith and Nemotron showed: at small body-splice sizes the
median pair is fine but a non-trivial fraction fails hard, and
that risk is not predictable from cheap text-similarity proxies
(reagent r ≈ ±0.3 ceiling on prediction).

## Bottom line

For Gemma-4 on real piloted-agent traffic generated under our own
control:

1. **Body-splice opportunities are rare and small.** ~3% of
   cross-session task-pair combinations have any byte-exact body
   match past the first turn; matches that exist are sub-1k
   tokens, dominated by deterministic tool outputs.
2. **Splice correctness is not uniform even at sub-1k sizes.**
   2/3 of measured pairs were bit-exact; 1/3 failed
   catastrophically (sim=0.85, top-1 disagrees). With only 3
   pairs we can't put error bars on it, but the failure mode is
   real and matches the larger reagent corpus.
3. **Implication for production cache layers.** The cache-value
   mass is in the first-turn structural prefix (~10.5k tokens of
   system + tools per OpenCode session, byte-stable across all
   sessions of the same agent build). That's the prefix-cache
   regime, already implemented in every serving stack.
   **Post-prefix splice on agent traces — the question
   `llama.cpp --cache-reuse` style addresses — is a small,
   risky-when-it-fails marginal addition for this workload.**

## Limitations

- One model (Gemma-4 E4B), one codebase (lmcache-src), one agent
  framework (OpenCode), 15 sessions. Estimates here are
  suggestive, not definitive. A larger run with deeper trajectories
  (more sessions, more diverse codebases) might surface more / larger
  body matches.
- Two of 15 prompts let the model hallucinate without using tools.
  Those sessions contribute zero body-match opportunity. Better
  prompt design might raise the tool-call rate but our instruction
  was no contrived scaffolding.
- Splice correctness was measured on bnb-4bit weights for memory
  reasons (bf16 OOMs on the second pair, see commit message of
  `tserve/splice_correctness_phase2_5.py`). Quantization may shift
  sim values slightly; bf16 spot-check on pair 1 gave sim=0.91 vs
  0.85 nf4, but both are well below the bit-exact threshold, so
  the qualitative outcome stands.

## Future directions

- Deeper trajectories. Increase max session steps. Use OpenCode in
  TUI mode on real engineering tasks (with file edits, test
  runs, multi-step debugging) rather than one-shot Q&A.
- Test on Qwen3.5+ once we have a proper environment (per
  preference order: Gemma-4 first, Qwen3.5+ second).
- Explore whether a write-time signal (attention entropy, K/V
  perturbation sensitivity, structural section identity from the
  manifest) can correctly classify which sub-1k body matches will
  splice cleanly. Reagent's earlier work showed best r ≈ 0.32 on
  these proxies; with only 3 measured pairs here we can't
  validate that on agent-trace data, but a phase-2.5-at-scale run
  would.
