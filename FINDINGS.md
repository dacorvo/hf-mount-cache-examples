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

## Phase 2.6 — scale-up (30 sessions)

Re-ran the same protocol with 30 natural prompts on lmcache-src to
get tighter statistics. Results (analysis_v2.json + splice_correctness.json
in `runs/phase2_6-2026-05-01-1700/`):

| metric | phase 2.5 | phase 2.6 |
|---|---|---|
| sessions captured | 15 (real agent) | ~27 (real agent, after machinery filter) |
| ordered pairs | 105 | 465 |
| pairs with body match | 3 / 105 (2.9%) | **11 / 465 (2.4%)** |
| match sizes (median) | 497 tok | **2954 tok** |
| match sizes (max) | 524 tok | 2954 tok |
| splice correctness — sim mean | 0.95 (3 pairs) | **0.934 (11 pairs)** |
| splice correctness — sim min | 0.85 | **0.877** |
| **splice correctness — top-1 agreement** | 2/3 | **0/11** |
| KL (mean across pairs) | mixed (~6) | **~12 nats (uniformly high)** |

Two big shifts vs phase 2.5:

1. **Larger matches.** Median match jumped from ~500 to ~2954 tokens.
   This is the substrate body-splice was supposed to act on. The
   2954-tok match recurred across multiple pair clusters — three
   sessions about vLLM-integration / chunk-policy / Redis-serializer
   all read the same large file (likely `lmcache/v1/cache_engine.py`).
   *Now we're above the typical disk-cache load break-even.*

2. **Splice correctness collapsed at this size.** 0 / 11 splice
   pairs preserve the next-token argmax. KL is uniformly ~12 nats.
   Sim mean 0.93 *looks* close but isn't telling the real story —
   see caveat below.

### Caveat: garbage-attractor mode collapse at 15k+ context

Inspecting fresh vs reused texts on phase 2.6 pairs reveals both
outputs are pseudo-refusal templates ("I do not have specific
documentation on…", "<eos>The documentation for OpenCode does not
contain…"). At 15k+ token contexts on Gemma-4 E4B (bf16 +
chunked-SDPA), the model collapses into generic-refusal mode for
both fresh and reused forwards, regardless of whether the splice
introduced any divergence. Cosine sim of two similarly-degenerate
outputs is ~0.93 trivially.

The user flagged this exact failure mode earlier
(`feedback_garbage_attractors.md`): a fresh-vs-reused similarity
test is invalidated when the model collapses to a stable garbage
attractor. **Sim values from phase 2.6 are uninformative.**

The clean signal that survives: **top-1 disagrees on every single
pair (0/11)**. Even when both outputs are degenerate refusals, the
splice consistently shifts the very first generated token. Combined
with the uniformly high KL (~12 nats), this confirms the splice is
introducing real distributional drift — but the cosine-sim metric
doesn't measure it because the model floor is the refusal template.

### Refined bottom line

For Gemma-4 E4B + lmcache-src + OpenCode:

- Body-splice opportunities exist but are rare (~2-3% of pairs)
  and most are deterministic tool outputs (file lists, file
  contents).
- The interesting larger matches (~3k-token shared file reads,
  above disk-cache break-even) appear *only* when the agent is
  actively reading the same file in multiple sessions.
- At the 3k-token splice scale, the reused forward's next-token
  distribution differs meaningfully from the fresh forward's
  (KL ≈ 12 nats; top-1 disagrees uniformly).
- The model's tendency to mode-collapse into refusal templates at
  long contexts makes per-pair sim measurement uninformative on
  this stack — the comparison metric needs to either run on a
  model that doesn't degrade at 15k+ tokens, or be replaced by a
  metric that doesn't trivially equate two degenerate outputs
  (e.g. KL on logits, top-1 agreement, or a task-success metric).

## Limitations (consolidated)

- One model (Gemma-4 E4B), one codebase, one agent framework, 45
  total sessions across phases 2.5 + 2.6. Estimates here are
  suggestive, not definitive.
- Several prompts let the model hallucinate without using tools
  (~30% of phase 2.6 sessions had zero tool calls). The "how/where
  is X" framings tend to elicit generic answers from priors, while
  "find/audit/walk through" framings force tool use. We didn't
  filter or rewrite these — that's real-piloted-agent behaviour.
- Splice correctness was measured on bnb-4bit weights to fit
  multiple pairs back-to-back on 4×A10G; quantization shifts sim
  values slightly (pair 1 of phase 2.5: 0.91 bf16 → 0.85 nf4) but
  qualitative outcomes are stable.
- The chunked-SDPA fork patch is what made 21k+ context measurable
  on Gemma-4. Without it, every phase 2.6 second-turn prefill OOMs.
  The patch is mathematically equivalent to a single SDPA call
  (per-row softmax independence) but does multiple kernel launches
  per attention; expect a wallclock cost.
- The garbage-attractor invalidation at long contexts (above) is
  the main caveat on phase 2.6 splice numbers. The clean signal
  remaining is top-1 disagreement (0/11) and KL (uniformly high).

## Future directions

- Deeper trajectories with file edits / test runs, not one-shot Q&A.
- Test on Qwen3.5+ once we have a proper environment (preference
  order: Gemma-4 first, Qwen3.5+ second).
- Explore whether a write-time signal (attention entropy, K/V
  perturbation sensitivity, structural section identity from the
  manifest) can correctly classify which body matches will splice
  with usable correctness. Reagent's earlier work showed best r ≈
  0.32 on these proxies; we'd want at least 30 measured pairs at
  consistent splice size to validate.
- Pick a model variant or generation config that doesn't
  garbage-attractor at 15k+ contexts so the sim metric becomes
  meaningful again. Or move to a task-success metric (does the
  reused agent get the right answer?) instead of textual sim.
