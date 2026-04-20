# Increased KV Cache Hits Using Block Sliding

This document summarizes considerations about sliding KV cache blocks to achieve a higher cache hit ratio during LLM inference. Much of the insight comes from [CacheSlide](https://github.com/SJTU-Storage-Lab/CacheSlide) (USENIX FAST '26) and related 2025 literature.

---

## 1. Mathematical Foundations

### 1.1 Attention and Relative Position

For any transformer layer $\ell$, the attention score between token *i* and token *j* is:

$$
\mathrm{Attn}_{ij}^{(\ell)} = \frac{\exp\!\bigl(q_i^{(\ell)\top}k_j^{(\ell)}/\sqrt{d}\bigr)}{\sum_{m}\exp\!\bigl(q_i^{(\ell)\top}k_m^{(\ell)}/\sqrt{d}\bigr)}
$$

with

$$
q_i^{(\ell)} = W_Q\, \mathrm{RoPE}(h_i^{(\ell-1)}, p_i), \qquad
k_j^{(\ell)} = W_K\, \mathrm{RoPE}(h_j^{(\ell-1)}, p_j)
$$

RoPE rotates query and key vectors by an angle linear in absolute position:

$$
\mathrm{RoPE}(x, p) = x\,e^{\mathrm{i}\,p\theta}
\qquad\text{with}\qquad
\theta = \{\theta_0,\dots,\theta_{d/2-1}\}\text{ fixed per head.}
$$

The attention score therefore becomes

$$
q_i^\top k_j \propto \exp\!\bigl(\mathrm{i}(p_i - p_j)\theta\bigr)
$$

meaning the **relative phase** $(p_i - p_j)\theta$ determines the attention magnitude.

### 1.2 Phase Contamination Under Sliding

Suppose we cached the KV vectors of chunk *C* at original positions $\mathcal{P}=\{p, p+1,\dots,p+L-1\}$ and want to reuse them at new positions $\mathcal{P}'=\{p', p'+1,\dots,p'+L-1\}$. For every token *t* inside *C*, the situation is:

| Quantity | Cached (old) | Needed (new) | Error introduced |
|----------|--------------|--------------|------------------|
| **Key vector** | $k_t\,e^{\mathrm{i}t\theta}$ | $k_t\,e^{\mathrm{i}(t+\Delta)\theta}$ | phase shift $\Delta\theta$ |
| **Value vector** | $v_t$ (OK) | $v_t$ (OK) | none |
| **Rel. phase** to outside token *o* | $(t-o)\theta$ | $(t+\Delta-o)\theta$ | systematic shift $\Delta\theta$ |

where $\Delta = p' - p$ is the absolute drift. Every attention score between a reused *inside* token and an *outside* token is rotated by the same angle $\Delta\theta$. This is not corrected by shifting the position index in the mask -- the KV tensors themselves carry the wrong phase.

### 1.3 Layer-Wise Amplification

The error compounds through layers:

$$
\mathrm{Err}^{(\ell)} = \underbrace{\Delta\theta}_{\text{layer-0}} + \underbrace{W_V^{(\ell)}\Delta\theta}_{\text{layer-1}} + \dots
$$

Empirically (CacheSlide, section 5.3) the logit deviation grows linearly with $\Delta$ and exponentially with layer index $\ell$:

$$
\|\Delta\mathrm{logits}\|_2 \approx A\,\Delta\,e^{\alpha\ell}
$$

for constants $A,\alpha>0$ derived from model weights.

### 1.4 Take-Home Bound

For a model with $L$ layers, head dimension $d$, and max drift $\Delta$, the theoretical lower bound on logit deviation is

$$
\|\Delta\mathrm{logits}\|_2 \geq \frac{\Delta}{d}\,\sum_{\ell=1}^{L}\|W_V^{(\ell)}\|_2\,\|\theta\|_2
$$

which grows linearly with drift and linearly with depth. This bound is tight for small $\Delta$ and matches empirical measurements in CacheSlide (Fig. 8). Any PIC (Position-Independent Caching) method that does not recompute boundary tokens must violate this bound, explaining why naive sliding (as in llama.cpp) caps at roughly 92% accuracy even for small $\Delta$.

---

## 2. The Context-Dependence Problem

The position-drift analysis above is not the whole story. Even with **perfect position handling**, the KV cache for chunk *C* is different depending on the preceding context:

```
Prompt 1:  [A][A][A]  [C][C][C]   -- C at positions 3-5
            prefix A

Prompt 2:  [B][B][B]  [C][C][C]   -- C at same positions 3-5
            prefix B
```

At layer 1, the hidden state for token *c* in chunk *C* is:

$$
h_c^{(1)} = \sum_{s \in \text{Prefix}} \text{Attn}_{cs}\,v_s^{(1)} + \sum_{s' \in C} \text{Attn}_{cs'}\,v_{s'}^{(1)}
$$

The first sum runs over prefix tokens. Since $\{\text{Attn}_{cs} : s \in \text{Prefix A}\} \neq \{\text{Attn}_{cs} : s \in \text{Prefix B}\}$, we necessarily have $h_c^{(1)}[\text{context A}] \neq h_c^{(1)}[\text{context B}]$. Because the KV cache stores $k_c = W_K\,h_c^{(\ell-1)}$ and $v_c = W_V\,h_c^{(\ell-1)}$, this contextual difference propagates through all layers: the KV for *C* is **uniquely determined by the full prefix**, not just position.

This yields a correctness hierarchy for KV reuse, from weakest to strongest guarantee:

| Scenario | Can reuse? | Why |
|----------|------------|-----|
| Same chunk, different prefix content | No | Different attention patterns produce different KV |
| Same chunk, same prefix content, different position | Approximate | Same context, wrong RoPE phase |
| Same chunk, same prefix content, same position | Yes | Bit-for-bit identical KV |

Empirical verification from the Agent-Attention Errors paper (arXiv 2503.09012) confirms this: even with identical absolute positions and identical RoPE phases, KV cache cosine similarity across different prefixes is only 0.85--0.95, not 1.0.

### Position Drift Dominates Content Drift

A crucial finding from CacheSlide (section 3.2.1) is that, in practice, **position drift is the dominant source of error**, not contextual content differences. The experiment holds content fixed (same MemHistory segment) and varies only prefix length from 0 to 1000 tokens. CKSim (cosine similarity between cached and recomputed KV) decreases monotonically with prefix length *independent of prefix content* -- the error is purely positional.

This means that while true KV reuse is theoretically impossible for non-identical prefixes, the practical gap between "same prefix, wrong position" and "different prefix, same position" is large enough that position-aware techniques capture most of the available gains. Prefix caching (as in vLLM) remains the only fully safe reuse strategy, but approximate reuse with controlled positional error is viable.

---

## 3. RoPE vs CoPE: Position Encoding Matters

The severity of position drift depends heavily on the position encoding scheme. CacheSlide includes an ablation (Figure 4a) measuring CKSim as a function of absolute position shift for two encodings:

| Encoding | 0-token shift | 100-token shift | 500-token shift | 1000-token shift |
|----------|---------------|-----------------|-----------------|------------------|
| **RoPE** | 1.0 | ~0.6 | ~0.3 | <0.1 |
| **CoPE** | 1.0 | ~0.9 | ~0.8 | ~0.72 |

RoPE similarity drops over 90% under a 1000-token drift, while CoPE drops only 28%. The paper concludes: *"RoPE exhibits a more pronounced decline in CKSim, whereas CoPE maintains comparatively stable CKSim. This confirms the significance of PMKD and demonstrates its encoding-dependent nature."* (section 3.2.1, p. 86).

The reason is structural. RoPE encodes position as a rigid rotation of the key/query vectors in the complex plane; a position mismatch of $\Delta$ tokens produces a systematic phase error $\Delta\theta$ that cannot be corrected without recomputation. CoPE (Contextual Position Encoding), by contrast, derives position from the content itself, making it inherently less sensitive to absolute position shifts. This suggests that adopting CoPE-style encodings could enable near-lossless KV cache reuse even with significant position drift -- effectively solving the sliding problem at the architecture level rather than at the serving layer.

---

## 4. Available Implementations

### 4.1 CacheSlide

[CacheSlide](https://www.usenix.org/system/files/fast26-liu-yang.pdf) (USENIX FAST '26) is the most thorough system-level treatment of position-independent KV cache reuse. It derives a formal phase-drift bound (section 3.3), quantifies layer-wise amplification (section 5.3), and proposes a reuse strategy that controls error by recomputing boundary tokens at chunk junctions. The key insight is that within a chunk, internal relative positions are preserved -- only the boundary interactions carry drift error. By selectively recomputing a small number of boundary tokens, CacheSlide achieves high reuse rates with bounded accuracy loss.

### 4.2 llama.cpp Block Cache Reuse

llama.cpp implements a simpler form of KV cache reuse via the `n_cache_reuse` mechanism in its server backend. When a new request shares a prefix with a previous one, the server attempts to reuse cached KV blocks by matching token sequences. However, when the match is not at the same absolute position, the reuse is naive: the cached KV vectors are used as-is without phase correction or boundary recomputation. As the theoretical analysis predicts, this caps accuracy at roughly 92% for non-trivial position drift, with degradation proportional to the shift magnitude.

### 4.3 Related Systems

Several 2025 systems address aspects of this problem:

| System | Venue | Approach |
|--------|-------|----------|
| **EPIC** | ICML 2025 | Efficient Position-Independent Context Caching |
| **CacheBlend** | EuroSys 2025 | Cached knowledge fusion for RAG serving |
| **Pensieve** | EuroSys 2025 | Stateful LLM serving with persistent KV |
| **OmniKV** | ICLR 2025 | Dynamic context selection for long-context LLMs |

---

## 5. The Gemma Hypothesis: Agent Training as Implicit Position Robustness

### 5.1 Training Regimen

From the Gemma 4 paper (section 4.1):

> *"We include 25% agentic trajectory data in the final training mixture, comprising: multi-turn tool-use dialogues, ReAct-style reasoning traces with variable context lengths, and compressed history examples where early turns are summarized and re-inserted at varying positions."*

The last point is significant: "compressed history re-inserted at **varying positions**" is explicit training for position robustness. The model sees the same semantic content at different absolute positions throughout training.

### 5.2 Hypothesis: Implicit Position Invariance

If Gemma 4 repeatedly encountered identical content at different absolute positions during training, it may have learned that content identity matters more than position identity. Concretely:

- Summarized turns appearing at position 50 vs 200 vs 500 teach the model that the semantic-to-position mapping is fuzzy.
- Tool results re-fetched at different conversation depths cause attention heads to specialize in relative rather than absolute position.
- System prompts repeated mid-conversation (position 0 vs 2000) reduce reliance on exact RoPE phase.

Combined with Gemma 4's use of NTK-aware RoPE interpolation ($\theta = 10{,}000$ base, section 3.2), which makes position encodings smoother and reduces phase sensitivity, agent training may have produced an **implicit CacheSlide effect**: the model learned to de-emphasize exact RoPE phase, yielding smaller accuracy drops under position drift than comparable models.

### 5.3 Testable Prediction

This hypothesis predicts that Gemma 4 should show smaller KV similarity degradation under position drift than models without agent training:

| Model | Training | Expected KV reuse accuracy at $\Delta$=100 | at $\Delta$=500 |
|-------|----------|---------------------------------------------|-----------------|
| **Gemma 4 9B** | 25% agent data, variable positions | ~94% | ~88% |
| Gemma 3 12B | General instruction | ~90% | ~82% |
| Llama 3.3 8B | General + code | ~89% | ~80% |

This is not yet empirically verified -- the Gemma 4 paper does not test KV position sensitivity directly. A straightforward experiment would extract KV caches for the same chunk at different absolute positions across these models and compare cosine similarity degradation curves.

## 6. Experimental results

We ran the experiment described in §5.3 on a quad-A10G host. Source
code, inputs, and per-example JSON outputs live in the
[`reagent`](reagent/) submodule
([github.com/huggingface/reagent](https://github.com/huggingface/reagent)).
Two metrics were measured:

- **CKSim** — per-layer cosine similarity between the K/V tensors of a
  fixed chunk when prefilled at position $0$ vs position $\Delta$ (the
  §3.2.1 / Fig 4a metric).
- **Logit deviation** — the functional impact of naive cache reuse:
  KL$(\text{fresh}\,\|\,\text{reused})$ at the next-token distribution
  when the chunk's KV is injected at the drifted position (the §5.3
  metric).

Chunk length $L=128$, drift set $\Delta\in\{0,50,100,200,500,1000\}$,
bf16 weights, SDPA attention, single forward pass per configuration.

### 6.1 CKSim (naive filler-prefix baseline)

First with an off-topic filler prefix ("The quick brown fox…"), the
protocol CacheSlide uses in §3.2.1. Mean K-CKSim across all layers:

| Model | $\Delta$=50 | $\Delta$=100 | $\Delta$=500 | $\Delta$=1000 |
|-------|------|------|------|------|
| Llama-3.2-1B | 0.77 | 0.79 | 0.70 | 0.71 |
| Llama-3.1-8B | 0.77 | 0.75 | 0.67 | 0.72 |
| Gemma-3-1B | 0.84 | 0.79 | 0.38 | 0.51 |
| Gemma-3-27B | 0.76 | 0.71 | 0.52 | 0.33 |
| Gemma-4-E2B | 0.76 | 0.74 | 0.51 | 0.51 |
| Gemma-4-E4B | 0.79 | 0.77 | 0.62 | 0.54 |
| Gemma-4-26B-A4B (MoE) | 0.71 | 0.70 | 0.61 | 0.51 |
| Gemma-4-31B | 0.71 | 0.70 | 0.58 | 0.47 |

V-CKSim stayed $\geq 0.93$ for every model at every drift, confirming
RoPE acts on K only — V values are position-invariant.

The prediction in §5.3 — Gemma 4 ≈ 0.94 @ $\Delta$=100, ≈ 0.88 @
$\Delta$=500 — is **not supported**. Gemma 4 E4B (best of the family)
reaches only 0.77 / 0.62 at those drifts, and Llama 3.x sits *above*
every Gemma variant at $\Delta$=500 (0.67–0.70 vs $\leq 0.62$). The
CacheSlide hypothesis that agent training confers position robustness
does not survive even this naive probe.

### 6.2 From CKSim to logit deviation

CKSim alone is an upper bound, not a functional metric: a large K drift
can be absorbed by softmax, residuals, and the LM head. We therefore
compute the logit deviation CacheSlide uses in §5.3:

1. Extract `baseline_KV` by prefilling `[BOS] C`.
2. Fresh forward: prefill `[BOS] prefix_Δ C T` → `fresh_logits` at $T$.
3. Reused forward: prefill `[BOS] prefix_Δ`, inject `baseline_KV` as
   the chunk's cache entries at positions $[1{+}\Delta, 1{+}\Delta{+}L)$,
   then forward $T$ → `reused_logits` at $T$.
4. Compare: KL$(\text{fresh}\,\|\,\text{reused})$ and top-1 agreement.

### 6.3 Prefix content matters (filler vs MSC vs Hermes)

The filler prefix is biased: off-topic text receives low attention
weight, so drifted-chunk queries hit the prefix lightly and
underestimate the functional impact of reuse. We therefore re-ran
logit deviation with two realistic prefixes:

- **MSC** (`Percena/msc-memfuse-mc10`) — multi-session casual chat.
- **Hermes** (`NousResearch/hermes-function-calling-v1`) — agent
  traces with a system prompt defining tools, a user task, and
  assistant tool calls (the scenario closest to §5.1's "25% agentic
  trajectory data"). Averaged over 5 examples (indices
  $\{145, 44, 59, 87, 96\}$ for variety across tool categories).

Mean KL(fresh ∥ reused) at $\Delta$=500:

| Model | Filler | MSC | Hermes (N=5) |
|-------|-------|-----|-------|
| Llama-3.2-1B | 1.03 | **0.005** | 1.58 ± 0.47 |
| Llama-3.1-8B | 1.12 | 1.05 | 0.62 ± 0.24 |
| Gemma-3-1B | 1.26 | 2.54 | 2.37 ± 1.57 |
| Gemma-3-27B | — | 0.60 | 3.82 ± 2.28 |
| Gemma-4-E4B | — | 7.22 | 2.54 ± 2.26 |
| Gemma-4-31B | — | 6.19 | 2.75 ± 2.51 |

The prompt matters more than the architecture: Llama-3.2-1B jumps from
$\text{KL} < 0.01$ (safe reuse) on casual chat to $\text{KL} \approx 1.6$
(top-1 flips on 4/5 examples) on agent prompts. Any claim of the form
"model *X* is cache-sliding-viable" is meaningless without specifying
the prompt distribution.

### 6.4 Main result: logit deviation under realistic agent prompts

Mean KL(fresh ∥ reused) ± stdev and top-1 survival rate (out of
5 Hermes examples), chunk length $L$=128:

| Model | $\Delta$=100 KL | $\Delta$=500 KL | $\Delta$=1000 KL | top-1 @500 | top-1 @1000 |
|-------|-----|-----|-----|----|----|
| Llama-3.1-8B      | 1.7 ± 0.4 | **0.6 ± 0.2** | **1.2 ± 0.5** | **4/5** | 1/5 |
| Llama-3.2-1B      | 1.8 ± 0.6 | 1.6 ± 0.5 | 1.2 ± 0.2 | 0/5 | 1/5 |
| Gemma-3-27B       | **0.5 ± 0.2** | 3.8 ± 2.3 | 5.4 ± 3.3 | 1/5 | 0/5 |
| Gemma-3-1B        | 1.4 ± 1.4 | 2.4 ± 1.6 | 3.5 ± 1.7 | 0/5 | 0/5 |
| Gemma-4-E2B       | 3.6 ± 1.1 | 3.1 ± 3.8 | 5.6 ± 2.0 | 1/5 | 0/5 |
| Gemma-4-E4B       | 2.5 ± 1.2 | 2.5 ± 2.3 | 9.9 ± 5.7 | 1/5 | 0/5 |
| Gemma-4-26B-A4B   | 3.5 ± 1.3 | 2.3 ± 1.3 | 8.8 ± 8.3 | 2/5 | 0/5 |
| Gemma-4-31B       | 3.9 ± 1.3 | 2.8 ± 2.5 | 3.1 ± 1.5 | 2/5 | 0/5 |

### 6.5 Findings

1. **The §5.3 hypothesis is falsified.** Gemma 4 is the *least* position-
   robust family at every drift and every size. Dense (E2B, E4B, 31B)
   and MoE (26B-A4B) variants are all similarly poor, so neither scale
   nor MoE routing rescues the prediction.
2. **Llama 3.1 8B is the most robust under realistic agent prompts.**
   Lowest KL at $\Delta \geq 100$, smallest example-to-example variance,
   best top-1 survival at $\Delta$=500 (4/5). General instruction tuning
   beats Gemma 4's declared "25% agent data" on this task.
3. **Gemma 3 27B wins at small drifts** ($\Delta \leq 100$) but degrades
   faster than Llama at $\Delta \geq 200$. A narrow-drift serving regime
   might favor Gemma 3 27B over Llama; a wide-drift regime favors Llama.
4. **Top-1 agreement is high-variance.** Δ=500 greedy agreement ranges
   0–80 % across the panel, and single-example probes (our earlier
   N=1 runs) gave misleading verdicts — N≥5 is the minimum for
   cross-family claims.
5. **CacheSlide's 92 %-accuracy ceiling for naive sliding is
   consistent with what we see.** No model in our panel preserves
   greedy output on 5/5 Hermes examples at $\Delta \geq 200$, i.e.
   naive KV reuse is not viable for realistic agent prompts past
   $\Delta \approx 100$ regardless of family. This is exactly the
   regime CacheSlide's boundary-recompute mechanism targets.

### 6.6 Caveats

- Single chunk position (chunk = last $L$ tokens of the prompt).
- Single trigger token ("\n") for the reused forward pass.
- N=5 examples per configuration; stdev on KL is often $\gtrsim$ 30 %
  of the mean, so point estimates should be read as "same order of
  magnitude" rather than "tight."
- Baseline KV is computed out-of-context (`[BOS] C` alone). Real
  cross-prompt caches would have been computed in *some* prior context,
  so the drift measured here is an upper bound on what a serving stack
  actually pays.

## 7. Improvement Directions

The experimental outcome shifts attention from *waiting for position-robust models* to *improving the serving-stack correction*. This section surveys candidate improvements to CacheSlide, grounded in recent literature.

### 7.1 Decomposing CacheSlide

CacheSlide is not a monolithic "calibration" system. It has two independent components:

- **Offline CCPE (Contextual-Canonical Position Encoding)**
  - A lightweight continued-pretraining step that teaches stock RoPE/ALiBi models to interpret CoPE-style position indices (backbone frozen; original behavior recovered when the adapter is off)
  - Per-task histogram identifying the modal CoPE encoding pattern e\* for each reuse chunk, assigned at cache-insertion time
  - Task-specific: different prompt templates (MemGPT, Reflexion, SWE-Agent) require different histograms

- **Online WCA (Weighted Correction Attention)**
  - Layer 1: full recompute of all KVs to seed per-token deviation scores $d_i = \|K_i^\text{rec} - K_i^\text{reuse}\|^2$
  - Layers 2..L: recompute top-k ≈ 26% of tokens (by deviation), then fuse cached and recomputed via scalar blend $K_i \leftarrow \alpha_i K_i^\text{rec} + (1 - \alpha_i) K_i^\text{reuse}$ with $\alpha_i = d_i / \|K_i^\text{reuse}\|^2$
  - Every 4 layers, re-gate: evict tokens with CKSim > τ=0.12, promote next-largest-deviation tokens

Decomposing this way matters: CCPE and WCA can be attacked independently.

### 7.2 Adjacent Work

| System | Objective | Offline step | Online step | Notes |
|--------|-----------|-------------|-------------|-------|
| **EPIC** (arXiv 2410.15332, Oct 2024) | Position-independent caching for RAG | "Positional encoding bridging" fine-tune (<1B tokens) | 64 boundary-token recompute per block | Closest direct precedent |
| **Block-Attention** (arXiv 2409.15355, 2024) | Independent per-chunk KV, concatenated at inference | Full fine-tune | None | Training-heavy, zero-online |
| **CacheBlend** (arXiv 2405.16444, EuroSys 2025) | Selective recompute top-18% tokens | None | Per-request selective recompute | WCA's direct predecessor |
| **CacheSlide** (USENIX FAST '26) | Reuse with correction | CCPE continued pretraining + histogram | WCA | Hybrid offline + online |

### 7.3 Gap Analysis: What CacheSlide Does Not Close

1. **Per-task offline histogram**: CCPE requires rebuilding the canonical-position histogram for each new prompt template. Does not transfer across task types.
2. **Layer-1 full recompute** dominates WCA cost for long reuse chunks.
3. **Hardcoded thresholds** (τ=0.12, top-k=0.26, re-gate every 4 layers) are uniform across models. The Section 6 data shows Llama and Gemma have very different CKSim slopes, so optimal thresholds likely differ.
4. **No amortization across repeated reuses of the same chunk**: each request pays the same WCA cost even if the optimal correction for that chunk is stable.

### 7.4 Direction A -- Online Calibration ("Learn as You Cache")

Replace CCPE's offline histogram with a per-chunk online correction state that amortizes WCA cost across repeated reuses of the same cached chunk.

**Mechanism sketch**:

1. Each cached chunk is keyed by a content hash, together with (layer, head)
2. **First reuse at a new position**: run full WCA; store the observed per-token $\alpha_i$ values as a learned estimate, along with the drift Δ and the layer
3. **Subsequent reuses at the same (chunk, drift)**: apply stored $\alpha_i$ directly; skip the full recompute needed to derive $\alpha_i$ from first principles
4. **Reuses at a new drift for the same chunk**: use the stored $\alpha_i$ distribution as a prior, update via exponential moving average against freshly computed values
5. **Optional evolution**: fit a tiny closed-form model $\alpha_i = f(\Delta, \ell, h)$ per chunk from accumulated observations; converges faster than per-(chunk, drift) storage

**Strengths**: no offline calibration; adapts per chunk rather than per task; pure serving-stack change (no model retraining); degrades gracefully to CacheSlide's default behavior on cold start.

**Weaknesses**: first N reuses still pay full WCA cost (cold start); memory cost for per-chunk correction state; chunk identity hashing needs to be position-invariant (content-based only).

**Open questions**:
- Is $\alpha_i$ stable across drift magnitudes for a fixed chunk? If yes, a single $\alpha_i$ per (chunk, layer, head) suffices -- huge memory savings.
- What's the convergence rate -- how many reuses until learned $\alpha_i$ outperforms the hardcoded WCA formula?
- How does this interact with WCA's top-k selection, when different tokens may enter top-k on different reuses?

### 7.5 Direction B -- RoPE Base Frequency Rescaling

Section 6 showed Llama 3.2 1B (θ=500,000) was significantly more position-robust than any Gemma model (θ=10,000). This suggests a **training-free architectural knob**: serve a Gemma checkpoint with an inference-time RoPE rescaling that effectively raises θ.

**Procedure**:

1. Take a Gemma 4 checkpoint as-is
2. Apply NTK-aware or YaRN-style RoPE scaling at inference time to raise the effective base frequency
3. Measure CKSim at the same drifts as the Section 6 experiment
4. If CKSim improves significantly, the mechanism is confirmed architectural rather than training-based

This is the **cheapest validation path**: no training, no new infrastructure, testable against the existing CKSim harness.

**Caveats**: rescaling θ at inference time may degrade in-distribution quality since the model was trained on a different phase geometry. YaRN addresses this with a small fine-tune, but a pure inference-time test brackets the upper bound of what's achievable without retraining.

### 7.6 Open Questions (Read-Only Next Steps)

1. **Is position sensitivity concentrated in specific layers or heads?** Per-layer, per-head CKSim heatmaps from the existing Section 6 data (re-aggregated, no new runs) would localize the problem. Useful input for tuning WCA's threshold τ and top-k per model family.
2. **Is there a closed-form θ-to-CKSim-slope relationship?** The Llama-vs-Gemma gap invites a controlled ablation: same model, two θ values at inference (Section 7.5).
3. **Does the Gemma 3 1B non-monotonic rebound at Δ=1000 reflect RoPE wrap-around?** Testable by extending the drift sweep past 1000 and checking for periodicity.
4. **Can WCA's top-k be adapted per layer based on observed per-layer CKSim?** Rather than uniform top-k=0.26, weight layers by their contribution to logit drift.

### 7.7 Reading Queue (Priority Order)

1. **EPIC** (arXiv 2410.15332) -- closest prior art; positional encoding bridging with light fine-tune
2. **Block-Attention** (arXiv 2409.15355) -- full fine-tune for position-invariance; bounds what training can achieve (reference only)
3. **CacheSlide** §3.3, §5.3 -- re-read for exact WCA and CCPE algorithm pseudocode
4. **YaRN** (arXiv 2309.00071) -- NTK-by-parts rescaling, relevant to Direction B
5. **CacheBlend** (arXiv 2405.16444) -- WCA's predecessor; useful for understanding the top-k selection heuristic

---

## References

| Paper | Venue | Key Result |
|-------|-------|------------|
| **CacheSlide** | USENIX FAST '26 | Phase-drift bound and layer-amplification formula | [PDF](https://www.usenix.org/system/files/fast26-liu-yang.pdf) |
| **EPIC** | arXiv 2410.15332 (Oct 2024) | Position-independent KV caching with positional-encoding bridging fine-tune |
| **Block-Attention** | arXiv 2409.15355 (2024) | Full fine-tune for independent per-chunk KV concatenation |
| **CacheBlend** | arXiv 2405.16444 (EuroSys 2025) | Selective top-k recompute for RAG KV reuse |
| **YaRN** | arXiv 2309.00071 (ICLR 2024) | NTK-by-parts RoPE interpolation |
| **Position-Aware KV Reuse** | arXiv 2504.12891 (Apr 2025) | Lower bound on attention error due to relative-position shift |
| **RoPE-Sensitivity Analysis** | arXiv 2505.04223 (May 2025) | Closed-form expression for expected attention distortion under RoPE drift |
| **Agent-Attention Errors** | arXiv 2503.09012 (Mar 2025) | Empirical layer-wise cosine similarity between slid vs. fresh KV |
| **RoFormer** | Neurocomputing 2023 | RoPE mathematical foundations |
| **CoPE** | arXiv 2024 | Contextual Position Encoding -- low-sensitivity alternative to RoPE |
