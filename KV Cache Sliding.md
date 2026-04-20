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

---

## References

| Paper | Venue | Key Result |
|-------|-------|------------|
| **CacheSlide** | USENIX FAST '26 | Phase-drift bound and layer-amplification formula | [PDF](https://www.usenix.org/system/files/fast26-liu-yang.pdf) |
| **Position-Aware KV Reuse** | arXiv 2504.12891 (Apr 2025) | Lower bound on attention error due to relative-position shift |
| **RoPE-Sensitivity Analysis** | arXiv 2505.04223 (May 2025) | Closed-form expression for expected attention distortion under RoPE drift |
| **Agent-Attention Errors** | arXiv 2503.09012 (Mar 2025) | Empirical layer-wise cosine similarity between slid vs. fresh KV |
| **RoFormer** | Neurocomputing 2023 | RoPE mathematical foundations |
| **CoPE** | arXiv 2024 | Contextual Position Encoding -- low-sensitivity alternative to RoPE |
