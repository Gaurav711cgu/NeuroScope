# NeuroScope — Production Requirements Document

**Domain:** Mechanistic Interpretability Research / AI Safety  
**Status:** 🟡 RESEARCH-IN-PROGRESS — expand N before publishing, Nov 2026 MATS deadline  
**Target:** MATS Spring 2027 Fellowship, AI Safety Labs (Anthropic, DeepMind), Research Roles  
**Stack:** TransformerLens · GemmaScope SAE · Neuronpedia API · PyTorch · Jupyter

---

## 1. Problem Statement

### Research Question
Can we identify, precisely locate, and causally confirm the specific attention heads and MLP sublayers responsible for the Indirect Object Identification (IOI) circuit in GPT-2 Small? And can we use Sparse Autoencoder (SAE) features from GemmaScope to label what these components are computing in human-interpretable terms?

### Why This Matters (Research Significance)
The IOI task — "Mary gave John a gift. She told ___" (answer: John) — requires the model to:
1. Identify that "She" refers to Mary (pronoun resolution)
2. Identify that Mary is the indirect object recipient
3. Predict "John" as the answer (subject of the second clause)

Understanding which attention heads handle each sub-computation is a foundational mech interp problem. The Anthropic IOI paper (Wang et al., 2022) identified this circuit; replicating it with modern tooling (TransformerLens, GemmaScope SAE) and verifying it causally (not just correlationally) is a contribution to the field.

### What "Causal" Means Here
Finding attention heads that activate on IOI examples is **correlation**. Verifying causality requires **path patching**: ablate a head → measure if IOI performance degrades. If yes, the head is causally necessary. This is the key distinction between observational mech interp and causal mech interp.

---

## 2. Research Architecture

```
IOI Dataset (Synthetic prompts)
     │
     ├── N=50 (current) → N=200+ (target)
     │
     ▼
TransformerLens (GPT-2 Small)
     │
     ├── Forward pass with activation caching
     │   ├── Attention head activations: [num_layers × num_heads × seq_len × d_head]
     │   └── MLP sublayer activations: [num_layers × seq_len × d_mlp]
     │
     ├── IOI Faithfulness Metric
     │   ├── logit_diff = P(John) - P(Mary) on corrupted IOI prompt
     │   └── Normalized: (full_model - ablated) / (full_model - zero_ablated)
     │
     ├── Path Patching (Causal)
     │   ├── Corrupt prompt: "Mary gave John → ABC gave XYZ"
     │   ├── Patch one head's output from corrupt → clean run
     │   └── Measure Δ IOI faithfulness → causal importance score
     │
     └── GemmaScope SAE Integration
         ├── Neuronpedia API: feature activation lookup
         ├── Feature autointerp: what concept does this SAE feature encode?
         └── Steering experiments: push feature activation → does behavior change?
```

---

## 3. Methodology Decisions (Research ADRs)

| Decision | Alternatives | Chosen | Rationale |
|---|---|---|---|
| Model | GPT-3.5 (opaque), LLaMA, Pythia, GPT-2 Small | **GPT-2 Small** | Fully open weights + activations; TransformerLens has complete support; sufficiently complex for IOI circuit but small enough for ablation sweeps on a T4 |
| Circuit discovery method | Activation patching, Logit attribution, Causal tracing | **Path patching** | Activation patching can conflate direct and indirect effects. Path patching (Goldowsky-Dill et al., 2023) isolates specific information flow paths — more causally precise. |
| SAE tool | Custom SAE training, Eleuther SAE, OpenAI SAE | **GemmaScope SAE + Neuronpedia** | GemmaScope provides pre-trained, high-quality SAEs for Gemma models; Neuronpedia API provides human-readable feature labels without manual annotation. Correct tooling choice for applied mech interp. |
| Faithfulness metric | Raw accuracy, KL divergence, Logit diff | **Normalized logit diff** | Raw accuracy has ceiling effects on easy IOI examples; logit diff (P(correct) - P(incorrect)) is more sensitive; normalization allows comparison across ablation levels. |
| Sample size | N=50 (current), N=100, N=200+ | **Target: N=200+** | N=50: p-values unreliable (large confidence intervals on path patching scores). N=200: sufficient for statistical claims. N=500: publishable to Alignment Forum as research note. |

---

## 4. Current Results (N=50 — Preliminary)

> ⚠️ **Statistical caveat:** All values below have wide confidence intervals at N=50. Do not publish or cite until N≥200. MATS reviewers will ask about statistical power.

| Finding | Metric | Value (N=50) | Status |
|---|---|---|---|
| IOI circuit replicated | Faithfulness score | Measured | ⚠️ Preliminary |
| Key attention heads identified | Path patching importance | Head rankings computed | ⚠️ Preliminary |
| SAE feature labels | Neuronpedia auto-interp | Features labeled | ⚠️ Qualitative only |
| Causal confirmation | Ablation faithfulness drop | Measured per head | ⚠️ Needs N=200+ |

### Statistical Power Analysis
```python
# At N=50, effect detection requires Cohen's d > 0.4 (medium-large effect)
# At N=200, effect detection requires Cohen's d > 0.2 (small-medium effect)
# At N=500, effect detection requires Cohen's d > 0.13 (publishable)

from scipy import stats
import numpy as np

# Current N=50 confidence intervals are ~2.5× wider than N=200
# This means: a "significant" finding at N=50 may not replicate
# Expand to N=200 BEFORE publishing ANY quantitative claims
```

---

## 5. Research Checklist

| Item | Status | Notes |
|---|---|---|
| IOI faithfulness metric implementation | ✅ Correct | Matches Wang et al. (2022) definition |
| Path patching implementation | ✅ Implemented | Causal (not just activation) patching |
| GemmaScope SAE integration | ✅ Correct | Neuronpedia API connected |
| N=200+ sample | ❌ In progress (N=50) | Must complete before any publication |
| Steering experiment integrated | ❌ Disconnected | Steering → connect to main circuit findings |
| Findings post published | ❌ Not yet | Target: Alignment Forum / LessWrong |
| Replication of Wang et al. | ⚠️ Partial | N too small for confidence |

---

## 6. System Contract (Research Reproducibility)

### Reproducibility Requirements
| Requirement | Status | Notes |
|---|---|---|
| Random seed fixed | Must verify | All stochastic operations seeded |
| Environment pinned | `requirements.txt` present | TransformerLens version critical |
| Dataset specification | Synthetic IOI generation script | Reproducible prompt generation |
| Results committed | `/results/` directory | All runs committed, not just best |

### Honest Limitations (Must State in Any Publication)
1. **Sample size:** N=50 is insufficient for statistical confidence. All findings are preliminary.
2. **Model scope:** Results are for GPT-2 Small only. Whether the IOI circuit generalizes to larger models is unknown.
3. **Correlation vs. causation:** Path patching provides causal evidence within the model's computation graph, but does not imply the same circuit exists in biological neural networks.
4. **SAE feature quality:** GemmaScope SAEs are trained on Gemma, not GPT-2. Feature transfers are approximate.

---

## 7. MATS Spring 2027 Roadmap

### What MATS Reviewers Look For
- Clear research question with stated hypothesis
- Rigorous methodology (causal, not correlational)
- Statistical validity (N is always the first question)
- Novel contribution beyond replication (what's NEW?)
- Public output (Alignment Forum post is evidence of research communication)

### Timeline
```
Aug-Sep 2026: Expand IOI dataset N=50 → N=200+ (1 week compute)
              Run path patching at N=200+
              Compute confidence intervals on all findings

Oct 2026:    Novel angle — apply GemmaScope SAE to a different circuit
              Options: (1) Greater-than circuit, (2) Docstring attribution heads,
              (3) In-context learning heads

Nov 2026:    Write findings_post.md
              Publish to Alignment Forum as research note
              Submit MATS Spring 2027 application with link to published post

Nov-Dec 2026: Application review period
```

### The Novel Angle (Required for MATS)
Replication is necessary but not sufficient. Your novel contribution options:
1. **Cross-model IOI**: Does the GPT-2 IOI circuit structure transfer to Pythia-1.4B? What's preserved vs. different?
2. **GemmaScope SAE → different circuit**: The same SAE tooling applied to a circuit beyond IOI (e.g., factual recall heads in Gemma-2)
3. **Entropy spike analysis**: You have entropy-spike causal circuit discovery — make this the headline finding rather than IOI replication

---

## 8. Interview Defense Points (AI Safety Lab Context)

**Q: What is the difference between activation patching and path patching?**  
A: Activation patching replaces a component's output with a corrupted run's output and measures the effect. Path patching goes further: it traces a specific information path through the model (e.g., query vectors from head 4.7 to the output of head 10.0) and patches only along that path. This isolates indirect effects — head A might affect head B which affects the output, and you want to attribute the credit to A, not just B.

**Q: Why is N=50 a problem?**  
A: The path patching scores have variance across different IOI prompt structures (different names, different syntactic forms). With N=50, the confidence interval on each head's importance score is ±20-30 percentage points. A head that looks causally important at N=50 may not replicate at N=200. For any quantitative claim about "head X is causally necessary for IOI", you need N≥200 for the confidence interval to be meaningful.

**Q: What does GemmaScope SAE tell you that activation analysis doesn't?**  
A: Activation analysis tells you which neurons fire. SAE decomposition finds sparse, interpretable features that are superposed in the neuron activations — neurons compute many features simultaneously (the superposition hypothesis). GemmaScope SAE decomposes neuron activations into monosemantic features that Neuronpedia labels with human-interpretable descriptions. This is the difference between "layer 6 head 3 is active" and "layer 6 head 3 is computing indirect object binding".

---

*Last updated: Aug 2026 | Maintainer: Gaurav Kumar Nayak | Target: MATS Spring 2027, AI Safety Labs*
