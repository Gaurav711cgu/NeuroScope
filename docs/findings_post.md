# Early Hallucination Detection in Multi-Step Reasoning: Entropy vs Attention Diffusion vs Feature Drift

**Author:** NeuroScope Interpretability Team  
**Published:** June 2026  
**Forum:** LessWrong / Alignment Forum  

---

## Executive Summary
How do we know when a chain-of-thought (CoT) reasoning model has gone off the rails before it emits the final incorrect answer? In this post, we present findings from evaluating **50 multi-step reasoning trajectories** of `Gemma-2-2b-it` on factual question-answering tasks (TriviaQA). 

We compare three step-level diagnostic signals:
1. **Next-Token Output Entropy** (representing model uncertainty at the vocabulary unembedding layer).
2. **Attention Diffusion** (measured as the entropy of key-value attention distributions in the model's final layer).
3. **Feature Drift** (quantified as the variance of top active SAE features reconstructed via 16k-width GemmaScope JumpReLU Sparse Autoencoders at Layer 12).

Our results show a stark difference in predictive horizon: **Next-token vocabulary entropy predicts factual errors 1.8 steps earlier than attention diffusion**, correlating strongly with final correctness ($\rho = -0.71$). Attention diffusion correlates moderately ($\rho = -0.43$) but serves as a late-stage indicator, spiking only when the model is on the verge of emitting the incorrect claim. Feature drift shows very weak correlation ($\rho = -0.18$), suggesting that representations drift even during successful reasoning chains, making it a poor signal for factual uncertainty.

---

## 1. Introduction: The Silent Failure of Chains of Thought
Multi-step reasoning (Chain-of-Thought) has dramatically improved the capability of Large Language Models (LLMs) on complex tasks. However, CoT models remain highly susceptible to **hallucination propagation**: once a false claim or mathematical error is introduced in step $N$, the model treats it as ground truth in its context, leading to a cascade of errors that guarantees a wrong final answer in step $N+K$.

Detecting these failures *early*—before the model generates the final incorrect answer—is critical for safety and alignment. If we can identify the exact step where the reasoning chain breaks, we can intervene (e.g., via activation steering, backtracking, or token rejection). 

Using **NeuroScope**, we capture the internal states of `Gemma-2-2b-it` as it solves 50 TriviaQA questions using 5-step CoT reasoning. We analyze whether the model's internal representations signal factual uncertainty early in the trajectory.

---

## 2. Methodology & Signal Definition
We prompt `Gemma-2-2b-it` to answer 50 TriviaQA questions with a strict 5-step CoT reasoning template:
```text
Step 1: Thought: <reasoning>
Step 2: Thought: <reasoning>
...
Step 5: Thought: <reasoning and final answer>
```
At each step, we hook the model's forward pass using `TransformerLens` and extract three diagnostic metrics.

### 2.1 Next-Token Output Entropy (Entropy)
We measure the Shannon entropy of the model's next-token vocabulary distribution at the final token of each step:
$$\text{Entropy}(P) = -\sum_{w \in V} P(w) \log P(w)$$
Where $P$ is the softmax distribution over the vocabulary $V$ (size $\approx 256,000$ for Gemma-2). We normalize this to $[0, 1]$ by dividing by a reference ceiling of $10.0$ nats. High entropy indicates the model is uncertain about the next token to generate.

### 2.2 Last-Layer Attention Diffusion (Attn Diffusion)
Factual retrieval is typically mediated by attention heads routing info from key tokens. Attention Diffusion measures the entropy of the attention pattern in the final layer's heads:
$$\text{Diffusion} = \frac{1}{H} \sum_{h=1}^{H} \text{Entropy}(A_h)$$
Where $A_h$ is the attention distribution over input positions for head $h$. High diffusion indicates the model is attending broadly across the prompt history rather than focused retrieval. Wentao Shi et al. (2023) demonstrated that attention diffusion correlates with hallucination in single-turn QA. We evaluate its performance in multi-step settings.

### 2.3 Feature Drift (Drift Proxy)
To ground the analysis in sparse semantic concepts, we project the last-token residual stream at Layer 12 through a 16k-width canonical JumpReLU GemmaScope SAE. 
Feature drift tracks the mean activation of the features that exhibit the highest activation variance across the trajectory steps. High drift suggests that the model's representation is shifting rapidly from step to step, signaling semantic instability.

---

## 3. Results: The Predictive Horizon
Each of the 50 trajectories was classified as **Correct** (if the final output contained the ground-truth TriviaQA answer) or **Incorrect**. Out of 50 runs, 35 finished correctly (70%) and 15 finished incorrectly (30%).

We computed the Spearman rank correlation ($\rho$) between the step-level signals and the final correctness (binary variable $Y \in \{0, 1\}$).

### 3.1 Spearman Rank Correlation Table
| Diagnostic Signal | Correlation with Correctness ($\rho$) | Warning Horizon (Steps Early) |
|:---|:---:|:---:|
| **Next-Token Vocabulary Entropy** | **-0.71** | **1.8 steps** |
| **Attention Diffusion** | **-0.43** | **0.9 steps** |
| **SAE Feature Drift** | **-0.18** | **0.2 steps** |

*Note: Negative correlation means higher signal values correlate with incorrect final answers.*

### 3.2 Analysis of the Temporal Warning Horizon
The most striking result is the difference in **when** these signals spike during incorrect runs:

```
Step 1       Step 2       Step 3       Step 4       Step 5 (Error)
  |------------|------------|------------|------------|------------|
               * Entropy Spikes (Warning ~1.8 steps early)
                            
                                         * Attn Diffusion Spikes (~0.9 steps early)
```

1. **Entropy as an Early Warning (1.8 Steps Early):**
   In incorrect trajectories, next-token entropy spikes to $>0.7$ as early as **Step 2 or 3**, long before the model prints the final incorrect claim. This suggests that even when the model's output tokens look superficially fluent and "on-track", the underlying probability distribution is already highly disordered, indicating latent uncertainty.
   
2. **Attention Diffusion as a Late-Stage Indicator (0.9 Steps Early):**
   Attention diffusion remains low during the early steps of incorrect runs. It only spikes significantly at **Step 4 or 5**, when the model is on the verge of emitting the incorrect factual claim. This indicates that attention diffusion is a symptom of collapse, rather than a predictive early warning.

3. **Feature Drift is Uncorrelated:**
   Feature drift correlates poorly ($\rho = -0.18$). When inspecting feature timelines, we found that even in correct runs, the model's active SAE features change significantly as it progresses through different sub-tasks, leading to high baseline drift. Thus, drift is a natural property of multi-hop reasoning, not a signal of factual error.

---

## 4. Discussion & Implications for AI Safety
These findings have direct implications for alignment and control:
* **Passive Monitoring:** Vocabulary entropy serves as an excellent lightweight diagnostic tool. By monitoring entropy during multi-step runs, we can halt generation at Step 2 or 3 and save 50% of compute cost by not completing the doomed run.
* **Semantic Grounding via SAEs:** By linking entropy spikes to specific Layer 12 GemmaScope features, we can inspect *what* concept is causing the uncertainty.
* **Steering over Explanations:** Instead of asking another LLM to explain the uncertain features (which can introduce secondary hallucinations), we can actively steer the model. By amplifying the active SAE features via residual addition ($\alpha \cdot W_{dec}$), we can force the model to complete its thoughts under different steering conditions, letting the steered output explain the feature's true causal role.

---

## 5. Limitations & Future Work
* **Sample Size:** This study uses 50 trajectories of `Gemma-2-2b-it`. Extending this to larger datasets (e.g. HotpotQA) and larger models (Gemma-2-9b, Llama-3) is necessary to verify scalability.
* **Quantization Effects:** The models were evaluated in float16. Quantizing models (e.g., 4-bit) changes the softmax logits and may shift the entropy baseline.
* **Causal Attribution:** Pearson co-activation graphs show which features activate together, but true circuit mapping requires path-patching (ablating feature A and measuring feature B), which we plan to explore in the next version of the NeuroScope platform.
