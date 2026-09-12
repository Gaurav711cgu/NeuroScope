# Mapping Hallucination Dynamics in Gemma-2: A Sparse Autoencoder Approach

## Introduction

As large language models scale, detecting and mitigating hallucinations at the latent level remains a critical challenge for alignment. Building on the k-SAE methodology presented by Gao et al. (OpenAI, 2024) and the GemmaScope architecture by Lieberum et al. (Google DeepMind, 2024), we investigate the activation dynamics of hallucinations in `google/gemma-2-2b-it`. 

Specifically, we apply a survival analysis framework—modeling hallucination onset via a Cox Proportional Hazards model—using latent feature activations extracted from a 16K-width JumpReLU Sparse Autoencoder (SAE) at Layer 12. This post details our methodology, evaluation metrics, and preliminary baseline validations designed to meet stringent interpretability standards.

## Methodology

### Latent Feature Extraction
We utilize the canonical GemmaScope SAE for Layer 12 of `gemma-2-2b-it`. The SAE enforces sparsity via a JumpReLU activation function. To avoid memory bottlenecks when computing dense dictionary orthogonality metrics (Gram matrix allocations exceeding 1 GB), we subsample the SAE decoder weight matrix to 2,048 directions.

### Survival Analysis of Hallucinations
Rather than treating hallucination as a static binary classification problem, we frame it as a time-to-event process during autoregressive generation. We extract the latent activations across the sequence and fit a Cox Proportional Hazards model with $L_2$ regularization:

$$ h(t | x) = h_0(t) \exp(\beta^T x) $$

Where $x$ represents the SAE feature activations. To validate the statistical significance of the distinct survival curves between factual and hallucinated generation paths, we compute the log-rank test $p$-value comparing the two underlying distributions.

### Quality and Sparsity Metrics
To ensure the integrity of our SAE and the isolated features, we strictly replicate the evaluation suites from recent literature:

1. **SAE Quality (Lieberum et al.)**:
   - **$L_0$ Norm**: Measures the average number of active features per token.
   - **Fraction of Variance Unexplained (FVU)**: Evaluates reconstruction fidelity.
   - **$\Delta$ LM Loss**: Quantifies the downstream cross-entropy degradation when the residual stream is entirely replaced by the SAE reconstruction.

2. **Ablation Sparsity (Gao et al.)**:
   - For a given highly predictive hallucination feature, we perform causal intervention (ablation).
   - We compute the downstream sparsity of the logit changes as $\frac{(L_1 / L_2)^2}{V}$, validating whether the feature represents a concentrated semantic concept or a diffuse polysemantic vector.

3. **Auto-Interpretability (Bills et al.)**:
   - Top-activating sequences for the hallucination feature are extracted.
   - An LLM explainer generates a semantic hypothesis and simulated activation scores.
   - We compute the Pearson correlation ($r$) between simulated and ground-truth activations, targeting $r \ge 0.10$ for validation.

### Causal Steering
We validate the causal role of the identified hallucination features through activation steering (residual addition). By clamping a composite vector of top features multiplied by a scaling factor $\alpha$, we can theoretically suppress hallucinations. Crucially, we monitor the perplexity delta ($\Delta \text{PPL}$) to guarantee we do not breach the semantic collapse boundary ($\Delta \text{PPL} \ge 1.0$), ensuring the model retains coherent linguistic capabilities.

## Preliminary Pipeline Validation

Our end-to-end evaluation script (`scripts/paper_results.py`) validates the infrastructure required for this analysis. 

- **Dictionary Orthogonality**: The $W_{dec}^T W_{dec}$ matrix exhibits an acceptable Frobenius norm for off-diagonal elements, suggesting a well-conditioned latent space.
- **Probe Training**: Our 1D logistic probes (trained via Newton-Raphson) successfully converge without single-class fabrication errors.
- **Cox Model Convergence**: The L-BFGS-B optimization for the partial log-likelihood converges reliably across batch sizes up to 128.

## Next Steps

With the evaluation harnesses fully aligned with the OpenAI and DeepMind SOPs, our immediate next step involves scaling the evaluation over the TruthfulQA and clinical trial benchmark datasets. We expect to isolate a sparse subset of features ($k < 5$) that reliably predict hallucination onset at $p < 0.01$ (log-rank test), maintaining steering efficacy well within the $\Delta \text{PPL} < 1.0$ safety threshold.
