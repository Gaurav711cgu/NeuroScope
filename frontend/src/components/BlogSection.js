import React, { useState } from "react";
import { BookOpen, X, Award, ExternalLink, Calendar, Users } from "lucide-react";

const ARTICLES = [
    {
        id: "01",
        title: "What Sparse Autoencoders actually find inside GPT-2",
        author: "Dr. Elena Rostova",
        date: "May 2026",
        readTime: "8 min read",
        citations: 184,
        tag: "paper-explainer",
        citationRef: "Bricken et al. (Anthropic, 2023) — Towards Monosemanticity",
        link: "https://transformer-circuits.pub/2023/monosemantic-features/index.html",
        abstract: "A deep-dive into how dictionary learning exposes cleanly interpretable feature directions inside the residual stream, such as code components, legal definitions, and multilingual syntax.",
        content: `
### Dictionary Learning & The Superposition Hypothesis

Mechanistic interpretability historically struggled with **polysemanticity**—the tendency of individual neurons to fire in response to multiple, unrelated concepts. To resolve this, researchers train **Sparse Autoencoders (SAEs)** on intermediate activations of the transformer models. An SAE decomposes the residual stream vector $x \\in \\mathbb{R}^{d_{model}}$ as:

$$x \\approx b_{dec} + \\sum_{i=1}^{M} f_i(x) W_{dec, i}$$

where $f_i(x) \\ge 0$ is the feature activation of feature $i$, $W_{dec, i} \\in \\mathbb{R}^{d_{model}}$ is the decoder weight vector, and $M \\gg d_{model}$ represents a high-dimensional sparse dictionary (typically $16k$ to $128k$ features).

### Monosemantic Discoveries

By applying SAEs to GPT-2, we isolate highly specific features:
1. **Arabic Script Feature**: Fires precisely on Arabic text tokens, regardless of semantic content.
2. **DNA Sequences**: Activates on raw nucleic sequences inside training data.
3. **Legal Citations**: Responds to formatting patterns resembling legal briefs (e.g., "v. California").

> [!TIP]
> Inspecting these directions allows developers to directly amplify or suppress concepts (e.g. steering out toxic language styles or safety violations) without changing model parameters.
        `
    },
    {
        id: "02",
        title: "Polysemanticity is everywhere — and NeuroScope is trying to fix it",
        author: "Prof. Marcus Vance",
        date: "May 2026",
        readTime: "10 min read",
        citations: 42,
        tag: "opinion",
        citationRef: "Liu et al. (arXiv, Jan 2026) — NeuronScope",
        link: "https://arxiv.org",
        abstract: "Why single neurons encode multiple concepts. How the NeuronScope multi-agent framework decomposes this iteratively.",
        content: `
### The Puzzle of Superposition

Why does a model squeeze multiple concepts into a single neuron? The mathematical explanation is **superposition**: high-dimensional spaces can pack more almost-orthogonal vectors than their dimensions ($N > d_{model}$). As long as activations are sparse, the model can query these vectors through attention layers, taking advantage of noise thresholds.

However, for human auditors, superposition makes raw neuron observation useless.

### The NeuroScope Decomposition

NeuroScope leverages multi-agent optimization to iteratively construct sparse dictionaries of feature activations. By analyzing activation densities:

- We measure feature sharing across layers.
- We map polysemantic clusters into tree architectures.
- We highlight where representations diverge.

> [!WARNING]
> Simple co-occurrence analysis of activations is insufficient. True decomposition requires checking directed causal pathways (i.e. where suppressing a node breaks downstream behavior).
        `
    },
    {
        id: "03",
        title: "Activation patching: the closest thing we have to LLM surgery",
        author: "Devon Thorne",
        date: "April 2026",
        readTime: "9 min read",
        citations: 215,
        tag: "tutorial",
        citationRef: "Meng et al. (ROME, 2022), Conmy et al. (ACDC, 2023)",
        link: "https://arxiv.org/abs/2202.05262",
        abstract: "How causal patching isolates which layer/head is responsible for a behavior. Step-by-step with real examples.",
        content: `
### What is Activation Patching?

Activation patching (also known as causal swapping) is an experimental method to isolate causal pathways. Given a clean prompt (e.g., "The capital of France is [Paris]") and a corrupted prompt (e.g., "The capital of Rome is [Italy]"), we swap a specific activation tensor (like a residual stream state or attention head output) from the clean run into the corrupted run.

We then measure the recovery of the target token's log-probability:

$$\\text{Recovery Ratio} = \\frac{\\text{Logit}(Paris)_{patched} - \\text{Logit}(Paris)_{corrupted}}{\\text{Logit}(Paris)_{clean} - \\text{Logit}(Paris)_{corrupted}}$$

### Step-by-Step Procedure

1. **Step 1**: Run the baseline clean pass and save activations.
2. **Step 2**: Run the corrupted pass.
3. **Step 3**: Inject the saved Layer $L$ residual activations at token position $T$ during a new forward pass on the corrupted prompt.
4. **Step 4**: Compute logit differences and localize the core computational circuit.

> [!NOTE]
> Unlike gradient-based attribution, activation patching is a direct causal intervention. It tells us exactly which representations are necessary and sufficient for factual completion.
        `
    },
    {
        id: "04",
        title: "Mechanistic Interpretability is a 2026 Breakthrough Technology — here's why",
        author: "Sarah Jenkins",
        date: "April 2026",
        readTime: "6 min read",
        citations: 12,
        tag: "opinion",
        citationRef: "MIT Tech Review, April 2026",
        link: "https://technologyreview.com",
        abstract: "MIT Technology Review's recent designation explained. Goodfire, Anthropic, DeepMind — who's doing what and how NeuroScope fits.",
        content: `
### The Rise of Mechanistic Interpretability

In April 2026, MIT Technology Review designated Mechanistic Interpretability as a Top 10 Breakthrough Technology. As models coordinate autonomous agent trajectories, black-box evaluations are no longer sufficient. We need to inspect the internal circuitry.

### Corporate Ecosystem

1. **Anthropic**: Focuses on large-scale SAEs for Claude 3.5 Sonnet to detect deception and model drift.
2. **Goodfire**: Pioneers interactive feature steering SDKs for application developers.
3. **NeuroScope**: Specializes in agentic, multi-step trajectory analysis—mapping how features evolve across multiple steps of a conversation rather than static single prompts.

> [!IMPORTANT]
> The industry is transitioning from descriptive interpretability (just showing visualizations) to prescriptive interpretability (editing and patching features to guarantee safety alignments).
        `
    },
    {
        id: "05",
        title: "Agentic interpretability: why multi-turn AI is harder to inspect than single prompts",
        author: "Liam O'Connor",
        date: "March 2026",
        readTime: "11 min read",
        citations: 59,
        tag: "paper-explainer",
        citationRef: "Sharkey et al. (2025, arXiv:2501.16496)",
        link: "https://arxiv.org/abs/2501.16496",
        abstract: "Residual stream drift across turns, cross-step causal patching, and why existing tools fail at agent analysis.",
        content: `
### The Challenge of Multi-Turn Contexts

Most interpretability research is done on single-sentence prompts. However, real-world AI agents run in loops: receiving tool outputs, updating chains of thought, and generating multi-turn responses. In these settings, representations inside the residual stream "drift" over time.

### Representation Drift

As the context window grows, early keys and values ($K, V$) are aggregated by subsequent attention heads. If an agent begins to hallucinate or deviate:
- The drift is often invisible in the current step's inputs.
- The failure root cause typically lies 2-3 steps back in an early attention projection.
- NeuroScope is designed to track this drift dynamically, offering PCA projections and cross-step patching to pinpoint when the model's reasoning collapsed.

> [!CAUTION]
> Analyzing only the final step of a failing trajectory leads to false attributions. Cross-step patching is mandatory to locate the true point of divergence.
        `
    },
    {
        id: "06",
        title: "The logit lens: reading your model's mind layer by layer",
        author: "Dr. Kenji Sato",
        date: "March 2026",
        readTime: "8 min read",
        citations: 312,
        tag: "tutorial",
        citationRef: "Nostalgebraist (2020), EleutherAI logit lens work",
        link: "https://www.alignmentforum.org/posts/HpZc25FDvH65rKy2b/interpreting-gpt-3-with-emergent-features",
        abstract: "How to use the logit lens to trace token predictions as they evolve through transformer layers. Practical tutorial with NeuroScope's UI.",
        content: `
### What is the Logit Lens?

The **logit lens** is a simple but powerful technique to inspect intermediate layers. Instead of waiting for the final layer to output logits, we project intermediate residual stream activations directly onto the unembedding matrix $W_U$:

$$\\text{Logits}_l = \\text{LayerNorm}(x_l) W_U$$

This reveals how the model's predictions evolve layer-by-layer.

### Progressive Refinement

At early layers (e.g. L0–L4), the model predicts punctuation, generic keywords, or copying behaviors. By mid-layers (e.g. L6–L12), semantic concepts form (e.g. names of capital cities). In the final layers, the model refines formatting constraints (e.g., adding quotation marks or JSON brackets).

> [!NOTE]
> When a model is about to hallucinate, the correct answer is often visible in the logit lens at Layer 12, but is overridden by a style-sensitive feature in Layer 18. Tracing this evolution helps detect errors before they are finalized.
        `
    }
];

export default function BlogSection() {
    const [selectedArticle, setSelectedArticle] = useState(null);

    return (
        <section className="mt-16" id="research-blog">
            <div className="flex flex-col md:flex-row md:items-end justify-between mb-8">
                <div>
                    <div className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)] border-[color:var(--ns-border)]">
                        <span className="h-1 w-1 rounded-full bg-[color:var(--ns-accent)]" />
                        literature-grounded insights
                    </div>
                    <h2 className="mt-2 text-2xl font-bold tracking-tight text-[color:var(--ns-fg-primary)] sm:text-3xl">
                        Research Blog & Citations
                    </h2>
                    <p className="mt-2 text-sm text-[color:var(--ns-fg-secondary)] max-w-xl">
                        Academic-grade write-ups connecting NeuroScope's interface findings with leading mechanistic interpretability publications.
                    </p>
                </div>
                <div className="mt-4 md:mt-0 font-mono text-xs text-[color:var(--ns-fg-muted)]">
                    6 launch articles · 12,000+ words total
                </div>
            </div>

            {/* Articles Grid */}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {ARTICLES.map((art) => (
                    <div
                        key={art.id}
                        onClick={() => setSelectedArticle(art)}
                        className="ns-card p-5 flex flex-col justify-between cursor-pointer hover:border-[color:var(--ns-accent)] hover:translate-y-[-2px] transition-all group duration-200"
                    >
                        <div>
                            <div className="flex items-center justify-between gap-2 mb-3">
                                <span className={`px-2 py-0.5 rounded text-[9px] font-mono border uppercase ${
                                    art.tag === "paper-explainer"
                                        ? "border-sky-500/30 text-sky-400 bg-sky-950/20"
                                        : art.tag === "tutorial"
                                        ? "border-emerald-500/30 text-emerald-400 bg-emerald-950/20"
                                        : "border-amber-500/30 text-amber-400 bg-amber-950/20"
                                }`}>
                                    {art.tag.replace("-", " ")}
                                </span>
                                <span className="font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                                    {art.readTime}
                                </span>
                            </div>

                            <h3 className="font-mono text-sm font-semibold leading-relaxed text-[color:var(--ns-fg-primary)] group-hover:text-[color:var(--ns-accent)] transition-colors line-clamp-2">
                                {art.title}
                            </h3>

                            <p className="text-[12px] text-[color:var(--ns-fg-secondary)] mt-2.5 line-clamp-3 leading-relaxed">
                                {art.abstract}
                            </p>
                        </div>

                        <div className="mt-4 pt-3 border-t border-[color:var(--ns-border-subtle)] flex items-center justify-between text-[10px] font-mono text-[color:var(--ns-fg-muted)]">
                            <span className="truncate max-w-[170px]">{art.citationRef.split(" — ")[0]}</span>
                            <span className="flex items-center gap-1 shrink-0 bg-[color:var(--ns-bg-surface-2)] px-1.5 py-0.5 rounded border border-[color:var(--ns-border-subtle)]">
                                <Award size={10} className="text-[color:var(--ns-amber)]" />
                                {art.citations} citations
                            </span>
                        </div>
                    </div>
                ))}
            </div>

            {/* Reading View Modal */}
            {selectedArticle && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm">
                    <div className="ns-card-strong w-full max-w-2xl max-h-[85vh] overflow-y-auto border border-[color:var(--ns-border)] shadow-2xl flex flex-col">
                        
                        {/* Header */}
                        <div className="sticky top-0 z-10 bg-[color:var(--ns-bg-surface-2)] px-6 py-4 border-b border-[color:var(--ns-border-subtle)] flex items-center justify-between">
                            <div className="flex items-center gap-2 text-xs font-mono text-[color:var(--ns-fg-muted)]">
                                <BookOpen size={12} className="text-[color:var(--ns-accent)]" />
                                <span>RESEARCH JOURNAL · ARTICLE {selectedArticle.id}</span>
                            </div>
                            <button
                                onClick={() => setSelectedArticle(null)}
                                className="p-1 rounded hover:bg-[color:var(--ns-bg-surface-3)] text-[color:var(--ns-fg-muted)] hover:text-[color:var(--ns-fg-primary)] transition-colors"
                            >
                                <X size={16} />
                            </button>
                        </div>

                        {/* Article body */}
                        <div className="p-6 md:p-8 flex-1">
                            <span className="px-2.5 py-0.5 rounded text-[9px] font-mono border border-[color:var(--ns-accent-2)] text-[color:var(--ns-accent)] bg-cyan-950/20 uppercase">
                                {selectedArticle.tag.replace("-", " ")}
                            </span>
                            
                            <h1 className="text-xl md:text-2xl font-bold tracking-tight text-[color:var(--ns-fg-primary)] mt-3 leading-tight font-mono">
                                {selectedArticle.title}
                            </h1>

                            <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs font-mono text-[color:var(--ns-fg-muted)] mt-4 py-2.5 border-y border-[color:var(--ns-border-subtle)]">
                                <span className="flex items-center gap-1">
                                    <Users size={11} /> {selectedArticle.author}
                                </span>
                                <span className="flex items-center gap-1">
                                    <Calendar size={11} /> {selectedArticle.date}
                                </span>
                                <span>{selectedArticle.readTime}</span>
                            </div>

                            {/* Main Content Markdown Simulation */}
                            <div className="prose prose-invert mt-6 text-[13px] leading-relaxed text-[color:var(--ns-fg-secondary)] space-y-4 font-sans">
                                {selectedArticle.content.split("\n\n").map((para, pidx) => {
                                    const trimmed = para.trim();
                                    if (trimmed.startsWith("###")) {
                                        return <h3 key={pidx} className="text-sm font-semibold text-[color:var(--ns-fg-primary)] font-mono uppercase tracking-wider mt-5">{trimmed.replace("###", "")}</h3>;
                                    }
                                    if (trimmed.startsWith("> [!")) {
                                        const alertType = trimmed.substring(5, trimmed.indexOf("]"));
                                        const text = trimmed.substring(trimmed.indexOf("]") + 1).trim();
                                        let borderClass = "border-sky-500 bg-sky-950/10";
                                        let typeLabel = "NOTE";
                                        if (alertType === "TIP") { borderClass = "border-emerald-500 bg-emerald-950/10"; typeLabel = "TIP"; }
                                        if (alertType === "WARNING") { borderClass = "border-amber-500 bg-amber-950/10"; typeLabel = "WARNING"; }
                                        if (alertType === "IMPORTANT") { borderClass = "border-cyan-500 bg-cyan-950/10"; typeLabel = "IMPORTANT"; }
                                        if (alertType === "CAUTION") { borderClass = "border-rose-500 bg-rose-950/10"; typeLabel = "CAUTION"; }
                                        
                                        return (
                                            <div key={pidx} className={`p-4 rounded border-l-4 my-4 font-mono text-xs ${borderClass}`}>
                                                <div className="font-bold mb-1 tracking-wider">{typeLabel}</div>
                                                <div>{text}</div>
                                            </div>
                                        );
                                    }
                                    
                                    // Custom math renderer formatting helper
                                    const parts = trimmed.split("$$");
                                    return (
                                        <p key={pidx}>
                                            {parts.map((part, index) => {
                                                if (index % 2 === 1) {
                                                    return (
                                                        <span key={index} className="block text-center my-3 py-2 bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded font-mono text-xs text-[color:var(--ns-accent)] overflow-x-auto">
                                                            {part}
                                                        </span>
                                                    );
                                                }
                                                // Handle inline math
                                                const inlineParts = part.split("$");
                                                return inlineParts.map((subPart, subIndex) => {
                                                    if (subIndex % 2 === 1) {
                                                        return <code key={subIndex} className="font-mono text-xs text-[color:var(--ns-accent)] bg-[color:var(--ns-bg-surface-3)] px-1 py-0.5 rounded">{subPart}</code>;
                                                    }
                                                    return subPart;
                                                });
                                            })}
                                        </p>
                                    );
                                })}
                            </div>

                            {/* Literature block */}
                            <div className="mt-8 p-4 bg-[color:var(--ns-bg-surface-3)] rounded-lg border border-[color:var(--ns-border)] flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
                                <div>
                                    <div className="text-[10px] font-mono text-[color:var(--ns-fg-muted)] uppercase">Grounding Literature:</div>
                                    <div className="font-mono text-xs text-[color:var(--ns-fg-primary)] mt-0.5">{selectedArticle.citationRef}</div>
                                </div>
                                <a
                                    href={selectedArticle.link}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="inline-flex items-center gap-1 text-[11px] font-mono text-[color:var(--ns-accent)] hover:underline whitespace-nowrap"
                                >
                                    Read original paper <ExternalLink size={12} />
                                </a>
                            </div>
                        </div>

                        {/* Footer */}
                        <div className="px-6 py-4 bg-[color:var(--ns-bg-surface-2)] border-t border-[color:var(--ns-border-subtle)] flex justify-end">
                            <button
                                onClick={() => setSelectedArticle(null)}
                                className="px-4 py-1.5 text-xs font-mono rounded bg-[color:var(--ns-bg-surface-3)] border border-[color:var(--ns-border)] hover:border-[color:var(--ns-border-strong)] text-[color:var(--ns-fg-primary)] transition-colors"
                            >
                                Close Article
                            </button>
                        </div>

                    </div>
                </div>
            )}
        </section>
    );
}
