import { Link } from "react-router-dom";
import { ChevronLeft } from "lucide-react";

export default function Docs() {
    return (
        <div className="mx-auto max-w-[1000px] px-4 pb-16 pt-10 sm:px-6">
            <Link
                to="/"
                className="inline-flex items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-[color:var(--ns-fg-muted)] hover:text-[color:var(--ns-fg-primary)]"
            >
                <ChevronLeft size={12} /> back
            </Link>
            <h1 className="mt-3 text-3xl font-semibold tracking-tight">Methodology & limitations</h1>
            <p className="mt-3 max-w-2xl text-[13px] leading-7 text-[color:var(--ns-fg-secondary)]">
                NeuroScope is an open research diagnostic for multi-step transformer agents. It captures
                the residual stream and SAE features at every step of a ReAct-style trajectory and lets
                researchers patch activations across steps to identify the causal origin of failures.
            </p>

            <Section title="Architecture">
                <ul className="list-disc space-y-1 pl-5">
                    <li>Subject + agent model: <code>google/gemma-2-2b-it</code> via TransformerLens HookedTransformer (26 layers, d_model=2304).</li>
                    <li>SAE: <code>gemma-scope-2b-pt-res</code> (GemmaScope, canonical 16k-width) on <code>hook_resid_post</code> at the captured layers.</li>
                    <li>Hooks: <code>hook_resid_post</code> at layers 6, 12, 18, 24 (4 of 26 layers) — matches GemmaScope training convention (resid_POST, not resid_PRE). Attention pattern on layer 25. MLP_out on layers 6 and 12.</li>
                    <li>Activations stored as <code>float16</code> .npz in Firebase / Local Disk Storage, ~2–4 MB per step.</li>
                    <li>LLM explainer (NL query): Google Gemini via official <code>google-generativeai</code> Python SDK.</li>
                </ul>
            </Section>

            <Section title="Hypotheses">
                <ul className="list-disc space-y-1 pl-5">
                    <li><b>H1</b> · hallucination-associated GemmaScope features activate 1–2 steps BEFORE the failed output.</li>
                    <li><b>H2</b> · layer-12 activations predict the next tool call before the action token is emitted.</li>
                    <li><b>H3</b> · GemmaScope feature drift variance correlates with task failure rate.</li>
                </ul>
            </Section>

            <Section title="Cross-step causal patching">
                <p>
                    Standard activation patching compares a clean and a corrupted version of the SAME
                    prompt. NeuroScope patches BETWEEN steps: it takes step N's last-token residual stream
                    at a chosen layer and substitutes it into step M's last-token position in the forward
                    pass, then measures <code>KL(patched || baseline)</code> — the divergence of the patched
                    distribution FROM the baseline. KL &gt; 0.05 is treated as
                    causally significant per common interpretability practice.
                    Patching only the last token (not the full sequence) avoids prompt-length semantic
                    mismatch across steps where the context history differs in length.
                </p>
            </Section>

            <Section title="Hallucination risk signal">
                <p>
                    A composite of three measurements per step: output token entropy (40%), attention
                    diffusion (entropy of attention weights across keys at the final layer, 30%), and a
                    feature drift proxy (mean activation of the top-drifting GemmaScope features from
                    this run, 30%). It is reported as a 0–1 score with a flag threshold at 0.65.
                    <br /><br />
                    <strong>No hardcoded feature IDs:</strong> v1 used fixed GPT-2 feature IDs [12059, 4521, 7291, 1842]
                    as an "uncertainty" signal — these had no scientific basis for Gemma-2. v2 uses
                    dynamic drift: the top-drifting features from each specific run, making the signal
                    self-referential and scientifically defensible.
                </p>
            </Section>

            <Section title="Known limitations">
                <ul className="list-disc space-y-1 pl-5">
                    <li>Gemma-2-2b-it is a 2B-parameter open model. Findings here may not generalize to frontier closed models.</li>
                    <li>GemmaScope SAE features are approximate; labels can be misleading. Always validate with causal tests.</li>
                    <li>SAE-based steering has ~20–30% success at correcting outputs (Mar 2026 literature). Treat the hallucination signal as an early warning, not a ground truth.</li>
                    <li>Cross-step patching replaces only the last-token position to avoid prompt-length semantic mismatch. Even so, interpret KL deltas comparatively across conditions, not as absolute causal magnitudes.</li>
                    <li>The co-activation graph (Pearson correlation) is exploratory — it is not causal attribution. Do not claim features are "the circuit for X"; claim they "co-activate with X".</li>
                </ul>
            </Section>

            <Section title="Reproducibility">
                <pre
                    className="overflow-x-auto rounded-md border px-3 py-3 font-mono text-[11px] leading-5"
                    style={{ background: "var(--ns-bg-codeblock)", borderColor: "var(--ns-border-subtle)" }}
                >
{`# locally
pip install transformer_lens sae_lens torch firebase-admin google-generativeai
cd backend && python -m neuroscope.runner   # see /backend/neuroscope/runner.py
# or via the API:
curl -X POST $API/v1/runs -H 'Content-Type: application/json' \\
     -d '{"task":"<your task>","n_steps":3,"sae_layer":12}'`}
                </pre>
            </Section>
        </div>
    );
}

function Section({ title, children }) {
    return (
        <section className="mt-8">
            <h2 className="text-[15px] font-semibold text-[color:var(--ns-fg-primary)]">{title}</h2>
            <div className="mt-2 text-[13px] leading-7 text-[color:var(--ns-fg-secondary)]">{children}</div>
        </section>
    );
}
