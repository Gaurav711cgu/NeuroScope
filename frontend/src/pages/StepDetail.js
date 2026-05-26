import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import api from "@/lib/api";
import AttributionGraph from "@/components/AttributionGraph";
import VizPanel from "@/components/VizPanel";
import { ChevronLeft, ExternalLink } from "lucide-react";

export default function StepDetail() {
    const { id, n } = useParams();
    const stepN = parseInt(n, 10);
    const [step, setStep] = useState(null);
    const [run, setRun] = useState(null);
    const [graph, setGraph] = useState(null);
    const [loadingGraph, setLoadingGraph] = useState(false);

    useEffect(() => {
        api.getStep(id, stepN).then(setStep).catch(() => {});
        api.getRun(id).then(setRun).catch(() => {});
    }, [id, stepN]);

    async function loadGraph() {
        setLoadingGraph(true);
        try {
            const res = await api.attribution(id, { step_n: stepN, layer: 12, top_k: 12 });
            setGraph(res.graph);
        } finally {
            setLoadingGraph(false);
        }
    }

    useEffect(() => {
        if (step) loadGraph();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [step]);

    if (!step) {
        return <div className="mx-auto max-w-[1100px] px-6 py-10 text-[12px] text-[color:var(--ns-fg-muted)]">loading…</div>;
    }

    const h = step.hallucination;

    return (
        <div className="mx-auto max-w-[1280px] px-4 pb-12 pt-6 sm:px-6" data-testid="step-detail-panel">
            <Link
                to={`/run/${id}`}
                className="inline-flex items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-[color:var(--ns-fg-muted)] hover:text-[color:var(--ns-fg-primary)]"
                data-testid="step-detail-back"
            >
                <ChevronLeft size={12} /> back to analysis
            </Link>
            <h1 className="mt-2 text-2xl font-semibold tracking-tight">
                Step {step.step_n} <span className="font-mono text-[14px] text-[color:var(--ns-fg-muted)]">· deep dive</span>
            </h1>
            <div className="mt-1 font-mono text-[11px] text-[color:var(--ns-fg-muted)]">
                run {id?.slice(0, 8)}… · prompt {step.prompt_tokens} tokens · elapsed {step.elapsed_ms}ms
            </div>

            <div className="mt-5 grid gap-4 lg:grid-cols-[1.4fr_1fr]">
                <div className="space-y-4">
                    <VizPanel title="step output" subtitle={"tool: " + (step.tool_called || "—")} testid="step-output-card">
                        <pre
                            className="max-h-72 overflow-auto whitespace-pre-wrap rounded-md px-3 py-3 font-mono text-[12px] leading-6 text-[color:var(--ns-fg-secondary)]"
                            style={{ background: "var(--ns-bg-codeblock)" }}
                            data-testid="step-output-text"
                        >
                            {step.output || "—"}
                        </pre>
                    </VizPanel>

                    <VizPanel
                        title={`SAE co-activation graph @ layer ${graph?.layer ?? 12}`}
                        subtitle="top GemmaScope features as nodes · edges weighted by Pearson correlation (not causal attribution)"
                        methodNote="green edges = positive correlation · red edges = negative correlation · node size = activation strength."
                        testid="viz-attribution-graph"
                    >
                        {graph ? (
                            <AttributionGraph graph={graph} width={620} height={340} />
                        ) : (
                            <div className="text-[11px] text-[color:var(--ns-fg-muted)]">
                                {loadingGraph ? "computing co-activation graph…" : "no graph yet"}
                            </div>
                        )}
                    </VizPanel>
                </div>

                <div className="space-y-4">
                    <VizPanel title="hallucination signals" subtitle="3-signal composite" testid="step-hallucination-signal-breakdown">
                        <div className="space-y-3 text-[11px]">
                            <Signal label="composite risk" value={h.composite} max={1} hi={h.flag} bigger />
                            <Signal label="output entropy" value={h.entropy} max={1} />
                            <Signal label="attention diffusion" value={h.attention_diffusion} max={1} />
                            <Signal label="feature drift proxy" value={h.drift_proxy} max={1} />
                            <div
                                className="mt-3 rounded-md border px-3 py-2 text-[10.5px] leading-5 text-[color:var(--ns-fg-muted)]"
                                style={{ borderColor: "var(--ns-border-subtle)", background: "rgba(242,193,78,0.04)" }}
                            >
                                {h.note}
                            </div>
                        </div>
                    </VizPanel>

                    <VizPanel title="top GemmaScope features" subtitle={`layer ${run?.sae_layer ?? 12} · gemma-scope-2b-pt-res 16k`} testid="step-top-features-table">
                        <table className="w-full font-mono text-[11px]">
                            <thead>
                                <tr className="text-left text-[color:var(--ns-fg-muted)]">
                                    <th className="py-1">id</th>
                                    <th>activation</th>
                                    <th>neuronpedia</th>
                                </tr>
                            </thead>
                            <tbody>
                                {(step.top_features || []).slice(0, 12).map((f, i) => (
                                    <tr key={i} style={{ borderTop: "1px solid var(--ns-border-subtle)" }}>
                                        <td className="py-1 text-[color:var(--ns-fg-primary)]">#{f.feature_id}</td>
                                        <td>{f.activation.toFixed(2)}</td>
                                        <td>
                                            <a
                                                href={`https://www.neuronpedia.org/gemma-2-2b/${run?.sae_layer ?? 12}-gemmascope-res-16k/${f.feature_id}`}
                                                target="_blank"
                                                rel="noreferrer"
                                                className="inline-flex items-center gap-0.5 text-[10px] text-[color:var(--ns-accent)] hover:underline"
                                                data-testid={`step-feature-link-${f.feature_id}`}
                                            >
                                                view <ExternalLink size={9} />
                                            </a>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </VizPanel>
                </div>
            </div>
        </div>
    );
}

function Signal({ label, value, max, hi = false, bigger = false }) {
    const pct = Math.min(1, (value ?? 0) / max);
    return (
        <div>
            <div className="flex items-center justify-between">
                <span className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">{label}</span>
                <span
                    className={`font-mono ${bigger ? "text-[14px]" : "text-[12px]"}`}
                    style={{ color: hi ? "var(--ns-warning)" : "var(--ns-fg-primary)" }}
                >
                    {value?.toFixed(3)}
                </span>
            </div>
            <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full" style={{ background: "var(--ns-bg-surface-2)" }}>
                <div
                    className="h-full"
                    style={{
                        width: `${Math.max(2, pct * 100)}%`,
                        background: hi ? "var(--ns-warning)" : "var(--ns-accent)",
                    }}
                />
            </div>
        </div>
    );
}
