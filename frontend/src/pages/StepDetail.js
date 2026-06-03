import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import api from "@/lib/api";
import AttributionGraph from "@/components/AttributionGraph";
import VizPanel from "@/components/VizPanel";
import { ChevronLeft, ExternalLink, RefreshCw } from "lucide-react";

export default function StepDetail() {
    const { id, n } = useParams();
    const stepN = parseInt(n, 10);
    const [step, setStep] = useState(null);
    const [run, setRun] = useState(null);
    const [graph, setGraph] = useState(null);
    const [loadingGraph, setLoadingGraph] = useState(false);

    const [selectedFeature, setSelectedFeature] = useState(null);
    const [steeringPrompt, setSteeringPrompt] = useState("");
    const [steeringAlpha, setSteeringAlpha] = useState(10.0);
    const [steeringReal, setSteeringReal] = useState(false);
    const [steeringResult, setSteeringResult] = useState(null);
    const [steeringRunning, setSteeringRunning] = useState(false);
    const [graphReal, setGraphReal] = useState(false);

    useEffect(() => {
        api.getStep(id, stepN).then(setStep).catch(() => {});
        api.getRun(id).then(setRun).catch(() => {});
    }, [id, stepN]);

    async function loadGraph() {
        setLoadingGraph(true);
        try {
            const res = await api.attribution(id, { step_n: stepN, layer: 12, top_k: 12, real: graphReal });
            setGraph(res.graph);
        } finally {
            setLoadingGraph(false);
        }
    }

    const selectFeatureForSteering = (f) => {
        setSelectedFeature(f);
        setSteeringPrompt(step ? step.prompt : "");
        setSteeringResult(null);
    };

    async function runSteering() {
        if (!selectedFeature) return;
        setSteeringRunning(true);
        setSteeringResult(null);
        try {
            const res = await api.steer({
                prompt: steeringPrompt,
                layer: run?.sae_layer ?? 12,
                feature_id: selectedFeature.feature_id,
                alpha: steeringAlpha,
                real: steeringReal
            });
            setSteeringResult(res);
        } catch (e) {
            console.error("Steering failed", e);
        } finally {
            setSteeringRunning(false);
        }
    }

    useEffect(() => {
        if (step) loadGraph();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [step, graphReal]);

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
                        title={`Causal Attribution Graph @ Layer ${graph?.layer ?? 12}`}
                        subtitle="top GemmaScope features as nodes · edges represent causal path patching coefficients (ablate A → change in B)"
                        methodNote="green arrows = causal driving · red arrows = causal inhibition · node size = activation strength."
                        testid="viz-attribution-graph"
                    >
                        <div className="mb-3 flex items-center justify-between font-mono text-[10px]">
                            <div className="flex items-center gap-1.5">
                                <input
                                    type="checkbox"
                                    id="graph-real"
                                    checked={graphReal}
                                    onChange={(e) => setGraphReal(e.target.checked)}
                                    className="rounded border-[color:var(--ns-border)] bg-[color:var(--ns-bg-surface-2)] accent-[color:var(--ns-accent)] cursor-pointer"
                                />
                                <label htmlFor="graph-real" className="text-[color:var(--ns-fg-secondary)] cursor-pointer select-none">
                                    Real GPU Causal Sweep
                                </label>
                            </div>
                            <span className="text-[color:var(--ns-fg-faint)]">Method: {graph?.method || "causal_path_patching"}</span>
                        </div>
                        {graph ? (
                            <AttributionGraph graph={graph} width={620} height={340} />
                        ) : (
                            <div className="text-[11px] text-[color:var(--ns-fg-muted)] py-8 text-center">
                                {loadingGraph ? "running causal path patching..." : "no graph yet"}
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
                                    <tr 
                                        key={i} 
                                        style={{ borderTop: "1px solid var(--ns-border-subtle)" }}
                                        className={selectedFeature?.feature_id === f.feature_id ? "bg-[color:var(--ns-bg-surface-3)]" : ""}
                                    >
                                        <td className="py-1 text-[color:var(--ns-fg-primary)]">#{f.feature_id}</td>
                                        <td>{f.activation.toFixed(2)}</td>
                                        <td>
                                            <div className="flex items-center gap-2">
                                                <button
                                                    onClick={() => selectFeatureForSteering(f)}
                                                    className="text-[10px] text-[color:var(--ns-accent)] hover:underline font-mono"
                                                >
                                                    steer
                                                </button>
                                                <span className="text-[color:var(--ns-fg-faint)]">·</span>
                                                <a
                                                    href={`https://www.neuronpedia.org/gemma-2-2b/${run?.sae_layer ?? 12}-gemmascope-res-16k/${f.feature_id}`}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    className="inline-flex items-center gap-0.5 text-[10px] text-[color:var(--ns-fg-muted)] hover:underline"
                                                    data-testid={`step-feature-link-${f.feature_id}`}
                                                >
                                                    view <ExternalLink size={9} />
                                                </a>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </VizPanel>
                </div>
            </div>

            {selectedFeature && (
                <div className="mt-6">
                    <VizPanel
                        title={`Feature Steering Sandbox — Feature #${selectedFeature.feature_id}`}
                        subtitle={`Target Layer: ${run?.sae_layer ?? 12} · Current Step: ${stepN}`}
                        testid="feature-steering-card"
                    >
                        <div className="grid gap-6 md:grid-cols-[300px_1fr] font-mono text-[11px]">
                            {/* Left Configuration Panel */}
                            <div className="flex flex-col gap-4 border-r border-[color:var(--ns-border-subtle)] pr-6 text-left">
                                <div className="flex flex-col gap-1">
                                    <span className="text-[color:var(--ns-fg-muted)] uppercase tracking-wider text-[10px]">Active Feature Info</span>
                                    <div className="text-white text-xs font-semibold">GemmaScope Feature #{selectedFeature.feature_id}</div>
                                    <div className="text-[color:var(--ns-fg-secondary)]">Baseline Activation: {selectedFeature.activation.toFixed(4)}</div>
                                </div>

                                <div className="flex flex-col gap-2">
                                    <label className="text-[color:var(--ns-fg-secondary)] text-[10px] uppercase tracking-wider">Steering Strength (α)</label>
                                    <div className="flex items-center gap-3">
                                        <input
                                            type="range"
                                            min="-30"
                                            max="30"
                                            step="1"
                                            value={steeringAlpha}
                                            onChange={(e) => setSteeringAlpha(parseFloat(e.target.value))}
                                            className="flex-1 accent-[color:var(--ns-accent)]"
                                        />
                                        <span className="text-white font-semibold w-10 text-right">{steeringAlpha > 0 ? `+${steeringAlpha}` : steeringAlpha}</span>
                                    </div>
                                </div>

                                <div className="flex flex-col gap-2">
                                    <label className="text-[color:var(--ns-fg-secondary)] text-[10px] uppercase tracking-wider">Steering Prompt</label>
                                    <textarea
                                        value={steeringPrompt}
                                        onChange={(e) => setSteeringPrompt(e.target.value)}
                                        rows={3}
                                        className="w-full bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border)] rounded-md px-3 py-2 text-[11px] leading-relaxed text-[color:var(--ns-fg-primary)] focus:outline-none focus:ring-1 focus:ring-[color:var(--ns-focus)]"
                                    />
                                </div>

                                <div className="flex items-center gap-2">
                                    <input
                                        type="checkbox"
                                        id="steer-real"
                                        checked={steeringReal}
                                        onChange={(e) => setSteeringReal(e.target.checked)}
                                        className="rounded border-[color:var(--ns-border)] bg-[color:var(--ns-bg-surface-2)] accent-[color:var(--ns-accent)]"
                                    />
                                    <label htmlFor="steer-real" className="text-[color:var(--ns-fg-secondary)] cursor-pointer select-none">
                                        Real GPU Execution
                                    </label>
                                </div>

                                <button
                                    onClick={runSteering}
                                    disabled={steeringRunning}
                                    className="flex h-9 items-center justify-center gap-2 rounded-md bg-[color:var(--ns-accent)] hover:bg-[color:var(--ns-accent-2)] text-[color:var(--ns-bg-canvas)] font-semibold tracking-wide disabled:opacity-50 transition-colors mt-2"
                                >
                                    {steeringRunning ? (
                                        <RefreshCw size={13} className="animate-spin" />
                                    ) : (
                                        "Inject Steering Vector"
                                    )}
                                </button>
                            </div>

                            {/* Right Panel: Side-by-Side Outputs */}
                            <div className="flex flex-col gap-4 text-left">
                                <span className="text-[color:var(--ns-fg-muted)] uppercase tracking-wider text-[10px]">Validation Output</span>
                                
                                <div className="grid gap-4 md:grid-cols-2 flex-1">
                                    {/* Baseline Box */}
                                    <div className="flex flex-col gap-2">
                                        <span className="text-xs font-semibold text-[color:var(--ns-fg-secondary)]">Baseline Completion</span>
                                        <div
                                            className="flex-1 min-h-[120px] rounded-md border border-[color:var(--ns-border-subtle)] p-3 text-[11px] leading-relaxed text-[color:var(--ns-fg-secondary)] overflow-y-auto"
                                            style={{ background: "var(--ns-bg-codeblock)" }}
                                        >
                                            {steeringResult ? steeringResult.baseline : (steeringRunning ? "generating..." : "Waiting for injection...")}
                                        </div>
                                    </div>

                                    {/* Steered Box */}
                                    <div className="flex flex-col gap-2">
                                        <span className="text-xs font-semibold text-[color:var(--ns-accent)]">Steered Completion (Baseline + α · W_dec)</span>
                                        <div
                                            className="flex-1 min-h-[120px] rounded-md border border-[color:var(--ns-accent)] p-3 text-[11px] leading-relaxed text-white overflow-y-auto"
                                            style={{ background: "rgba(99, 102, 241, 0.05)" }}
                                        >
                                            {steeringResult ? steeringResult.steered : (steeringRunning ? "generating..." : "Waiting for injection...")}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </VizPanel>
                </div>
            )}
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
