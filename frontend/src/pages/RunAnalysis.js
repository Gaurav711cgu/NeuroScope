import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import api from "@/lib/api";
import ActivationTimelineHeatmap from "@/components/ActivationTimelineHeatmap";
import SaeFeatureDriftChart from "@/components/SaeFeatureDriftChart";
import HallucinationRiskTimeline from "@/components/HallucinationRiskTimeline";
import PatchMatrix from "@/components/PatchMatrix";
import CircuitQueryBox from "@/components/CircuitQueryBox";
import StepList from "@/components/StepList";
import ReproSnippet from "@/components/ReproSnippet";
import VizPanel from "@/components/VizPanel";
import LoadingProgress from "@/components/LoadingProgress";
import StatusDot from "@/components/StatusDot";
import { Wand2, Loader2 } from "lucide-react";
import { toast } from "sonner";

export default function RunAnalysis({ presetRun = null }) {
    const params = useParams();
    const navigate = useNavigate();
    const id = params.id || presetRun?.id;
    const [run, setRun] = useState(presetRun);
    const [activeStep, setActiveStep] = useState(1);
    const [pickedFeatureId, setPickedFeatureId] = useState(null);
    const [patchLayer, setPatchLayer] = useState(12);
    const [patchInspector, setPatchInspector] = useState(null);
    const [matrixLoading, setMatrixLoading] = useState(false);
    const timer = useRef(null);

    // poll while running
    useEffect(() => {
        if (!id || presetRun) return;
        let alive = true;
        async function tick() {
            try {
                const r = await api.getRun(id);
                if (!alive) return;
                setRun(r);
                if (r.status !== "done" && r.status !== "error") {
                    timer.current = setTimeout(tick, 2000);
                }
            } catch {
                timer.current = setTimeout(tick, 4000);
            }
        }
        tick();
        return () => {
            alive = false;
            if (timer.current) clearTimeout(timer.current);
        };
    }, [id, presetRun]);

    const trajectoryReady = run?.status === "done";
    const steps = run?.steps || [];
    const timelines = run?.feature_timelines || [];
    const patchMatrix = run?.patch_matrix || [];

    const filteredTimelines = useMemo(() => {
        if (!pickedFeatureId) return timelines;
        return timelines.filter((t) => t.feature_id === pickedFeatureId).concat(timelines.filter((t) => t.feature_id !== pickedFeatureId));
    }, [timelines, pickedFeatureId]);

    async function runMatrix() {
        setMatrixLoading(true);
        try {
            const res = await api.patchMatrix(id, { layers: [6, 12, 18] });
            setRun((r) => ({ ...r, patch_matrix: res.patch_matrix }));
            toast.success(`patch matrix swept · ${res.patch_matrix.length} cells`);
        } catch (e) {
            toast.error(`patch matrix failed: ${e.message}`);
        } finally {
            setMatrixLoading(false);
        }
    }

    async function runPatch(cell) {
        // If viewing a pre-computed experiment, look up the cell in the cached matrix.
        if (presetRun) {
            const cached = (run.patch_matrix || []).find(
                (m) =>
                    m.source_step === cell.source &&
                    m.target_step === cell.target &&
                    m.patch_layer === cell.layer,
            );
            if (cached) {
                setPatchInspector({
                    source_step: cell.source,
                    target_step: cell.target,
                    patch_layer: cell.layer,
                    kl: cached.kl,
                    significant: cached.significant,
                    token_changes: cached.top_token_change ? [cached.top_token_change] : [],
                    interpretation:
                        `Patching layer ${cell.layer} activations from step ${cell.source} into step ${cell.target} ` +
                        `${cached.significant ? "significantly changes" : "does not significantly change"} ` +
                        `the output distribution (KL=${cached.kl.toFixed(4)}). ` +
                        `${cached.significant ? "Causal link between these steps confirmed." : "No strong causal link at this layer."} ` +
                        `(Precomputed from experiment cache.)`,
                });
                return;
            }
        }
        try {
            const res = await api.patch(id, {
                source_step: cell.source,
                target_step: cell.target,
                patch_layer: cell.layer,
            });
            setPatchInspector({ ...cell, ...res });
        } catch (e) {
            toast.error(`patch failed: ${e.message}`);
        }
    }

    if (!run) {
        return (
            <div className="mx-auto max-w-[1440px] px-4 py-12">
                <div className="ns-card p-6 text-[12px] text-[color:var(--ns-fg-muted)]">loading run…</div>
            </div>
        );
    }

    return (
        <div className="mx-auto max-w-[1440px] px-4 pb-12 pt-5 sm:px-6" data-testid="run-analysis">
            {/* header */}
            <div className="mb-4 flex items-start justify-between">
                <div>
                    <div className="flex items-center gap-3">
                        <StatusDot status={run.status} />
                        <span className="font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                            run {id?.slice(0, 8)}… · n_steps {run.n_steps} · layer {run.sae_layer}
                        </span>
                        {run.total_elapsed_ms ? (
                            <span className="font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                                · {(run.total_elapsed_ms / 1000).toFixed(1)}s
                            </span>
                        ) : null}
                    </div>
                    <h1 className="mt-1.5 max-w-3xl text-[17px] font-medium text-[color:var(--ns-fg-primary)]">
                        {run.task}
                    </h1>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        type="button"
                        onClick={() => navigate("/run")}
                        className="rounded-md border px-3 py-1 font-mono text-[11px] text-[color:var(--ns-fg-secondary)] hover:bg-[color:var(--ns-bg-surface-2)]"
                        style={{ borderColor: "var(--ns-border)" }}
                        data-testid="run-analysis-new-run"
                    >
                        + new run
                    </button>
                </div>
            </div>

            {/* error banner */}
            {run.status === "error" && (
                <div
                    className="mb-4 rounded-md border p-3 font-mono text-[11px] text-[color:var(--ns-danger)]"
                    style={{ borderColor: "var(--ns-danger)", background: "rgba(248,113,113,0.06)" }}
                >
                    run failed: {run.error}
                </div>
            )}

            {/* loading state */}
            {run.status !== "done" && run.status !== "error" && (
                <LoadingProgress
                    stage={run.progress?.stage}
                    completedSteps={run.progress?.completed_steps}
                    totalSteps={run.n_steps}
                />
            )}

            {/* main grid (3-pane) */}
            <div className="mt-4 grid gap-4 lg:grid-cols-[260px_1fr_360px]">
                <aside className="order-2 lg:order-1">
                    <StepList
                        runId={id}
                        steps={steps}
                        n_steps={run.n_steps}
                        activeStep={activeStep}
                        onSelect={setActiveStep}
                        status={run.status}
                        progress={run.progress}
                    />
                </aside>

                <main className="order-1 space-y-4 lg:order-2">
                    {/* activation timeline heatmap */}
                    <VizPanel
                        title="activation timeline"
                        subtitle="residual stream L2 norm per (layer, step)"
                        methodNote="y = captured layer (L6, L12, L18, L24) · x = agent step 1..n · color = L2 norm of last-token residual."
                        testid="viz-activation-timeline"
                    >
                        <div className="overflow-x-auto">
                            <ActivationTimelineHeatmap
                                steps={steps}
                                n_layers={4}
                                capture_layers={run.capture_layers || [6, 12, 18, 24]}
                                selected={{ step: activeStep, layer: -1 }}
                                onSelect={(c) => setActiveStep(c.step)}
                            />
                        </div>
                    </VizPanel>

                    {/* SAE drift */}
                    <VizPanel
                        title="SAE feature drift"
                        subtitle={`top GemmaScope features at layer ${run.sae_layer} across steps`}
                        methodNote="each line = one SAE feature · click a chip to pin a feature · ranked by variance across steps (drift score)."
                        testid="viz-sae-drift"
                    >
                        {steps.length ? (
                            <SaeFeatureDriftChart
                                timelines={filteredTimelines}
                                onPick={setPickedFeatureId}
                                pickedFeatureId={pickedFeatureId}
                            />
                        ) : (
                            <div className="text-[11px] text-[color:var(--ns-fg-muted)]">waiting for steps…</div>
                        )}
                    </VizPanel>

                    {/* hallucination + patch matrix */}
                    <div className="grid gap-4 lg:grid-cols-2">
                        <VizPanel
                            title="hallucination risk timeline"
                            subtitle="composite of entropy + attention diffusion + feature drift proxy"
                            testid="viz-hallucination"
                        >
                            {steps.length ? (
                                <HallucinationRiskTimeline steps={steps} />
                            ) : (
                                <div className="text-[11px] text-[color:var(--ns-fg-muted)]">no risk data yet</div>
                            )}
                        </VizPanel>

                        <VizPanel
                            title="cross-step patch matrix"
                            subtitle="KL(patched || baseline) at chosen layer · click a cell to run patch"
                            testid="viz-patch-matrix"
                            right={
                                <button
                                    type="button"
                                    onClick={runMatrix}
                                    disabled={!trajectoryReady || matrixLoading}
                                    className="flex items-center gap-1 rounded-md border px-2.5 py-1 font-mono text-[10px] text-[color:var(--ns-fg-secondary)] hover:bg-[color:var(--ns-bg-surface-2)]"
                                    style={{ borderColor: "var(--ns-border)" }}
                                    data-testid="run-matrix-button"
                                >
                                    {matrixLoading ? (
                                        <Loader2 size={11} className="animate-spin" />
                                    ) : (
                                        <Wand2 size={11} />
                                    )}
                                    {patchMatrix.length ? "re-sweep" : "sweep matrix"}
                                </button>
                            }
                        >
                            {patchMatrix.length ? (
                                <PatchMatrix
                                    matrix={patchMatrix}
                                    layers={[6, 12, 18]}
                                    n_steps={run.n_steps}
                                    layer={patchLayer}
                                    setLayer={setPatchLayer}
                                    onCellClick={runPatch}
                                    selectedCell={
                                        patchInspector
                                            ? {
                                                  source: patchInspector.source_step ?? patchInspector.source,
                                                  target: patchInspector.target_step ?? patchInspector.target,
                                                  layer: patchInspector.patch_layer ?? patchInspector.layer,
                                              }
                                            : null
                                    }
                                />
                            ) : (
                                <div className="text-[11px] text-[color:var(--ns-fg-muted)]">
                                    No patch sweep yet. Click “sweep matrix” to run all (source, target, layer)
                                    cross-step patches — ≈18 cells, ~{(run.n_steps * (run.n_steps - 1) * 3 * 1.5).toFixed(0)}s on CPU.
                                </div>
                            )}
                        </VizPanel>
                    </div>
                </main>

                <aside className="order-3 space-y-4">
                    {/* inspector */}
                    <div className="ns-card overflow-hidden" data-testid="cross-step-patch-inspector">
                        <header
                            className="px-4 pt-3.5 pb-2"
                            style={{ borderBottom: "1px solid var(--ns-border-subtle)" }}
                        >
                            <div className="ns-section-title">patch inspector</div>
                        </header>
                        {patchInspector ? (
                            <div className="px-4 py-3 text-[11px]">
                                <div className="flex items-center justify-between font-mono text-[11px] text-[color:var(--ns-fg-secondary)]">
                                    <span>
                                        s{patchInspector.source_step ?? patchInspector.source}{" "}
                                        → s{patchInspector.target_step ?? patchInspector.target}{" "}
                                        @ L{patchInspector.patch_layer ?? patchInspector.layer}
                                    </span>
                                    <span
                                        style={{
                                            color: patchInspector.significant
                                                ? "var(--ns-success)"
                                                : "var(--ns-fg-muted)",
                                        }}
                                    >
                                        {patchInspector.significant ? "● significant" : "○ not sig"}
                                    </span>
                                </div>
                                <div className="mt-1 font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                                    KL(patched || baseline) = {patchInspector.kl?.toFixed(4)}
                                </div>
                                <div className="mt-3 font-mono text-[10px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">
                                    top token shifts
                                </div>
                                <table className="mt-1 w-full font-mono text-[11px]">
                                    <thead>
                                        <tr className="text-left text-[color:var(--ns-fg-muted)]">
                                            <th className="py-1">token</th>
                                            <th>base</th>
                                            <th>patched</th>
                                            <th>Δ</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {(patchInspector.token_changes || []).map((c, i) => (
                                            <tr key={i} style={{ borderTop: "1px solid var(--ns-border-subtle)" }}>
                                                <td className="py-1 text-[color:var(--ns-fg-primary)]">
                                                    {JSON.stringify(c.token)}
                                                </td>
                                                <td>{c.baseline_p.toFixed(3)}</td>
                                                <td>{c.patched_p.toFixed(3)}</td>
                                                <td
                                                    style={{
                                                        color:
                                                            c.delta > 0
                                                                ? "var(--ns-success)"
                                                                : "var(--ns-danger)",
                                                    }}
                                                >
                                                    {c.delta > 0 ? "+" : ""}
                                                    {c.delta.toFixed(3)}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                                <div className="mt-3 text-[11px] leading-5 text-[color:var(--ns-fg-secondary)]">
                                    {patchInspector.interpretation}
                                </div>
                            </div>
                        ) : (
                            <div className="px-4 py-4 text-[11px] text-[color:var(--ns-fg-muted)]">
                                Select a source/target cell in the patch matrix to run causal patching and inspect
                                token-level effects.
                            </div>
                        )}
                    </div>

                    {/* selected step quick info */}
                    {steps.find((s) => s.step_n === activeStep) ? (
                        <div className="ns-card overflow-hidden" data-testid="step-quick">
                            <header
                                className="px-4 pt-3.5 pb-2"
                                style={{ borderBottom: "1px solid var(--ns-border-subtle)" }}
                            >
                                <div className="ns-section-title">selected step · s{activeStep}</div>
                            </header>
                            <div className="space-y-2 px-4 py-3 text-[11px]">
                                <div className="font-mono text-[10px] text-[color:var(--ns-fg-muted)]">output</div>
                                <pre
                                    className="max-h-32 overflow-auto whitespace-pre-wrap rounded-md px-2 py-1.5 font-mono text-[11px] text-[color:var(--ns-fg-secondary)]"
                                    style={{ background: "var(--ns-bg-codeblock)" }}
                                >
                                    {steps.find((s) => s.step_n === activeStep).output || "—"}
                                </pre>
                                <button
                                    type="button"
                                    onClick={() => navigate(`/run/${id}/step/${activeStep}`)}
                                    className="rounded-md border px-2 py-0.5 font-mono text-[10px] text-[color:var(--ns-fg-secondary)] hover:bg-[color:var(--ns-bg-surface-2)]"
                                    style={{ borderColor: "var(--ns-border)" }}
                                    data-testid="open-step-detail"
                                >
                                    deep dive →
                                </button>
                            </div>
                        </div>
                    ) : null}

                    <ReproSnippet runId={id} />
                </aside>
            </div>

            {/* NL query box — full width below */}
            <div className="mt-4">
                <CircuitQueryBox runId={id} trajectoryReady={trajectoryReady} />
            </div>
        </div>
    );
}
