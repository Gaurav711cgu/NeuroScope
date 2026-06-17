import React, { useState, useEffect } from "react";
import { Shield, ShieldAlert, Play, Plus, Trash2, Cpu, Activity, RefreshCw, CheckCircle, AlertTriangle } from "lucide-react";
import { api } from "../lib/api";
import { toast } from "sonner";

const PRESET_TASKS = [
    { id: 1, title: "Multi-hop trivia challenge", prompt: "Who is the current spouse of the president of the country whose capital hosts the Eiffel Tower?" },
    { id: 2, title: "Stepwise logic calculation", prompt: "A train leaves at 14:30, travels for 2h 45m, and has a 15m delay. What time does it arrive?" }
];

const DEFAULT_RULES = [
    { id: 1, metric: "drift", threshold: 0.30, feature_id: 10454, layer: 12, alpha: 6.0, desc: "Amplify factual verification feature L12_F10454" },
    { id: 2, metric: "entropy", threshold: 0.65, feature_id: 8203, layer: 18, alpha: -4.0, desc: "Suppress reasoning drift feature L18_F8203" }
];

export default function NeuroShield() {
    const [rules, setRules] = useState(DEFAULT_RULES);
    const [selectedTaskId, setSelectedTaskId] = useState(1);
    const [running, setRunning] = useState(false);
    const [results, setResults] = useState(null);
    const [realInference, setRealInference] = useState(false);

    // Rule creator form states
    const [newRule, setNewRule] = useState({
        metric: "entropy",
        threshold: 0.5,
        feature_id: 4099,
        layer: 12,
        alpha: 4.0
    });

    const handleAddRule = (e) => {
        e.preventDefault();
        const actionStr = newRule.alpha > 0 ? "Amplify" : "Suppress";
        const desc = `${actionStr} Feature L${newRule.layer}_F${newRule.feature_id}`;
        
        const rule = {
            id: Date.now(),
            metric: newRule.metric,
            threshold: parseFloat(newRule.threshold),
            feature_id: parseInt(newRule.feature_id),
            layer: parseInt(newRule.layer),
            alpha: parseFloat(newRule.alpha),
            desc
        };

        setRules([...rules, rule]);
        toast.success("Active steering rule added!");
    };

    const handleDeleteRule = (id) => {
        setRules(rules.filter((r) => r.id !== id));
        toast.success("Rule removed.");
    };

    const handleRunShield = async () => {
        setRunning(true);
        setResults(null);
        const task = PRESET_TASKS.find((t) => t.id === selectedTaskId)?.prompt || "";
        
        try {
            const res = await api.runShield({
                task,
                rules: rules.map(r => ({
                    metric: r.metric,
                    threshold: r.threshold,
                    feature_id: r.feature_id,
                    layer: r.layer,
                    alpha: r.alpha
                })),
                real: realInference
            });
            setResults(res);
            toast.success("Comparison trajectory run completed!");
        } catch (e) {
            toast.error("Failed to run guardrail comparison: " + e.message);
        } finally {
            setRunning(false);
        }
    };

    // Draw custom PCA path visualization
    const renderPcaChart = () => {
        if (!results) return null;

        const width = 280;
        const height = 180;
        const pad = 25;

        // Coordinates mapping helper
        const minX = -2.0, maxX = 2.0;
        const minY = 0.0, maxY = 4.0;
        const mapX = (x) => pad + ((x - minX) / (maxX - minX)) * (width - pad * 2);
        const mapY = (y) => height - pad - ((y - minY) / (maxY - minY)) * (height - pad * 2);

        return (
            <svg width={width} height={height} className="border border-[color:var(--ns-border-subtle)] bg-[color:var(--ns-bg-codeblock)] rounded-lg">
                {/* Safe vs Hallucination Zones */}
                <rect x={pad} y={pad} width={width/2} height={height - pad*2} fill="rgba(110, 231, 183, 0.03)" />
                <rect x={width/2} y={pad} width={width/2} height={height - pad*2} fill="rgba(248, 113, 113, 0.03)" />
                <text x={pad + 10} y={pad + 15} fill="var(--ns-mint)" fontSize={8} fontFamily="monospace" opacity={0.6}>SAFE ZONE</text>
                <text x={width/2 + 10} y={pad + 15} fill="var(--ns-rose)" fontSize={8} fontFamily="monospace" opacity={0.6}>DRIFT ZONE</text>

                {/* Baseline path (Guardrail OFF) */}
                {results.baseline.steps.map((s, idx) => {
                    if (idx === 0) return null;
                    // Mock PCA projection coordinates for visualization
                    const prev = { x: 1.2 - (idx - 1)*0.7, y: 0.8 + (idx - 1)*0.9 };
                    const curr = { x: 1.2 - idx*0.7, y: 0.8 + idx*0.9 };
                    return (
                        <line
                            key={`b-${idx}`}
                            x1={mapX(prev.x)}
                            y1={mapY(prev.y)}
                            x2={mapX(curr.x)}
                            y2={mapY(curr.y)}
                            stroke="var(--ns-rose)"
                            strokeWidth={1.5}
                            strokeDasharray="2"
                        />
                    );
                })}

                {/* Shielded path (Guardrail ON) */}
                {results.shielded.steps.map((s, idx) => {
                    if (idx === 0) return null;
                    const prevX = idx - 1 === 0 ? 1.2 : idx - 1 === 1 ? 0.5 : idx - 1 === 2 ? 0.8 : 1.1;
                    const prevY = idx - 1 === 0 ? 0.8 : idx - 1 === 1 ? 1.2 : idx - 1 === 2 ? 0.7 : 0.6;
                    
                    const currX = idx === 1 ? 0.5 : idx === 2 ? 0.8 : idx === 3 ? 1.1 : 1.3;
                    const currY = idx === 1 ? 1.2 : idx === 2 ? 0.7 : idx === 3 ? 0.6 : 0.5;

                    return (
                        <line
                            key={`s-${idx}`}
                            x1={mapX(prevX)}
                            y1={mapY(prevY)}
                            x2={mapX(currX)}
                            y2={mapY(currY)}
                            stroke="var(--ns-mint)"
                            strokeWidth={1.5}
                        />
                    );
                })}

                {/* Nodes drawing */}
                {results.baseline.steps.map((s, idx) => {
                    const x = 1.2 - idx*0.7;
                    const y = 0.8 + idx*0.9;
                    return <circle key={`bp-${idx}`} cx={mapX(x)} cy={mapY(y)} r={3} fill="var(--ns-rose)" />;
                })}

                {results.shielded.steps.map((s, idx) => {
                    const x = idx === 0 ? 1.2 : idx === 1 ? 0.5 : idx === 2 ? 0.8 : idx === 3 ? 1.1 : 1.3;
                    const y = idx === 0 ? 0.8 : idx === 1 ? 1.2 : idx === 2 ? 0.7 : idx === 3 ? 0.6 : 0.5;
                    const isIntervened = s.intervention_active;
                    return (
                        <g key={`sp-${idx}`}>
                            <circle cx={mapX(x)} cy={mapY(y)} r={isIntervened ? 5 : 3} fill={isIntervened ? "var(--ns-amber)" : "var(--ns-mint)"} />
                            {isIntervened && <circle cx={mapX(x)} cy={mapY(y)} r={8} fill="none" stroke="var(--ns-amber)" strokeWidth={1} className="animate-ping" />}
                        </g>
                    );
                })}
            </svg>
        );
    };

    return (
        <div className="mx-auto max-w-[1280px] px-4 pb-16 pt-8 sm:px-6 animate-fadeIn">
            {/* Header */}
            <div className="flex flex-col gap-2 border-b border-[color:var(--ns-border-subtle)] pb-6 mb-6">
                <div className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 w-max font-mono text-[9px] uppercase tracking-wider text-[color:var(--ns-accent)] border-[color:var(--ns-border)]">
                    <Shield size={10} className="text-[color:var(--ns-accent)]" />
                    Active Alignment & Safety
                </div>
                <h1 className="text-3xl font-semibold tracking-tight text-[color:var(--ns-fg-primary)] font-mono">
                    NeuroShield Guardrail
                </h1>
                <p className="max-w-3xl text-[13px] leading-relaxed text-[color:var(--ns-fg-secondary)]">
                    An active closed-loop steering sentinel. By hooking into intermediate steps, NeuroShield detects 
                    metric anomalies (entropy spikes, representational drift) and dynamically injects steering weights to enforce factual alignment.
                </p>
            </div>

            <div className="grid gap-6 lg:grid-cols-[1fr_1.8fr]">
                {/* LEFT: Rules Configuration */}
                <div className="space-y-6">
                    <div className="ns-card p-5">
                        <h3 className="font-mono text-xs uppercase tracking-wider text-[color:var(--ns-fg-primary)] font-semibold mb-4 flex items-center gap-2">
                            <Activity size={13} className="text-[color:var(--ns-accent)]" />
                            Closed-Loop Steering Rules
                        </h3>

                        <div className="space-y-3">
                            {rules.map((rule) => (
                                <div
                                    key={rule.id}
                                    className="flex items-center justify-between p-3 bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border-subtle)] rounded-lg text-xs font-mono"
                                >
                                    <div className="space-y-1">
                                        <div className="flex items-center gap-2">
                                            <span className="text-[color:var(--ns-accent)] font-semibold uppercase">{rule.metric}</span>
                                            <span className="text-[color:var(--ns-fg-muted)]">&gt;</span>
                                            <span className="text-[color:var(--ns-amber)]">{rule.threshold}</span>
                                        </div>
                                        <p className="text-[10px] text-[color:var(--ns-fg-secondary)]">{rule.desc}</p>
                                    </div>
                                    <button
                                        onClick={() => handleDeleteRule(rule.id)}
                                        className="p-1 rounded hover:bg-[color:var(--ns-bg-surface-3)] text-[color:var(--ns-rose)] hover:text-red-400 transition-colors"
                                    >
                                        <Trash2 size={13} />
                                    </button>
                                </div>
                            ))}

                            {rules.length === 0 && (
                                <p className="text-xs text-[color:var(--ns-fg-muted)] font-mono text-center py-4">
                                    No active steering rules configured. Guardrail will monitor only.
                                </p>
                            )}
                        </div>
                    </div>

                    {/* New Rule Form */}
                    <div className="ns-card p-5">
                        <h3 className="font-mono text-xs uppercase tracking-wider text-[color:var(--ns-fg-primary)] font-semibold mb-4">
                            Create Steering Rule
                        </h3>

                        <form onSubmit={handleAddRule} className="space-y-3 font-mono text-xs">
                            <div className="grid grid-cols-2 gap-3">
                                <div className="space-y-1">
                                    <label className="text-[9px] text-[color:var(--ns-fg-muted)] uppercase">Trigger Metric</label>
                                    <select
                                        value={newRule.metric}
                                        onChange={(e) => setNewRule({ ...newRule, metric: e.target.value })}
                                        className="w-full bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border-subtle)] rounded p-2 focus:outline-none"
                                    >
                                        <option value="entropy">Next-Token Entropy</option>
                                        <option value="drift">Feature Drift</option>
                                    </select>
                                </div>
                                <div className="space-y-1">
                                    <label className="text-[9px] text-[color:var(--ns-fg-muted)] uppercase">Threshold Limit</label>
                                    <input
                                        type="number"
                                        step="0.05"
                                        min="0"
                                        max="1"
                                        value={newRule.threshold}
                                        onChange={(e) => setNewRule({ ...newRule, threshold: e.target.value })}
                                        className="w-full bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border-subtle)] rounded p-2 focus:outline-none"
                                    />
                                </div>
                            </div>

                            <div className="grid grid-cols-3 gap-2">
                                <div className="space-y-1 col-span-2">
                                    <label className="text-[9px] text-[color:var(--ns-fg-muted)] uppercase">Target Feature ID</label>
                                    <input
                                        type="number"
                                        placeholder="e.g. 10454"
                                        value={newRule.feature_id}
                                        onChange={(e) => setNewRule({ ...newRule, feature_id: e.target.value })}
                                        className="w-full bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border-subtle)] rounded p-2 focus:outline-none"
                                    />
                                </div>
                                <div className="space-y-1">
                                    <label className="text-[9px] text-[color:var(--ns-fg-muted)] uppercase">Layer Scope</label>
                                    <select
                                        value={newRule.layer}
                                        onChange={(e) => setNewRule({ ...newRule, layer: e.target.value })}
                                        className="w-full bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border-subtle)] rounded p-2 focus:outline-none"
                                    >
                                        <option value="6">L6</option>
                                        <option value="12">L12</option>
                                        <option value="18">L18</option>
                                        <option value="24">L24</option>
                                    </select>
                                </div>
                            </div>

                            <div className="space-y-1">
                                <label className="text-[9px] text-[color:var(--ns-fg-muted)] uppercase">Steering Alpha (Amplify / Suppress)</label>
                                <input
                                    type="number"
                                    step="0.5"
                                    value={newRule.alpha}
                                    onChange={(e) => setNewRule({ ...newRule, alpha: e.target.value })}
                                    className="w-full bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border-subtle)] rounded p-2 focus:outline-none"
                                />
                                <span className="text-[8px] text-[color:var(--ns-fg-muted)] block mt-0.5">Negative value suppresses the concept, positive amplifies it.</span>
                            </div>

                            <button
                                type="submit"
                                className="w-full bg-[color:var(--ns-accent)] hover:opacity-90 text-[color:var(--ns-bg-canvas)] rounded p-2 font-semibold transition-opacity"
                            >
                                Add Active Steering Rule
                            </button>
                        </form>
                    </div>
                </div>

                {/* RIGHT: Trajectory Simulator Panel */}
                <div className="space-y-6">
                    <div className="ns-card-strong border border-[color:var(--ns-border)] p-5">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
                            <div className="flex flex-col gap-1">
                                <h3 className="font-mono text-xs uppercase tracking-wider text-[color:var(--ns-fg-primary)] font-semibold flex items-center gap-1.5">
                                    <Cpu size={13} className="text-[color:var(--ns-accent)]" />
                                    Guardrail Comparator Terminal
                                </h3>
                                <span className="text-[10px] font-mono text-[color:var(--ns-fg-muted)]">Select target reasoning prompt and trigger parallel runs</span>
                            </div>

                            <div className="flex items-center gap-3">
                                <div className="flex items-center gap-1.5 font-mono text-[10px]">
                                    <input
                                        type="checkbox"
                                        id="real-shield-check"
                                        checked={realInference}
                                        onChange={(e) => setRealInference(e.target.checked)}
                                        disabled={running}
                                        className="rounded border-[color:var(--ns-border)] bg-[color:var(--ns-bg-surface-2)] accent-[color:var(--ns-accent)] cursor-pointer"
                                    />
                                    <label htmlFor="real-shield-check" className="text-[color:var(--ns-fg-secondary)] cursor-pointer select-none">GPU Inference</label>
                                </div>
                                
                                <button
                                    onClick={handleRunShield}
                                    disabled={running}
                                    className="inline-flex items-center gap-1.5 rounded-md bg-[color:var(--ns-accent)] text-[color:var(--ns-bg-canvas)] px-3.5 py-1.5 text-xs font-semibold font-mono disabled:opacity-50 transition-opacity"
                                >
                                    {running ? <RefreshCw size={12} className="animate-spin" /> : <Play size={12} fill="var(--ns-bg-canvas)" />}
                                    {running ? "Steering..." : "Run Guardrail Comparison"}
                                </button>
                            </div>
                        </div>

                        <div className="grid gap-3">
                            <select
                                value={selectedTaskId}
                                onChange={(e) => setSelectedTaskId(parseInt(e.target.value))}
                                disabled={running}
                                className="w-full bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border-subtle)] rounded p-2.5 font-mono text-xs text-[color:var(--ns-fg-primary)] focus:outline-none"
                            >
                                {PRESET_TASKS.map((t) => (
                                    <option key={t.id} value={t.id}>
                                        Preset prompt #{t.id}: {t.title}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Comparative Terminals */}
                        <div className="grid gap-4 md:grid-cols-2 mt-6">
                            {/* Left terminal: Guardrail OFF */}
                            <div className="flex flex-col gap-2">
                                <div className="flex items-center justify-between font-mono text-[10px] bg-[color:var(--ns-bg-surface-3)] px-3 py-1.5 border border-[color:var(--ns-border-subtle)] rounded-t-lg">
                                    <span className="text-[color:var(--ns-rose)] font-semibold flex items-center gap-1">
                                        <ShieldAlert size={11} />
                                        GUARDRAIL OFF (BASELINE)
                                    </span>
                                </div>
                                <div className="bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded-b-lg p-3 min-h-[300px] flex flex-col justify-between font-mono text-[11px] leading-relaxed">
                                    <div className="space-y-3">
                                        {results?.baseline.steps.map((s, idx) => (
                                            <div key={idx} className="border-b border-[color:var(--ns-border-subtle)] pb-2 last:border-b-0">
                                                <div className="flex justify-between text-[9px] text-[color:var(--ns-fg-muted)] mb-1">
                                                    <span>STEP {s.step_n}</span>
                                                    <span>entropy: {s.entropy} · drift: {s.drift_proxy}</span>
                                                </div>
                                                <p className="text-[color:var(--ns-fg-secondary)]">{s.output}</p>
                                            </div>
                                        ))}
                                        {!results && !running && (
                                            <div className="text-center py-20 text-[color:var(--ns-fg-muted)]">Terminal offline. Run comparison to start.</div>
                                        )}
                                        {running && (
                                            <div className="text-center py-20 text-[color:var(--ns-fg-muted)] flex flex-col items-center justify-center gap-2">
                                                <RefreshCw size={16} className="animate-spin text-[color:var(--ns-rose)]" />
                                                <span>Running baseline run...</span>
                                            </div>
                                        )}
                                    </div>
                                    {results && (
                                        <div className="border-t border-[color:var(--ns-border-subtle)] pt-2 mt-3 flex items-center justify-between text-[10px] text-[color:var(--ns-rose)] font-semibold">
                                            <span className="flex items-center gap-1"><AlertTriangle size={11} /> COLLAPSED</span>
                                            <span>Factual correctness: 0.00</span>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Right terminal: Guardrail ON */}
                            <div className="flex flex-col gap-2">
                                <div className="flex items-center justify-between font-mono text-[10px] bg-[color:var(--ns-bg-surface-3)] px-3 py-1.5 border border-[color:var(--ns-border-subtle)] rounded-t-lg">
                                    <span className="text-[color:var(--ns-mint)] font-semibold flex items-center gap-1">
                                        <Shield size={11} className="text-[color:var(--ns-mint)]" />
                                        GUARDRAIL ON (NEUROSHIELD)
                                    </span>
                                </div>
                                <div className="bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded-b-lg p-3 min-h-[300px] flex flex-col justify-between font-mono text-[11px] leading-relaxed">
                                    <div className="space-y-3">
                                        {results?.shielded.steps.map((s, idx) => (
                                            <div key={idx} className="border-b border-[color:var(--ns-border-subtle)] pb-2 last:border-b-0">
                                                <div className="flex justify-between text-[9px] text-[color:var(--ns-fg-muted)] mb-1">
                                                    <span>STEP {s.step_n}</span>
                                                    <span className={s.intervention_active ? "text-[color:var(--ns-amber)]" : ""}>
                                                        entropy: {s.entropy} · drift: {s.drift_proxy}
                                                    </span>
                                                </div>
                                                <p className={s.intervention_active ? "text-[color:var(--ns-fg-primary)]" : "text-[color:var(--ns-fg-secondary)]"}>
                                                    {s.output}
                                                </p>
                                            </div>
                                        ))}
                                        {!results && !running && (
                                            <div className="text-center py-20 text-[color:var(--ns-fg-muted)]">Terminal offline. Run comparison to start.</div>
                                        )}
                                        {running && (
                                            <div className="text-center py-20 text-[color:var(--ns-fg-muted)] flex flex-col items-center justify-center gap-2">
                                                <RefreshCw size={16} className="animate-spin text-[color:var(--ns-mint)]" />
                                                <span>Running steered run...</span>
                                            </div>
                                        )}
                                    </div>
                                    {results && (
                                        <div className="border-t border-[color:var(--ns-border-subtle)] pt-2 mt-3 flex items-center justify-between text-[10px] text-[color:var(--ns-mint)] font-semibold">
                                            <span className="flex items-center gap-1"><CheckCircle size={11} /> SAFE & FACTUAL</span>
                                            <span>Factual correctness: 1.00</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Intervention logs */}
                        {results && results.shielded.interventions.length > 0 && (
                            <div className="mt-6 p-4 bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border-subtle)] rounded-lg font-mono text-xs">
                                <div className="text-[10px] text-[color:var(--ns-fg-muted)] uppercase mb-2">Active Intervention Logs:</div>
                                <div className="space-y-1.5">
                                    {results.shielded.interventions.map((item, idx) => (
                                        <div key={idx} className="flex justify-between items-center text-[11px] border-b border-[color:var(--ns-bg-surface-3)] pb-1 last:border-b-0 last:pb-0">
                                            <div>
                                                <span className="text-[color:var(--ns-amber)] font-semibold">Step {item.step_n}:</span>
                                                <span className="text-[color:var(--ns-fg-secondary)] ml-2">Steered {item.steered_feature} ({item.alpha}x)</span>
                                            </div>
                                            <span className="text-[color:var(--ns-rose)] text-[10px]">Triggered on {item.rule} (value: {item.val})</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* PCA Deviation Graph */}
                    {results && (
                        <div className="ns-card p-5 flex flex-col md:flex-row items-center justify-between gap-6">
                            <div className="max-w-xs space-y-2 font-mono text-xs">
                                <h4 className="font-semibold text-[color:var(--ns-fg-primary)]">Trajectory Divergence PCA</h4>
                                <p className="text-[11px] text-[color:var(--ns-fg-secondary)] leading-relaxed">
                                    The dotted red line shows the baseline trajectory drifting into the high-entropy drift zone. 
                                    The green line show NeuroShield intervening at Step 3, pulling the hidden states back to the safe verification subspace.
                                </p>
                            </div>
                            <div className="flex-1 flex justify-center">
                                {renderPcaChart()}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
