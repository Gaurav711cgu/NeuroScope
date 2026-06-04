import React, { useState } from "react";
import { Play, RotateCcw, HelpCircle, Activity } from "lucide-react";
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from "recharts";

const SAMPLES = [
    {
        name: "Factual Retrieval (Paris vs Rome)",
        source: "The Eiffel Tower is located in the beautiful city of Paris.",
        target: "The Colosseum is located in the beautiful city of Rome.",
        layersData: [
            { layer: "L0", recovery: 0.00, logitDiff: 0.01, activeFeature: "None" },
            { layer: "L6", recovery: 0.12, logitDiff: 0.42, activeFeature: "L6_F992 (Token location)" },
            { layer: "L12", recovery: 0.86, logitDiff: 3.12, activeFeature: "L12_F10454 (Factual city coordinates)" },
            { layer: "L18", recovery: 0.94, logitDiff: 3.45, activeFeature: "L18_F8203 (Semantic country query)" },
            { layer: "L24", recovery: 0.98, logitDiff: 3.58, activeFeature: "L24_F1312 (Style formatting)" }
        ]
    },
    {
        name: "DNA Translation (Alanine vs Methionine)",
        source: "Translate DNA codons: GCT corresponds to Alanine.",
        target: "Translate DNA codons: ATG corresponds to Methionine.",
        layersData: [
            { layer: "L0", recovery: 0.00, logitDiff: 0.00, activeFeature: "None" },
            { layer: "L6", recovery: 0.05, logitDiff: 0.15, activeFeature: "L6_F2203 (Syntactic codon bracket)" },
            { layer: "L12", recovery: 0.78, logitDiff: 2.89, activeFeature: "L12_F8891 (Amino acid vocabulary)" },
            { layer: "L18", recovery: 0.89, logitDiff: 3.10, activeFeature: "L18_F1102 (Biological translation)" },
            { layer: "L24", recovery: 0.95, logitDiff: 3.32, activeFeature: "L24_F1312 (Style formatting)" }
        ]
    }
];

export default function CausalPatchingPlayground() {
    const [selectedIdx, setSelectedIdx] = useState(0);
    const [layerStart, setLayerStart] = useState(6);
    const [layerEnd, setLayerEnd] = useState(18);
    const [isRunning, setIsRunning] = useState(false);
    const [results, setResults] = useState(null);

    const activeSample = SAMPLES[selectedIdx];

    const handleRunPatching = () => {
        setIsRunning(true);
        setResults(null);
        setTimeout(() => {
            setIsRunning(false);
            // Simulate outcome based on selected range
            const data = activeSample.layersData.map((d) => {
                const lNum = parseInt(d.layer.replace("L", ""));
                // If layer falls within start/end range, simulate high recovery
                const inRange = lNum >= layerStart && lNum <= layerEnd;
                return {
                    ...d,
                    recovery: inRange ? d.recovery : parseFloat((d.recovery * 0.1).toFixed(2)),
                    logitDiff: inRange ? d.logitDiff : parseFloat((d.logitDiff * 0.1).toFixed(2)),
                };
            });
            setResults(data);
        }, 800);
    };

    const handleReset = () => {
        setResults(null);
        setLayerStart(6);
        setLayerEnd(18);
    };

    return (
        <section className="mt-16 animate-fadeIn" id="patching-playground">
            <div className="flex flex-col md:flex-row md:items-end justify-between mb-8">
                <div>
                    <div className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)] border-[color:var(--ns-border)]">
                        <span className="h-1 w-1 rounded-full bg-[color:var(--ns-accent)]" />
                        causal intervention playground
                    </div>
                    <h2 className="mt-2 text-2xl font-bold tracking-tight text-[color:var(--ns-fg-primary)] sm:text-3xl">
                        Causal Patching Sandbox
                    </h2>
                    <p className="mt-2 text-sm text-[color:var(--ns-fg-secondary)] max-w-xl">
                        Ablate or patch intermediate residual stream slices between two prompts to locate the physical layers representing facts.
                    </p>
                </div>
            </div>

            <div className="w-full grid gap-6 lg:grid-cols-[1.2fr_1fr]">
                {/* Control Panel */}
                <div className="ns-card p-5 flex flex-col justify-between">
                    <div>
                        <div className="flex items-center justify-between mb-4">
                            <span className="font-mono text-xs uppercase tracking-wider text-[color:var(--ns-fg-primary)] font-semibold">
                                Patching Setup
                            </span>
                            <select
                                value={selectedIdx}
                                onChange={(e) => {
                                    setSelectedIdx(parseInt(e.target.value));
                                    setResults(null);
                                }}
                                className="bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded-md px-2.5 py-1 font-mono text-xs text-[color:var(--ns-fg-primary)] focus:outline-none"
                            >
                                {SAMPLES.map((s, idx) => (
                                    <option key={idx} value={idx}>
                                        {s.name}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Prompt visualizer */}
                        <div className="space-y-3 mb-6">
                            <div className="p-3 bg-[color:var(--ns-bg-surface-2)] rounded border border-[color:var(--ns-border-subtle)] font-mono text-xs">
                                <div className="text-[9px] text-[color:var(--ns-fg-muted)] uppercase mb-1">Source (Clean state donor):</div>
                                <div className="text-[color:var(--ns-mint)]">"{activeSample.source}"</div>
                            </div>
                            <div className="p-3 bg-[color:var(--ns-bg-surface-2)] rounded border border-[color:var(--ns-border-subtle)] font-mono text-xs">
                                <div className="text-[9px] text-[color:var(--ns-fg-muted)] uppercase mb-1">Target (Corrupted receiver):</div>
                                <div className="text-[color:var(--ns-rose)]">"{activeSample.target}"</div>
                            </div>
                        </div>

                        {/* Layer range slider */}
                        <div className="mb-6">
                            <div className="flex justify-between font-mono text-xs text-[color:var(--ns-fg-primary)] mb-2">
                                <span className="uppercase font-semibold">Slice Layer Hook Range</span>
                                <span className="text-[color:var(--ns-accent)]">Layer {layerStart} → Layer {layerEnd}</span>
                            </div>
                            <div className="flex gap-4 items-center">
                                <div className="flex-1 space-y-3">
                                    <div className="flex items-center gap-2">
                                        <span className="text-[10px] font-mono text-[color:var(--ns-fg-muted)] w-8">START</span>
                                        <input
                                            type="range"
                                            min="0"
                                            max="24"
                                            step="6"
                                            value={layerStart}
                                            onChange={(e) => {
                                                const val = parseInt(e.target.value);
                                                if (val <= layerEnd) setLayerStart(val);
                                            }}
                                            className="w-full h-1 bg-[color:var(--ns-bg-surface-3)] rounded appearance-none cursor-pointer accent-[color:var(--ns-accent)]"
                                        />
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-[10px] font-mono text-[color:var(--ns-fg-muted)] w-8">END</span>
                                        <input
                                            type="range"
                                            min="0"
                                            max="24"
                                            step="6"
                                            value={layerEnd}
                                            onChange={(e) => {
                                                const val = parseInt(e.target.value);
                                                if (val >= layerStart) setLayerEnd(val);
                                            }}
                                            className="w-full h-1 bg-[color:var(--ns-bg-surface-3)] rounded appearance-none cursor-pointer accent-[color:var(--ns-accent)]"
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="flex gap-3 border-t border-[color:var(--ns-border-subtle)] pt-4">
                        <button
                            onClick={handleRunPatching}
                            disabled={isRunning}
                            className="flex-1 inline-flex items-center justify-center gap-1.5 rounded-md bg-[color:var(--ns-accent)] text-[color:var(--ns-bg-canvas)] px-4 py-2 text-xs font-semibold font-mono hover:opacity-90 transition-opacity disabled:opacity-50"
                        >
                            <Play size={12} fill="var(--ns-bg-canvas)" />
                            {isRunning ? "Running Intervention..." : "Run Activation Patching"}
                        </button>
                        <button
                            onClick={handleReset}
                            className="inline-flex items-center justify-center rounded-md border border-[color:var(--ns-border)] hover:border-[color:var(--ns-border-strong)] px-3 py-2 text-xs font-mono text-[color:var(--ns-fg-secondary)] transition-colors"
                        >
                            <RotateCcw size={12} />
                        </button>
                    </div>
                </div>

                {/* Visualizer Panel */}
                <div className="ns-card-strong border border-[color:var(--ns-border)] p-5 flex flex-col justify-between min-h-[300px]">
                    {isRunning ? (
                        <div className="flex-1 flex flex-col items-center justify-center py-12">
                            <Activity className="h-8 w-8 text-[color:var(--ns-accent)] animate-spin" />
                            <p className="font-mono text-xs text-[color:var(--ns-fg-secondary)] mt-4">
                                Patching activations... Computing cross-step KL(patched || baseline)
                            </p>
                        </div>
                    ) : results ? (
                        <div className="flex-1 flex flex-col justify-between">
                            <div>
                                <h3 className="font-mono text-xs uppercase tracking-wider text-[color:var(--ns-fg-primary)] font-semibold mb-3">
                                    Causal Recovery Matrix
                                </h3>

                                <div className="h-[150px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart
                                            data={results}
                                            margin={{ top: 10, right: 10, left: -25, bottom: 0 }}
                                        >
                                            <defs>
                                                <linearGradient id="colorRecovery" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="var(--ns-accent)" stopOpacity={0.4} />
                                                    <stop offset="95%" stopColor="var(--ns-accent)" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <XAxis dataKey="layer" stroke="var(--ns-fg-muted)" fontSize={9} fontFamily="monospace" />
                                            <YAxis domain={[0, 1]} stroke="var(--ns-fg-muted)" fontSize={9} fontFamily="monospace" />
                                            <Tooltip
                                                contentStyle={{
                                                    background: "var(--ns-bg-surface-2)",
                                                    border: "1px solid var(--ns-border)",
                                                    borderRadius: "6px",
                                                    fontFamily: "monospace",
                                                    fontSize: "11px"
                                                }}
                                            />
                                            <Area
                                                type="monotone"
                                                dataKey="recovery"
                                                stroke="var(--ns-accent)"
                                                strokeWidth={2}
                                                fillOpacity={1}
                                                fill="url(#colorRecovery)"
                                                name="Target Logit Recovery"
                                            />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            <div className="mt-4 border-t border-[color:var(--ns-border-subtle)] pt-4 space-y-2">
                                <div className="text-[10px] font-mono text-[color:var(--ns-fg-muted)] uppercase">Key Causal Node:</div>
                                {results.map((r, idx) => {
                                    if (r.recovery > 0.5) {
                                        return (
                                            <div key={idx} className="flex justify-between items-center bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded p-2 text-xs font-mono">
                                                <div>
                                                    <span className="text-[color:var(--ns-accent)] font-semibold">{r.layer}</span>
                                                    <span className="text-[color:var(--ns-fg-secondary)] ml-2">{r.activeFeature}</span>
                                                </div>
                                                <span className="text-[color:var(--ns-mint)] font-bold">{(r.recovery * 100).toFixed(0)}% recovery</span>
                                            </div>
                                        );
                                    }
                                    return null;
                                })}
                            </div>
                        </div>
                    ) : (
                        <div className="flex-1 flex flex-col items-center justify-center text-center py-12">
                            <HelpCircle size={24} className="text-[color:var(--ns-fg-muted)] mb-2" />
                            <p className="font-mono text-xs text-[color:var(--ns-fg-muted)] max-w-xs">
                                Configure the layer range and press "Run Activation Patching" to run the causal intervention simulation.
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </section>
    );
}
