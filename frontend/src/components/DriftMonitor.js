import React, { useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { AlertTriangle, TrendingUp, Cpu, RefreshCw } from "lucide-react";

const DRIFT_DATASETS = [
    {
        name: "Factual Path (Stable)",
        turns: [
            { turn: "T1: Initial Query", pc1: 1.2, pc2: 0.8, entropy: 0.18, diffusion: 0.22, drift: 0.05, activeFeat: "L12_F10454" },
            { turn: "T2: Retrieval", pc1: 1.1, pc2: 0.9, entropy: 0.20, diffusion: 0.25, drift: 0.08, activeFeat: "L12_F1410" },
            { turn: "T3: Verification", pc1: 1.3, pc2: 0.7, entropy: 0.15, diffusion: 0.21, drift: 0.06, activeFeat: "L24_F1312" },
            { turn: "T4: Output", pc1: 1.4, pc2: 0.6, entropy: 0.12, diffusion: 0.18, drift: 0.07, activeFeat: "L24_F1312" }
        ],
        status: "stable",
        desc: "Representations remain bound within the factual subspace. Feature activations are monosemantic and low entropy."
    },
    {
        name: "Reasoning Loop (Hallucination Collapse)",
        turns: [
            { turn: "T1: Initial Query", pc1: 1.2, pc2: 0.8, entropy: 0.18, diffusion: 0.22, drift: 0.05, activeFeat: "L12_F10454" },
            { turn: "T2: Code Block", pc1: 0.8, pc2: 1.4, entropy: 0.35, diffusion: 0.41, drift: 0.18, activeFeat: "L6_F2203" },
            { turn: "T3: Self-Correction", pc1: -0.2, pc2: 2.1, entropy: 0.68, diffusion: 0.62, drift: 0.45, activeFeat: "L24_F4099" },
            { turn: "T4: Contradiction", pc1: -1.5, pc2: 3.5, entropy: 0.92, diffusion: 0.84, drift: 0.78, activeFeat: "L18_F8203" }
        ],
        status: "drifted",
        desc: "Significant representation decay detected starting at Turn 3. PC1/PC2 vectors deviate rapidly as entropy surges."
    }
];

export default function DriftMonitor() {
    const [datasetIdx, setDatasetIdx] = useState(1);
    const activeSet = DRIFT_DATASETS[datasetIdx];

    // Helper to scale PCA coordinates for a beautiful Canvas grid view
    const drawPcaGrid = () => {
        const width = 300;
        const height = 180;
        const padding = 30;

        // Find min/max values to fit coordinates dynamically
        const pc1Values = activeSet.turns.map((t) => t.pc1);
        const pc2Values = activeSet.turns.map((t) => t.pc2);
        
        const minPc1 = -2.0;
        const maxPc1 = 2.0;
        const minPc2 = 0.0;
        const maxPc2 = 4.0;

        const mapX = (x) => padding + ((x - minPc1) / (maxPc1 - minPc1)) * (width - padding * 2);
        const mapY = (y) => height - padding - ((y - minPc2) / (maxPc2 - minPc2)) * (height - padding * 2);

        return (
            <svg width={width} height={height} className="border border-[color:var(--ns-border-subtle)] bg-[color:var(--ns-bg-codeblock)] rounded-lg">
                {/* Background grid lines */}
                <line x1={width / 2} y1={0} x2={width / 2} y2={height} stroke="var(--ns-border-subtle)" strokeDasharray="3" />
                <line x1={0} y1={height / 2} x2={width} y2={height / 2} stroke="var(--ns-border-subtle)" strokeDasharray="3" />

                {/* Path line connecting turns */}
                {activeSet.turns.map((t, idx) => {
                    if (idx === 0) return null;
                    const prev = activeSet.turns[idx - 1];
                    return (
                        <line
                            key={idx}
                            x1={mapX(prev.pc1)}
                            y1={mapY(prev.pc2)}
                            x2={mapX(t.pc1)}
                            y2={mapY(t.pc2)}
                            stroke={activeSet.status === "stable" ? "var(--ns-mint)" : "var(--ns-rose)"}
                            strokeWidth={1.5}
                            strokeDasharray={activeSet.status === "stable" ? "0" : "2"}
                        />
                    );
                })}

                {/* Scatter plot points */}
                {activeSet.turns.map((t, idx) => {
                    const cx = mapX(t.pc1);
                    const cy = mapY(t.pc2);
                    const isLast = idx === activeSet.turns.length - 1;
                    
                    let pointColor = "var(--ns-accent)";
                    if (activeSet.status === "drifted" && idx >= 2) {
                        pointColor = "var(--ns-rose)";
                    } else if (activeSet.status === "stable") {
                        pointColor = "var(--ns-mint)";
                    }

                    return (
                        <g key={idx}>
                            <circle
                                cx={cx}
                                cy={cy}
                                r={isLast ? 6 : 4}
                                fill={pointColor}
                                className="transition-all hover:scale-150 cursor-pointer"
                            />
                            {isLast && (
                                <circle
                                    cx={cx}
                                    cy={cy}
                                    r={10}
                                    fill="none"
                                    stroke={pointColor}
                                    strokeWidth={1.2}
                                    className="animate-ping"
                                />
                            )}
                            <text
                                x={cx + 8}
                                y={cy + 4}
                                fill="var(--ns-fg-secondary)"
                                fontSize={9}
                                fontFamily="monospace"
                            >
                                T{idx + 1}
                            </text>
                        </g>
                    );
                })}

                {/* Axis Labels */}
                <text x={10} y={height - 10} fill="var(--ns-fg-muted)" fontSize={8} fontFamily="monospace">PC1 (Query Alignment)</text>
                <text x={10} y={15} fill="var(--ns-fg-muted)" fontSize={8} fontFamily="monospace">PC2 (Decay/Drift)</text>
            </svg>
        );
    };

    return (
        <section className="mt-16" id="drift-monitor">
            <div className="flex flex-col md:flex-row md:items-end justify-between mb-8">
                <div>
                    <div className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)] border-[color:var(--ns-border)]">
                        <span className="h-1 w-1 rounded-full bg-[color:var(--ns-accent)]" />
                        Multi-turn trajectory diagnostics
                    </div>
                    <h2 className="mt-2 text-2xl font-bold tracking-tight text-[color:var(--ns-fg-primary)] sm:text-3xl">
                        Agent Drift & PCA Monitor
                    </h2>
                    <p className="mt-2 text-sm text-[color:var(--ns-fg-secondary)] max-w-xl">
                        Trace how residual stream hidden representations warp across turns. Detect conceptual drift before hallucinations reach the tokenizer.
                    </p>
                </div>
                
                <div className="mt-4 md:mt-0 flex gap-2">
                    {DRIFT_DATASETS.map((d, idx) => (
                        <button
                            key={idx}
                            onClick={() => setDatasetIdx(idx)}
                            className={`px-3 py-1.5 rounded text-xs font-mono border transition-all ${
                                datasetIdx === idx
                                    ? "border-[color:var(--ns-accent)] text-[color:var(--ns-accent)] bg-[color:var(--ns-bg-surface-2)] font-semibold"
                                    : "border-[color:var(--ns-border)] text-[color:var(--ns-fg-muted)] hover:text-[color:var(--ns-fg-primary)]"
                            }`}
                        >
                            {d.name}
                        </button>
                    ))}
                </div>
            </div>

            <div className="w-full grid gap-6 lg:grid-cols-[1fr_1.5fr]">
                {/* Scatter plot grid */}
                <div className="ns-card p-5 flex flex-col justify-between">
                    <div>
                        <div className="flex items-center justify-between mb-3">
                            <span className="font-mono text-xs uppercase tracking-wider text-[color:var(--ns-fg-primary)] font-semibold flex items-center gap-1.5">
                                <Cpu size={12} className="text-[color:var(--ns-accent)]" />
                                2D Representation Space
                            </span>
                            {activeSet.status === "drifted" ? (
                                <span className="inline-flex items-center gap-1 font-mono text-[10px] text-[color:var(--ns-rose)] border border-rose-500/20 bg-rose-950/20 px-1.5 py-0.5 rounded">
                                    <AlertTriangle size={10} /> DRIFT DETECTED
                                </span>
                            ) : (
                                <span className="inline-flex items-center gap-1 font-mono text-[10px] text-[color:var(--ns-mint)] border border-emerald-500/20 bg-emerald-950/20 px-1.5 py-0.5 rounded">
                                    STABLE PATH
                                </span>
                            )}
                        </div>
                        
                        <div className="flex justify-center p-2">
                            {drawPcaGrid()}
                        </div>
                    </div>

                    <div className="mt-4 border-t border-[color:var(--ns-border-subtle)] pt-3 text-xs">
                        <div className="font-semibold text-[color:var(--ns-fg-primary)] mb-1 font-mono">
                            Diagnostic Analysis:
                        </div>
                        <p className="text-[12px] text-[color:var(--ns-fg-secondary)] leading-relaxed font-sans">
                            {activeSet.desc}
                        </p>
                    </div>
                </div>

                {/* Turn metrics chart */}
                <div className="ns-card-strong border border-[color:var(--ns-border)] p-5 flex flex-col justify-between">
                    <div>
                        <span className="font-mono text-xs uppercase tracking-wider text-[color:var(--ns-fg-primary)] font-semibold flex items-center gap-1.5 mb-4">
                            <TrendingUp size={12} className="text-[color:var(--ns-accent)]" />
                            Turn-by-turn activation metrics
                        </span>

                        <div className="h-[180px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart
                                    data={activeSet.turns}
                                    margin={{ top: 5, right: 10, left: -25, bottom: 0 }}
                                >
                                    <XAxis dataKey="turn" stroke="var(--ns-fg-muted)" fontSize={9} fontFamily="monospace" />
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
                                    <Line
                                        type="monotone"
                                        dataKey="entropy"
                                        stroke="var(--ns-accent)"
                                        strokeWidth={2}
                                        name="Next-token Entropy"
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="diffusion"
                                        stroke="var(--ns-amber)"
                                        strokeWidth={2}
                                        name="Attention Diffusion"
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="drift"
                                        stroke="var(--ns-rose)"
                                        strokeWidth={2}
                                        name="Feature Drift"
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="flex justify-between items-center text-[10px] font-mono text-[color:var(--ns-fg-muted)] mt-4 pt-3 border-t border-[color:var(--ns-border-subtle)]">
                        <span>Turn 3/4 warning overlap threshold = 0.40</span>
                        <span>Auto-refreshing KV-cache states</span>
                    </div>
                </div>
            </div>
        </section>
    );
}
