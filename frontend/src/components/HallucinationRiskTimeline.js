import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    Tooltip,
    ReferenceLine,
    ResponsiveContainer,
    CartesianGrid,
    Dot,
} from "recharts";

export default function HallucinationRiskTimeline({ steps = [], threshold = 0.65, height = 180 }) {
    const data = steps.map((s) => ({
        step: `s${s.step_n}`,
        composite: s.hallucination.composite,
        entropy: s.hallucination.entropy,
        attn: s.hallucination.attention_diffusion,
        uncertainty: s.hallucination.uncertainty_features,
        flag: s.hallucination.flag,
    }));
    return (
        <div data-testid="hallucination-risk-timeline">
            <div style={{ width: "100%", height }}>
                <ResponsiveContainer>
                    <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -10 }}>
                        <defs>
                            <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#4FB3C8" stopOpacity={0.4} />
                                <stop offset="100%" stopColor="#4FB3C8" stopOpacity={0.05} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="2 4" />
                        <XAxis dataKey="step" tickLine={false} axisLine={{ stroke: "var(--ns-border)" }} />
                        <YAxis domain={[0, 1]} tickLine={false} axisLine={{ stroke: "var(--ns-border)" }} width={32} />
                        <Tooltip
                            content={({ active, payload, label }) =>
                                active && payload && payload.length ? (
                                    <div className="ns-d3-tooltip" style={{ position: "static" }}>
                                        <div className="mb-1 text-[10px] text-[color:var(--ns-fg-muted)]">{label}</div>
                                        <div>risk: {payload[0].payload.composite.toFixed(3)}</div>
                                        <div style={{ color: "#93C5FD" }}>entropy: {payload[0].payload.entropy.toFixed(2)}</div>
                                        <div style={{ color: "#F2C14E" }}>attn diffusion: {payload[0].payload.attn.toFixed(2)}</div>
                                        <div style={{ color: "#6EE7B7" }}>uncertainty: {payload[0].payload.uncertainty.toFixed(2)}</div>
                                    </div>
                                ) : null
                            }
                        />
                        <ReferenceLine
                            y={threshold}
                            stroke="#F2C14E"
                            strokeDasharray="4 4"
                            label={{ value: `risk ≥ ${threshold}`, position: "insideTopRight", fill: "#F2C14E", fontSize: 10, fontFamily: "JetBrains Mono" }}
                        />
                        <Area
                            type="monotone"
                            dataKey="composite"
                            stroke="#4FB3C8"
                            strokeWidth={2}
                            fill="url(#riskFill)"
                            dot={(props) => {
                                const { cx, cy, payload } = props;
                                if (cx == null || cy == null) return null;
                                return (
                                    <Dot
                                        cx={cx}
                                        cy={cy}
                                        r={payload.flag ? 5 : 3.5}
                                        fill={payload.flag ? "#F2C14E" : "#4FB3C8"}
                                        stroke={payload.flag ? "#F2C14E" : "#0B0F14"}
                                        strokeWidth={payload.flag ? 1.5 : 1}
                                    />
                                );
                            }}
                            isAnimationActive={false}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
            <div
                className="mt-2 rounded-md border px-3 py-2 text-[10.5px] leading-5 text-[color:var(--ns-fg-muted)]"
                style={{ borderColor: "var(--ns-border-subtle)", background: "rgba(242,193,78,0.04)" }}
                data-testid="hallucination-risk-integrity-note"
            >
                <span style={{ color: "var(--ns-warning)" }}>integrity note —</span> early warning
                diagnostic only. SAE-steering intervention has ~20–30% correction success rate in literature
                (Mar 2026). Use this to flag trajectories for human review, not as ground truth.
            </div>
        </div>
    );
}
