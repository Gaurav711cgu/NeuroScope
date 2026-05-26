import { useMemo, useState } from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    CartesianGrid,
} from "recharts";
import { CATEGORICAL_8 } from "@/lib/colors";

export default function SaeFeatureDriftChart({
    timelines = [],
    height = 220,
    onPick,
    pickedFeatureId,
}) {
    const top = useMemo(
        () => timelines.slice(0, 12),
        [timelines],
    );

    const data = useMemo(() => {
        if (top.length === 0) return [];
        const n = top[0].activations.length;
        return Array.from({ length: n }, (_, i) => {
            const row = { step: `s${i + 1}` };
            top.forEach((t) => {
                row[`f${t.feature_id}`] = t.activations[i];
            });
            return row;
        });
    }, [top]);

    const [hoverFid, setHoverFid] = useState(null);

    return (
        <div data-testid="sae-feature-drift-chart">
            <div style={{ width: "100%", height }}>
                <ResponsiveContainer>
                    <LineChart
                        data={data}
                        margin={{ top: 8, right: 12, bottom: 0, left: -10 }}
                    >
                        <CartesianGrid strokeDasharray="2 4" />
                        <XAxis dataKey="step" tickLine={false} axisLine={{ stroke: "var(--ns-border)" }} />
                        <YAxis tickLine={false} axisLine={{ stroke: "var(--ns-border)" }} width={32} />
                        <Tooltip
                            content={({ active, payload, label }) =>
                                active && payload && payload.length ? (
                                    <div className="ns-d3-tooltip" style={{ position: "static" }}>
                                        <div className="mb-1 text-[10px] text-[color:var(--ns-fg-muted)]">
                                            {label}
                                        </div>
                                        {payload
                                            .sort((a, b) => b.value - a.value)
                                            .slice(0, 6)
                                            .map((p) => (
                                                <div key={p.dataKey} style={{ color: p.color }}>
                                                    {p.dataKey}: {p.value?.toFixed(2)}
                                                </div>
                                            ))}
                                    </div>
                                ) : null
                            }
                        />
                        {top.map((t, i) => {
                            const fid = `f${t.feature_id}`;
                            const c = CATEGORICAL_8[i % CATEGORICAL_8.length];
                            const active =
                                pickedFeatureId === t.feature_id ||
                                hoverFid === t.feature_id;
                            return (
                                <Line
                                    key={fid}
                                    dataKey={fid}
                                    stroke={c}
                                    strokeWidth={active ? 2.5 : 1.25}
                                    strokeOpacity={
                                        pickedFeatureId && pickedFeatureId !== t.feature_id
                                            ? 0.2
                                            : 1
                                    }
                                    dot={false}
                                    isAnimationActive={false}
                                />
                            );
                        })}
                    </LineChart>
                </ResponsiveContainer>
            </div>
            {/* feature legend */}
            <div className="mt-2 flex flex-wrap gap-1.5" data-testid="sae-feature-drift-legend">
                {top.map((t, i) => {
                    const c = CATEGORICAL_8[i % CATEGORICAL_8.length];
                    const picked = pickedFeatureId === t.feature_id;
                    return (
                        <button
                            key={t.feature_id}
                            type="button"
                            onClick={() => onPick && onPick(picked ? null : t.feature_id)}
                            onMouseEnter={() => setHoverFid(t.feature_id)}
                            onMouseLeave={() => setHoverFid(null)}
                            className="flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[10px]"
                            style={{
                                borderColor: picked ? c : "var(--ns-border-subtle)",
                                background: picked ? "rgba(79,179,200,0.08)" : "transparent",
                                color: picked ? c : "var(--ns-fg-secondary)",
                            }}
                            data-testid={`drift-legend-${t.feature_id}`}
                        >
                            <span className="h-1.5 w-1.5 rounded-full" style={{ background: c }} />
                            #{t.feature_id} · drift {t.drift_score.toFixed(2)}
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
