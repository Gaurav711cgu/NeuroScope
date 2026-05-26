import { useMemo, useState } from "react";
import { seqColor } from "@/lib/colors";

/**
 * D3-style heatmap. Rows = layers (0..n_layers-1), cols = steps (1..n).
 * Color = L2 norm of last-token residual at (layer, step).
 */
const DEFAULT_CAPTURE_LAYERS = [6, 12, 18, 24];

export default function ActivationTimelineHeatmap({
    steps = [],
    n_layers = 4,            // 4 captured layers in v2 (not all 26)
    capture_layers = DEFAULT_CAPTURE_LAYERS,  // actual layer numbers for labels
    onSelect,
    selected,
}) {
    const { matrix, min, max } = useMemo(() => {
        const m = Array.from({ length: n_layers }, () =>
            Array(steps.length).fill(0),
        );
        let lo = Infinity;
        let hi = -Infinity;
        steps.forEach((s, sIdx) => {
            (s.layer_l2_norms || []).forEach((v, lIdx) => {
                if (lIdx >= n_layers) return;
                m[lIdx][sIdx] = v;
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            });
        });
        if (!Number.isFinite(lo)) lo = 0;
        if (!Number.isFinite(hi)) hi = 1;
        return { matrix: m, min: lo, max: hi };
    }, [steps, n_layers]);

    const cellW = 28;
    const cellH = 20;
    const padL = 38;
    const padT = 22;
    const w = padL + steps.length * (cellW + 2);
    const h = padT + n_layers * (cellH + 2);

    const [hover, setHover] = useState(null);

    return (
        <div
            className="relative"
            data-testid="activation-timeline-heatmap"
        >
            <svg width={w} height={h} className="block">
                {/* col headers */}
                {steps.map((_, sIdx) => (
                    <text
                        key={`c${sIdx}`}
                        x={padL + sIdx * (cellW + 2) + cellW / 2}
                        y={14}
                        textAnchor="middle"
                        className="font-mono"
                        fontSize={10}
                        fill="var(--ns-fg-muted)"
                    >
                        s{sIdx + 1}
                    </text>
                ))}
                {/* row labels — show actual layer numbers (L6, L12, L18, L24) */}
                {Array.from({ length: n_layers }).map((_, lIdx) => (
                    <text
                        key={`r${lIdx}`}
                        x={padL - 6}
                        y={padT + lIdx * (cellH + 2) + cellH / 2 + 3}
                        textAnchor="end"
                        className="font-mono"
                        fontSize={9}
                        fill="var(--ns-fg-muted)"
                    >
                        L{capture_layers[lIdx] ?? lIdx}
                    </text>
                ))}
                {matrix.map((row, lIdx) =>
                    row.map((v, sIdx) => {
                        const t = max > min ? (v - min) / (max - min) : 0;
                        const isSel =
                            selected &&
                            selected.layer === lIdx &&
                            selected.step === sIdx + 1;
                        return (
                            <rect
                                key={`${lIdx}-${sIdx}`}
                                x={padL + sIdx * (cellW + 2)}
                                y={padT + lIdx * (cellH + 2)}
                                width={cellW}
                                height={cellH}
                                rx={2}
                                fill={seqColor(t)}
                                stroke={
                                    isSel
                                        ? "var(--ns-focus)"
                                        : "var(--ns-border-subtle)"
                                }
                                strokeWidth={isSel ? 2 : 0.5}
                                onMouseEnter={() =>
                                    setHover({
                                        layer: lIdx,
                                        step: sIdx + 1,
                                        value: v,
                                    })
                                }
                                onMouseLeave={() => setHover(null)}
                                onClick={() =>
                                    onSelect &&
                                    onSelect({
                                        layer: lIdx,
                                        step: sIdx + 1,
                                        value: v,
                                    })
                                }
                                style={{ cursor: "pointer" }}
                                data-testid={`heatmap-cell-${lIdx}-${sIdx + 1}`}
                            />
                        );
                    }),
                )}
            </svg>
            {hover && (
                <div
                    className="ns-d3-tooltip"
                    style={{ left: padL + (hover.step - 1) * (cellW + 2) + cellW + 8, top: padT + hover.layer * (cellH + 2) - 4 }}
                    data-testid="activation-heatmap-tooltip"
                >
                    L{capture_layers[hover.layer] ?? hover.layer} · step {hover.step}
                    <br />
                    L2 = {hover.value.toFixed(2)}
                </div>
            )}
        </div>
    );
}
