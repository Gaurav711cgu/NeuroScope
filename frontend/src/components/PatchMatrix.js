import { useMemo, useState } from "react";
import { seqColor } from "@/lib/colors";

export default function PatchMatrix({
    matrix = [],
    layers = [3, 7, 10],
    n_steps = 0,
    layer,
    setLayer,
    onCellClick,
    selectedCell,
}) {
    const view = useMemo(() => {
        const out = matrix.filter((m) => m.patch_layer === layer);
        const max = out.reduce((a, b) => Math.max(a, b.kl), 0) || 1;
        return { rows: out, max };
    }, [matrix, layer]);

    const cellSize = 36;
    const padL = 28;
    const padT = 22;
    const w = padL + n_steps * (cellSize + 2);
    const h = padT + n_steps * (cellSize + 2);
    const [hover, setHover] = useState(null);

    function cellAt(src, tgt) {
        return view.rows.find(
            (r) => r.source_step === src && r.target_step === tgt,
        );
    }

    return (
        <div data-testid="cross-step-patch-matrix">
            <div className="mb-2 flex items-center gap-2">
                <span className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">
                    layer
                </span>
                <div className="flex overflow-hidden rounded-md border" style={{ borderColor: "var(--ns-border)" }}>
                    {layers.map((L) => (
                        <button
                            key={L}
                            type="button"
                            onClick={() => setLayer(L)}
                            data-testid={`patch-layer-${L}`}
                            className="px-2 py-0.5 font-mono text-[11px]"
                            style={{
                                background:
                                    layer === L
                                        ? "var(--ns-accent)"
                                        : "var(--ns-bg-surface-2)",
                                color:
                                    layer === L
                                        ? "var(--ns-bg-canvas)"
                                        : "var(--ns-fg-secondary)",
                            }}
                        >
                            L{L}
                        </button>
                    ))}
                </div>
                <span className="ml-auto font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                    max KL {view.max.toFixed(3)}
                </span>
            </div>
            <div className="relative">
                <svg width={w} height={h} className="block">
                    {Array.from({ length: n_steps }).map((_, i) => (
                        <text
                            key={`th${i}`}
                            x={padL + i * (cellSize + 2) + cellSize / 2}
                            y={14}
                            textAnchor="middle"
                            className="font-mono"
                            fontSize={10}
                            fill="var(--ns-fg-muted)"
                        >
                            t{i + 1}
                        </text>
                    ))}
                    {Array.from({ length: n_steps }).map((_, i) => (
                        <text
                            key={`sh${i}`}
                            x={padL - 4}
                            y={padT + i * (cellSize + 2) + cellSize / 2 + 3}
                            textAnchor="end"
                            className="font-mono"
                            fontSize={10}
                            fill="var(--ns-fg-muted)"
                        >
                            s{i + 1}
                        </text>
                    ))}
                    {Array.from({ length: n_steps }).map((_, srcIdx) =>
                        Array.from({ length: n_steps }).map((_, tgtIdx) => {
                            const src = srcIdx + 1;
                            const tgt = tgtIdx + 1;
                            const cell = cellAt(src, tgt);
                            const isDiag = src === tgt;
                            const v = cell?.kl ?? 0;
                            const t = view.max > 0 ? v / view.max : 0;
                            const isSel =
                                selectedCell &&
                                selectedCell.source === src &&
                                selectedCell.target === tgt &&
                                selectedCell.layer === layer;
                            return (
                                <rect
                                    key={`${src}-${tgt}`}
                                    x={padL + (tgt - 1) * (cellSize + 2)}
                                    y={padT + (src - 1) * (cellSize + 2)}
                                    width={cellSize}
                                    height={cellSize}
                                    rx={3}
                                    fill={
                                        isDiag
                                            ? "var(--ns-bg-surface-1)"
                                            : seqColor(t)
                                    }
                                    stroke={
                                        isSel
                                            ? "var(--ns-focus)"
                                            : "var(--ns-border-subtle)"
                                    }
                                    strokeWidth={isSel ? 2 : 0.5}
                                    onMouseEnter={() =>
                                        setHover({ src, tgt, v, sig: cell?.significant })
                                    }
                                    onMouseLeave={() => setHover(null)}
                                    onClick={() =>
                                        !isDiag &&
                                        cell &&
                                        onCellClick &&
                                        onCellClick({ source: src, target: tgt, layer })
                                    }
                                    style={{ cursor: isDiag ? "default" : "pointer" }}
                                    data-testid={`cross-step-patch-matrix-cell-${src}-${tgt}`}
                                />
                            );
                        }),
                    )}
                </svg>
                {hover && !Number.isNaN(hover.v) && (
                    <div className="ns-d3-tooltip" style={{ left: padL + hover.tgt * (cellSize + 2) + 6, top: padT + (hover.src - 1) * (cellSize + 2) }}>
                        src=s{hover.src} → tgt=s{hover.tgt}
                        <br />
                        KL = {hover.v?.toFixed(4) ?? "—"}
                        {hover.sig ? " · sig" : ""}
                    </div>
                )}
            </div>
            <div className="mt-2 font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                row = source step, col = target step · patch resid_post @ L{layer}
            </div>
        </div>
    );
}
