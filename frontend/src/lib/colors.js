/* Diverging viz color scale + helpers for D3 components. */
export const HEATMAP_SEQ = [
    "#0B0F14",
    "#102033",
    "#163A5A",
    "#1F5F7A",
    "#2E8FA6",
    "#4FB3C8",
    "#8BE3F0",
];

export function seqColor(t) {
    // t ∈ [0, 1]
    const clamped = Math.max(0, Math.min(1, t));
    const n = HEATMAP_SEQ.length - 1;
    const idx = clamped * n;
    const lo = Math.floor(idx);
    const hi = Math.min(n, lo + 1);
    const f = idx - lo;
    return lerpColor(HEATMAP_SEQ[lo], HEATMAP_SEQ[hi], f);
}

function lerpColor(c1, c2, t) {
    const a = hexToRgb(c1);
    const b = hexToRgb(c2);
    const r = Math.round(a[0] + (b[0] - a[0]) * t);
    const g = Math.round(a[1] + (b[1] - a[1]) * t);
    const bl = Math.round(a[2] + (b[2] - a[2]) * t);
    return `rgb(${r}, ${g}, ${bl})`;
}

function hexToRgb(hex) {
    const m = hex.replace("#", "");
    return [
        parseInt(m.substring(0, 2), 16),
        parseInt(m.substring(2, 4), 16),
        parseInt(m.substring(4, 6), 16),
    ];
}

export function divergingColor(t) {
    // t in [-1, 1]
    if (t >= 0) {
        return lerpColor("#1E2A3B", "#6EE7B7", Math.min(1, t));
    }
    return lerpColor("#1E2A3B", "#F87171", Math.min(1, -t));
}

export const CATEGORICAL_8 = [
    "#4FB3C8",
    "#6EE7B7",
    "#F2C14E",
    "#F87171",
    "#93C5FD",
    "#34D399",
    "#F59E0B",
    "#FB7185",
];
