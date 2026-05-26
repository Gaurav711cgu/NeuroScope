import { Link } from "react-router-dom";
import { Activity, Brain, GitBranch } from "lucide-react";

export default function StepList({ runId, steps = [], n_steps, activeStep, onSelect, status, progress }) {
    const placeholder = Array.from({ length: n_steps || 3 });
    const rows = steps.length
        ? steps
        : placeholder.map((_, i) => ({ step_n: i + 1, _placeholder: true }));
    return (
        <div className="ns-card overflow-hidden" data-testid="step-list">
            <header
                className="flex items-center justify-between px-4 pt-3.5 pb-2"
                style={{ borderBottom: "1px solid var(--ns-border-subtle)" }}
            >
                <span className="ns-section-title">trajectory</span>
                <span className="font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                    {steps.length}/{n_steps} steps
                </span>
            </header>
            <ul>
                {rows.map((s) => {
                    const isActive = activeStep === s.step_n;
                    const done = !s._placeholder;
                    return (
                        <li
                            key={s.step_n}
                            className="cursor-pointer"
                            style={{
                                background: isActive
                                    ? "var(--ns-bg-surface-2)"
                                    : "transparent",
                                borderLeft: isActive
                                    ? "2px solid var(--ns-accent)"
                                    : "2px solid transparent",
                            }}
                            onClick={() => onSelect && onSelect(s.step_n)}
                            data-testid={`step-list-row-${s.step_n}`}
                        >
                            <div
                                className="flex items-start gap-3 px-4 py-2.5"
                                style={{ borderBottom: "1px solid var(--ns-border-subtle)" }}
                            >
                                <div className="flex flex-col items-center pt-0.5">
                                    <span
                                        className="font-mono text-[10px]"
                                        style={{ color: done ? "var(--ns-accent)" : "var(--ns-fg-muted)" }}
                                    >
                                        s{s.step_n}
                                    </span>
                                    <span
                                        className="mt-1 h-1.5 w-1.5 rounded-full"
                                        style={{
                                            background: done
                                                ? s.hallucination?.flag
                                                    ? "var(--ns-warning)"
                                                    : "var(--ns-success)"
                                                : status === "running" && progress?.completed_steps + 1 === s.step_n
                                                ? "var(--ns-accent)"
                                                : "var(--ns-fg-faint)",
                                        }}
                                    />
                                </div>
                                <div className="min-w-0 flex-1">
                                    <div className="flex items-center gap-2 text-[11px]">
                                        <Brain size={11} color="var(--ns-fg-muted)" />
                                        <span className="truncate text-[color:var(--ns-fg-secondary)]">
                                            {done ? s.output : "…"}
                                        </span>
                                    </div>
                                    {done && (
                                        <div className="mt-1 flex items-center gap-3 font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                                            <span title="hallucination risk">
                                                <Activity size={9} className="inline" /> risk{" "}
                                                <span style={{ color: s.hallucination.flag ? "var(--ns-warning)" : "var(--ns-fg-secondary)" }}>
                                                    {s.hallucination.composite.toFixed(2)}
                                                </span>
                                            </span>
                                            <span title="tool routed">
                                                <GitBranch size={9} className="inline" />{" "}
                                                {s.tool_called || "—"}
                                            </span>
                                            {runId && (
                                                <Link
                                                    to={`/run/${runId}/step/${s.step_n}`}
                                                    className="ml-auto text-[color:var(--ns-accent)] hover:underline"
                                                    onClick={(e) => e.stopPropagation()}
                                                    data-testid={`step-list-detail-${s.step_n}`}
                                                >
                                                    detail →
                                                </Link>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </li>
                    );
                })}
            </ul>
        </div>
    );
}
