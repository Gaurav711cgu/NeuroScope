import { useState, useRef, useEffect } from "react";
import { Send, Sparkles } from "lucide-react";
import api from "@/lib/api";

const SUGGESTIONS = [
    "Why is the hallucination risk highest at step 1?",
    "Did internal state at step 1 causally affect step 3?",
    "Which features drift the most across the trajectory?",
    "Summarize the cross-step patch matrix findings.",
];

export default function CircuitQueryBox({ runId, trajectoryReady }) {
    const [msgs, setMsgs] = useState([]);
    const [val, setVal] = useState("");
    const [busy, setBusy] = useState(false);
    const endRef = useRef(null);

    useEffect(() => {
        if (!runId) return;
        api.listQueries(runId)
            .then((d) => {
                const ordered = (d.queries || []).slice().reverse();
                setMsgs(
                    ordered.flatMap((q) => [
                        { role: "user", text: q.query },
                        { role: "assistant", text: q.answer },
                    ]),
                );
            })
            .catch(() => {});
    }, [runId]);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [msgs.length]);

    async function send(text) {
        const t = (text ?? val).trim();
        if (!t || !trajectoryReady) return;
        setMsgs((m) => [...m, { role: "user", text: t }]);
        setVal("");
        setBusy(true);
        try {
            const res = await api.query(runId, { query: t });
            setMsgs((m) => [...m, { role: "assistant", text: res.answer }]);
        } catch (e) {
            setMsgs((m) => [
                ...m,
                { role: "assistant", text: `[error: ${e.message}]` },
            ]);
        } finally {
            setBusy(false);
        }
    }

    return (
        <div
            className="ns-card flex h-[400px] flex-col"
            data-testid="nl-circuit-query-box"
        >
            <header
                className="flex items-center gap-2 px-4 pt-3.5 pb-2"
                style={{ borderBottom: "1px solid var(--ns-border-subtle)" }}
            >
                <Sparkles size={14} color="var(--ns-accent)" />
                <span className="ns-section-title">NL circuit query</span>
                <span className="font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                    grounded in run artifacts • claude-sonnet-4-5
                </span>
            </header>
            <div className="flex-1 overflow-y-auto px-4 py-3 text-[12px] leading-6">
                {msgs.length === 0 ? (
                    <div className="space-y-2">
                        <div className="text-[11px] text-[color:var(--ns-fg-muted)]">
                            Ask any question about the trajectory. Suggestions:
                        </div>
                        {SUGGESTIONS.map((s, i) => (
                            <button
                                key={i}
                                type="button"
                                onClick={() => send(s)}
                                className="block w-full rounded-md border px-2.5 py-1.5 text-left font-mono text-[11px] text-[color:var(--ns-fg-secondary)] hover:bg-[color:var(--ns-bg-surface-2)]"
                                style={{ borderColor: "var(--ns-border-subtle)" }}
                                data-testid={`nl-query-suggestion-${i}`}
                            >
                                {s}
                            </button>
                        ))}
                    </div>
                ) : (
                    msgs.map((m, i) => (
                        <div
                            key={i}
                            className="mb-3"
                            data-testid="nl-circuit-query-message"
                        >
                            <div
                                className="mb-0.5 font-mono text-[10px] uppercase tracking-wider"
                                style={{
                                    color:
                                        m.role === "user"
                                            ? "var(--ns-accent)"
                                            : "var(--ns-mint)",
                                }}
                            >
                                {m.role === "user" ? "you" : "neuroscope"}
                            </div>
                            <div
                                className="whitespace-pre-wrap text-[color:var(--ns-fg-primary)]"
                                style={{ fontSize: 12 }}
                            >
                                {m.text}
                            </div>
                        </div>
                    ))
                )}
                {busy && (
                    <div className="font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                        thinking… (model + sae + patch context being summarized for the LLM)
                    </div>
                )}
                <div ref={endRef} />
            </div>
            <div
                className="flex items-center gap-2 px-3 py-2"
                style={{ borderTop: "1px solid var(--ns-border-subtle)" }}
            >
                <input
                    value={val}
                    onChange={(e) => setVal(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && send()}
                    placeholder={
                        trajectoryReady
                            ? "e.g. why is risk highest at step 3?"
                            : "trajectory still running…"
                    }
                    disabled={!trajectoryReady || busy}
                    className="flex-1 rounded-md border bg-transparent px-2 py-1.5 font-mono text-[12px] text-[color:var(--ns-fg-primary)] placeholder:text-[color:var(--ns-fg-faint)] focus:outline-none"
                    style={{ borderColor: "var(--ns-border-subtle)" }}
                    data-testid="nl-circuit-query-input"
                />
                <button
                    type="button"
                    onClick={() => send()}
                    disabled={!trajectoryReady || busy || !val.trim()}
                    className="flex h-7 items-center gap-1 rounded-md px-2.5 text-[11px]"
                    style={{
                        background: trajectoryReady && val.trim() ? "var(--ns-accent)" : "var(--ns-bg-surface-2)",
                        color: trajectoryReady && val.trim() ? "var(--ns-bg-canvas)" : "var(--ns-fg-muted)",
                    }}
                    data-testid="nl-circuit-query-send-button"
                >
                    <Send size={11} /> send
                </button>
            </div>
        </div>
    );
}
