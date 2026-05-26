import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Play, Loader2, Beaker } from "lucide-react";
import api from "@/lib/api";

export default function RunCreate() {
    const navigate = useNavigate();
    const [tasks, setTasks] = useState([]);
    const [task, setTask] = useState("");
    const [n_steps, setNSteps] = useState(3);
    const [sae_layer, setSaeLayer] = useState(7);
    const [busy, setBusy] = useState(false);
    const [recent, setRecent] = useState([]);

    useEffect(() => {
        api.suggestedTasks().then((d) => setTasks(d.tasks || [])).catch(() => {});
        api.listRuns().then((d) => setRecent(d.runs || [])).catch(() => {});
    }, []);

    async function start() {
        const t = task.trim();
        if (!t || busy) return;
        setBusy(true);
        try {
            const res = await api.createRun({ task: t, n_steps, sae_layer });
            navigate(`/run/${res.run_id}`);
        } finally {
            setBusy(false);
        }
    }

    return (
        <div className="mx-auto max-w-[1100px] px-4 pb-20 pt-10 sm:px-6">
            <header>
                <div className="ns-section-title">/run · create</div>
                <h1 className="mt-2 text-2xl font-semibold tracking-tight">Start a new trajectory</h1>
                <p className="mt-1 text-[13px] text-[color:var(--ns-fg-secondary)]">
                    GPT-2 Small runs as both the agent and the analysis subject. Capture begins on every step.
                </p>
            </header>
            <div className="mt-6 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
                <section className="ns-card p-5">
                    <label className="ns-section-title">task</label>
                    <textarea
                        value={task}
                        onChange={(e) => setTask(e.target.value)}
                        placeholder="e.g. The Eiffel Tower is located in which city, and what country is that city the capital of?"
                        className="mt-2 h-28 w-full resize-none rounded-md border bg-transparent px-3 py-2 font-mono text-[12px] text-[color:var(--ns-fg-primary)] placeholder:text-[color:var(--ns-fg-faint)] focus:outline-none"
                        style={{ borderColor: "var(--ns-border)" }}
                        data-testid="run-create-task-input"
                    />
                    <div className="mt-4 grid grid-cols-2 gap-4">
                        <div>
                            <label className="ns-section-title">n_steps</label>
                            <div className="mt-2 flex items-center gap-2">
                                <input
                                    type="range"
                                    min={2}
                                    max={6}
                                    value={n_steps}
                                    onChange={(e) => setNSteps(parseInt(e.target.value, 10))}
                                    className="flex-1 accent-[#4FB3C8]"
                                    data-testid="run-create-n-steps-slider"
                                />
                                <span className="ns-card-strong rounded-md px-2 py-0.5 font-mono text-[11px]">
                                    {n_steps}
                                </span>
                            </div>
                            <div className="mt-1 font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                                est. runtime ~{n_steps * 9}s on CPU
                            </div>
                        </div>
                        <div>
                            <label className="ns-section-title">sae layer</label>
                            <div className="mt-2 flex flex-wrap gap-1.5">
                                {[0, 3, 5, 7, 9, 11].map((L) => (
                                    <button
                                        key={L}
                                        type="button"
                                        onClick={() => setSaeLayer(L)}
                                        className="rounded-md border px-2.5 py-0.5 font-mono text-[11px]"
                                        style={{
                                            borderColor:
                                                sae_layer === L ? "var(--ns-accent)" : "var(--ns-border)",
                                            background:
                                                sae_layer === L ? "rgba(79,179,200,0.12)" : "transparent",
                                            color:
                                                sae_layer === L ? "var(--ns-accent)" : "var(--ns-fg-secondary)",
                                        }}
                                        data-testid={`run-create-sae-layer-${L}`}
                                    >
                                        L{L}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                    <button
                        type="button"
                        onClick={start}
                        disabled={!task.trim() || busy}
                        className="mt-6 inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium"
                        style={{
                            background: task.trim() && !busy ? "var(--ns-accent)" : "var(--ns-bg-surface-2)",
                            color: task.trim() && !busy ? "var(--ns-bg-canvas)" : "var(--ns-fg-muted)",
                        }}
                        data-testid="run-create-start-analysis-button"
                    >
                        {busy ? (
                            <Loader2 size={14} className="animate-spin" />
                        ) : (
                            <Play size={14} />
                        )}
                        Start analysis
                    </button>
                </section>
                <section>
                    <div className="ns-card overflow-hidden">
                        <header
                            className="px-4 pt-3.5 pb-2"
                            style={{ borderBottom: "1px solid var(--ns-border-subtle)" }}
                        >
                            <div className="ns-section-title">suggested tasks</div>
                        </header>
                        <ul>
                            {tasks.map((t, i) => (
                                <li
                                    key={t.id}
                                    className="cursor-pointer px-4 py-2.5 text-[12px] hover:bg-[color:var(--ns-bg-surface-2)]"
                                    style={{
                                        borderBottom:
                                            i < tasks.length - 1 ? "1px solid var(--ns-border-subtle)" : "none",
                                    }}
                                    onClick={() => {
                                        setTask(t.task);
                                        setNSteps(t.n_steps);
                                    }}
                                    data-testid={`suggested-task-${t.id}`}
                                >
                                    <div className="flex items-center gap-2">
                                        <Beaker size={11} color="var(--ns-fg-muted)" />
                                        <span className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">
                                            {t.category}
                                        </span>
                                    </div>
                                    <div className="mt-0.5 text-[color:var(--ns-fg-primary)]">{t.title}</div>
                                    <div className="mt-0.5 line-clamp-2 font-mono text-[10.5px] text-[color:var(--ns-fg-muted)]">
                                        {t.task}
                                    </div>
                                </li>
                            ))}
                        </ul>
                    </div>
                    {recent.length > 0 && (
                        <div className="ns-card mt-4 overflow-hidden">
                            <header
                                className="px-4 pt-3.5 pb-2"
                                style={{ borderBottom: "1px solid var(--ns-border-subtle)" }}
                            >
                                <div className="ns-section-title">recent runs</div>
                            </header>
                            <ul>
                                {recent.slice(0, 5).map((r) => (
                                    <li
                                        key={r.id}
                                        className="flex items-center justify-between px-4 py-2 text-[11px] hover:bg-[color:var(--ns-bg-surface-2)] cursor-pointer"
                                        onClick={() => navigate(`/run/${r.id}`)}
                                        data-testid={`recent-run-${r.id}`}
                                    >
                                        <span className="truncate text-[color:var(--ns-fg-primary)]">{r.task}</span>
                                        <span className="ml-3 font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                                            {r.status} · {r.n_steps} steps
                                        </span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </section>
            </div>
        </div>
    );
}
