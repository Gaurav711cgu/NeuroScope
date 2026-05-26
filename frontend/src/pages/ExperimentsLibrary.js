import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import api from "@/lib/api";
import { FlaskConical, ArrowRight, Sparkles } from "lucide-react";

export default function ExperimentsLibrary() {
    const [experiments, setExperiments] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        api.listExperiments()
            .then((d) => setExperiments(d.experiments || []))
            .catch(() => {})
            .finally(() => setLoading(false));
    }, []);

    return (
        <div className="mx-auto max-w-[1200px] px-4 pb-16 pt-10 sm:px-6">
            <header>
                <div className="ns-section-title">/experiments</div>
                <h1 className="mt-2 text-2xl font-semibold tracking-tight">Pre-built experiments</h1>
                <p className="mt-1 max-w-2xl text-[13px] text-[color:var(--ns-fg-secondary)]">
                    Five precomputed trajectories with full activation captures, SAE drift, cross-step patch
                    matrices, and a one-paragraph research finding. No GPU time required — load instantly.
                </p>
            </header>

            <div className="mt-6 grid gap-4 lg:grid-cols-2" data-testid="experiments-library-grid">
                {loading && (
                    <div className="ns-card p-6 text-[11px] text-[color:var(--ns-fg-muted)]">loading…</div>
                )}
                {!loading && experiments.length === 0 && (
                    <div className="ns-card p-6">
                        <div className="text-[12px] text-[color:var(--ns-fg-primary)]">
                            No experiments seeded yet.
                        </div>
                        <div className="mt-1 font-mono text-[11px] text-[color:var(--ns-fg-muted)]">
                            Run <code>python -m seed_experiments</code> from /app/backend to populate this library.
                        </div>
                    </div>
                )}
                {experiments.map((e) => (
                    <Link
                        key={e.slug}
                        to={`/experiments/${e.slug}`}
                        className="ns-card p-5 hover:bg-[color:var(--ns-bg-surface-2)]"
                        data-testid="experiments-library-card"
                    >
                        <div className="flex items-center gap-2">
                            <FlaskConical size={13} color="var(--ns-accent)" />
                            <span className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">
                                {e.category}
                            </span>
                        </div>
                        <div className="mt-2 text-[15px] font-semibold text-[color:var(--ns-fg-primary)]">{e.title}</div>
                        <pre className="mt-2 whitespace-pre-wrap font-mono text-[11px] leading-5 text-[color:var(--ns-fg-secondary)]">
                            {e.hypothesis}
                        </pre>
                        {e.finding ? (
                            <div
                                className="mt-3 rounded-md border px-3 py-2 text-[12px] leading-5 text-[color:var(--ns-fg-secondary)]"
                                style={{ borderColor: "var(--ns-border-subtle)", background: "rgba(110,231,183,0.04)" }}
                            >
                                <span className="mr-1 font-mono text-[10px] uppercase tracking-wider" style={{ color: "var(--ns-mint)" }}>
                                    <Sparkles size={9} className="mr-0.5 inline" /> finding
                                </span>
                                {e.finding.slice(0, 220)}…
                            </div>
                        ) : null}
                        <div className="mt-4 flex items-center justify-between font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                            <span>n_steps {e.n_steps} · layer {e.sae_layer}</span>
                            <span className="inline-flex items-center gap-1 text-[color:var(--ns-accent)]">
                                open <ArrowRight size={11} />
                            </span>
                        </div>
                    </Link>
                ))}
            </div>
        </div>
    );
}
