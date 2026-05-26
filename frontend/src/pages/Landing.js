import { Link } from "react-router-dom";
import { Activity, Layers, Zap, GitBranch, ArrowRight, Cpu, FlaskConical } from "lucide-react";

export default function Landing() {
    return (
        <div className="relative">
            <div className="ns-noise-bg absolute inset-0" />
            <div className="relative mx-auto max-w-[1280px] px-4 pb-20 pt-12 sm:px-6">
                {/* hero */}
                <div className="grid gap-10 lg:grid-cols-[1.2fr_1fr]">
                    <div>
                        <div className="inline-flex items-center gap-2 rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]" style={{ borderColor: "var(--ns-border)" }} data-testid="landing-badge">
                            <span className="h-1.5 w-1.5 rounded-full" style={{ background: "var(--ns-accent)" }} />
                            agentic mechanistic interpretability · v2.0
                        </div>
                        <h1 className="mt-5 text-4xl font-semibold tracking-tight text-[color:var(--ns-fg-primary)] sm:text-5xl lg:text-6xl">
                            Look <span className="font-mono" style={{ color: "var(--ns-accent)" }}>inside</span> a multi-step
                            agent’s reasoning.
                        </h1>
                        <p className="mt-5 max-w-2xl text-base leading-7 text-[color:var(--ns-fg-secondary)]">
                            NeuroScope captures the residual stream, SAE features, and attention
                            patterns at every step of a multi-turn agent. Then it lets you patch
                            activations across steps to find{" "}
                            <span className="font-mono" style={{ color: "var(--ns-mint)" }}>which step + layer caused which failure</span>{"  "}
                            — a question single-prompt tools cannot answer.
                        </p>
                        <div className="mt-7 flex flex-wrap items-center gap-3">
                            <Link
                                to="/run"
                                className="inline-flex items-center gap-1.5 rounded-md px-4 py-2 text-sm font-medium"
                                style={{ background: "var(--ns-accent)", color: "var(--ns-bg-canvas)" }}
                                data-testid="landing-cta-run"
                            >
                                <Activity size={14} /> Start a trajectory
                            </Link>
                            <Link
                                to="/experiments"
                                className="inline-flex items-center gap-1.5 rounded-md border px-4 py-2 text-sm text-[color:var(--ns-fg-primary)]"
                                style={{ borderColor: "var(--ns-border)" }}
                                data-testid="landing-cta-experiments"
                            >
                                <FlaskConical size={14} /> Browse experiments
                                <ArrowRight size={13} />
                            </Link>
                            <Link
                                to="/docs"
                                className="font-mono text-[11px] text-[color:var(--ns-fg-muted)] hover:underline"
                                data-testid="landing-cta-docs"
                            >
                                methodology & limitations →
                            </Link>
                        </div>
                        <div className="mt-8 grid grid-cols-2 gap-3 text-[11px] sm:grid-cols-4">
                            <Stat label="model" value="Gemma-2-2b-it" />
                            <Stat label="layers hooked" value="L6 L12 L18 L24" />
                            <Stat label="SAE" value="GemmaScope 16k" />
                            <Stat label="features" value="16,384" />
                        </div>
                    </div>
                    <div className="ns-card relative overflow-hidden" data-testid="landing-preview">
                        <div
                            className="px-4 pt-3.5 pb-2"
                            style={{ borderBottom: "1px solid var(--ns-border-subtle)" }}
                        >
                            <div className="ns-section-title">preview · activation timeline</div>
                            <div className="mt-0.5 font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                                steps × layers × residual L2 norm · darker = lower energy
                            </div>
                        </div>
                        <div className="flex items-end justify-center p-4">
                            <MiniHeatmap />
                        </div>
                        <div className="flex items-center justify-between px-4 pb-3 font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                            <span>cross-step KL × 0.140</span>
                            <span style={{ color: "var(--ns-mint)" }}>significant</span>
                        </div>
                    </div>
                </div>

                {/* how it works */}
                <section className="mt-16">
                    <div className="ns-section-title">how it works</div>
                    <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                        <PipelineCard icon={Cpu} title="Hook" line="resid_post at layers 6, 12, 18, 24 + attention patterns captured per agent step" />
                        <PipelineCard icon="capture" title="Capture" line="float16 activations saved per step (≈2–4 MB each) to Firebase / Local Disk Storage" />
                        <PipelineCard icon={Layers} title="SAE" line="GemmaScope 16k decomposes layer-12 residual into 16,384 interpretable JumpReLU features" />
                        <PipelineCard icon={GitBranch} title="Patch" line="patch last-token residual from one step into another to measure KL(patched||baseline) causal effect" />
                    </div>
                </section>

                {/* limitations */}
                <section className="mt-16 grid gap-6 lg:grid-cols-[1.2fr_1fr]">
                    <div>
                        <div className="ns-section-title">positioning</div>
                        <h2 className="mt-2 text-2xl font-semibold">
                            A research diagnostic. <span className="text-[color:var(--ns-fg-muted)]">Not a correction engine.</span>
                        </h2>
                        <p className="mt-3 text-[13px] leading-6 text-[color:var(--ns-fg-secondary)]">
                            Recent literature shows SAE-based steering interventions have ~20–30% success at
                            actually correcting hallucinations. NeuroScope is designed as an{" "}
                            <span style={{ color: "var(--ns-mint)" }}>early-warning monitor</span> — it flags
                            risky trajectories for human review and surfaces the step + layer where a failure
                            originated. That is a more honest and more defensible application of
                            interpretability given current evidence.
                        </p>
                    </div>
                    <div
                        className="ns-card-strong p-4 font-mono text-[11px] leading-6 text-[color:var(--ns-fg-secondary)]"
                    >
                        <div className="mb-2 text-[10px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">research goals</div>
                        <div>H1 · hallucination features activate 1–2 steps BEFORE bad output</div>
                        <div>H2 · layer-12 activations predict the next tool call</div>
                        <div>H3 · GemmaScope feature drift correlates with failure rate</div>
                        <div className="mt-3 text-[10px] text-[color:var(--ns-fg-muted)]">target: ICML 2026 mech interp workshop</div>
                    </div>
                </section>
            </div>
        </div>
    );
}

function Stat({ label, value }) {
    return (
        <div className="ns-card-strong px-3 py-2">
            <div className="font-mono text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">{label}</div>
            <div className="mt-0.5 font-mono text-[12px] text-[color:var(--ns-fg-primary)]">{value}</div>
        </div>
    );
}

function PipelineCard({ icon: Icon, title, line }) {
    return (
        <div className="ns-card p-4">
            <div className="flex items-center gap-2">
                {typeof Icon === "function" ? <Icon size={14} color="var(--ns-accent)" /> : <Zap size={14} color="var(--ns-accent)" />}
                <span className="font-mono text-[11px] uppercase tracking-wider text-[color:var(--ns-fg-primary)]">{title}</span>
            </div>
            <div className="mt-2 text-[12px] leading-5 text-[color:var(--ns-fg-secondary)]">{line}</div>
        </div>
    );
}

function MiniHeatmap() {
    const rows = 4;   // CAPTURE_LAYERS: [6, 12, 18, 24]
    const cols = 4;
    const cellW = 30;
    const cellH = 18;
    const pad = 18;
    const w = pad + cols * (cellW + 2);
    const h = pad + rows * (cellH + 2);
    const seed = (i, j) => Math.abs(Math.sin((i + 1) * (j + 2) * 0.7)) * 0.9 + 0.05;
    const HEATMAP_SEQ = ["#0B0F14", "#102033", "#163A5A", "#1F5F7A", "#2E8FA6", "#4FB3C8", "#8BE3F0"];
    function color(t) {
        const n = HEATMAP_SEQ.length - 1;
        const idx = Math.min(n, Math.floor(t * n));
        return HEATMAP_SEQ[idx];
    }
    return (
        <svg width={w} height={h}>
            {Array.from({ length: cols }).map((_, j) => (
                <text key={`c${j}`} x={pad + j * (cellW + 2) + cellW / 2} y={12} textAnchor="middle" fontSize={9} fontFamily="JetBrains Mono" fill="var(--ns-fg-muted)">
                    s{j + 1}
                </text>
            ))}
            {Array.from({ length: rows }).map((_, i) =>
                Array.from({ length: cols }).map((_, j) => (
                    <rect key={`${i}-${j}`} x={pad + j * (cellW + 2)} y={pad + i * (cellH + 2)} width={cellW} height={cellH} rx={2} fill={color(seed(i, j))} />
                )),
            )}
        </svg>
    );
}
