import { Loader2 } from "lucide-react";

export default function LoadingProgress({ stage, completedSteps, totalSteps }) {
    const lines = [
        "● model.load(\"gpt2\") → HookedTransformer (12 layers, 768 d_model)",
        "● sae.load(\"gpt2-small-res-jb\", layer=7) → 24576 features",
        "● agent.run(react_loop, n_steps=" + (totalSteps ?? "?") + ")",
        "● hook.capture(resid_post[0..11], attn_pattern[11], mlp_out[3,7,11])",
        "● sae.decompose(layer=7, top_k=25)",
        "● hallucination.score(entropy + attn_diffusion + uncertainty)",
    ];
    const stageIdx =
        {
            queued: 0,
            loading_model: 0,
            model_ready: 1,
            step_done: 3,
            done: 5,
        }[stage] ?? 0;
    return (
        <div className="ns-card p-4" data-testid="analysis-loading-state">
            <div className="mb-3 flex items-center gap-2">
                <Loader2
                    size={14}
                    className="animate-spin text-[color:var(--ns-accent)]"
                />
                <span className="font-mono text-xs uppercase tracking-wider text-[color:var(--ns-fg-secondary)]">
                    capturing trajectory · step {completedSteps ?? 0} / {totalSteps ?? "?"}
                </span>
            </div>
            <pre
                className="font-mono text-[11px] leading-6 text-[color:var(--ns-fg-secondary)]"
                data-testid="analysis-loading-log"
            >
                {lines
                    .map((l, i) => {
                        const done = i <= stageIdx;
                        const active = i === stageIdx + 1;
                        const prefix = done ? " \u2713 " : active ? " \u2022 " : "   ";
                        return (
                            <div
                                key={i}
                                style={{
                                    color: done
                                        ? "var(--ns-success)"
                                        : active
                                        ? "var(--ns-accent)"
                                        : "var(--ns-fg-muted)",
                                }}
                            >
                                {prefix}{l}
                            </div>
                        );
                    })}
            </pre>
        </div>
    );
}
