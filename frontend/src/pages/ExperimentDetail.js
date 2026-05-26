import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import api from "@/lib/api";
import RunAnalysis from "@/pages/RunAnalysis";
import { ChevronLeft, Sparkles } from "lucide-react";

export default function ExperimentDetail() {
    const { slug } = useParams();
    const [exp, setExp] = useState(null);

    useEffect(() => {
        api.getExperiment(slug).then(setExp).catch(() => {});
    }, [slug]);

    if (!exp) {
        return (
            <div className="mx-auto max-w-[1100px] px-6 py-10 text-[12px] text-[color:var(--ns-fg-muted)]">
                loading experiment…
            </div>
        );
    }

    // adapt the experiment doc into a "run" shape for RunAnalysis
    const preset = {
        id: exp.slug,
        task: exp.task,
        n_steps: exp.n_steps,
        sae_layer: exp.sae_layer,
        model: exp.model,
        status: "done",
        steps: exp.steps,
        feature_timelines: exp.feature_timelines,
        patch_matrix: exp.patch_matrix,
        total_elapsed_ms: exp.total_elapsed_ms,
        progress: { stage: "done", completed_steps: exp.n_steps },
    };

    return (
        <div>
            <div className="mx-auto max-w-[1440px] px-4 pt-6 sm:px-6">
                <Link
                    to="/experiments"
                    className="inline-flex items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-[color:var(--ns-fg-muted)] hover:text-[color:var(--ns-fg-primary)]"
                    data-testid="experiment-back"
                >
                    <ChevronLeft size={12} /> all experiments
                </Link>
                <div className="mt-2 flex items-start justify-between gap-4">
                    <div>
                        <h1 className="text-2xl font-semibold tracking-tight">{exp.title}</h1>
                        <pre className="mt-2 whitespace-pre-wrap font-mono text-[12px] leading-5 text-[color:var(--ns-fg-secondary)]">
                            {exp.hypothesis}
                        </pre>
                    </div>
                    <span
                        className="rounded-full border px-2.5 py-0.5 font-mono text-[10px]"
                        style={{ borderColor: "var(--ns-border)", color: "var(--ns-mint)" }}
                    >
                        precomputed
                    </span>
                </div>
                {exp.finding && (
                    <div
                        className="mt-4 rounded-md border px-4 py-3 text-[13px] leading-6 text-[color:var(--ns-fg-primary)]"
                        style={{ borderColor: "rgba(110,231,183,0.4)", background: "rgba(110,231,183,0.05)" }}
                        data-testid="experiment-finding"
                    >
                        <div
                            className="mb-1 font-mono text-[10px] uppercase tracking-wider"
                            style={{ color: "var(--ns-mint)" }}
                        >
                            <Sparkles size={10} className="mr-1 inline" />
                            finding
                        </div>
                        {exp.finding}
                    </div>
                )}
            </div>
            <RunAnalysis presetRun={preset} />
        </div>
    );
}
