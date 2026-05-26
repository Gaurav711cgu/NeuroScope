import { Copy } from "lucide-react";
import { toast } from "sonner";

export default function ReproSnippet({ runId }) {
    const text = `# NeuroScope reproducibility snippet
# Re-run this analysis locally with TransformerLens + SAELens (CPU friendly).
import os, requests

BACKEND = os.environ.get("NEUROSCOPE_API", "${process.env.REACT_APP_BACKEND_URL || "http://localhost:8001"}/api/v1")
run_id = "${runId}"

run = requests.get(f"{BACKEND}/runs/{run_id}").json()
print("task:", run["task"])
for s in run["steps"]:
    print(f"  step {s['step_n']}", s["output"][:80], "risk=", s["hallucination"]["composite"])

patch = requests.post(f"{BACKEND}/runs/{run_id}/patch",
    json={"source_step": 1, "target_step": 3, "patch_layer": 7}).json()
print("cross-step KL:", patch["kl"], "significant:", patch["significant"])
`;
    function copy() {
        navigator.clipboard.writeText(text);
        toast.success("snippet copied");
    }
    return (
        <div className="ns-card" data-testid="reproducibility-snippet">
            <header
                className="flex items-center justify-between px-4 pt-3.5 pb-2"
                style={{ borderBottom: "1px solid var(--ns-border-subtle)" }}
            >
                <span className="ns-section-title">reproducibility</span>
                <button
                    type="button"
                    onClick={copy}
                    className="flex items-center gap-1 rounded-md border px-2 py-0.5 font-mono text-[10px] text-[color:var(--ns-fg-secondary)] hover:bg-[color:var(--ns-bg-surface-2)]"
                    style={{ borderColor: "var(--ns-border-subtle)" }}
                    data-testid="reproducibility-copy-button"
                >
                    <Copy size={10} /> copy
                </button>
            </header>
            <pre
                className="overflow-x-auto px-4 py-3 font-mono text-[10.5px] leading-5 text-[color:var(--ns-fg-secondary)]"
                style={{ background: "var(--ns-bg-codeblock)" }}
            >
                {text}
            </pre>
        </div>
    );
}
