import { cn } from "@/lib/utils";

const statusMap = {
    queued: { color: "#7F8CA3", label: "queued" },
    running: { color: "#4FB3C8", label: "running" },
    done: { color: "#6EE7B7", label: "done" },
    error: { color: "#F87171", label: "error" },
    pending: { color: "#7F8CA3", label: "pending" },
};

export default function StatusDot({ status = "queued", showLabel = true, className }) {
    const cfg = statusMap[status] || statusMap.queued;
    return (
        <span
            className={cn(
                "inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider",
                className,
            )}
            data-testid={`status-${status}`}
        >
            <span
                className={cn(
                    "ns-status-dot",
                    status === "running" && "animate-pulse",
                )}
                style={{ background: cfg.color }}
            />
            {showLabel ? cfg.label : null}
        </span>
    );
}
