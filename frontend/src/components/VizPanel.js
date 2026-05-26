import { cn } from "@/lib/utils";

export default function VizPanel({
    title,
    subtitle,
    methodNote,
    right,
    children,
    className,
    testid,
    bodyClass,
}) {
    return (
        <section
            className={cn("ns-card flex flex-col", className)}
            data-testid={testid}
        >
            <header
                className="flex items-start justify-between gap-3 px-4 pt-3.5 pb-2"
                style={{ borderBottom: "1px solid var(--ns-border-subtle)" }}
            >
                <div>
                    <div className="ns-section-title">{title}</div>
                    {subtitle ? (
                        <div className="mt-0.5 text-[11px] text-[color:var(--ns-fg-secondary)]">
                            {subtitle}
                        </div>
                    ) : null}
                </div>
                {right ? <div className="flex items-center gap-2">{right}</div> : null}
            </header>
            <div className={cn("flex-1 px-4 py-3", bodyClass)}>{children}</div>
            {methodNote ? (
                <div
                    className="px-4 pb-3 pt-1 text-[10.5px] text-[color:var(--ns-fg-muted)]"
                    style={{ borderTop: "1px solid var(--ns-border-subtle)" }}
                >
                    {methodNote}
                </div>
            ) : null}
        </section>
    );
}
