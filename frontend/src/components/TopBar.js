import { Link, NavLink, useLocation } from "react-router-dom";
import { Activity, FlaskConical, BookOpen, Play, Github } from "lucide-react";

export default function TopBar() {
    const { pathname } = useLocation();
    const inAnalysis = pathname.startsWith("/run/");
    return (
        <div
            className="sticky top-0 z-40 w-full border-b"
            style={{
                borderColor: "var(--ns-border-subtle)",
                background: "rgba(11,15,20,0.85)",
                backdropFilter: "blur(8px)",
            }}
            data-testid="topbar"
        >
            <div className="mx-auto flex h-12 max-w-[1440px] items-center justify-between px-4 sm:px-6">
                <Link
                    to="/"
                    className="flex items-center gap-2.5"
                    data-testid="topbar-home"
                >
                    <span
                        className="flex h-5 w-5 items-center justify-center rounded"
                        style={{ background: "var(--ns-accent)" }}
                    >
                        <Activity
                            size={12}
                            color="var(--ns-bg-canvas)"
                            strokeWidth={2.5}
                        />
                    </span>
                    <span className="font-mono text-[13px] font-semibold tracking-tight text-[color:var(--ns-fg-primary)]">
                        NeuroScope
                    </span>
                    <span className="font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                        v1.0 · gpt2-small
                    </span>
                </Link>
                <nav className="flex items-center gap-1">
                    <NavTab to="/run" icon={Play} label="Run" testid="nav-run" />
                    <NavTab
                        to="/experiments"
                        icon={FlaskConical}
                        label="Experiments"
                        testid="nav-experiments"
                    />
                    <NavTab
                        to="/findings"
                        icon={Activity}
                        label="Findings"
                        testid="nav-findings"
                    />
                    <NavTab
                        to="/docs"
                        icon={BookOpen}
                        label="Docs"
                        testid="nav-docs"
                    />
                    <a
                        href="https://github.com"
                        target="_blank"
                        rel="noreferrer"
                        className="ml-2 flex h-8 w-8 items-center justify-center rounded-md text-[color:var(--ns-fg-muted)] hover:bg-[color:var(--ns-bg-surface-2)] hover:text-[color:var(--ns-fg-primary)]"
                        title="View on GitHub"
                        data-testid="nav-github"
                    >
                        <Github size={14} />
                    </a>
                </nav>
            </div>
            {inAnalysis ? (
                <div
                    className="px-4 py-1 sm:px-6"
                    style={{
                        borderTop: "1px solid var(--ns-border-subtle)",
                        background: "var(--ns-bg-surface-1)",
                    }}
                >
                    <div className="mx-auto max-w-[1440px] font-mono text-[10px] text-[color:var(--ns-fg-muted)]">
                        ANALYSIS · GPT-2 SMALL (12 LAYERS, 768 d_model) · SAE
                        gpt2-small-res-jb — read-only research view
                    </div>
                </div>
            ) : null}
        </div>
    );
}

function NavTab({ to, icon: Icon, label, testid }) {
    return (
        <NavLink
            to={to}
            data-testid={testid}
            className={({ isActive }) =>
                `flex h-8 items-center gap-1.5 rounded-md px-2.5 text-xs ${
                    isActive
                        ? "bg-[color:var(--ns-bg-surface-2)] text-[color:var(--ns-fg-primary)]"
                        : "text-[color:var(--ns-fg-muted)] hover:bg-[color:var(--ns-bg-surface-2)] hover:text-[color:var(--ns-fg-primary)]"
                }`
            }
        >
            <Icon size={13} />
            {label}
        </NavLink>
    );
}
