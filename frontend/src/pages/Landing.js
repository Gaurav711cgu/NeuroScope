import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Activity, ArrowRight, Github, BookOpen, MessageSquare, Mail, Award, Check, X, Star } from "lucide-react";
import AnimatedResidualStream from "../components/AnimatedResidualStream";
import LiveDemoWidget from "../components/LiveDemoWidget";
import FeatureExplorer from "../components/FeatureExplorer";
import CausalPatchingPlayground from "../components/CausalPatchingPlayground";
import CircuitLeaderboard from "../components/CircuitLeaderboard";
import DriftMonitor from "../components/DriftMonitor";
import BlogSection from "../components/BlogSection";

export default function Landing() {
    const [activeTab, setActiveTab] = useState("explorer");
    const [newsletterEmail, setNewsletterEmail] = useState("");
    const [subscribed, setSubscribed] = useState(false);

    const handleSubscribe = (e) => {
        e.preventDefault();
        if (newsletterEmail) {
            setSubscribed(true);
            setNewsletterEmail("");
        }
    };

    return (
        <div className="relative overflow-hidden min-h-screen bg-[color:var(--ns-bg-canvas)]">
            {/* noise grid */}
            <div className="ns-noise-bg absolute inset-0 pointer-events-none" />

            {/* HERO SECTION */}
            <header className="relative border-b border-[color:var(--ns-border-subtle)] py-20 lg:py-28 overflow-hidden">
                <AnimatedResidualStream />
                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[color:var(--ns-bg-canvas)]/30 to-[color:var(--ns-bg-canvas)] pointer-events-none" />
                
                <div className="relative mx-auto max-w-[1280px] px-4 sm:px-6 z-10 flex flex-col items-center text-center">
                    <div className="inline-flex items-center gap-2 rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-[color:var(--ns-fg-muted)] border-[color:var(--ns-border)] bg-[color:var(--ns-bg-surface-1)]/75 backdrop-blur">
                        <span className="h-1.5 w-1.5 rounded-full bg-[color:var(--ns-accent)]" />
                        mechanistic interpretability terminal · v2.0
                    </div>

                    <h1 className="mt-6 text-4xl font-extrabold tracking-tight text-[color:var(--ns-fg-primary)] sm:text-5xl lg:text-6xl font-mono max-w-4xl leading-tight">
                        Look <span className="text-[color:var(--ns-accent)]">inside</span> a multi-step
                        agent’s reasoning.
                    </h1>

                    <p className="mt-6 max-w-2xl text-base leading-relaxed text-[color:var(--ns-fg-secondary)]">
                        NeuroScope captures intermediate residual streams, SAE dictionary directions, and attention weights. 
                        Trace, steer, and causally patch representations across multiple steps to identify failure modes before they reach output tokens.
                    </p>

                    {/* 3-tier CTA Funnel */}
                    <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
                        <Link
                            to="/run"
                            className="inline-flex items-center gap-1.5 rounded-md bg-[color:var(--ns-accent)] text-[color:var(--ns-bg-canvas)] px-5 py-2.5 text-sm font-semibold font-mono hover:opacity-90 transition-opacity"
                        >
                            <Activity size={15} /> Launch Trajectory Editor
                        </Link>
                        
                        <a
                            href="#research-blog"
                            className="inline-flex items-center gap-1.5 rounded-md border border-[color:var(--ns-border)] bg-[color:var(--ns-bg-surface-1)]/40 hover:bg-[color:var(--ns-bg-surface-2)] hover:border-[color:var(--ns-border-strong)] px-5 py-2.5 text-sm font-mono text-[color:var(--ns-fg-primary)] transition-all"
                        >
                            <BookOpen size={14} /> Read Research Blog
                            <ArrowRight size={13} className="text-[color:var(--ns-fg-muted)]" />
                        </a>

                        <a
                            href="https://github.com"
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1.5 rounded-md bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border)] px-4 py-2.5 text-sm font-mono text-[color:var(--ns-fg-secondary)] hover:text-[color:var(--ns-fg-primary)] transition-colors"
                        >
                            <Github size={14} /> Star on Github
                            <span className="flex items-center gap-0.5 text-[10px] text-[color:var(--ns-amber)] bg-[color:var(--ns-bg-surface-3)] px-1 rounded ml-1 border border-[color:var(--ns-border-subtle)]">
                                <Star size={10} fill="var(--ns-amber)" /> 1.2k
                            </span>
                        </a>
                    </div>

                    {/* Stat Grid */}
                    <div className="mt-12 grid grid-cols-2 gap-3 text-[11px] sm:grid-cols-4 w-full max-w-4xl font-mono">
                        <div className="ns-card-strong px-4 py-2.5 border border-[color:var(--ns-border)]">
                            <div className="text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">Target Model</div>
                            <div className="mt-0.5 text-xs text-[color:var(--ns-fg-primary)] font-semibold">Gemma-2-2b-it</div>
                        </div>
                        <div className="ns-card-strong px-4 py-2.5 border border-[color:var(--ns-border)]">
                            <div className="text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">SAE Dictionary</div>
                            <div className="mt-0.5 text-xs text-[color:var(--ns-fg-primary)] font-semibold">GemmaScope 16k</div>
                        </div>
                        <div className="ns-card-strong px-4 py-2.5 border border-[color:var(--ns-border)]">
                            <div className="text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">Captures</div>
                            <div className="mt-0.5 text-xs text-[color:var(--ns-fg-primary)] font-semibold">L6, L12, L18, L24</div>
                        </div>
                        <div className="ns-card-strong px-4 py-2.5 border border-[color:var(--ns-border)]">
                            <div className="text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">Causal Edges</div>
                            <div className="mt-0.5 text-xs text-[color:var(--ns-fg-primary)] font-semibold">Attribution Swaps</div>
                        </div>
                    </div>
                </div>
            </header>

            {/* LIVE DEMO EMBED */}
            <main className="relative mx-auto max-w-[1280px] px-4 py-16 sm:px-6 space-y-20 z-10">
                <section>
                    <div className="text-center max-w-xl mx-auto mb-10">
                        <div className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)] border-[color:var(--ns-border)]">
                            Interactive Demonstration
                        </div>
                        <h2 className="mt-2 text-2xl font-bold tracking-tight text-[color:var(--ns-fg-primary)] sm:text-3xl font-mono">
                            Paste text. See activations.
                        </h2>
                        <p className="mt-2 text-sm text-[color:var(--ns-fg-secondary)]">
                            Experience real-time dictionary decomposition. Paste any sentence below to isolate SAE features and trigger causal steering.
                        </p>
                    </div>
                    
                    <LiveDemoWidget />
                </section>

                {/* FEATURE SHOWCASE TABS */}
                <section>
                    <div className="text-center max-w-xl mx-auto mb-8">
                        <div className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)] border-[color:var(--ns-border)]">
                            Diagnostic Showcase
                        </div>
                        <h2 className="mt-2 text-2xl font-bold tracking-tight text-[color:var(--ns-fg-primary)] sm:text-3xl font-mono">
                            Deep-Dive Tools
                        </h2>
                        <p className="mt-2 text-sm text-[color:var(--ns-fg-secondary)]">
                            Explore discovered computational features, run patching simulations, and analyze multi-turn KV cache drift.
                        </p>
                    </div>

                    {/* Tab Navigation */}
                    <div className="flex flex-wrap justify-center gap-2 mb-6 border-b border-[color:var(--ns-border-subtle)] pb-4 font-mono">
                        <button
                            onClick={() => setActiveTab("explorer")}
                            className={`px-4 py-2 text-xs rounded-md border transition-all ${
                                activeTab === "explorer"
                                    ? "bg-[color:var(--ns-bg-surface-2)] border-[color:var(--ns-accent)] text-[color:var(--ns-accent)] font-semibold"
                                    : "border-transparent text-[color:var(--ns-fg-muted)] hover:text-[color:var(--ns-fg-primary)]"
                            }`}
                        >
                            SAE Feature Explorer
                        </button>
                        <button
                            onClick={() => setActiveTab("patching")}
                            className={`px-4 py-2 text-xs rounded-md border transition-all ${
                                activeTab === "patching"
                                    ? "bg-[color:var(--ns-bg-surface-2)] border-[color:var(--ns-accent)] text-[color:var(--ns-accent)] font-semibold"
                                    : "border-transparent text-[color:var(--ns-fg-muted)] hover:text-[color:var(--ns-fg-primary)]"
                            }`}
                        >
                            Causal Patching Sandbox
                        </button>
                        <button
                            onClick={() => setActiveTab("drift")}
                            className={`px-4 py-2 text-xs rounded-md border transition-all ${
                                activeTab === "drift"
                                    ? "bg-[color:var(--ns-bg-surface-2)] border-[color:var(--ns-accent)] text-[color:var(--ns-accent)] font-semibold"
                                    : "border-transparent text-[color:var(--ns-fg-muted)] hover:text-[color:var(--ns-fg-primary)]"
                            }`}
                        >
                            Representation Drift PCA
                        </button>
                        <button
                            onClick={() => setActiveTab("leaderboard")}
                            className={`px-4 py-2 text-xs rounded-md border transition-all ${
                                activeTab === "leaderboard"
                                    ? "bg-[color:var(--ns-bg-surface-2)] border-[color:var(--ns-accent)] text-[color:var(--ns-accent)] font-semibold"
                                    : "border-transparent text-[color:var(--ns-fg-muted)] hover:text-[color:var(--ns-fg-primary)]"
                            }`}
                        >
                            Community Discoveries
                        </button>
                    </div>

                    <div className="transition-all duration-300">
                        {activeTab === "explorer" && <FeatureExplorer />}
                        {activeTab === "patching" && <CausalPatchingPlayground />}
                        {activeTab === "drift" && <DriftMonitor />}
                        {activeTab === "leaderboard" && <CircuitLeaderboard />}
                    </div>
                </section>

                {/* RESEARCH BLOG SECTION */}
                <BlogSection />

                {/* HOW IT COMPARES */}
                <section className="mt-20">
                    <div className="text-center max-w-xl mx-auto mb-10">
                        <div className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)] border-[color:var(--ns-border)]">
                            Competitive Alignment
                        </div>
                        <h2 className="mt-2 text-2xl font-bold tracking-tight text-[color:var(--ns-fg-primary)] sm:text-3xl font-mono">
                            Honest Tool Comparison
                        </h2>
                        <p className="mt-2 text-sm text-[color:var(--ns-fg-secondary)]">
                            Why NeuroScope is engineered specifically for agentic multi-turn tracing, not just single-prompt visualizations.
                        </p>
                    </div>

                    <div className="ns-card border border-[color:var(--ns-border-subtle)] overflow-hidden">
                        <div className="overflow-x-auto">
                            <table className="w-full text-left font-mono text-xs border-collapse">
                                <thead>
                                    <tr className="bg-[color:var(--ns-bg-surface-2)] border-b border-[color:var(--ns-border-subtle)] text-[color:var(--ns-fg-muted)] uppercase text-[9px]">
                                        <th className="px-5 py-3">Feature capabilities</th>
                                        <th className="px-5 py-3 text-[color:var(--ns-accent)]">NeuroScope</th>
                                        <th className="px-5 py-3">TransformerLens</th>
                                        <th className="px-5 py-3">COGNITWIN</th>
                                        <th className="px-5 py-3">Goodfire</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-[color:var(--ns-border-subtle)]">
                                    <tr className="hover:bg-[color:var(--ns-bg-surface-1)]">
                                        <td className="px-5 py-3.5 font-sans font-semibold text-[color:var(--ns-fg-primary)]">Multi-turn trajectory diagnostics</td>
                                        <td className="px-5 py-3.5"><Check size={14} className="text-[color:var(--ns-mint)]" /></td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">No (Single-prompt)</td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">No</td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">No (Single-prompt)</td>
                                    </tr>
                                    <tr className="hover:bg-[color:var(--ns-bg-surface-1)]">
                                        <td className="px-5 py-3.5 font-sans font-semibold text-[color:var(--ns-fg-primary)]">SAE Dictionary Support</td>
                                        <td className="px-5 py-3.5"><Check size={14} className="text-[color:var(--ns-mint)]" /></td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">Manual setup</td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">No</td>
                                        <td className="px-5 py-3.5"><Check size={14} className="text-[color:var(--ns-mint)]" /></td>
                                    </tr>
                                    <tr className="hover:bg-[color:var(--ns-bg-surface-1)]">
                                        <td className="px-5 py-3.5 font-sans font-semibold text-[color:var(--ns-fg-primary)]">Causal Cross-Step Patching</td>
                                        <td className="px-5 py-3.5"><Check size={14} className="text-[color:var(--ns-mint)]" /></td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">Manual script only</td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">No</td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">No</td>
                                    </tr>
                                    <tr className="hover:bg-[color:var(--ns-bg-surface-1)]">
                                        <td className="px-5 py-3.5 font-sans font-semibold text-[color:var(--ns-fg-primary)]">Interactive Concept Steering</td>
                                        <td className="px-5 py-3.5"><Check size={14} className="text-[color:var(--ns-mint)]" /></td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">No</td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">No</td>
                                        <td className="px-5 py-3.5"><Check size={14} className="text-[color:var(--ns-mint)]" /></td>
                                    </tr>
                                    <tr className="hover:bg-[color:var(--ns-bg-surface-1)]">
                                        <td className="px-5 py-3.5 font-sans font-semibold text-[color:var(--ns-fg-primary)]">KV-Cache representation drift PCA</td>
                                        <td className="px-5 py-3.5"><Check size={14} className="text-[color:var(--ns-mint)]" /></td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">No</td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">No</td>
                                        <td className="px-5 py-3.5 text-[color:var(--ns-fg-muted)]">No</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>

                {/* FOOTER CTA & NEWSLETTER */}
                <section className="ns-card-strong border border-[color:var(--ns-border)] p-6 md:p-10 flex flex-col md:flex-row justify-between items-center gap-6">
                    <div className="max-w-md">
                        <div className="flex items-center gap-1.5 font-mono text-[11px] text-[color:var(--ns-accent)] font-semibold uppercase mb-1">
                            <Mail size={12} />
                            Interpretability papers digest
                        </div>
                        <h3 className="text-lg font-bold text-[color:var(--ns-fg-primary)] font-mono leading-tight">
                            Subscribe to the Paper Feed
                        </h3>
                        <p className="text-xs text-[color:var(--ns-fg-secondary)] mt-1.5 leading-relaxed">
                            Receive bi-weekly reviews of emerging mechanistic interpretability papers and sparse dictionary updates.
                        </p>
                    </div>

                    <div className="w-full md:w-auto shrink-0">
                        {subscribed ? (
                            <div className="font-mono text-xs text-[color:var(--ns-mint)] border border-emerald-500/20 bg-emerald-950/10 px-4 py-2.5 rounded-lg">
                                Subscribed successfully. Check your feed soon!
                            </div>
                        ) : (
                            <form onSubmit={handleSubscribe} className="flex flex-col sm:flex-row gap-2">
                                <input
                                    type="email"
                                    required
                                    placeholder="Enter researcher email..."
                                    value={newsletterEmail}
                                    onChange={(e) => setNewsletterEmail(e.target.value)}
                                    className="bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded-lg px-3 py-2 font-mono text-xs text-[color:var(--ns-fg-primary)] focus:outline-none focus:border-[color:var(--ns-accent)] w-full sm:w-64"
                                />
                                <button
                                    type="submit"
                                    className="bg-[color:var(--ns-accent)] text-[color:var(--ns-bg-canvas)] font-mono text-xs font-semibold px-4 py-2 rounded-lg hover:opacity-90 transition-opacity"
                                >
                                    Join Digest
                                </button>
                            </form>
                        )}
                    </div>
                </section>
            </main>

            {/* FOOTER BADGES */}
            <footer className="border-t border-[color:var(--ns-border-subtle)] py-6 z-10 relative">
                <div className="mx-auto max-w-[1280px] px-4 sm:px-6 flex flex-col sm:flex-row justify-between items-center gap-4 text-[10px] font-mono text-[color:var(--ns-fg-muted)]">
                    <div className="flex flex-wrap items-center gap-3">
                        <span>© 2026 NeuroScope Team. All rights reserved.</span>
                        <span>·</span>
                        <a href="https://github.com" target="_blank" rel="noreferrer" className="hover:underline flex items-center gap-0.5">
                            MIT License <ExternalLinkIcon />
                        </a>
                    </div>
                    <div className="flex gap-4">
                        <a href="https://github.com" target="_blank" rel="noreferrer" className="hover:underline flex items-center gap-0.5">
                            Github Repository <ExternalLinkIcon />
                        </a>
                        <a href="https://discord.com" target="_blank" rel="noreferrer" className="hover:underline flex items-center gap-0.5">
                            Discord Server <ExternalLinkIcon />
                        </a>
                    </div>
                </div>
            </footer>
        </div>
    );
}

function ExternalLinkIcon() {
    return <ArrowRight size={10} className="transform rotate-[-45deg] inline" />;
}
