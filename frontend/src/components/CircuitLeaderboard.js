import React, { useState } from "react";
import { ThumbsUp, Plus, ShieldCheck, User } from "lucide-react";

const INITIAL_LEADERBOARD = [
    {
        id: 1,
        name: "Indirect Object Identification (IOI)",
        layers: "L8-L12",
        author: "@charlie_researcher",
        description: "Orchestrates the transfer of target indirect object representations from name-mover attention heads to final unembeddings.",
        upvotes: 142,
        verified: true
    },
    {
        id: 2,
        name: "DNA Replication Transcription Matcher",
        layers: "L12-L18",
        author: "@bio_interpret",
        description: "A compact circuit mapping nucleotide combinations directly to their standard biochemical code completions.",
        upvotes: 89,
        verified: true
    },
    {
        id: 3,
        name: "Arabic Translation Script Hook",
        layers: "L6-L12",
        author: "@sae_fanatic",
        description: "Responsible for routing Arabic alphabet tokens to Layer 12 multi-hop dictionary features.",
        upvotes: 67,
        verified: false
    },
    {
        id: 4,
        name: "Chain-of-thought Contradiction Watchdog",
        layers: "L18-L24",
        author: "@neuroscope_dev",
        description: "Fires when intermediate reasoning tokens mismatch factual database lookups, triggering next-step correction warnings.",
        upvotes: 56,
        verified: true
    }
];

export default function CircuitLeaderboard() {
    const [leaderboard, setLeaderboard] = useState(INITIAL_LEADERBOARD);
    const [showForm, setShowForm] = useState(false);
    const [newCircuit, setNewCircuit] = useState({
        name: "",
        layers: "L12-L18",
        author: "",
        description: ""
    });

    const handleUpvote = (id) => {
        setLeaderboard((prev) =>
            prev.map((c) => (c.id === id ? { ...c, upvotes: c.upvotes + 1 } : c))
        );
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!newCircuit.name || !newCircuit.description || !newCircuit.author) return;

        const record = {
            id: leaderboard.length + 1,
            name: newCircuit.name,
            layers: newCircuit.layers,
            author: newCircuit.author.startsWith("@") ? newCircuit.author : `@${newCircuit.author}`,
            description: newCircuit.description,
            upvotes: 1,
            verified: false
        };

        setLeaderboard((prev) => [record, ...prev]);
        setNewCircuit({ name: "", layers: "L12-L18", author: "", description: "" });
        setShowForm(false);
    };

    return (
        <section className="mt-16 animate-fadeIn" id="circuit-leaderboard">
            <div className="flex flex-col md:flex-row md:items-end justify-between mb-8">
                <div>
                    <div className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)] border-[color:var(--ns-border)]">
                        <span className="h-1 w-1 rounded-full bg-[color:var(--ns-accent)]" />
                        collective research database
                    </div>
                    <h2 className="mt-2 text-2xl font-bold tracking-tight text-[color:var(--ns-fg-primary)] sm:text-3xl">
                        Community Circuit discoveries
                    </h2>
                    <p className="mt-2 text-sm text-[color:var(--ns-fg-secondary)] max-w-xl">
                        A crowdsourced index of computational circuits verified by mechanistic interpretability researchers.
                    </p>
                </div>
                <button
                    onClick={() => setShowForm(!showForm)}
                    className="mt-4 md:mt-0 inline-flex items-center gap-1.5 rounded-md bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border)] hover:border-[color:var(--ns-border-strong)] px-3 py-1.5 text-xs font-mono text-[color:var(--ns-fg-primary)] transition-colors"
                >
                    <Plus size={12} />
                    {showForm ? "Cancel" : "Submit Finding"}
                </button>
            </div>

            {showForm && (
                <form
                    onSubmit={handleSubmit}
                    className="ns-card p-5 mb-6 border border-[color:var(--ns-border)] grid gap-4 max-w-lg mx-auto"
                >
                    <h3 className="font-mono text-xs uppercase tracking-wider text-[color:var(--ns-fg-primary)] font-semibold border-b border-[color:var(--ns-border-subtle)] pb-2">
                        Submit Circuit Finding
                    </h3>
                    
                    <div className="grid gap-2">
                        <label className="font-mono text-[10px] text-[color:var(--ns-fg-muted)] uppercase">Circuit Name</label>
                        <input
                            type="text"
                            required
                            placeholder="e.g. Induction Head Subgraph"
                            value={newCircuit.name}
                            onChange={(e) => setNewCircuit({ ...newCircuit, name: e.target.value })}
                            className="bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded p-2 font-mono text-xs text-[color:var(--ns-fg-primary)] focus:outline-none"
                        />
                    </div>

                    <div className="grid gap-2 grid-cols-2">
                        <div className="grid gap-1">
                            <label className="font-mono text-[10px] text-[color:var(--ns-fg-muted)] uppercase">Layer Scope</label>
                            <select
                                value={newCircuit.layers}
                                onChange={(e) => setNewCircuit({ ...newCircuit, layers: e.target.value })}
                                className="bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded p-2 font-mono text-xs text-[color:var(--ns-fg-primary)] focus:outline-none"
                            >
                                <option value="L0-L8">L0-L8 (Early)</option>
                                <option value="L8-L12">L8-L12 (Mid-Early)</option>
                                <option value="L12-L18">L12-L18 (Mid-Late)</option>
                                <option value="L18-L24">L18-L24 (Late)</option>
                            </select>
                        </div>
                        <div className="grid gap-1">
                            <label className="font-mono text-[10px] text-[color:var(--ns-fg-muted)] uppercase">Discovered By (Twitter/Github)</label>
                            <input
                                type="text"
                                required
                                placeholder="e.g. charlie_research"
                                value={newCircuit.author}
                                onChange={(e) => setNewCircuit({ ...newCircuit, author: e.target.value })}
                                className="bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded p-2 font-mono text-xs text-[color:var(--ns-fg-primary)] focus:outline-none"
                            />
                        </div>
                    </div>

                    <div className="grid gap-2">
                        <label className="font-mono text-[10px] text-[color:var(--ns-fg-muted)] uppercase">Mathematical/Semantic Description</label>
                        <textarea
                            required
                            rows="3"
                            placeholder="Detail the attention heads, key vectors, and mathematical pathways..."
                            value={newCircuit.description}
                            onChange={(e) => setNewCircuit({ ...newCircuit, description: e.target.value })}
                            className="bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded p-2 font-mono text-xs text-[color:var(--ns-fg-primary)] resize-none focus:outline-none"
                        />
                    </div>

                    <button
                        type="submit"
                        className="w-full bg-[color:var(--ns-accent)] text-[color:var(--ns-bg-canvas)] rounded p-2 font-mono text-xs font-semibold hover:opacity-90 transition-opacity"
                    >
                        Publish Finding
                    </button>
                </form>
            )}

            {/* Leaderboard database table */}
            <div className="ns-card overflow-hidden border border-[color:var(--ns-border-subtle)]">
                <div className="overflow-x-auto">
                    <table className="w-full text-left font-mono border-collapse text-xs">
                        <thead>
                            <tr className="bg-[color:var(--ns-bg-surface-2)] border-b border-[color:var(--ns-border-subtle)] text-[color:var(--ns-fg-muted)] uppercase text-[9px] tracking-wider">
                                <th className="px-5 py-3.5">Circuit Name</th>
                                <th className="px-5 py-3.5">Scope</th>
                                <th className="px-5 py-3.5">Description</th>
                                <th className="px-5 py-3.5">Author</th>
                                <th className="px-5 py-3.5 text-right">Attribution Rating</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-[color:var(--ns-border-subtle)]">
                            {leaderboard.map((item) => (
                                <tr key={item.id} className="hover:bg-[color:var(--ns-bg-surface-2)] transition-colors">
                                    <td className="px-5 py-4 font-semibold text-[color:var(--ns-fg-primary)]">
                                        <div className="flex items-center gap-1.5">
                                            {item.name}
                                            {item.verified && (
                                                <ShieldCheck size={13} className="text-[color:var(--ns-mint)]" title="Verified by Core Team" />
                                            )}
                                        </div>
                                    </td>
                                    <td className="px-5 py-4">
                                        <span className="px-2 py-0.5 rounded text-[10px] bg-[color:var(--ns-bg-surface-3)] border border-[color:var(--ns-border-subtle)] text-[color:var(--ns-accent)]">
                                            {item.layers}
                                        </span>
                                    </td>
                                    <td className="px-5 py-4 text-[color:var(--ns-fg-secondary)] max-w-sm font-sans leading-relaxed">
                                        {item.description}
                                    </td>
                                    <td className="px-5 py-4 text-[color:var(--ns-fg-muted)]">
                                        <span className="flex items-center gap-1">
                                            <User size={10} />
                                            {item.author}
                                        </span>
                                    </td>
                                    <td className="px-5 py-4 text-right">
                                        <button
                                            onClick={() => handleUpvote(item.id)}
                                            className="inline-flex items-center gap-1.5 rounded bg-[color:var(--ns-bg-surface-3)] hover:bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border)] hover:border-[color:var(--ns-accent-2)] px-2.5 py-1 text-xs text-[color:var(--ns-fg-primary)] transition-all cursor-pointer"
                                        >
                                            <ThumbsUp size={11} className="text-[color:var(--ns-amber)]" />
                                            <span>{item.upvotes}</span>
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
    );
}
