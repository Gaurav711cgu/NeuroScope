import React, { useState } from "react";
import { Search, Filter, Sliders, ArrowUpRight, HelpCircle } from "lucide-react";

const FEATURES_DB = [
    {
        id: "L12_F8891",
        layer: 12,
        fid: 8891,
        label: "Autoencoder architectures",
        category: "Semantic",
        polysemanticity: 0.08,
        activeTokens: ["autoencoder", "dictionary", "SAE", "reconstruction"],
        desc: "Fires on machine learning architectures and vector dictionary learning terms.",
        density: "0.0014%",
        neuronpediaUrl: "https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/8891"
    },
    {
        id: "L12_F10454",
        layer: 12,
        fid: 10454,
        label: "Mechanistic Interpretability",
        category: "Semantic",
        polysemanticity: 0.02,
        activeTokens: ["interpretability", "circuits", "monosemantic", "superposition"],
        desc: "Fires on transformer-circuits research discussions.",
        density: "0.0008%",
        neuronpediaUrl: "https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/10454"
    },
    {
        id: "L6_F2203",
        layer: 6,
        fid: 2203,
        label: "JSON opening brackets",
        category: "Structural",
        polysemanticity: 0.04,
        activeTokens: ["{", "JSON", '{"', "data"],
        desc: "Tracks starting brackets and syntax declaration for data serialization.",
        density: "0.0120%",
        neuronpediaUrl: "https://www.neuronpedia.org/gemma-2-2b/6-gemmascope-res-16k/2203"
    },
    {
        id: "L18_F1102",
        layer: 18,
        fid: 1102,
        label: "Passive voice constructions",
        category: "Syntactic",
        polysemanticity: 0.18,
        activeTokens: ["been", "was", "were", "had"],
        desc: "Activates in passive voice syntactic structures.",
        density: "0.0450%",
        neuronpediaUrl: "https://www.neuronpedia.org/gemma-2-2b/18-gemmascope-res-16k/1102"
    },
    {
        id: "L24_F4099",
        layer: 24,
        fid: 4099,
        label: "Reasoning failure detection",
        category: "Error Detection",
        polysemanticity: 0.05,
        activeTokens: ["wait", "actually", "incorrect", "instead"],
        desc: "High-activation triggers when the model self-corrects or realizes a contradiction in context.",
        density: "0.0003%",
        neuronpediaUrl: "https://www.neuronpedia.org/gemma-2-2b/24-gemmascope-res-16k/4099"
    },
    {
        id: "L12_F12044",
        layer: 12,
        fid: 12044,
        label: "Residual stream operations",
        category: "Semantic",
        polysemanticity: 0.12,
        activeTokens: ["residual", "stream", "post_resid", "additive"],
        desc: "Activates on mathematical additive stream operations.",
        density: "0.0031%",
        neuronpediaUrl: "https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/12044"
    },
    {
        id: "L18_F8203",
        layer: 18,
        fid: 8203,
        label: "Drift deviation warning",
        category: "Error Detection",
        polysemanticity: 0.09,
        activeTokens: ["therefore", "thus", "consequently", "conclusion"],
        desc: "Flags cumulative representation shifts away from initial query constraints.",
        density: "0.0022%",
        neuronpediaUrl: "https://www.neuronpedia.org/gemma-2-2b/18-gemmascope-res-16k/8203"
    },
    {
        id: "L6_F992",
        layer: 6,
        fid: 992,
        label: "Arabic script writing",
        category: "Syntactic",
        polysemanticity: 0.01,
        activeTokens: ["العربية", "في", "من", "الله"],
        desc: "Fires on Arabic characters and letter sequences.",
        density: "0.0055%",
        neuronpediaUrl: "https://www.neuronpedia.org/gemma-2-2b/6-gemmascope-res-16k/992"
    },
    {
        id: "L12_F1410",
        layer: 12,
        fid: 1410,
        label: "Causal patching terminology",
        category: "Semantic",
        polysemanticity: 0.10,
        activeTokens: ["causal", "intervention", "patching", "ablation"],
        desc: "Fires on causal attribution and structural patching concepts.",
        density: "0.0019%",
        neuronpediaUrl: "https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/1410"
    },
    {
        id: "L24_F1312",
        layer: 24,
        fid: 1312,
        label: "Code block termination",
        category: "Structural",
        polysemanticity: 0.15,
        activeTokens: ["```", "return", "export", "class"],
        desc: "Traces markdown code blocks ending markers.",
        density: "0.0098%",
        neuronpediaUrl: "https://www.neuronpedia.org/gemma-2-2b/24-gemmascope-res-16k/1312"
    }
];

export default function FeatureExplorer() {
    const [searchQuery, setSearchQuery] = useState("");
    const [layerFilter, setLayerFilter] = useState("All");
    const [categoryFilter, setCategoryFilter] = useState("All");
    const [polyFilter, setPolyFilter] = useState("All");
    
    // Steering interactive popup states
    const [steeredFeature, setSteeredFeature] = useState(null);
    const [steerFactor, setSteerFactor] = useState(4.0);

    const filteredFeatures = FEATURES_DB.filter((feat) => {
        const matchesSearch =
            feat.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
            feat.desc.toLowerCase().includes(searchQuery.toLowerCase()) ||
            feat.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
            feat.activeTokens.some((t) => t.toLowerCase().includes(searchQuery.toLowerCase()));

        const matchesLayer = layerFilter === "All" || feat.layer === parseInt(layerFilter);
        const matchesCategory = categoryFilter === "All" || feat.category === categoryFilter;
        
        let matchesPoly = true;
        if (polyFilter === "Monosemantic") matchesPoly = feat.polysemanticity < 0.05;
        else if (polyFilter === "Moderate") matchesPoly = feat.polysemanticity >= 0.05 && feat.polysemanticity <= 0.12;
        else if (polyFilter === "Polysemantic") matchesPoly = feat.polysemanticity > 0.12;

        return matchesSearch && matchesLayer && matchesCategory && matchesPoly;
    });

    const triggerSteerSimulator = (feat) => {
        setSteeredFeature(feat);
        setSteerFactor(4.0);
    };

    return (
        <section className="mt-16" id="feature-explorer">
            <div className="flex flex-col md:flex-row md:items-end justify-between mb-8">
                <div>
                    <div className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-[color:var(--ns-fg-muted)] border-[color:var(--ns-border)]">
                        <span className="h-1 w-1 rounded-full bg-[color:var(--ns-accent)]" />
                        GemmaScope SAE Dictionary
                    </div>
                    <h2 className="mt-2 text-2xl font-bold tracking-tight text-[color:var(--ns-fg-primary)] sm:text-3xl">
                        SAE Feature Explorer
                    </h2>
                    <p className="mt-2 text-sm text-[color:var(--ns-fg-secondary)] max-w-xl">
                        Browse, filter, and inspect sparse autoencoder directions discovered at Layer 6, 12, 18, and 24.
                    </p>
                </div>
            </div>

            {/* Filters panel */}
            <div className="ns-card-strong p-4 mb-6 border border-[color:var(--ns-border)] grid gap-4 md:grid-cols-4 items-center">
                <div className="relative">
                    <Search className="absolute left-3 top-2.5 h-4 w-4 text-[color:var(--ns-fg-muted)]" />
                    <input
                        type="text"
                        placeholder="Search features or tokens..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded-md pl-9 pr-3 py-1.5 font-mono text-xs text-[color:var(--ns-fg-primary)] focus:outline-none focus:border-[color:var(--ns-accent)]"
                    />
                </div>

                <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-[color:var(--ns-fg-muted)] uppercase whitespace-nowrap">Layer</span>
                    <select
                        value={layerFilter}
                        onChange={(e) => setLayerFilter(e.target.value)}
                        className="w-full bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded-md px-2.5 py-1.5 font-mono text-xs text-[color:var(--ns-fg-primary)] focus:outline-none"
                    >
                        <option value="All">All Layers</option>
                        <option value="6">Layer 6 (Early)</option>
                        <option value="12">Layer 12 (Mid)</option>
                        <option value="18">Layer 18 (Late)</option>
                        <option value="24">Layer 24 (Output)</option>
                    </select>
                </div>

                <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-[color:var(--ns-fg-muted)] uppercase whitespace-nowrap">Category</span>
                    <select
                        value={categoryFilter}
                        onChange={(e) => setCategoryFilter(e.target.value)}
                        className="w-full bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded-md px-2.5 py-1.5 font-mono text-xs text-[color:var(--ns-fg-primary)] focus:outline-none"
                    >
                        <option value="All">All Categories</option>
                        <option value="Semantic">Semantic</option>
                        <option value="Syntactic">Syntactic</option>
                        <option value="Structural">Structural</option>
                        <option value="Error Detection">Error Detection</option>
                    </select>
                </div>

                <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-[color:var(--ns-fg-muted)] uppercase whitespace-nowrap">Poly</span>
                    <select
                        value={polyFilter}
                        onChange={(e) => setPolyFilter(e.target.value)}
                        className="w-full bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded-md px-2.5 py-1.5 font-mono text-xs text-[color:var(--ns-fg-primary)] focus:outline-none"
                    >
                        <option value="All">All Scores</option>
                        <option value="Monosemantic">Monosemantic (&lt;0.05)</option>
                        <option value="Moderate">Moderate (0.05-0.12)</option>
                        <option value="Polysemantic">Polysemantic (&gt;0.12)</option>
                    </select>
                </div>
            </div>

            {/* Features list */}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {filteredFeatures.map((feat) => (
                    <div
                        key={feat.id}
                        className="ns-card p-5 border border-[color:var(--ns-border-subtle)] flex flex-col justify-between"
                    >
                        <div>
                            <div className="flex items-center justify-between gap-2 mb-3">
                                <span className="font-mono text-xs font-semibold text-[color:var(--ns-accent)]">
                                    {feat.id}
                                </span>
                                <span className="px-2 py-0.5 rounded text-[8px] font-mono border uppercase bg-[color:var(--ns-bg-surface-2)] text-[color:var(--ns-fg-secondary)] border-[color:var(--ns-border)]">
                                    {feat.category}
                                </span>
                            </div>

                            <h3 className="font-mono text-sm font-semibold text-[color:var(--ns-fg-primary)]">
                                {feat.label}
                            </h3>
                            
                            <p className="text-xs text-[color:var(--ns-fg-secondary)] mt-2 leading-relaxed">
                                {feat.desc}
                            </p>

                            <div className="mt-4">
                                <div className="text-[9px] font-mono text-[color:var(--ns-fg-muted)] uppercase mb-1.5">Top Activating Tokens:</div>
                                <div className="flex flex-wrap gap-1.5">
                                    {feat.activeTokens.map((token, tIdx) => (
                                        <span
                                            key={tIdx}
                                            className="px-2 py-0.5 rounded font-mono text-[11px] bg-[color:var(--ns-bg-codeblock)] text-[color:var(--ns-amber)] border border-[color:var(--ns-border-subtle)]"
                                        >
                                            {token}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </div>

                        <div className="mt-5 pt-3 border-t border-[color:var(--ns-border-subtle)] flex items-center justify-between">
                            <button
                                onClick={() => triggerSteerSimulator(feat)}
                                className="inline-flex items-center gap-1 text-[11px] font-mono text-[color:var(--ns-mint)] hover:underline"
                            >
                                <Sliders size={11} /> Steer Feature
                            </button>
                            <a
                                href={feat.neuronpediaUrl}
                                target="_blank"
                                rel="noreferrer"
                                className="inline-flex items-center gap-0.5 text-[10px] font-mono text-[color:var(--ns-fg-muted)] hover:text-[color:var(--ns-fg-primary)]"
                            >
                                Neuronpedia <ArrowUpRight size={10} />
                            </a>
                        </div>
                    </div>
                ))}

                {filteredFeatures.length === 0 && (
                    <div className="col-span-full py-12 text-center ns-card border border-[color:var(--ns-border-subtle)]">
                        <p className="font-mono text-xs text-[color:var(--ns-fg-muted)]">
                            No features matches your search queries or filter selections.
                        </p>
                    </div>
                )}
            </div>

            {/* Steering Modal */}
            {steeredFeature && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm">
                    <div className="ns-card-strong w-full max-w-md border border-[color:var(--ns-border)] shadow-2xl p-6">
                        <div className="flex items-center justify-between border-b border-[color:var(--ns-border-subtle)] pb-3 mb-4">
                            <h3 className="font-mono text-sm font-semibold text-[color:var(--ns-fg-primary)]">
                                Steer Feature: <span className="text-[color:var(--ns-accent)]">{steeredFeature.id}</span>
                            </h3>
                            <button
                                onClick={() => setSteeredFeature(null)}
                                className="text-xs font-mono text-[color:var(--ns-fg-muted)] hover:text-[color:var(--ns-fg-primary)]"
                            >
                                Close
                            </button>
                        </div>

                        <div className="mb-4">
                            <div className="text-[10px] font-mono text-[color:var(--ns-fg-muted)] uppercase mb-1">Target concept:</div>
                            <div className="text-xs text-[color:var(--ns-fg-primary)] font-mono font-medium">{steeredFeature.label}</div>
                            <p className="text-[11px] text-[color:var(--ns-fg-secondary)] mt-1">{steeredFeature.desc}</p>
                        </div>

                        <div className="bg-[color:var(--ns-bg-codeblock)] rounded p-4 border border-[color:var(--ns-border-subtle)] mb-5">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-[10px] font-mono text-[color:var(--ns-fg-muted)]">STEERING AMPLIFICATION</span>
                                <span className="text-xs font-mono text-[color:var(--ns-mint)] font-bold">{steerFactor > 0 ? `+${steerFactor}` : steerFactor}x</span>
                            </div>
                            <input
                                type="range"
                                min="-10"
                                max="10"
                                step="0.5"
                                value={steerFactor}
                                onChange={(e) => setSteerFactor(parseFloat(e.target.value))}
                                className="w-full h-1 bg-[color:var(--ns-bg-surface-3)] rounded-lg appearance-none cursor-pointer accent-[color:var(--ns-mint)]"
                            />
                            <div className="flex justify-between text-[8px] font-mono text-[color:var(--ns-fg-muted)] uppercase mt-1">
                                <span>ablate</span>
                                <span>neutral</span>
                                <span>saturate</span>
                            </div>
                        </div>

                        <div className="mb-5">
                            <div className="text-[10px] font-mono text-[color:var(--ns-fg-muted)] uppercase mb-1.5">Simulation completions:</div>
                            <div className="space-y-2 text-xs font-mono">
                                <div className="p-2.5 bg-[color:var(--ns-bg-surface-3)] rounded border border-[color:var(--ns-border-subtle)]">
                                    <span className="text-[9px] text-[color:var(--ns-fg-muted)] uppercase block mb-1">Baseline:</span>
                                    <span className="text-[color:var(--ns-fg-secondary)]">"We need to execute the query verification loops inside the client module."</span>
                                </div>
                                <div className="p-2.5 bg-[color:var(--ns-bg-surface-3)] rounded border border-[color:var(--ns-accent-2)]">
                                    <span className="text-[9px] text-[color:var(--ns-mint)] uppercase block mb-1">Steered ({steerFactor > 0 ? `+${steerFactor}` : steerFactor}x):</span>
                                    <span className="text-[color:var(--ns-fg-primary)]">
                                        {steerFactor === 0
                                            ? `"We need to execute the query verification loops inside the client module."`
                                            : steerFactor > 0
                                            ? `"We need to execute the ${steeredFeature.activeTokens[0]} dictionaries to extract ${steeredFeature.activeTokens[1]} attributes inside the module."`
                                            : `"We run the loops inside the client module."`}
                                    </span>
                                </div>
                            </div>
                        </div>

                        <div className="flex justify-end gap-2">
                            <button
                                onClick={() => setSteeredFeature(null)}
                                className="px-3.5 py-1.5 text-xs font-mono rounded bg-[color:var(--ns-bg-surface-3)] border border-[color:var(--ns-border)] hover:border-[color:var(--ns-border-strong)] text-[color:var(--ns-fg-primary)]"
                            >
                                Done
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </section>
    );
}
