import React, { useState, useEffect } from "react";
import { Sparkles, Sliders, Info, Zap } from "lucide-react";

const PRESETS = [
    {
        name: "Sparse Autoencoders",
        text: "Sparse Autoencoders extract monosemantic feature directions inside model residual streams.",
        highlights: {
            "Autoencoders": { id: 8891, layer: 12, act: 3.42, poly: 0.08, label: "Autoencoders & representation learning", desc: "Fires heavily on ML terms, specifically self-supervised compression architectures." },
            "monosemantic": { id: 10454, layer: 12, act: 4.88, poly: 0.02, label: "Monosemantic feature lookup", desc: "Tracks clean single-concept activations, isolating individual token interpretations." },
            "residual": { id: 12044, layer: 12, act: 2.15, poly: 0.15, label: "Residual stream operations", desc: "Fires on discussions regarding transformer additive streams and identity paths." },
            "feature": { id: 5521, layer: 12, act: 1.89, poly: 0.22, label: "Feature directions & activations", desc: "Activates for features, activations, and weights inside neural networks." }
        }
    },
    {
        name: "Causal Patching",
        text: "The causal patching intervention isolates the computational paths of factual retrieval.",
        highlights: {
            "causal": { id: 1410, layer: 12, act: 2.95, poly: 0.11, label: "Causal attribution & mechanisms", desc: "Fires on causal dependencies, patching, and structural interventions." },
            "patching": { id: 3912, layer: 12, act: 3.81, poly: 0.05, label: "Activation patching / hook interventions", desc: "Activates on activation replacement, swapping, and ablation." },
            "retrieval": { id: 7122, layer: 12, act: 2.44, poly: 0.19, label: "Factual recall / database query", desc: "Fires during multi-hop lookup and fact retrieval steps." }
        }
    },
    {
        name: "Agentic Drift",
        text: "Our multi-turn agent exhibits drift in the residual stream when generating incorrect answers.",
        highlights: {
            "multi-turn": { id: 1102, layer: 12, act: 3.21, poly: 0.12, label: "Dialogue state persistence", desc: "Tracks conversation context retention across multiple execution cycles." },
            "drift": { id: 8203, layer: 12, act: 4.12, poly: 0.06, label: "Representation drift / target decay", desc: "Fires when residual vectors drift away from the base task query representation." },
            "incorrect": { id: 4099, layer: 12, act: 3.65, poly: 0.09, label: "Incorrect reasoning & error states", desc: "Fires when model internal states begin generating hallucinatory tokens." }
        }
    }
];

export default function LiveDemoWidget() {
    const [currentPresetIdx, setCurrentPresetIdx] = useState(0);
    const [inputText, setInputText] = useState(PRESETS[0].text);
    const [tokens, setTokens] = useState([]);
    const [selectedToken, setSelectedToken] = useState(null);
    const [steeringAlpha, setSteeringAlpha] = useState(1.0);
    const [steeredOutput, setSteeredOutput] = useState("");

    // Process input text to generate simulated tokens and activations
    useEffect(() => {
        const words = inputText.split(/(\s+)/);
        const mappedTokens = words.map((word, idx) => {
            const cleanWord = word.trim().replace(/[.,/#!$%^&*;:{}=\-_`~()]/g, "");
            
            // Check current preset highlights
            let highlight = null;
            if (inputText === PRESETS[currentPresetIdx]?.text) {
                highlight = PRESETS[currentPresetIdx].highlights[cleanWord] || null;
            } else {
                // Generative mock feature if custom text is inputted
                if (cleanWord.length > 4 && (idx % 4 === 0)) {
                    const fid = Math.floor(Math.sin(idx) * 8000) + 8000;
                    highlight = {
                        id: fid,
                        layer: 12,
                        act: parseFloat((Math.abs(Math.sin(idx)) * 4 + 0.5).toFixed(2)),
                        poly: parseFloat((Math.abs(Math.cos(idx)) * 0.4).toFixed(2)),
                        label: `Custom concept feature for "${cleanWord}"`,
                        desc: `Fires on words resembling "${cleanWord}" inside custom prompt contexts.`
                    };
                }
            }

            return {
                raw: word,
                clean: cleanWord,
                highlight,
                index: idx
            };
        });

        setTokens(mappedTokens);

        // Auto-select first active token if available
        const firstActive = mappedTokens.find(t => t.highlight);
        if (firstActive) {
            setSelectedToken(firstActive);
        } else {
            setSelectedToken(null);
        }
    }, [inputText, currentPresetIdx]);

    // Update steered output simulation
    useEffect(() => {
        if (!selectedToken || !selectedToken.highlight) {
            setSteeredOutput("");
            return;
        }

        const feat = selectedToken.highlight;
        const alphaStr = steeringAlpha > 0 ? `+${steeringAlpha}` : steeringAlpha;
        
        if (steeringAlpha === 0) {
            setSteeredOutput("No steering active. Generating standard model output...");
        } else if (steeringAlpha > 2) {
            setSteeredOutput(`[Steering Active: Feature ${feat.id} amplified by ${alphaStr}x]\nGenerating context heavily biased towards: ${feat.label}.\nOutput: "Indeed, using sparse autoencoders, we isolate these exact ${selectedToken.clean.toLowerCase()} pathways directly inside the residual stream block."`);
        } else if (steeringAlpha < -2) {
            setSteeredOutput(`[Steering Active: Feature ${feat.id} suppressed by ${alphaStr}x]\nSuppressed target concept: ${selectedToken.clean.toLowerCase()}.\nOutput: "We run the basic mathematical linear algebra transformations inside the hidden layer matrices..."`);
        } else {
            setSteeredOutput(`[Steering Active: Feature ${feat.id} minor steering ${alphaStr}x]\nOutput: "We utilize standard ${selectedToken.clean.toLowerCase()} features for parsing the transformer outputs."`);
        }
    }, [selectedToken, steeringAlpha]);

    const handlePresetClick = (idx) => {
        setCurrentPresetIdx(idx);
        setInputText(PRESETS[idx].text);
    };

    return (
        <div className="w-full grid gap-6 lg:grid-cols-[1.5fr_1fr]" data-testid="live-demo-panel">
            <div className="ns-card p-5 flex flex-col justify-between">
                <div>
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <span className="h-2 w-2 rounded-full bg-[color:var(--ns-accent)]" />
                            <span className="font-mono text-xs uppercase tracking-wider text-[color:var(--ns-fg-primary)]">Interactive Activation Sandbox</span>
                        </div>
                        <div className="flex gap-2">
                            {PRESETS.map((preset, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => handlePresetClick(idx)}
                                    className={`px-2.5 py-1 rounded text-[10px] font-mono border transition-all ${
                                        currentPresetIdx === idx
                                            ? "border-[color:var(--ns-accent)] text-[color:var(--ns-accent)] bg-[color:var(--ns-bg-surface-2)]"
                                            : "border-[color:var(--ns-border)] text-[color:var(--ns-fg-muted)] hover:text-[color:var(--ns-fg-primary)]"
                                    }`}
                                >
                                    {preset.name}
                                </button>
                            ))}
                        </div>
                    </div>

                    <textarea
                        value={inputText}
                        onChange={(e) => {
                            setCurrentPresetIdx(-1);
                            setInputText(e.target.value);
                        }}
                        placeholder="Paste a custom sentence here to parse features..."
                        className="w-full h-20 bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded-lg p-3 text-[13px] text-[color:var(--ns-fg-primary)] font-mono resize-none focus:outline-none focus:border-[color:var(--ns-accent)]"
                    />

                    <div className="mt-5">
                        <div className="text-[10px] font-mono text-[color:var(--ns-fg-muted)] uppercase mb-2">Token activation heatmap (click tokens to inspect):</div>
                        <div className="flex flex-wrap items-center gap-x-1 gap-y-2 p-4 bg-[color:var(--ns-bg-surface-2)] rounded-lg border border-[color:var(--ns-border-subtle)] min-h-[70px]">
                            {tokens.map((token, idx) => {
                                if (token.raw.match(/^\s+$/)) {
                                    return <span key={idx} className="w-1" />;
                                }
                                
                                const isActive = token.highlight;
                                const isSelected = selectedToken && selectedToken.index === token.index;
                                
                                // Heatmap background color styling
                                let bgStyle = {};
                                let textClass = "text-[color:var(--ns-fg-secondary)]";
                                if (isActive) {
                                    const intensity = Math.min(1, token.highlight.act / 5);
                                    bgStyle = {
                                        background: `rgba(242, 193, 78, ${0.1 + intensity * 0.65})`,
                                        borderColor: isSelected ? "var(--ns-accent)" : "rgba(242, 193, 78, 0.4)"
                                    };
                                    textClass = "text-[color:var(--ns-fg-primary)] font-medium";
                                } else {
                                    bgStyle = {
                                        background: "transparent",
                                        borderColor: isSelected ? "var(--ns-border-strong)" : "transparent"
                                    };
                                }

                                return (
                                    <button
                                        key={idx}
                                        onClick={() => setSelectedToken(token)}
                                        style={bgStyle}
                                        className={`px-1.5 py-0.5 rounded border font-mono text-[12px] transition-all cursor-pointer select-none ${textClass} ${
                                            isSelected ? "ring-1 ring-[color:var(--ns-accent)]" : ""
                                        }`}
                                    >
                                        {token.raw}
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                </div>

                <div className="flex items-center justify-between text-[11px] font-mono text-[color:var(--ns-fg-muted)] mt-4 pt-3 border-t border-[color:var(--ns-border-subtle)]">
                    <span className="flex items-center gap-1">
                        <Sparkles size={11} className="text-[color:var(--ns-amber)]" />
                        Amber highlights = active SAE features
                    </span>
                    <span>No subscription or model download required</span>
                </div>
            </div>

            {/* Token details panel */}
            <div className="ns-card-strong border border-[color:var(--ns-border)] p-5 flex flex-col justify-between">
                {selectedToken && selectedToken.highlight ? (
                    <div className="flex flex-col h-full justify-between">
                        <div>
                            <div className="flex justify-between items-start mb-4">
                                <div>
                                    <span className="text-[10px] font-mono px-2 py-0.5 bg-[color:var(--ns-bg-surface-3)] border border-[color:var(--ns-border-subtle)] rounded text-[color:var(--ns-fg-secondary)]">
                                        Token: "{selectedToken.clean}"
                                    </span>
                                    <h3 className="text-[15px] font-mono text-[color:var(--ns-accent)] mt-2 font-semibold">
                                        Feature L12_F{selectedToken.highlight.id}
                                    </h3>
                                </div>
                                <span className="font-mono text-[11px] text-[color:var(--ns-fg-muted)]">
                                    Act: <strong className="text-[color:var(--ns-amber)]">{selectedToken.highlight.act}</strong>
                                </span>
                            </div>

                            <div className="bg-[color:var(--ns-bg-codeblock)] rounded p-3 text-xs mb-4 border border-[color:var(--ns-border-subtle)]">
                                <div className="font-mono font-semibold text-[color:var(--ns-fg-primary)] mb-1">
                                    {selectedToken.highlight.label}
                                </div>
                                <div className="text-[color:var(--ns-fg-secondary)] leading-relaxed">
                                    {selectedToken.highlight.desc}
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-2 mb-4 font-mono text-[10px]">
                                <div className="p-2 bg-[color:var(--ns-bg-surface-3)] rounded border border-[color:var(--ns-border-subtle)]">
                                    <div className="text-[color:var(--ns-fg-muted)]">POLYSEMANTICITY</div>
                                    <div className="text-sm font-semibold text-[color:var(--ns-mint)] mt-0.5">
                                        {selectedToken.highlight.poly} <span className="text-[8px] text-[color:var(--ns-fg-muted)]">(low)</span>
                                    </div>
                                </div>
                                <div className="p-2 bg-[color:var(--ns-bg-surface-3)] rounded border border-[color:var(--ns-border-subtle)]">
                                    <div className="text-[color:var(--ns-fg-muted)]">ACTIVATION DENSITY</div>
                                    <div className="text-sm font-semibold text-[color:var(--ns-fg-primary)] mt-0.5">
                                        0.002%
                                    </div>
                                </div>
                            </div>

                            {/* Steering Slider */}
                            <div className="border-t border-[color:var(--ns-border-subtle)] pt-4">
                                <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-1.5 font-mono text-[11px] text-[color:var(--ns-fg-primary)] uppercase font-semibold">
                                        <Sliders size={12} className="text-[color:var(--ns-accent)]" />
                                        Interactive Steering Slider
                                    </div>
                                    <span className="font-mono text-[11px] text-[color:var(--ns-accent)] font-semibold">
                                        {steeringAlpha > 0 ? `+${steeringAlpha}` : steeringAlpha}x
                                    </span>
                                </div>
                                <input
                                    type="range"
                                    min="-10"
                                    max="10"
                                    step="0.5"
                                    value={steeringAlpha}
                                    onChange={(e) => setSteeringAlpha(parseFloat(e.target.value))}
                                    className="w-full h-1 bg-[color:var(--ns-bg-surface-3)] rounded-lg appearance-none cursor-pointer accent-[color:var(--ns-accent)]"
                                />
                                <div className="flex justify-between text-[8px] font-mono text-[color:var(--ns-fg-muted)] uppercase mt-1">
                                    <span>suppress concept</span>
                                    <span>neutral</span>
                                    <span>amplify concept</span>
                                </div>
                            </div>
                        </div>

                        <div className="mt-4">
                            <div className="text-[9px] font-mono text-[color:var(--ns-fg-muted)] uppercase mb-1">Steered Model Output Preview:</div>
                            <pre className="text-[11px] bg-[color:var(--ns-bg-codeblock)] border border-[color:var(--ns-border-subtle)] rounded p-2.5 font-mono text-[color:var(--ns-fg-secondary)] whitespace-pre-wrap min-h-[85px] leading-relaxed">
                                {steeredOutput}
                            </pre>
                        </div>
                    </div>
                ) : (
                    <div className="h-full flex flex-col items-center justify-center text-center py-10">
                        <Info size={24} className="text-[color:var(--ns-fg-muted)] mb-2" />
                        <p className="font-mono text-xs text-[color:var(--ns-fg-muted)]">
                            Select a highlighted token to inspect activations and trigger steering.
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}
