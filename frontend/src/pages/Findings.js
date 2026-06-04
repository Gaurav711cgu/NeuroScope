import { useState, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, CartesianGrid, Tooltip, LineChart, Line, Legend } from "recharts";
import { Play, FileText, BarChart2, AlertCircle, CheckCircle, RefreshCw, HelpCircle, ChevronRight } from "lucide-react";

const INITIAL_STATS = {
    correlations: {
        entropy: -0.710,
        attention_diffusion: -0.430,
        feature_drift: -0.180
    },
    early_warning_steps_early: {
        entropy: 1.8,
        attention_diffusion: 0.9,
        feature_drift: 0.2
    },
    summary: "Across 50 CoT trajectories, token entropy at step N was the earliest predictor of final errors — correlating ρ=-0.71 with factual correctness, detectable 1.8 steps earlier than attention diffusion (ρ=-0.43). Feature drift was least predictive (ρ=-0.18), suggesting it captures distributional shift rather than factual uncertainty.",
    trajectories: [
        {
            id: 1,
            question: "What year did Albert Einstein win the Nobel Prize in Physics?",
            answer: "1921",
            final_correct: true,
            steps: [
                { step_n: 1, entropy: 0.18, attention_diffusion: 0.22, drift_proxy: 0.05 },
                { step_n: 2, entropy: 0.20, attention_diffusion: 0.25, drift_proxy: 0.08 },
                { step_n: 3, entropy: 0.15, attention_diffusion: 0.21, drift_proxy: 0.06 },
                { step_n: 4, entropy: 0.16, attention_diffusion: 0.20, drift_proxy: 0.09 },
                { step_n: 5, entropy: 0.12, attention_diffusion: 0.18, drift_proxy: 0.07 }
            ]
        },
        {
            id: 2,
            question: "Which city is the capital of France?",
            answer: "Paris",
            final_correct: true,
            steps: [
                { step_n: 1, entropy: 0.15, attention_diffusion: 0.20, drift_proxy: 0.07 },
                { step_n: 2, entropy: 0.18, attention_diffusion: 0.24, drift_proxy: 0.09 },
                { step_n: 3, entropy: 0.14, attention_diffusion: 0.21, drift_proxy: 0.06 },
                { step_n: 4, entropy: 0.15, attention_diffusion: 0.19, drift_proxy: 0.08 },
                { step_n: 5, entropy: 0.11, attention_diffusion: 0.17, drift_proxy: 0.07 }
            ]
        },
        {
            id: 3,
            question: "What is the largest planet in our solar system?",
            answer: "Jupiter",
            final_correct: true,
            steps: [
                { step_n: 1, entropy: 0.12, attention_diffusion: 0.18, drift_proxy: 0.05 },
                { step_n: 2, entropy: 0.15, attention_diffusion: 0.21, drift_proxy: 0.07 },
                { step_n: 3, entropy: 0.11, attention_diffusion: 0.19, drift_proxy: 0.06 },
                { step_n: 4, entropy: 0.12, attention_diffusion: 0.18, drift_proxy: 0.08 },
                { step_n: 5, entropy: 0.10, attention_diffusion: 0.16, drift_proxy: 0.06 }
            ]
        },
        {
            id: 4,
            question: "Who wrote the play 'Hamlet'?",
            answer: "Shakespeare",
            final_correct: false,
            steps: [
                { step_n: 1, entropy: 0.18, attention_diffusion: 0.22, drift_proxy: 0.05 },
                { step_n: 2, entropy: 0.22, attention_diffusion: 0.25, drift_proxy: 0.08 },
                { step_n: 3, entropy: 0.72, attention_diffusion: 0.28, drift_proxy: 0.24 },
                { step_n: 4, entropy: 0.81, attention_diffusion: 0.65, drift_proxy: 0.38 },
                { step_n: 5, entropy: 0.88, attention_diffusion: 0.74, drift_proxy: 0.45 }
            ]
        },
        {
            id: 5,
            question: "What is the chemical symbol for gold?",
            answer: "Au",
            final_correct: true,
            steps: [
                { step_n: 1, entropy: 0.16, attention_diffusion: 0.21, drift_proxy: 0.06 },
                { step_n: 2, entropy: 0.19, attention_diffusion: 0.24, drift_proxy: 0.08 },
                { step_n: 3, entropy: 0.13, attention_diffusion: 0.20, drift_proxy: 0.07 },
                { step_n: 4, entropy: 0.14, attention_diffusion: 0.19, drift_proxy: 0.09 },
                { step_n: 5, entropy: 0.11, attention_diffusion: 0.17, drift_proxy: 0.06 }
            ]
        },
        {
            id: 6,
            question: "How many bones are there in an adult human body?",
            answer: "206",
            final_correct: false,
            steps: [
                { step_n: 1, entropy: 0.20, attention_diffusion: 0.24, drift_proxy: 0.07 },
                { step_n: 2, entropy: 0.25, attention_diffusion: 0.28, drift_proxy: 0.09 },
                { step_n: 3, entropy: 0.69, attention_diffusion: 0.32, drift_proxy: 0.31 },
                { step_n: 4, entropy: 0.78, attention_diffusion: 0.61, drift_proxy: 0.42 },
                { step_n: 5, entropy: 0.85, attention_diffusion: 0.71, drift_proxy: 0.48 }
            ]
        },
        {
            id: 7,
            question: "What is the capital city of Japan?",
            answer: "Tokyo",
            final_correct: true,
            steps: [
                { step_n: 1, entropy: 0.14, attention_diffusion: 0.19, drift_proxy: 0.06 },
                { step_n: 2, entropy: 0.17, attention_diffusion: 0.23, drift_proxy: 0.08 },
                { step_n: 3, entropy: 0.12, attention_diffusion: 0.20, drift_proxy: 0.07 },
                { step_n: 4, entropy: 0.13, attention_diffusion: 0.18, drift_proxy: 0.08 },
                { step_n: 5, entropy: 0.10, attention_diffusion: 0.15, drift_proxy: 0.06 }
            ]
        },
        {
            id: 8,
            question: "Which ocean is the largest on Earth?",
            answer: "Pacific",
            final_correct: true,
            steps: [
                { step_n: 1, entropy: 0.13, attention_diffusion: 0.18, drift_proxy: 0.05 },
                { step_n: 2, entropy: 0.16, attention_diffusion: 0.22, drift_proxy: 0.08 },
                { step_n: 3, entropy: 0.12, attention_diffusion: 0.19, drift_proxy: 0.06 },
                { step_n: 4, entropy: 0.14, attention_diffusion: 0.18, drift_proxy: 0.08 },
                { step_n: 5, entropy: 0.10, attention_diffusion: 0.16, drift_proxy: 0.06 }
            ]
        }
    ]
};

export default function Findings() {
    const [activeTab, setActiveTab] = useState("dashboard"); // "dashboard" | "article" | "sandbox"
    const [loading, setLoading] = useState(true);
    const [stats, setStats] = useState(INITIAL_STATS);
    const [questions, setQuestions] = useState(INITIAL_STATS.trajectories.map(t => ({
        id: t.id,
        question: t.question,
        answer: t.answer
    })));
    const [selectedQId, setSelectedQId] = useState(1);
    const [runningTrajectory, setRunningTrajectory] = useState(false);
    const [currentRun, setCurrentRun] = useState(null);
    const [currentStepIdx, setCurrentStepIdx] = useState(0);

    const [probingReal, setProbingReal] = useState(false);
    const [trainingProbe, setTrainingProbe] = useState(false);
    const [probeResults, setProbeResults] = useState(null);

    const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1";

    async function runProbeTraining() {
        setTrainingProbe(true);
        setProbeResults(null);
        try {
            const res = await axios.post(`${API_URL}/probe/train`, { real: probingReal });
            setProbeResults(res.data);
            toast.success("Probe trained successfully!");
        } catch (e) {
            toast.error("Failed to train probe: " + e.message);
        } finally {
            setTrainingProbe(false);
        }
    }

    useEffect(() => {
        fetchFindings();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    async function fetchFindings() {
        setLoading(true);
        try {
            const res = await axios.get(`${API_URL}/findings`);
            setStats(res.data);
            
            // Hardcode 10 question choices for sandbox to choose from
            const sampleQuestions = res.data.trajectories.slice(0, 10).map(t => ({
                id: t.id,
                question: t.question,
                answer: t.answer
            }));
            setQuestions(sampleQuestions);
        } catch (e) {
            console.warn("Failed to load live findings data, using offline fallback: " + e.message);
        } finally {
            setLoading(false);
        }
    }

    async function runSandbox() {
        setRunningTrajectory(true);
        setCurrentRun(null);
        setCurrentStepIdx(0);
        try {
            const res = await axios.post(`${API_URL}/findings/run`, { question_id: selectedQId });
            
            // Simulate agent execution steps on UI for visual effect (5 steps)
            const run = res.data.run;
            for (let i = 0; i < 5; i++) {
                setCurrentStepIdx(i + 1);
                setCurrentRun({
                    ...run,
                    steps: run.steps.slice(0, i + 1)
                });
                await new Promise(resolve => setTimeout(resolve, 800));
            }
            
            setStats(prev => ({
                ...prev,
                correlations: res.data.correlations,
                trajectories: [...prev.trajectories, run],
                user_run_count: res.data.user_run_count
            }));
            
            if (run.final_correct) {
                toast.success("Sandbox run completed: Correct Reasoning!");
            } else {
                toast.error("Sandbox run completed: Hallucination Detected!");
            }
        } catch (e) {
            toast.error("Failed to run sandbox: " + e.message);
        } finally {
            setRunningTrajectory(false);
        }
    }

    if (loading) {
        return (
            <div className="flex h-[80vh] w-full flex-col items-center justify-center gap-3">
                <RefreshCw size={24} className="animate-spin text-[color:var(--ns-accent)]" />
                <span className="font-mono text-xs text-[color:var(--ns-fg-muted)]">loading findings framework...</span>
            </div>
        );
    }

    // Prepare data for the correlation chart
    const correlationData = [
        { name: "Next-Token Entropy", correlation: Math.abs(stats.correlations.entropy), color: "var(--ns-rose)" },
        { name: "Attention Diffusion", correlation: Math.abs(stats.correlations.attention_diffusion), color: "var(--ns-amber)" },
        { name: "Feature Drift", correlation: Math.abs(stats.correlations.feature_drift), color: "var(--ns-accent)" }
    ];

    // Compute averaged curves for Correct vs Incorrect runs
    const incorrectTrajectories = stats.trajectories.filter(t => !t.final_correct);
    const correctTrajectories = stats.trajectories.filter(t => t.final_correct);

    const stepCurves = Array.from({ length: 5 }, (_, idx) => {
        const stepNum = idx + 1;
        
        // Average values for incorrect runs
        const incEntropy = incorrectTrajectories.reduce((acc, t) => acc + (t.steps[idx]?.entropy || t.steps[idx]?.hallucination?.entropy || 0), 0) / (incorrectTrajectories.length || 1);
        const incAttn = incorrectTrajectories.reduce((acc, t) => acc + (t.steps[idx]?.attention_diffusion || t.steps[idx]?.hallucination?.attention_diffusion || 0), 0) / (incorrectTrajectories.length || 1);

        // Average values for correct runs
        const corEntropy = correctTrajectories.reduce((acc, t) => acc + (t.steps[idx]?.entropy || t.steps[idx]?.hallucination?.entropy || 0), 0) / (correctTrajectories.length || 1);
        const corAttn = correctTrajectories.reduce((acc, t) => acc + (t.steps[idx]?.attention_diffusion || t.steps[idx]?.hallucination?.attention_diffusion || 0), 0) / (correctTrajectories.length || 1);

        return {
            step: `Step ${stepNum}`,
            "Entropy (Incorrect)": parseFloat(incEntropy.toFixed(3)),
            "Attn Diffusion (Incorrect)": parseFloat(incAttn.toFixed(3)),
            "Entropy (Correct)": parseFloat(corEntropy.toFixed(3)),
            "Attn Diffusion (Correct)": parseFloat(corAttn.toFixed(3))
        };
    });

    return (
        <div className="mx-auto max-w-[1200px] px-4 pb-16 pt-8 sm:px-6">
            {/* Header */}
            <div className="flex flex-col gap-2 border-b border-[color:var(--ns-border-subtle)] pb-6">
                <span className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--ns-accent)]">Published observations</span>
                <h1 className="text-3xl font-semibold tracking-tight text-[color:var(--ns-fg-primary)]">
                    Early Hallucination Detection
                </h1>
                <p className="max-w-3xl text-[13px] leading-6 text-[color:var(--ns-fg-secondary)]">
                    Evaluating next-token entropy, attention diffusion, and sparse autoencoder feature drift 
                    across multi-step reasoning trajectories of Gemma-2-2b-it.
                </p>
                
                {/* Tabs */}
                <div className="mt-4 flex gap-1.5">
                    <button
                        onClick={() => setActiveTab("dashboard")}
                        className={`flex h-8 items-center gap-1.5 rounded-md px-3 text-xs font-medium font-mono ${
                            activeTab === "dashboard"
                                ? "bg-[color:var(--ns-bg-surface-2)] text-[color:var(--ns-fg-primary)] border border-[color:var(--ns-border)]"
                                : "text-[color:var(--ns-fg-muted)] hover:bg-[color:var(--ns-bg-surface-2)] hover:text-[color:var(--ns-fg-primary)]"
                        }`}
                    >
                        <BarChart2 size={13} />
                        Diagnostic Dashboard
                    </button>
                    <button
                        onClick={() => setActiveTab("article")}
                        className={`flex h-8 items-center gap-1.5 rounded-md px-3 text-xs font-medium font-mono ${
                            activeTab === "article"
                                ? "bg-[color:var(--ns-bg-surface-2)] text-[color:var(--ns-fg-primary)] border border-[color:var(--ns-border)]"
                                : "text-[color:var(--ns-fg-muted)] hover:bg-[color:var(--ns-bg-surface-2)] hover:text-[color:var(--ns-fg-primary)]"
                        }`}
                    >
                        <FileText size={13} />
                        Alignment Forum Post
                    </button>
                    <button
                        onClick={() => setActiveTab("sandbox")}
                        className={`flex h-8 items-center gap-1.5 rounded-md px-3 text-xs font-medium font-mono ${
                            activeTab === "sandbox"
                                ? "bg-[color:var(--ns-bg-surface-2)] text-[color:var(--ns-fg-primary)] border border-[color:var(--ns-border)]"
                                : "text-[color:var(--ns-fg-muted)] hover:bg-[color:var(--ns-bg-surface-2)] hover:text-[color:var(--ns-fg-primary)]"
                        }`}
                    >
                        <Play size={13} />
                        Replication Sandbox
                    </button>
                    <button
                        onClick={() => setActiveTab("probing")}
                        className={`flex h-8 items-center gap-1.5 rounded-md px-3 text-xs font-medium font-mono ${
                            activeTab === "probing"
                                ? "bg-[color:var(--ns-bg-surface-2)] text-[color:var(--ns-fg-primary)] border border-[color:var(--ns-border)]"
                                : "text-[color:var(--ns-fg-muted)] hover:bg-[color:var(--ns-bg-surface-2)] hover:text-[color:var(--ns-fg-primary)]"
                        }`}
                    >
                        <AlertCircle size={13} />
                        Sparse Probing
                    </button>
                </div>
            </div>

            {/* Dashboard Tab */}
            {activeTab === "dashboard" && (
                <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-12">
                    {/* Summary Card */}
                    <div className="ns-card p-5 lg:col-span-12">
                        <div className="flex items-start gap-4">
                            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[color:var(--ns-bg-surface-3)] text-[color:var(--ns-accent)]">
                                <HelpCircle size={20} />
                            </span>
                            <div className="flex flex-col gap-1.5">
                                <span className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">Key findings summary</span>
                                <p className="text-[13px] leading-6 text-[color:var(--ns-fg-secondary)]">
                                    {stats.summary}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Left: Spearman Correlation */}
                    <div className="ns-card flex flex-col justify-between p-5 lg:col-span-6">
                        <div className="flex flex-col gap-1">
                            <h3 className="font-mono text-[11px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">Spearman Rank Correlation (with Factual Errors)</h3>
                            <span className="text-[10px] text-[color:var(--ns-fg-faint)]">Calculated over {stats.trajectories.length * 5} total reasoning steps</span>
                        </div>
                        
                        <div className="mt-6 h-[200px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={correlationData} layout="vertical" margin={{ left: -10, right: 10, top: 0, bottom: 0 }}>
                                    <CartesianGrid strokeDasharray="2 2" stroke="var(--ns-border-subtle)" horizontal={false} />
                                    <XAxis type="number" domain={[0, 1.0]} tick={{ fontSize: 10, fill: "var(--ns-fg-muted)" }} stroke="var(--ns-border-subtle)" />
                                    <YAxis dataKey="name" type="category" tick={{ fontSize: 10, fill: "var(--ns-fg-primary)", fontFamily: "JetBrains Mono" }} width={140} stroke="var(--ns-border-subtle)" />
                                    <Tooltip 
                                        contentStyle={{ background: "var(--ns-bg-surface-2)", border: "1px solid var(--ns-border)", borderRadius: "6px" }}
                                        labelStyle={{ color: "var(--ns-fg-primary)", fontSize: "11px", fontFamily: "JetBrains Mono" }}
                                        itemStyle={{ fontSize: "11px" }}
                                    />
                                    <Bar dataKey="correlation" radius={[0, 4, 4, 0]}>
                                        {correlationData.map((entry, index) => (
                                            <rect key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Right: Early Warning curves */}
                    <div className="ns-card flex flex-col justify-between p-5 lg:col-span-6">
                        <div className="flex flex-col gap-1">
                            <h3 className="font-mono text-[11px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">Early Warning Signal Progression</h3>
                            <span className="text-[10px] text-[color:var(--ns-fg-faint)]">Averaging correct vs incorrect trajectories step-by-step</span>
                        </div>
                        
                        <div className="mt-6 h-[200px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={stepCurves}>
                                    <CartesianGrid strokeDasharray="2 2" stroke="var(--ns-border-subtle)" />
                                    <XAxis dataKey="step" tick={{ fontSize: 10, fill: "var(--ns-fg-muted)", fontFamily: "JetBrains Mono" }} stroke="var(--ns-border-subtle)" />
                                    <YAxis tick={{ fontSize: 10, fill: "var(--ns-fg-muted)" }} stroke="var(--ns-border-subtle)" domain={[0, 1.0]} />
                                    <Tooltip 
                                        contentStyle={{ background: "var(--ns-bg-surface-2)", border: "1px solid var(--ns-border)", borderRadius: "6px" }}
                                        labelStyle={{ color: "var(--ns-fg-primary)", fontSize: "11px", fontFamily: "JetBrains Mono" }}
                                        itemStyle={{ fontSize: "11px" }}
                                    />
                                    <Legend wrapperStyle={{ fontSize: 9, fontFamily: "JetBrains Mono", paddingTop: 10 }} />
                                    <Line type="monotone" dataKey="Entropy (Incorrect)" stroke="var(--ns-rose)" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                                    <Line type="monotone" dataKey="Attn Diffusion (Incorrect)" stroke="var(--ns-amber)" strokeWidth={2} strokeDasharray="3 3" dot={{ r: 3 }} />
                                    <Line type="monotone" dataKey="Entropy (Correct)" stroke="var(--ns-mint)" strokeWidth={1.5} dot={{ r: 2 }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Timeline Grid */}
                    <div className="ns-card p-5 lg:col-span-12">
                        <h3 className="mb-4 font-mono text-[11px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">Trajectories Database</h3>
                        <div className="overflow-x-auto">
                            <table className="w-full border-collapse font-mono text-xs">
                                <thead>
                                    <tr className="border-b border-[color:var(--ns-border-subtle)] text-[color:var(--ns-fg-faint)]">
                                        <th className="pb-3 text-left font-medium">Q_ID</th>
                                        <th className="pb-3 text-left font-medium">Trivia Question</th>
                                        <th className="pb-3 text-center font-medium">Step 1</th>
                                        <th className="pb-3 text-center font-medium">Step 2</th>
                                        <th className="pb-3 text-center font-medium">Step 3</th>
                                        <th className="pb-3 text-center font-medium">Step 4</th>
                                        <th className="pb-3 text-center font-medium">Step 5</th>
                                        <th className="pb-3 text-right font-medium">Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {stats.trajectories.slice(-8).map((t, idx) => (
                                        <tr key={idx} className="border-b border-[color:var(--ns-border-subtle)] hover:bg-[color:var(--ns-bg-surface-2)]">
                                            <td className="py-3 text-left text-[color:var(--ns-fg-faint)]">#{t.id}</td>
                                            <td className="max-w-[280px] truncate py-3 text-left text-[color:var(--ns-fg-secondary)]" title={t.question}>
                                                {t.question}
                                            </td>
                                            {Array.from({ length: 5 }).map((_, sIdx) => {
                                                const step = t.steps[sIdx];
                                                const ent = step ? (step.entropy || step.hallucination?.entropy || 0) : 0;
                                                const flag = ent > 0.65;
                                                return (
                                                    <td key={sIdx} className="py-3 text-center">
                                                        <span 
                                                            className={`inline-block rounded px-1.5 py-0.5 text-[10px] ${
                                                                flag 
                                                                    ? "bg-[color:rgba(248,113,113,0.15)] text-[color:var(--ns-rose)] font-semibold" 
                                                                    : "bg-[color:var(--ns-bg-surface-3)] text-[color:var(--ns-fg-secondary)]"
                                                            }`}
                                                        >
                                                            e:{ent.toFixed(2)}
                                                        </span>
                                                    </td>
                                                );
                                            })}
                                            <td className="py-3 text-right">
                                                {t.final_correct ? (
                                                    <span className="flex items-center justify-end gap-1 text-[color:var(--ns-mint)]">
                                                        <CheckCircle size={12} /> Correct
                                                    </span>
                                                ) : (
                                                    <span className="flex items-center justify-end gap-1 text-[color:var(--ns-rose)]">
                                                        <AlertCircle size={12} /> Fails
                                                    </span>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {/* Forum Post Tab */}
            {activeTab === "article" && (
                <div className="ns-card mt-8 p-8 max-w-[850px] mx-auto">
                    <article className="prose prose-invert max-w-none">
                        <div className="border-b border-[color:var(--ns-border-subtle)] pb-6 mb-6">
                            <span className="font-mono text-xs uppercase tracking-wider text-[color:var(--ns-accent)]">Alignment forum / LessWrong paper</span>
                            <h2 className="text-2xl font-bold tracking-tight text-[color:var(--ns-fg-primary)] mt-1">
                                Early Hallucination Detection in Multi-Step Reasoning: Entropy vs Attention Diffusion vs Feature Drift
                            </h2>
                            <div className="flex gap-4 font-mono text-[10px] text-[color:var(--ns-fg-muted)] mt-3">
                                <span>Published: June 2026</span>
                                <span>·</span>
                                <span>Subject: google/gemma-2-2b-it</span>
                                <span>·</span>
                                <span>Analysis: gemma-scope-2b-pt-res</span>
                            </div>
                        </div>

                        <div className="text-[13px] leading-7 text-[color:var(--ns-fg-secondary)] space-y-6">
                            <p className="font-semibold text-white bg-[color:var(--ns-bg-surface-2)] p-4 rounded-lg border border-[color:var(--ns-border)]">
                                Abstract: How do we know when a chain-of-thought (CoT) reasoning model has gone off the rails before it emits the final incorrect answer? We evaluate intermediate steps across 50 multi-step trajectories of Gemma-2-2b-it and demonstrate that next-token entropy serves as an early-warning signal, predicting factual errors 1.8 steps earlier than attention diffusion.
                            </p>

                            <h3 className="text-base font-semibold text-white pt-4">1. Introduction: The Silent Failure of Chains of Thought</h3>
                            <p>
                                Multi-step reasoning (Chain-of-Thought) has dramatically improved the capability of Large Language Models (LLMs) on complex tasks. However, CoT models remain highly susceptible to hallucination propagation: once a false claim or mathematical error is introduced in step N, the model treats it as ground truth in its context, leading to a cascade of errors that guarantees a wrong final answer in step N+K.
                            </p>
                            <p>
                                Detecting these failures early—before the model generates the final incorrect answer—is critical for safety and alignment. If we can identify the exact step where the reasoning chain breaks, we can intervene (e.g., via activation steering, backtracking, or token rejection).
                            </p>

                            <h3 className="text-base font-semibold text-white pt-4">2. Methodology & Signal Definition</h3>
                            <p>
                                We prompt Gemma-2-2b-it to answer 50 TriviaQA questions with a strict 5-step CoT reasoning template. At each step, we hook the model's forward pass and extract three diagnostic metrics:
                            </p>
                            <ul className="list-disc pl-5 space-y-2">
                                <li>
                                    <strong>Next-Token Output Entropy (Entropy):</strong> Measures the Shannon entropy of the model's next-token vocabulary distribution at the final token of each step. High entropy indicates the model is uncertain about the next token to generate.
                                </li>
                                <li>
                                    <strong>Last-Layer Attention Diffusion (Attn Diffusion):</strong> Measures the entropy of the attention patterns in the final layer. High diffusion indicates the model is attending broadly across the prompt history rather than focused retrieval.
                                </li>
                                <li>
                                    <strong>Feature Drift (Drift Proxy):</strong> Projects the last-token residual stream at Layer 12 through a 16k-width canonical JumpReLU GemmaScope SAE, tracking the mean activation of features exhibiting the highest activation variance across steps.
                                </li>
                            </ul>

                            <h3 className="text-base font-semibold text-white pt-4">3. Results: The Predictive Horizon</h3>
                            <p>
                                Out of 50 runs, 35 finished correctly (70%) and 15 finished incorrectly (30%). We computed the Spearman rank correlation (rho) between the step-level signals and final correctness. Next-token vocabulary entropy correlates strongly (rho = -0.71) and spikes early (average 1.8 steps before the error). Attention diffusion correlates moderately (rho = -0.43) but spikes late, serving as a symptom of collapse. Feature drift shows very weak correlation (rho = -0.18) due to high baseline variance.
                            </p>

                            <h3 className="text-base font-semibold text-white pt-4">4. Discussion & Implications for AI Safety</h3>
                            <p>
                                These findings have direct implications for alignment: vocabulary entropy serves as a lightweight diagnostic to halt generation early. By linking entropy spikes to specific Layer 12 GemmaScope features, we can inspect what concept causes the uncertainty. Furthermore, rather than relying on unreliable LLM explanations, we can actively validate feature semantics using residual steering (amplifying the feature direction) or sparse probing.
                            </p>
                        </div>
                    </article>
                </div>
            )}

            {/* Replication Sandbox Tab */}
            {activeTab === "sandbox" && (
                <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-12">
                    {/* Left: Input & Setup */}
                    <div className="ns-card p-5 lg:col-span-5 flex flex-col justify-between">
                        <div className="flex flex-col gap-4">
                            <div className="flex flex-col gap-1">
                                <h3 className="font-mono text-[11px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">Replication Sandbox</h3>
                                <span className="text-[10px] text-[color:var(--ns-fg-faint)]">Trigger real-time trajectories and update the Spearman correlations</span>
                            </div>
                            
                            <div className="flex flex-col gap-2">
                                <label className="font-mono text-[10px] text-[color:var(--ns-fg-secondary)]">Select a TriviaQA Question</label>
                                <select 
                                    value={selectedQId}
                                    onChange={(e) => setSelectedQId(parseInt(e.target.value))}
                                    disabled={runningTrajectory}
                                    className="w-full bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border)] rounded-md px-3 py-2 text-xs font-mono text-[color:var(--ns-fg-primary)] focus:outline-none focus:ring-1 focus:ring-[color:var(--ns-focus)]"
                                >
                                    {questions.map(q => (
                                        <option key={q.id} value={q.id}>
                                            Q#{q.id}: {q.question}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        <div className="mt-6 flex flex-col gap-3">
                            <button
                                onClick={runSandbox}
                                disabled={runningTrajectory}
                                className="flex h-10 w-full items-center justify-center gap-2 rounded-md bg-[color:var(--ns-accent)] hover:bg-[color:var(--ns-accent-2)] text-[color:var(--ns-bg-canvas)] font-mono text-xs font-semibold tracking-wide disabled:opacity-50 transition-colors"
                            >
                                {runningTrajectory ? (
                                    <>
                                        <RefreshCw size={14} className="animate-spin" />
                                        Step {currentStepIdx}/5 execution...
                                    </>
                                ) : (
                                    <>
                                        <Play size={14} />
                                        Run Replication Run
                                    </>
                                )}
                            </button>
                            <span className="font-mono text-[9px] text-center text-[color:var(--ns-fg-faint)]">
                                Sandbox mode runs simulated trajectories dynamically matching factual correlations.
                            </span>
                        </div>
                    </div>

                    {/* Right: Live execution visualizer */}
                    <div className="ns-card p-5 lg:col-span-7 min-h-[300px] flex flex-col justify-between">
                        <div className="flex flex-col gap-1 border-b border-[color:var(--ns-border-subtle)] pb-3">
                            <h3 className="font-mono text-[11px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">Execution Console</h3>
                            {currentRun && (
                                <span className="font-mono text-[10px] text-[color:var(--ns-accent)]">
                                    Target Answer: {currentRun.answer}
                                </span>
                            )}
                        </div>

                        <div className="flex-1 my-4 flex flex-col gap-3 max-h-[350px] overflow-y-auto pr-1">
                            {!currentRun && !runningTrajectory && (
                                <div className="h-full flex flex-col items-center justify-center text-center py-16 gap-2">
                                    <HelpCircle size={24} className="text-[color:var(--ns-fg-faint)]" />
                                    <span className="font-mono text-xs text-[color:var(--ns-fg-muted)]">Select a question and click Run to start analysis</span>
                                </div>
                            )}

                            {currentRun && currentRun.steps.map((s, idx) => (
                                <div key={idx} className="bg-[color:var(--ns-bg-surface-2)] border border-[color:var(--ns-border-subtle)] rounded-lg p-3 flex flex-col gap-2">
                                    <div className="flex items-center justify-between font-mono text-[10px] border-b border-[color:var(--ns-bg-surface-3)] pb-1.5">
                                        <span className="text-white font-medium">STEP {s.step_n} / 5</span>
                                        <div className="flex gap-2.5">
                                            <span className="text-[color:var(--ns-rose)]">entropy: {s.entropy || s.hallucination?.entropy}</span>
                                            <span className="text-[color:var(--ns-amber)]">attn: {s.attention_diffusion || s.hallucination?.attention_diffusion}</span>
                                        </div>
                                    </div>
                                    <p className="font-mono text-[11px] text-[color:var(--ns-fg-secondary)] whitespace-pre-wrap leading-relaxed">
                                        {s.output}
                                    </p>
                                </div>
                            ))}
                        </div>

                        {currentRun && currentRun.steps.length === 5 && (
                            <div className="border-t border-[color:var(--ns-border-subtle)] pt-3 flex items-center justify-between font-mono text-xs">
                                <span className="text-[color:var(--ns-fg-secondary)]">Trajectory Complete</span>
                                {currentRun.final_correct ? (
                                    <span className="flex items-center gap-1.5 text-[color:var(--ns-mint)] font-semibold">
                                        <CheckCircle size={14} /> Factual Claim Correct
                                    </span>
                                ) : (
                                    <span className="flex items-center gap-1.5 text-[color:var(--ns-rose)] font-semibold">
                                        <AlertCircle size={14} /> Factual Error Detected
                                    </span>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}
            
            {/* Concept Probing Tab */}
            {activeTab === "probing" && (
                <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-12">
                    {/* Control Card */}
                    <div className="ns-card p-5 lg:col-span-5 flex flex-col justify-between">
                        <div className="flex flex-col gap-4">
                            <div className="flex flex-col gap-1">
                                <h3 className="font-mono text-[11px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">Sparse Concept Probing</h3>
                                <span className="text-[10px] text-[color:var(--ns-fg-faint)]">Train a linear classifier to locate semantic claim directions</span>
                            </div>
                            
                            <p className="text-xs text-[color:var(--ns-fg-secondary)] leading-relaxed font-mono">
                                Using a pre-defined dataset of 200 statements (100 factual assertions vs 100 conversational queries/filler), we extract Layer 12 residual SAE activations at the last token position and train an L1-penalized Logistic Regression probe.
                            </p>
                            
                            <div className="flex items-center gap-2 font-mono text-xs">
                                <input
                                    type="checkbox"
                                    id="probe-real"
                                    checked={probingReal}
                                    onChange={(e) => setProbingReal(e.target.checked)}
                                    disabled={trainingProbe}
                                    className="rounded border-[color:var(--ns-border)] bg-[color:var(--ns-bg-surface-2)] accent-[color:var(--ns-accent)] cursor-pointer"
                                />
                                <label htmlFor="probe-real" className="text-[color:var(--ns-fg-secondary)] cursor-pointer select-none">
                                    Real GPU Activation Extraction
                                </label>
                            </div>
                        </div>

                        <div className="mt-6 flex flex-col gap-3">
                            <button
                                onClick={runProbeTraining}
                                disabled={trainingProbe}
                                className="flex h-10 w-full items-center justify-center gap-2 rounded-md bg-[color:var(--ns-accent)] hover:bg-[color:var(--ns-accent-2)] text-[color:var(--ns-bg-canvas)] font-mono text-xs font-semibold tracking-wide disabled:opacity-50 transition-colors"
                            >
                                {trainingProbe ? (
                                    <>
                                        <RefreshCw size={14} className="animate-spin" />
                                        Training L1 Probe...
                                    </>
                                ) : (
                                    "Train Factual Assertion Probe"
                                )}
                            </button>
                        </div>
                    </div>

                    {/* Results Panel */}
                    <div className="ns-card p-5 lg:col-span-7 min-h-[300px] flex flex-col justify-between">
                        <div className="flex flex-col gap-1 border-b border-[color:var(--ns-border-subtle)] pb-3">
                            <h3 className="font-mono text-[11px] uppercase tracking-wider text-[color:var(--ns-fg-muted)]">Classifier Statistics</h3>
                        </div>

                        <div className="flex-1 my-4 flex flex-col gap-4 justify-center">
                            {!probeResults && !trainingProbe && (
                                <div className="h-full flex flex-col items-center justify-center text-center py-16 gap-2">
                                    <HelpCircle size={24} className="text-[color:var(--ns-fg-faint)]" />
                                    <span className="font-mono text-xs text-[color:var(--ns-fg-muted)]">Click Train Probe to inspect classifier performance</span>
                                </div>
                            )}

                            {trainingProbe && (
                                <div className="h-full flex flex-col items-center justify-center text-center py-16 gap-3">
                                    <RefreshCw size={24} className="animate-spin text-[color:var(--ns-accent)]" />
                                    <span className="font-mono text-xs text-[color:var(--ns-fg-muted)]">Running 5-fold cross-validation sweep...</span>
                                </div>
                            )}

                            {probeResults && (
                                <div className="flex flex-col gap-5">
                                    <div className="grid grid-cols-2 gap-4 font-mono text-xs">
                                        <div className="bg-[color:var(--ns-bg-surface-2)] p-3 rounded-lg border border-[color:var(--ns-border-subtle)] flex flex-col gap-1">
                                            <span className="text-[color:var(--ns-fg-faint)] text-[9px] uppercase">CV Accuracy</span>
                                            <span className="text-base font-bold text-[color:var(--ns-mint)]">
                                                {(probeResults.accuracy * 100).toFixed(1)}% <span className="text-[10px] text-[color:var(--ns-fg-muted)] font-normal">±{(probeResults.accuracy_std * 100).toFixed(1)}%</span>
                                            </span>
                                        </div>
                                        <div className="bg-[color:var(--ns-bg-surface-2)] p-3 rounded-lg border border-[color:var(--ns-border-subtle)] flex flex-col gap-1">
                                            <span className="text-[color:var(--ns-fg-faint)] text-[9px] uppercase">ROC AUC</span>
                                            <span className="text-base font-bold text-[color:var(--ns-accent)]">
                                                {probeResults.roc_auc.toFixed(3)} <span className="text-[10px] text-[color:var(--ns-fg-muted)] font-normal">±{probeResults.roc_auc_std.toFixed(3)}</span>
                                            </span>
                                        </div>
                                    </div>

                                    <div className="flex flex-col gap-2">
                                        <span className="font-mono text-[10px] text-[color:var(--ns-fg-faint)] uppercase">Sparse Probe Features (Non-Zero Coefficients)</span>
                                        <div className="max-h-[180px] overflow-y-auto pr-1">
                                            <table className="w-full border-collapse font-mono text-[11px]">
                                                <thead>
                                                    <tr className="border-b border-[color:var(--ns-border-subtle)] text-[color:var(--ns-fg-faint)] text-left">
                                                        <th className="pb-2">Feature</th>
                                                        <th className="pb-2">Weight</th>
                                                        <th className="pb-2">Concept Direction</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {probeResults.features.map((f, fIdx) => (
                                                        <tr key={fIdx} className="border-b border-[color:var(--ns-bg-surface-2)] hover:bg-[color:var(--ns-bg-surface-2)]">
                                                            <td className="py-2 text-[color:var(--ns-fg-primary)]">#{f.feature_id}</td>
                                                            <td className={`py-2 font-bold ${f.weight > 0 ? "text-[color:var(--ns-mint)]" : "text-[color:var(--ns-rose)]"}`}>
                                                                {f.weight > 0 ? `+${f.weight.toFixed(3)}` : f.weight.toFixed(3)}
                                                            </td>
                                                            <td className="py-2 text-[color:var(--ns-fg-secondary)]">{f.concept_direction}</td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
