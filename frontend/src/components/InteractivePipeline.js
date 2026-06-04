import React, { useState } from 'react';
import { ShieldAlert, ShieldCheck, Terminal, HelpCircle } from 'lucide-react';

const PIPELINE_STEPS = [
  {
    title: "1. Residual Stream Hooking",
    desc: "Intercepts the activations of the residual stream at target layer intervals (L6, L12, L18, L24) using PyTorch forward hooks during the agent's execution loop.",
    input: "x = [Batch, Seq_Len, Hidden_Dim] (Layer 12 Input)",
    output: "L12 resid_post vector extracted.\nShape: [1, 24, 2048] (Gemma-2-2B)\nRegistered hooks: 4/4 online.",
    vulnTitle: "Host Memory Blowup & OOM",
    vulnDesc: "Hooking and storing high-dimensional float16 activation matrices across long multi-step agent trajectories can deplete GPU VRAM, crashing the system.",
    mitTitle: "Dynamic RAM Offloading",
    mitDesc: "Implement active hook collection pruning and offload inactive step tensors to host system RAM, capping active VRAM usage to <50MB.",
    color: "#4fb3c8"
  },
  {
    title: "2. SAE Feature Decomposition",
    desc: "Projects the extracted dense residual stream vectors into a JumpReLU Sparse Autoencoder (GemmaScope 16k) to resolve them into 16,384 sparse, monosemantic feature dimensions.",
    input: "resid_post vector (2048 dimensions)",
    output: "[SAE Inference]: Reconstruction L2 loss = 0.082\nActive Features: [Feat_8891: 1.42, Feat_12044: 0.88] (2 active / 16,384)",
    vulnTitle: "Adversarial Feature Hijacking",
    vulnDesc: "Out-of-distribution prompts or jailbreaks trigger anomalous feature activations, causing downstream safety monitors to miss critical failure states.",
    mitTitle: "Sparsity Bounds Enforcement",
    mitDesc: "Set reconstruction loss validation limits and sound alarms if active feature density crosses the 1% threshold (L0 > 163).",
    color: "#6ee7b7"
  },
  {
    title: "3. Causal Attribution Probing",
    desc: "Constructs linear probing classifiers over active feature paths to identify the exact step and layer predicting target behaviors (such as token hallucinations or tool call routing).",
    input: "Active feature values history across steps 1-4",
    output: "Hallucination Probability = 87.2%\nTop Causal Driver: Layer 12 Feature 8891 (Attribution: 0.74)",
    vulnTitle: "Linear Approximation Divergence",
    vulnDesc: "Attribution methods (like Taylor Approximations) can report false positives by misattributing causal weight on highly correlated inactive paths.",
    mitTitle: "Mean Ablation Cross-Validation",
    mitDesc: "Cross-validate attribution outputs against true causal effects using Mean Ablation, enforcing an r-value correlation filter (r >= 0.912).",
    color: "#f2c14e"
  },
  {
    title: "4. Activation Steering",
    desc: "Injects steering vectors directly into the active attention heads or residual stream to steer model generation away from flagged failure paths.",
    input: "Inject steering vector: x_patched = x + 2.5 * v_steer",
    output: "Steering active.\nTarget logit delta = -3.42\nKL Divergence (steered || baseline) = 0.140 (Success: Hallucination averted)",
    vulnTitle: "Output Logit Collapse",
    vulnDesc: "Steering values exceeding model bounds cause catastrophic logit degradation, leading to repetitive or corrupted token generations.",
    mitTitle: "L2-Norm Scaling Constraints",
    mitDesc: "Apply dynamic L2-norm constraints that automatically scale the steering vector to preserve baseline perplexity ranges.",
    color: "#f87171"
  }
];

export function InteractivePipeline() {
  const [activeStep, setActiveStep] = useState(0);
  const step = PIPELINE_STEPS[activeStep];

  return (
    <div className="ns-card p-6" style={{ marginTop: '2.5rem', background: 'var(--ns-bg-surface-1)', border: '1px solid var(--ns-border)' }}>
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
        <span className="ns-section-title" style={{ color: 'var(--ns-accent)' }}>Interactive Flow Sandbox</span>
        <h3 className="text-xl font-semibold mt-1" style={{ color: 'var(--ns-fg-primary)' }}>Agent Reasoning Interpretability Flow</h3>
        <p style={{ color: 'var(--ns-fg-muted)', fontSize: '13px', marginTop: '4px' }}>
          Deconstruct the mechanistic lifecycle of a multi-turn agent from raw layer hook to causal steering activation.
        </p>
      </div>

      {/* Steps Selector */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'relative', maxWidth: '600px', margin: '0 auto 2.5rem', padding: '0 10px' }}>
        <div style={{ position: 'absolute', top: '20px', left: '30px', right: '30px', height: '2px', background: 'var(--ns-border-subtle)', zIndex: 1 }} />
        <div style={{
          position: 'absolute', top: '20px', left: '30px',
          width: `${activeStep * 33.33}%`, height: '2px',
          background: `linear-gradient(90deg, var(--ns-accent), var(--ns-mint), var(--ns-amber))`,
          transition: 'width 0.3s ease', zIndex: 2
        }} />

        {PIPELINE_STEPS.map((s, idx) => {
          const isActive = activeStep === idx;
          const isPassed = activeStep > idx;
          const stepColor = isActive ? s.color : isPassed ? s.color : 'var(--ns-border-strong)';
          return (
            <button
              key={idx}
              onClick={() => setActiveStep(idx)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                display: 'flex', flexDirection: 'column', alignItems: 'center',
                zIndex: 3, position: 'relative', outline: 'none'
              }}
            >
              <div style={{
                width: '40px', height: '40px', borderRadius: '50%',
                background: 'var(--ns-bg-canvas)',
                border: `2px solid ${stepColor}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: isActive || isPassed ? s.color : 'var(--ns-fg-muted)',
                fontWeight: 700, fontSize: '13px',
                transition: 'all 0.3s ease',
                boxShadow: isActive ? `0 0 12px ${s.color}33` : 'none'
              }}>
                {idx + 1}
              </div>
              <span className="font-mono" style={{ fontSize: '10px', color: isActive ? 'var(--ns-fg-primary)' : 'var(--ns-fg-faint)', marginTop: '8px', whiteSpace: 'nowrap' }}>
                {s.title.split('. ')[1].split(' ')[0]}
              </span>
            </button>
          );
        })}
      </div>

      {/* Grid Panels */}
      <div className="grid gap-6 md:grid-cols-2" style={{ maxWidth: '950px', margin: '0 auto' }}>
        {/* Left: Input/Output and logs */}
        <div style={{ background: 'var(--ns-bg-canvas)', border: '1px solid var(--ns-border-subtle)', borderRadius: '8px', padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: step.color }} />
              <span className="font-mono" style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--ns-fg-muted)' }}>
                State Pipeline Data
              </span>
            </div>
            <h4 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--ns-fg-primary)', marginBottom: '8px' }}>
              {step.title}
            </h4>
            <p style={{ fontSize: '13px', color: 'var(--ns-fg-secondary)', lineHeight: '1.6', marginBottom: '20px' }}>
              {step.desc}
            </p>
          </div>

          <div style={{ marginTop: 'auto' }}>
            <div style={{ background: 'var(--ns-bg-surface-2)', border: '1px solid var(--ns-border-subtle)', borderRadius: '6px', padding: '14px', fontFamily: 'var(--font-mono)', fontSize: '11.5px', lineHeight: '1.6' }}>
              <div style={{ color: 'var(--ns-fg-faint)', marginBottom: '4px' }}>// HOOK DATA / INPUT</div>
              <pre style={{ color: 'var(--ns-fg-primary)', whiteSpace: 'pre-wrap', marginBottom: '12px' }}>{step.input}</pre>
              <div style={{ color: 'var(--ns-fg-faint)', marginBottom: '4px' }}>// ACTIVATION METRICS</div>
              <pre style={{ color: step.color, whiteSpace: 'pre-wrap' }}>{step.output}</pre>
            </div>
          </div>
        </div>

        {/* Right: Security & Safeguards */}
        <div style={{ background: 'var(--ns-bg-canvas)', border: '1px solid var(--ns-border-subtle)', borderRadius: '8px', padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <ShieldAlert size={14} style={{ color: 'var(--ns-rose)' }} />
              <span className="font-mono" style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--ns-rose)' }}>
                Mechanistic Vulnerability
              </span>
            </div>
            <h4 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--ns-fg-primary)', marginBottom: '6px' }}>
              {step.vulnTitle}
            </h4>
            <p style={{ fontSize: '13px', color: 'var(--ns-fg-secondary)', lineHeight: '1.6' }}>
              {step.vulnDesc}
            </p>
          </div>

          <div style={{ borderTop: '1px solid var(--ns-border-subtle)', paddingTop: '16px', marginTop: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <ShieldCheck size={14} style={{ color: 'var(--ns-mint)' }} />
              <span className="font-mono" style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--ns-mint)' }}>
                NeuroScope Safeguard
              </span>
            </div>
            <h4 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--ns-fg-primary)', marginBottom: '6px' }}>
              {step.mitTitle}
            </h4>
            <p style={{ fontSize: '13px', color: 'var(--ns-fg-secondary)', lineHeight: '1.6' }}>
              {step.mitDesc}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
