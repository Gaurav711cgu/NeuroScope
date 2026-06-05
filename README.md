# NeuroScope v2: Mechanistic Interpretability & SAE Trajectory Patching

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=next.js&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)

NeuroScope v2 is a mechanistic interpretability platform for autonomous LLM agents. It hooks internal transformer states across multi-turn agent execution sequences, decomposes residual streams using 16k-width GemmaScope Sparse Autoencoders (SAEs), and executes causal activation patching to isolate neural circuit reasoning failures.

---

## Key Architectures

### 1. Multi-Turn Residual Stream Hooking
Autonomous agents fail over long context steps due to subtle attention drift. NeuroScope hooks the transformer at runtime:
- Intercepts residual stream layers, attention heads, and MLP activations using TransformerLens.
- Tracks internal representations sequentially across multiple conversation turns/context history.
- Builds a stateful database of token-by-token activation outputs.

### 2. GemmaScope SAE Feature Dictionary Decoding
Model MLPs compress concepts into dense, uninterpretable polysemantic spaces. NeuroScope resolves this via:
- Hooks Gemma-2-2b-it using SAELens.
- Projects intermediate neural states onto 16,384 monosemantic MLP features defined in GemmaScope.
- Features a visual feature dictionary explorer to lookup active features (e.g., JSON tags, mathematical operators).

### 3. Cross-Step Causal Activation Patching
Locates the exact sub-circuits responsible for reasoning slips:
- Runs clean (successful task execution) and corrupted (failed task execution due to prompt drift) runs.
- Sequentially replaces clean residual stream segments with corrupted ones.
- Measures KL-Divergence output scores to map exactly which neural nodes/layers causally control the reasoning outcome.

---

## Platform Pipeline Flow

(Add a ```mermaid tag around the block below)

    graph TD
        A[Agent Conversation History] -->|Multi-Turn Prompts| B[TransformerLens Hooks]
        B -->|Hook Residual Stream & MLP| C[Gemma-2-2b-it Inference]
        C -->|MLP Layer Activation| D[GemmaScope 16k SAE]
        D -->|Decompose to 16,384 features| E[Monosemantic Feature Explorer]
        
        C -->|Activation Patching Matrix| F[Causal Patching Heatmap Engine]
        F -->|Corrupted vs Clean KL-Div| G[Reasoning Circuit Isolation]

---

## Directory Structure

(Add a ```yaml tag around the block below)

    neuroscope/
      ├── backend/
      │   ├── runner.py         # Multi-turn hook controller
      │   ├── sae_decoder.py    # GemmaScope 16k SAE hooks & projections (SAELens)
      │   ├── patching.py       # Causal activation patching matrix builder
      │   └── main.py           # FastAPI server endpoints
      ├── frontend/
      │   ├── src/
      │   │   ├── components/
      │   │   │   ├── Timeline.tsx      # Interactive steps x layers activation timeline
      │   │   │   ├── PatchingGrid.tsx  # Causal patching interactive heatmap
      │   │   │   └── Dictionary.tsx    # GemmaScope feature dictionary search UI
      │   └── package.json      # React 19 / Next.js setup
      └── requirements.txt      # PyTorch, TransformerLens, and SAELens dependencies

---

## Getting Started

### 1. Backend Setup
Install PyTorch and interpretability requirements:

    cd backend
    pip install -r requirements.txt
    python main.py

### 2. Frontend Setup
Launch the Next.js activation explorer interface:

    cd frontend
    npm install
    npm run dev

Open http://localhost:3000 to browse hooked agent activations.

---

## Sample Interpretability Log

    [HOOK SUCCESS] Hooked 26 layers of Gemma-2-2b-it.
    [SAE ACTIVATION] Layer 12 MLP: Decoded feature #4321 (Concept: "Strict JSON Syntax") with activation L2 norm 8.92.
    [PATCHING RESULT] High causal attribution found at Layer 14 Attention Head 3. Corrupting this node increases KL-divergence by 4.25 (causing tool invocation failure).

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
