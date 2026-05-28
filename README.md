---
title: NeuroScope
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 🧠 NeuroScope v2 — Agentic Mechanistic Interpretability Platform

**NeuroScope** is an open-source mechanistic interpretability diagnostic platform designed for multi-step transformer reasoning agents. Rather than analyzing models in static, single-prompt settings, NeuroScope lets researchers hook, decompose, and causally patch model representations across active reasoning trajectories.

## 🚀 Key Features

* **Trajectory SAE Decomposition:** Hooks `hook_resid_post` at captured layers (layers 6, 12, 18, 24 for Gemma-2) and decomposes activations using 16k-width canonical JumpReLU **GemmaScope Sparse Autoencoders**.
* **Cross-Step Causal Patching:** Novel mechanism that patches last-token residual stream representations from step $N$ into step $M$'s forward pass, measuring $\text{KL}(\text{patched} \mid\mid \text{baseline})$ to locate the causal origin of reasoning failures.
* **Three-Signal Hallucination Scorer:** Real-time uncertainty tracking combining token entropy, attention key diffusion, and dynamic feature drift proxy (no hardcoded features).
* **AI Explainer Client:** Automated natural language explanations of active features and circuits grounded in Gemini 2.5.
* **100% Free Hybrid Backend:** Uses Google Firestore for low-cost metadata storage, coupled with automatic **Local Disk Fallback** for `.npz` float16 step activation storage.
* **Dynamic Local Scaling:** Gracefully scales down to `gpt2` and Neel Nanda's `gpt2-small-res-jb` SAEs if running in resource-constrained environments (like 8GB Mac CPU).

---

## 🛠️ Local Development Setup

### 1. Configure Credentials
Copy `backend/.env.template` to `backend/.env` and add:
* `FIREBASE_SERVICE_ACCOUNT_PATH`: path to your `firebase-service-account.json` file.
* `GEMINI_API_KEY`: your Google AI Studio API key.
* `HF_TOKEN`: Hugging Face read token.

### 2. Seed Database
Seeding automatically adjusts parameters dynamically based on your local model settings (e.g. GPT-2 fallback vs Gemma-2):
```bash
cd backend
python3 seed_experiments.py
```

### 3. Run Servers
**Backend:**
```bash
python3 -m uvicorn server:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```
