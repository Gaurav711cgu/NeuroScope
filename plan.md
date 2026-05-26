# NeuroScope v1 Plan (Free-tier, CPU-only)

## 1) Objectives
- Deliver a **research-grade, cost-free** mechanistic interpretability tool for **multi-step agent trajectories** where the **same model** (GPT-2 Small) is both **agent + analysis subject** (scientifically valid).
- Capture **per-step internals** (residual stream + selective attention stats), run **SAE decomposition** (gpt2-small-res-jb), compute **feature drift** across steps.
- Implement **cross-step causal patching** (patch step N activations into step M) and quantify effect via **KL divergence + token deltas**.
- Provide **hallucination early-warning timeline** (entropy + attention diffusion + uncertainty-feature activations) with an explicit integrity note (early warning only).
- Provide **NL circuit query** grounded in run artifacts via Anthropic SDK (Claude Sonnet 4).
- Ship a dense, dark, Jupyter-adjacent UI + **5 pre-built experiments** cached in Mongo.

**Current status:** All objectives above are implemented and operational. End-to-end tests passed **24/24** (backend + frontend).

---

## 2) Implementation Steps

### Phase 1 — Core Workflow POC (isolation; do not proceed until stable)
**Goal:** prove the entire core loop works on CPU with acceptable latency.

**Status: COMPLETED** ✅
- Implemented `backend/test_core.py` end-to-end POC:
  - Loaded TransformerLens GPT-2 (`HookedTransformer.from_pretrained('gpt2')`) on CPU.
  - Loaded SAE from SAELens: `gpt2-small-res-jb` (layer 7).
  - Ran 3-step ReAct-style agent loop with greedy decoding.
  - Hook-captured residual streams across all layers + attention pattern (last layer) + selected MLP outputs.
  - Serialized activations to `float16` `.npz` per step.
  - SAE feature decomposition for last-token residual; top-K features per step.
  - Feature drift scoring (variance across steps).
  - Cross-step patching (step1 → step3 @ L7) with KL divergence + token delta summary.
  - Hallucination scoring (entropy + attention diffusion + uncertainty features).
  - Anthropic SDK LLM explanation call (Claude Sonnet 4).

**POC result:** 50.5s total runtime (target < 90s) ✅

**Phase 1 user stories (all complete)**
1. Run `python backend/test_core.py` and see a 3-step trajectory complete.
2. Saved activation files per step; shapes match expected layers/tokens.
3. Top SAE features per step + drift ranking printed.
4. Cross-step patching returns KL divergence + top token changes.
5. Grounded LLM explanation references steps/layers/features.

---

### Phase 2 — V1 App Development (build around proven core)
**Goal:** production-quality MVP (no auth), full core research loop in UI.

**Status: COMPLETED** ✅

#### 2.1 Backend (FastAPI) + core library
- Built `/backend/neuroscope/` modules:
  - `loader.py` (singleton model + SAE loader)
  - `agent.py` (ReAct prompt + greedy decode)
  - `hooks.py` (hook capture + residual L2 extraction)
  - `storage.py` (float16 `.npz` activation persistence)
  - `sae.py` (top-K decomposition, drift timelines, SAE-based attribution graph)
  - `patching.py` (cross-step patch + KL divergence + token shifts)
  - `hallucination.py` (3-signal early-warning diagnostic)
  - `llm.py` (Anthropic SDK wrapper)
  - `runner.py` (trajectory orchestration)

#### 2.2 Data model (Mongo) + activation storage
- MongoDB collections used:
  - `agent_runs`, `causal_patches`, `attribution_graphs`, `experiments`, `queries`, `feature_labels`
- Activation blobs persisted to:
  - `/app/backend/data/activations/{run_id}/step_{n}.npz` (float16, compressed)

#### 2.3 API implementation (under `/api/v1`)
- Implemented and verified endpoints:
  - Runs:
    - `POST /runs` (creates run and starts background execution)
    - `GET /runs` (list)
    - `GET /runs/{id}` (full run)
    - `GET /runs/{id}/steps/{n}` (single-step)
  - Patching:
    - `POST /runs/{id}/patch`
    - `POST /runs/{id}/patch-matrix`
    - `GET /runs/{id}/patches`
  - Query:
    - `POST /runs/{id}/query`
    - `GET /runs/{id}/queries`
  - Attribution:
    - `POST /runs/{id}/attribution`
  - Experiments:
    - `GET /experiments`
    - `GET /experiments/{slug}`
  - Feature metadata:
    - `GET /feature/{layer}/{feature_id}` (best-effort label + Neuronpedia URL)
  - Utility:
    - `GET /health`
    - `GET /suggested-tasks`

**Important fix implemented:** backend `_get_run()` falls back to `experiments` collection when `{id}` is an experiment slug, enabling `/runs/{slug}/query` and `/runs/{slug}/patch`.

#### 2.4 Execution model
- Background execution implemented using FastAPI `BackgroundTasks`.
- Progress stages persisted to Mongo (`progress.stage`, `progress.completed_steps`).

#### 2.5 Frontend (React) — dense research UX
- Implemented pages:
  - `/` landing
  - `/run` create
  - `/run/:id` analysis
  - `/run/:id/step/:n` deep dive
  - `/experiments` library
  - `/experiments/:slug` experiment detail (read-only analysis)
  - `/docs` methodology + limitations

- Implemented core components/visualizations:
  - Activation timeline heatmap
  - SAE drift multi-line chart + clickable legend chips
  - Hallucination risk timeline + integrity note
  - Cross-step patch matrix heatmap + inspector panel
  - NL circuit query chat box
  - SAE-based attribution force graph (step detail)
  - Step list + selected-step rail card
  - Reproducibility snippet + copy-to-clipboard toast

**Design system:** dark, dense, research-grade (IBM Plex Sans + JetBrains Mono) with explicit palette tokens.

#### 2.6 Seed pre-built experiments
- Implemented `backend/seed_experiments.py`.
- Seeded **5 experiments** into MongoDB (all available in UI):
  - `hallucination-propagation`
  - `tool-call-prediction`
  - `reasoning-collapse`
  - `ioi-persistence`
  - `self-correction`

#### 2.7 Phase 2 conclude: comprehensive E2E test run
- Completed full end-to-end testing:
  - Backend tests: **12/12 passed**
  - Frontend tests: **12/12 passed**
  - Total: **24/24 passed** ✅

**Phase 2 user stories (all complete)**
1. Create a new run with task + n_steps and watch step-by-step progress.
2. Open a run and view activation heatmap.
3. Inspect SAE feature drift and hover/pin features.
4. Run causal patching and inspect KL + token deltas.
5. Ask an NL question and receive grounded answer citing steps/layers/features.

---

### Phase 3 — Hardening, UX polish, and comprehensive testing
**Goal:** stabilize performance, correctness, and reproducibility.

**Status: COMPLETED (for v1 scope)** ✅
- Comprehensive testing completed (24/24 passes).
- Docs page includes methodology + limitations, including the hallucination “early warning only” integrity note.
- Activation persistence verified on disk.

**Phase 3 user stories (all complete)**
1. Reload a run and render all artifacts from DB/files.
2. Run patch matrix sweep and visualize results.
3. Open step deep dive with attribution graph + hallucination breakdown.
4. Browse experiments that load instantly.
5. Copy reproducibility snippet and rerun analysis.

---

## 3) Next Actions
**v1 is feature-complete and operationally healthy.**

If you request additional work, recommended next actions (Phase 4 / v1.1+) are:
1. **Side-by-side run comparison** (`/compare`) with aligned step timelines and diffed drift/patch matrices.
2. **Feature label enrichment**: fetch and cache Neuronpedia labels/examples into Mongo for top features.
3. **Export**: run export to JSON (already mostly implicit) and optional PDF report.
4. **Run history pagination + filters** (status, date, n_steps, max risk).
5. **Auth + private workspaces** (if multi-user needed).
6. **External GemmaScope/Gemma-2-2b integration** via a user-provided HF Space (swap in bigger subject model).
7. **Optional heavier attribution** (Anthropic circuit-tracer style) if compute budget allows.

---

## 4) Success Criteria
**All success criteria met.** ✅
- **POC:** `test_core.py` completed end-to-end on CPU in **50.5s** (< 90s) with SAE drift, patching, hallucination signals, and LLM explanation.
- **V1:** All core endpoints operational; UI renders run artifacts without manual steps; experiments library seeded.
- **User outcomes:** All required flows work end-to-end; tests show **24/24 pass**.
- **Scientific validity:** same model for agent+analysis; patching uses real activations; SAE decomposition uses published SAEs; limitations disclosed in docs.
