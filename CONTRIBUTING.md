# Contributing to NeuroScope

Thank you for contributing to NeuroScope v3! As an enterprise-grade AI/ML interpretability platform, we maintain strict standards for code hygiene, security, concurrency safety, and system architecture.

---

## 1. Branching Strategy

We enforce a strict git branching model to protect production stability:

```
main (Production — Protected)
  ▲
  │  [PR + CI Checks Required]
develop (Staging / Integration)
  ▲
  ├── feature/circuit-discovery
  ├── feature/steering-hooks
  └── fix/outbox-worker-deadlock
```

### Branch Roles
- **`main`**: Production code. Direct pushes are disabled. Merges require 1 approving review and clean CI pipeline passes.
- **`develop`**: Integration branch for pre-release features.
- **`feature/<short-description>`**: Feature development. Branch off `develop`.
- **`fix/<short-description>`**: Bug fixes and security patches.
- **`release/vX.Y.Z`**: Release preparation branches.

---

## 2. Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat(mechanisms)`: Add IOI circuit faithfulness benchmark calculation
- `fix(outbox)`: Handle `LockNotAvailableError` gracefully during worker poll
- `perf(sae)`: Accelerate Top-K sparse latent tensor serialization using `npz`
- `sec(auth)`: Enforce Bearer token verification across meta endpoints
- `docs(arch)`: Update system topology and advisory lock sequence diagrams
- `refactor(api)`: De-monolithize `server.py` into modular domain routers

---

## 3. Local Development Setup

### Backend (Python 3.11 / PyTorch / FastAPI)
```bash
# Clone repository
git clone https://github.com/Gaurav711cgu/NeuroScope.git
cd NeuroScope/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.template .env

# Run local FastAPI server
uvicorn server:app --reload --port 8000
```

### Frontend (Next.js / React 18 / Tailwind)
```bash
cd NeuroScope/frontend
npm install
npm run dev
```

---

## 4. Testing & Quality Standards

Before submitting a PR, ensure all verification suites pass:

```bash
# Run integration test suite
python backend_test.py

# Run unit tests
pytest tests/ -v

# Run linting and code formatting
ruff check backend/
```

### Key Requirements for Approval
1. Zero secrets, tokens, or private keys in diffs (`gitleaks` / `bandit` compliant).
2. All 12 integration test scenarios passing.
3. Proper transactional outbox pattern adherence for async DB state modifications.
4. Descriptive commit history and filled PR template.
