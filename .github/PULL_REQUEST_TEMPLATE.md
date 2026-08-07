## Description
<!-- Provide a clear summary of the changes made and the problem solved. -->

## Category of Change
- [ ] 🚀 **Feature**: New capability or neural feature visualization tool
- [ ] 🐛 **Bug Fix**: Patch for active circuit analysis or UI bug
- [ ] ⚡ **Performance**: Benchmark & p99 optimization (GPU, Postgres, caching)
- [ ] 🔒 **Security**: Auth, secret sanitization, or input validation fix
- [ ] 🛠️ **Refactoring/Architecture**: Domain router split, outbox pattern update
- [ ] 📚 **Documentation**: Architecture docs, PRD, or benchmark updates

## System & Architectural Impact
- [ ] Has database schema migrated (`backend/neuroscope/schema.sql`)?
- [ ] Are breaking changes introduced to API endpoints (`/api/v1/*`)?
- [ ] Are concurrency/advisory lock constraints updated?

## Checklist & Verification
- [ ] Unit & Integration Tests pass (`python backend_test.py` - 12/12)
- [ ] Static analysis & type checks pass (`ruff check`, `mypy`)
- [ ] Security audit verified (no secrets, raw tokens, or PII exposed)
- [ ] Performance benchmarks verified (p99 latency < 250ms)
- [ ] Relevant documentation updated (`ARCHITECTURE.md`, `README.md`)

## Evidence / Visuals
<!-- Attach screenshots, trace plots, or benchmark logs if applicable -->
