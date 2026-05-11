# Changelog

## [1.1.0] — 2026-05-11

### Added
- **Identity system** — optional `IdentityConfig` in `MemoryConfig` lets you describe the agent's role. Identity shaping happens at retrieval time (importance scoring, seed topics), not at extraction time, so existing memory is unaffected when identity is added or changed.
- **Prebuilt identity configs** — Paralegal, Executive Assistant, and Personal AI Assistant roles are matched automatically without an LLM call. For other roles, pass a `derivation_api_key` to derive a configuration via LLM; the result is cached in the memory store.
- **`questioner.py`** — internal QA cache verification module with 3-layer validation (confidence floor, node ID resolution, answer term presence).
- **Dual-mode benchmark infrastructure** — test runner that ingests into two DBs (identity vs baseline) in one pass and queries both side-by-side. Confirmed identity net positive on legal case notes (+2.5 pp) and personal assistant (Vectra, +2 pp) datasets.

### Fixed
- Replaced all `datetime.utcnow()` calls with `datetime.now(timezone.utc)` — eliminates `DeprecationWarning` on Python 3.12.
- Pinned `requests>=2.28,<3` and `sentence-transformers>=2.2,<4` to prevent silent breaking upgrades.

## [1.0.5] — 2026-05-08

- Minor README updates. Re-tested and logged benchmark results.

## [1.0.4]

- Minor changes.
