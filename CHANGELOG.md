# Changelog

All notable codedoctor changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.1.5 - 2026-11-08

### Changed

- Run every stage through `codedoctor <stage>`.
- Rename houselint to `codedoctor lint` and migrate its configuration.
- Require `codedoctor install --notebooks` to add notebook tooling.
- Install only the pre-commit Git hook by default; pre-push remains opt-in.
- Let OpenBLAS detect the host's usable CPU instructions.

### Fixed

- Preserve long hook commands, block scalars, arguments, and YAML aliases
  during command migration.
- Cover projects with dynamic names when generating checker policy.

## 0.1.4 - 2026-11-07

### Changed

- Provision pinned, checksummed CodeGraph and skip indexing when unavailable.
- Generate checker policy from packaged defaults, preserving project overrides.
- Check for stale generated configuration.
- Require explicit project settings for user-directory lint rules.
- Apply coverage floors to sources owned by changed tests.

### Fixed

- Preserve leading whitespace in hook configuration.
- Prove integer bounds for NumPy `arange` defaults.
- Restrict mutation's vendored-tree filter to stub directories.
- Fix mutant import shadowing.

## 0.1.3 - 2026-11-07

### Changed

- `mutation` findings are advisory: they print and the run exits 0. `--blocking`,
  or `PRECOMMIT_MUTATION_BLOCKING` set to a positive integer, makes any finding
  exit 1 as before. A run that cannot start still exits 1.
- houselint's `cli-shape` wants a polyglot launcher to run `python3 -P "$0"` (or
  `python3 -P -m MODULE`), and its fix adds the `-P`. Without it Python puts the
  launcher's folder first on `sys.path`, where a module beside it, such as a
  `types.py`, shadows the standard library's.

## 0.1.2 - 2026-10-22

### Added

- The pre-commit gates: `gate` (spool, `run`, `threaded`), `typecheck`
  (ty, pyrefly, basedpyright), `testing` (paired tests and pytest tiers),
  `coverage_floor`, `mutation`, `lint` (houselint), `assert_isinstance`,
  `affected`, `nbstripout` and `selected_nodeids`.
- A `coverage` extra for the `coverage_floor` gate, which imports `coverage`.
- The `mutation` extra requires `pytest`, which its pytest plugins import.
