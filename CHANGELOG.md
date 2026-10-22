# Changelog

All notable codedoctor changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- A `coverage` extra for the `coverage_floor` gate, which imports `coverage`.
- The `mutation` extra requires `pytest`, which its pytest plugins import.

### Fixed

- The `coverage_floor` and `mutation` gates no longer import
  `codedoctor.lib.testing.resources`, which the wheel does not ship.

### Removed

- The wheel no longer ships `codedoctor.gate.fakes`, stand-ins used only by
  the gates' tests.

## 0.1.1 - 2026-10-21

### Added

- The pre-commit gates: `gate` (spool, `run`, `threaded`), `typecheck`
  (ty, pyrefly, basedpyright), `testing` (paired tests and pytest tiers),
  `coverage_floor`, `mutation`, `lint` (houselint), `assert_isinstance`,
  `affected`, `nbstripout` and `selected_nodeids`.
