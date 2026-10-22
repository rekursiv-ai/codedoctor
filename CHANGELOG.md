# Changelog

All notable codedoctor changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.1.2 - 2026-10-22

### Added

- The pre-commit gates: `gate` (spool, `run`, `threaded`), `typecheck`
  (ty, pyrefly, basedpyright), `testing` (paired tests and pytest tiers),
  `coverage_floor`, `mutation`, `lint` (houselint), `assert_isinstance`,
  `affected`, `nbstripout` and `selected_nodeids`.
- A `coverage` extra for the `coverage_floor` gate, which imports `coverage`.
- The `mutation` extra requires `pytest`, which its pytest plugins import.
