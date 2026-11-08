# codedoctor🩺

[![PyPI version](https://img.shields.io/pypi/v/codedoctor.svg)](https://pypi.org/project/codedoctor/)
[![CI](https://github.com/rekursiv-ai/codedoctor/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/codedoctor/actions/workflows/package-validation.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1530237005311639592?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/2GZFPPvCqn)

Pre-commit gates that run each heavy check once per commit, sized to the
machine and below interactive priority.

# Quickstart

```bash
cd /path/to/your/repo
uvx codedoctor install
uv add --group pre-commit rekursiv-ai-typings
uv run --group pre-commit pre-commit run --all-files
```

## Repository setup

`codedoctor install` adds a locked `pre-commit` dependency group and installs
pre-commit and pre-push Git hooks. It creates missing configuration and preserves
existing hooks, comments, hook arguments, and the project's Python support range.
Running it again does not duplicate hooks.

The installer writes Ruff, ty, basedpyright, and pyrefly settings from the
package's bundled policy into `pyproject.toml`. It also configures pytest,
coverage floors, and mutation source selection. That policy comes from Loop's
canonical configuration, with checkout-specific paths removed. Existing project
settings remain overrides; installation reports their names.

Codedoctor is development tooling, not an application dependency. Applications
can use their own user-directory helpers or inline XDG handling. The installer
uses the project name for `userdirs-namespace.namespace`; set it explicitly in
`[tool.houselint]` to choose another namespace. `xdg-literal.helper` is unset
unless the project requires a shared helper.

Setup uses the same hook configuration as this repository, without its
package-export and Rust-only checks. Commits autofix whitespace, spelling,
notebooks, house lint, and Ruff formatting; check types with ty, pyrefly, and
basedpyright; and run paired tests, coverage checks, and sampled mutation tests.
Pushes run repository-wide type checks and the fast, slow, smoke, and integration
test tiers. Dependency-graph setup precedes affected-file checks.

Tests may use either `test_*.py` or `*_test.py`; existing discovery settings and
coverage thresholds are preserved. Hooks that rewrite files abort that attempt;
review and stage the changes, then retry.

`rekursiv-ai-typings` is optional, not a Codedoctor dependency. The Quickstart
adds it explicitly to the development-tools group. This also works for projects
supporting Python below 3.12; those targeting 3.12+ can use `uv add --dev`
instead.

Alternatively, install Codedoctor in the project's environment first:

```bash
uv add --dev codedoctor
uv run codedoctor install
```

Codedoctor requires Python 3.12 or newer. For projects supporting older Python,
use the `uvx` route: its tools live in a Python 3.12+ dependency group without
raising the application's runtime floor. Git and uv must already be installed.

## Policy updates

Policy changes reach projects through a new Codedoctor release. Upgrade the
locked dependency, then reinstall:

```bash
uv lock --upgrade-package codedoctor
uv run --group pre-commit codedoctor install
uv run --group pre-commit codedoctor check
```

`[tool.codedoctor.managed]` records the last generated defaults. Reinstallation
updates unchanged managed settings and adds new defaults. It preserves project
overrides and deliberately deleted settings. Lists are replaced as whole values;
a customized list remains an override. Do not edit the recorded baseline.

`codedoctor check` exits nonzero when the installed package would refresh the
project configuration. It does not write files or install dependencies. Preserved
overrides do not make configuration stale.

pre-commit calls a hook once per chunk of file names, so `--all-files` starts a
type checker or pytest several times over. Each heavy gate here is two hooks: a
`-spool` twin collects every chunk, then the gate runs once over all of them,
at this machine's worker count.

## Example

A `.pre-commit-config.yaml` pair for `ty`:

```yaml
repos:
  - repo: local
    hooks:
      - id: ty-check-spool
        name: ty-check-spool
        alias: ty-check
        language: system
        require_serial: true
        types_or: [python, pyi]
        entry: uv run python -m codedoctor.gate spool ty-check
      - id: ty-check
        name: ty-check
        language: system
        require_serial: true
        pass_filenames: false
        types_or: [python, pyi]
        entry: uv run python -m codedoctor.typecheck ty-check
```

This repository's own [`.pre-commit-config.yaml`](.pre-commit-config.yaml) wires
every gate and is the fullest example.

## Gates

| Module | Runs |
|---|---|
| `codedoctor.gate` | `spool`, `run` and `threaded`: the shared machinery |
| `codedoctor.typecheck` | `ty-check`, `pyrefly-check`, `basedpyright` |
| `codedoctor.testing` | the tests paired with each changed file, whole-suite tiers, and the pytest plugin that selects and records node ids |
| `codedoctor.coverage_floor` | each changed source's coverage by its own tests, read from the `pytest` gate's run |
| `codedoctor.mutation` | mutation testing of the changed functions (`codedoctor[mutation]`) |
| `codedoctor.lint` | `houselint`, AST rules configured in `[tool.houselint]` |
| `codedoctor.assert_isinstance` | deletes each `assert isinstance` a checker proves redundant |
| `codedoctor.affected` | the tests and files a change reaches, from a CodeGraph index |
| `codedoctor.nbstripout` | notebook output stripping |

Worker counts: `WORKERS_<HOOK_ID>` wins, then `WORKERS_DEFAULT`, then a count
derived from the machine's cores and free memory.

## Install

```bash
uv add --dev codedoctor
```

If you're missing `uv`,
- Mac
  ```bash
  brew install uv
  ```
- Linux
  ```bash
  sudo apt install -y curl
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

Or with pip,
```bash
python -m pip install codedoctor
```
