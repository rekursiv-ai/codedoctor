# codedoctor🩺

[![PyPI version](https://img.shields.io/pypi/v/codedoctor.svg)](https://pypi.org/project/codedoctor/)
[![CI](https://github.com/rekursiv-ai/codedoctor/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/codedoctor/actions/workflows/package-validation.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1530237005311639592?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/2GZFPPvCqn)

Pre-commit gates that run each heavy check once per commit, sized to the
machine and below interactive priority.

## Quick Start

```bash
uv add --dev codedoctor
```

See [Install](#install) for additional details.

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
| `codedoctor.testing` | the tests paired with each changed file, and whole-suite tiers |
| `codedoctor.coverage_floor` | each changed source's coverage by its own tests |
| `codedoctor.mutation` | mutation testing of the changed functions (`codedoctor[mutation]`) |
| `codedoctor.lint` | `houselint`, AST rules configured in `[tool.houselint]` |
| `codedoctor.assert_isinstance` | deletes each `assert isinstance` a checker proves redundant |
| `codedoctor.affected` | the tests and files a change reaches, from a CodeGraph index |
| `codedoctor.nbstripout` | notebook output stripping |
| `codedoctor.selected_nodeids` | a pytest plugin writing a run's selected node ids |

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
