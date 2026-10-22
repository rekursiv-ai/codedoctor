# Security Policy

## Why this file exists

codedoctor runs inside `pre-commit` on a developer's machine and in CI. Its
gates start subprocesses (type checkers, pytest, mutation runs) over the files
a commit changes, and read settings from the repository's `pyproject.toml` and
environment. Security reports need a private path so exploit details are not
published before review.

## Reporting a vulnerability

Please report suspected security vulnerabilities privately by emailing hello@rekursiv.ai.

Include:

- Affected version or commit.
- Steps to reproduce.
- Expected impact.
- Any suggested mitigation.

Please do not open public issues for vulnerabilities until we have investigated and coordinated disclosure.

## Scope

Security reports are especially useful for:

- A file name, setting, or environment variable that makes a gate run a
  command other than the one its hook names.
- Spool or temporary files another local user can read or replace.
- Dependency or packaging issues that affect installed users.
- Supply-chain concerns in the published wheel or its dependency set.

## Out of scope

The gates run the tools and tests of the repository they check. Running them on
a repository you do not trust is equivalent to running its code; reports that
amount to "an untrusted repository's tests run" describe that behavior rather
than a vulnerability.
