# Fix a mutation-gate failure

1. Read mutant diffs, owning code, callers, and tests. Inspect production
   before changing tests.
2. Check redundant defaults, fallbacks on required typed fields, and duplicate
   checks. Keep only simplifications worthwhile without mutation testing;
   never redesign sound code just to eliminate a mutant.
3. Strengthen an existing behavioral assertion before adding the smallest
   missing input or isolated case. Assert meaningful results, exception types,
   causes, and fields -- not noncontractual diagnostic spelling.
4. Use a selective skip only when its declaration below holds. If none applies,
   retain the finding; do not invent an exclusion to obtain a pass.
5. Repair setup failures, no-tests results, timeouts, and inconclusive runs.
   Rerun the same failed gate without changing its scope, seed, or requirements.

## Selective skips

Use these through **CodeDoctor's mutation gate**. Unmodified native mutmut can
interpret bracketed labels as blanket skips.

| Label | You declare | Skipped change |
| --- | --- | --- |
| `diagnostic-text` | Exact wording is noncontractual. | One string changes; brace/percent field signatures match. |
| `case-insensitive` | The whole literal ignores case. | Different strings with equal `.lower()`. |
| `truthiness` | Only truth value matters. | One falsy pure literal replaces another. |

Other tokens, call targets, and arguments stay unchanged. The matcher checks
changes, not declarations' truth. No rule grants generic functional equivalence.
For text rules, put each literal on its own physical line. In multiline calls,
keep its pragma inside parentheses; a trailing comma preserves that placement
through formatting. A pragma after the final `)` targets the statement's first line.

```python
text = raw.decode(
    "utf-8",  # pragma: no mutate[case-insensitive]
    errors="replace",
)

# pragma: no mutate[diagnostic-text] start
raise ValueError("Cannot load the requested configuration.")
# pragma: no mutate end


def validate(name: str) -> None:  # pragma: no mutate[diagnostic-text] block
    if not name:
        raise ValueError("A name is required.")
```

- Combine applicable labels as `[diagnostic-text, case-insensitive]`. Order and
  duplicates do not matter; any matching label skips. Append no justification prose.
- Matching candidates never execute; other mutations remain active.
- Unknown labels, empty brackets, malformed scopes, and examined unused labels
  fail. Label observed findings, not speculation.
- Preserve existing blocks and code layout when migrating; keep closing `end`
  delimiters. Existing bare exclusions remain blanket skips until resolved.
