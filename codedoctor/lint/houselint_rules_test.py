"""Tests for ``houselint_rules``: one positive and one negative per rule."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import ast
import json
import textwrap

import pytest

from codedoctor.lint import houselint, houselint_rules
from codedoctor.lint.houselint_rules import (
    AllReexportRule,
    AnnotationImportRule,
    ArgsSectionRule,
    AssertRule,
    AttrBlankLineRule,
    AttrDocFormRule,
    CacheKwargRule,
    CliShapeRule,
    CommentSentenceRule,
    ConfigGlobalsRule,
    CwdFinalRule,
    DataclassRule,
    DefOrderRule,
    DigitsRule,
    DottedImportRule,
    EmDashRule,
    FinalInlineRule,
    FluentNameRule,
    FmtDirectiveRule,
    ForwarderHelperRule,
    GetLoggerNameRule,
    ImportAliasRule,
    ImportHoistRule,
    InitFacadeRule,
    LazyImportBindingRule,
    LazyImportCostRule,
    LoggingFstringRule,
    MkdirModeRule,
    ModuleSideEffectsRule,
    NegatedCompareRule,
    NegatedTernaryRule,
    NegativeBoolParamRule,
    NestedDefRule,
    NestedTryRule,
    PrivateOnelinerRule,
    PytestFooterRule,
    PytestNamingRule,
    ReturnsNamedRule,
    ReturnsSectionRule,
    SingleUseHelperRule,
    StaticMethodRule,
    SuperOnlyRule,
    SuppressionFormRule,
    TempfilePrefixRule,
    TensorValueGuardRule,
    TmpLiteralRule,
    TreekleTorchLoadRule,
    UnderscoreParamRule,
    UserdirsNamespaceRule,
    UserdirsWrapperRule,
    XdgLiteralRule,
    _is_literal_value,
    _sections,
    _target_name,
)
from codedoctor.lint.houselint_types import Finding, Unit, _tokenize


if TYPE_CHECKING:
    from collections.abc import Sequence

    from codedoctor.lint.houselint import Rule


def _findings(
    tmp_path: Path,
    rule: Rule,
    body: str,
    name: str = "mod.py",
) -> Sequence[Finding]:
    """Write ``body`` as ``name`` and run one rule's ``check`` over it."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    # The dispatcher sorts every finding (``houselint.run``), so order here
    # is not part of a rule's contract; the result is sorted to compare stably.
    findings = tuple(sorted(rule.check(unit)))
    for finding in findings:
        assert finding.path == path, finding
        assert finding.rule == rule.name, finding
        assert isinstance(finding.line, int), finding
        assert finding.line >= 1, finding
        assert isinstance(finding.message, str), finding
        assert finding.message, finding
    return findings


# Also asserts the contract the dispatcher's fix gate relies on: a fix that rewrites the
# file implies a check that reported it.
def _fixed(tmp_path: Path, rule: Rule, body: str, name: str = "mod.py") -> str | None:
    """Write ``body`` as ``name`` and return one rule's ``fix`` result."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    fixed = rule.fix(unit)
    if fixed is not None and fixed != body:
        assert rule.check(unit), f"{rule.name} fix rewrote a file its check passed"
    return fixed


_EMDASH: Final = chr(0x2014)
_TMP: Final = "/" + "tmp"

_FOOTER: Final = (
    'if __name__ == "__main__":\n'
    "    from codedoctor.lib.testing.runner import test_main\n\n"
    "    test_main(__file__)\n"
)

_UNREAD: Final = (
    "nothing reads through `{}`; delete it, or name the side effect it is for "
    "in a house-ignore reason"
)


def test_every_new_rule_is_registered_and_satisfies_the_protocol() -> None:
    names = [rule.name for rule in houselint.default_rules()]
    assert len(names) == len(set(names))
    assert "emdash" in names
    assert "module-side-effects" in names
    assert "fmt-directive" in names


# -- emdash ------------------------------------------------------------------


def test_emdash_is_reported_and_fixed(tmp_path: Path) -> None:
    body = f"x = 1  # a {_EMDASH} b\n"
    assert [f.line for f in _findings(tmp_path, EmDashRule(), body)] == [1]
    assert _fixed(tmp_path, EmDashRule(), body) == "x = 1  # a -- b\n"


def test_double_hyphen_is_clean(tmp_path: Path) -> None:
    assert not _findings(tmp_path, EmDashRule(), "x = 1  # a -- b\n")


def test_emdash_in_a_notebook_cell_is_fixed(tmp_path: Path) -> None:
    body = json.dumps(
        {"cells": [{"cell_type": "code", "source": [f"# a {_EMDASH} b\n"]}]},
    )
    fixed = _fixed(tmp_path, EmDashRule(), body, name="n.ipynb")
    assert fixed is not None
    assert _EMDASH not in fixed


# -- test-footer -------------------------------------------------------------


def test_missing_footer_is_reported_and_appended(tmp_path: Path) -> None:
    body = "def test_x() -> None:\n    assert True\n"
    assert _findings(tmp_path, PytestFooterRule(), body, name="a_test.py")
    fixed = _fixed(tmp_path, PytestFooterRule(), body, name="a_test.py")
    assert fixed is not None
    assert fixed.endswith(_FOOTER)


def test_facade_footer_is_reported_and_replaced(tmp_path: Path) -> None:
    # Assembled so this test file does not itself carry the banned import.
    facade_import = "    from codedoctor.lib.testing " + "import test_main\n\n"
    body = (
        "def test_x() -> None:\n    assert True\n\n\n"
        'if __name__ == "__main__":\n' + facade_import + "    test_main(__file__)\n"
    )
    findings = _findings(tmp_path, PytestFooterRule(), body, name="a_test.py")
    assert any("facade" in f.message for f in findings)
    fixed = _fixed(tmp_path, PytestFooterRule(), body, name="a_test.py")
    assert fixed is not None
    assert fixed.count("__main__") == 1
    assert fixed.endswith(_FOOTER)


def test_standard_footer_is_clean_and_non_test_is_skipped(tmp_path: Path) -> None:
    body = "def test_x() -> None:\n    assert True\n\n\n" + _FOOTER
    assert not _findings(tmp_path, PytestFooterRule(), body, name="a_test.py")
    assert not _findings(tmp_path, PytestFooterRule(), "x = 1\n", name="a.py")


def test_footer_runner_is_configurable_for_an_exported_package(
    tmp_path: Path,
) -> None:
    # An exported package keeps its runner under its own root, not ``loop``.
    rule = PytestFooterRule(runner="pkg.lib.testing.runner")
    exported = (
        'if __name__ == "__main__":\n'
        "    from pkg.lib.testing.runner import test_main\n\n"
        "    test_main(__file__)\n"
    )
    body = "def test_x() -> None:\n    assert True\n\n\n"
    assert not _findings(tmp_path, rule, body + exported, name="a_test.py")
    assert _findings(tmp_path, rule, body + _FOOTER, name="a_test.py")
    fixed = _fixed(tmp_path, rule, body + _FOOTER, name="a_test.py")
    assert fixed is not None
    assert fixed.count("__main__") == 1
    assert fixed.endswith(exported)


def test_footer_facade_check_follows_the_configured_runner(tmp_path: Path) -> None:
    rule = PytestFooterRule(runner="pkg.lib.testing.runner")
    body = "x = 1\nfrom pkg.lib.testing " + "import test_main\n"
    findings = _findings(tmp_path, rule, body, name="a_test.py")
    assert any(
        "use `from pkg.lib.testing.runner import test_main`" in f.message
        for f in findings
    )


# -- test-naming -------------------------------------------------------------


def test_test_prefix_naming_is_reported(tmp_path: Path) -> None:
    findings = _findings(tmp_path, PytestNamingRule(), "x = 1\n", name="test_foo.py")
    assert len(findings) == 1
    assert "foo_test.py" in findings[0].message


def test_test_prefix_naming_is_allowed_under_tools_tests(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        PytestNamingRule(),
        "x = 1\n",
        name="tools/tests/test_foo.py",
    )


# -- xdg-literal -------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        'x = os.environ.get("XDG_CONFIG_HOME")\n',
        "x = Path.home() / '.config'\n",
        'x = os.environ["HOME"]\n',
        'x = os.path.expanduser("~")\n',
    ],
)
def test_xdg_rederivation_is_reported(tmp_path: Path, body: str) -> None:
    assert _findings(tmp_path, XdgLiteralRule(), body)


def test_xdg_findings_name_each_rederivation(tmp_path: Path) -> None:
    xdg = "XDG" + "_CONFIG_HOME"
    body = (
        f'"""Module doc naming {xdg}."""\n'
        "class C:\n"
        f'    """Class doc naming {xdg}."""\n'
        "    def f(self):\n"
        f'        """Method doc naming {xdg}."""\n'
        f'        "{xdg} as a bare non-docstring string"\n'
        "        return Path.home()\n"
        f'x = os.environ["{xdg}"]\n'
        "y = os.path.expanduser('~'), expanduser('~')\n"
        "z = os.getenv('HOME'), environ.get('HOME'), os.getenv('USER')\n"
        "w = os.environ['HOME'], os.environ['USER'], env['HOME'], environ['HOME']\n"
        "v = Path.cwd(), other.home()\n"
    )
    tail = "; go through codedoctor.lib.userdirs"
    assert [
        (f.line, f.message) for f in _findings(tmp_path, XdgLiteralRule(), body)
    ] == [
        (6, f"literal {xdg + ' as a bare non-docstring string'!r}{tail}"),
        (7, f"Path.home(){tail}"),
        (8, f"literal {xdg!r}{tail}"),
        (9, f"expanduser{tail}"),
        (9, f"expanduser{tail}"),
        (10, f"$HOME lookup{tail}"),
        (10, f"$HOME lookup{tail}"),
        (11, f"$HOME lookup{tail}"),
        (11, f"$HOME lookup{tail}"),
    ]


def test_userdirs_helper_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        XdgLiteralRule(),
        'x = config_dir() / "rekursiv-ai"\n',
    )


# -- userdirs-namespace ------------------------------------------------------


def test_wrong_namespace_is_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        UserdirsNamespaceRule(),
        'x = state_dir() / "loop" / "a"\n'
        'y = userdirs.config_dir() / "rekursiv-ai"\n'
        "z = cache_dir() / name\n"
        'w = cache_dir() * "loop"\n'
        "v = cache_dir() / 3\n",
    )
    tail = 'the first segment must be "rekursiv-ai" (AGENTS.md Per-User Directories)'
    assert [(f.line, f.message) for f in findings] == [
        (1, f"namespace 'loop' after state_dir; {tail}"),
        (3, f"non-literal namespace after cache_dir; {tail}"),
        (5, f"non-literal namespace after cache_dir; {tail}"),
    ]


def test_non_literal_namespace_is_reported(tmp_path: Path) -> None:
    assert _findings(tmp_path, UserdirsNamespaceRule(), "x = state_dir() / name\n")


def test_rekursiv_and_vendor_namespaces_are_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        UserdirsNamespaceRule(),
        'a = config_dir() / "rekursiv-ai" / "x"\n'
        'b = cache_dir() / "uv"\n'
        'c = userdirs.config_dir() / "goa-1.0"\n'
        'd = config_dir() / "Google" / "Chrome"\n'
        'e = config_dir() / "google-chrome"\n',
    )


# -- userdirs-wrapper --------------------------------------------------------


def test_wrapper_def_and_global_are_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        UserdirsWrapperRule(),
        'ROOT = state_dir() / "rekursiv-ai" / "x"\n\n\n'
        "def runs() -> Path:\n"
        '    """Runs dir."""\n'
        '    return state_dir() / "rekursiv-ai" / "runs"\n'
        "async def logs():\n"
        '    return cache_dir() / "rekursiv-ai"\n'
        "def two():\n"
        "    x = 1\n"
        '    return cache_dir() / "rekursiv-ai"\n'
        "def other():\n"
        "    return compute()\n"
        "OTHER: int = 3\n",
    )
    assert [(f.line, f.message) for f in findings] == [
        (
            1,
            (
                "module global holds a userdirs path; build it inline at each use "
                "(AGENTS.md rule 2)"
            ),
        ),
        *(
            (
                line,
                (
                    f"{name} wraps a userdirs path; inline `<x>_dir() / ...` at each "
                    "call site (AGENTS.md rule 2)"
                ),
            )
            for line, name in ((4, "runs"), (7, "logs"))
        ),
    ]


def test_def_that_does_more_than_wrap_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        UserdirsWrapperRule(),
        "def runs() -> Path:\n"
        '    p = state_dir() / "rekursiv-ai" / "runs"\n'
        "    p.mkdir(mode=0o700, exist_ok=True)\n"
        "    return p\n",
    )


# -- cache-kwarg -------------------------------------------------------------


def test_cache_dir_kwarg_and_set_dir_are_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        CacheKwargRule(),
        "def helper(*, cache_dir): ...\n"
        'load(cache_dir="/x")\n'
        'hf.load(model_dir="/y")\n'
        'torch.hub.set_dir("/z")\n'
        "m.helper(cache_dir=c)\n"
        "load(other=1)\n",
    )
    kwarg = (
        "= overrides the provisioned cache; let the library read its env var "
        "(AGENTS.md Provisioned Paths)"
    )
    assert [(f.line, f.message) for f in findings] == [
        (2, f"explicit cache_dir{kwarg}"),
        (3, f"explicit model_dir{kwarg}"),
        (
            4,
            (
                "torch.hub.set_dir overrides TORCH_HOME; delete it "
                "(AGENTS.md Provisioned Paths)"
            ),
        ),
    ]


def test_plain_load_is_clean(tmp_path: Path) -> None:
    assert not _findings(tmp_path, CacheKwargRule(), 'load("x", revision="main")\n')


# -- lazy-import-binding -----------------------------------------------------


def test_aliased_and_attribute_lazy_import_are_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        LazyImportBindingRule(),
        "from wrapt import lazy_import as li, other as o\n"
        "from other import lazy_import as li2\n"
        "import wrapt\n"
        "x = wrapt.lazy_import('m')\n"
        "y = other.lazy_import('m'), wrapt.other('m'), f().lazy_import\n",
    )
    message = (
        "lazy_import bound other than `from wrapt import lazy_import` "
        "(STYLE.md Code Organization)"
    )
    assert [(f.line, f.message) for f in findings] == [(1, message), (4, message)]


def test_canonical_lazy_import_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        LazyImportBindingRule(),
        "from wrapt import lazy_import\nnp = lazy_import('numpy')\n",
    )


# -- tmp-literal -------------------------------------------------------------


def test_tmp_literal_is_reported(tmp_path: Path) -> None:
    body = (
        f'p = Path("{_TMP}/x")\n'
        f'q = "{_TMP}"\n'
        f'r = "{_TMP}x", "/var{_TMP}/y", b"{_TMP}/z", 3\n'
    )
    message = f"literal {_TMP} path; use tempfile (STYLE.md Executables)"
    assert [
        (f.line, f.message) for f in _findings(tmp_path, TmpLiteralRule(), body)
    ] == [
        (1, message),
        (2, message),
    ]


def test_tempfile_use_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        TmpLiteralRule(),
        'p = tempfile.mkdtemp(prefix="x-")\n',
    )


# -- tempfile-prefix ---------------------------------------------------------


def test_tempfile_without_prefix_is_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        TempfilePrefixRule(),
        "with tempfile.TemporaryDirectory() as d:\n    pass\n"
        "h, p = tempfile.mkstemp(dir=d)\n",
    )
    assert [(f.line, f.message) for f in findings] == [
        (1, "tempfile.TemporaryDirectory without prefix=; name the temporary"),
        (3, "tempfile.mkstemp without prefix=; name the temporary"),
    ]


def test_tempfile_prefix_fix_two_calls_on_one_line(tmp_path: Path) -> None:
    """Inserts apply right to left, so the left call's offset stays valid."""
    body = "import tempfile\na, b = tempfile.mkdtemp(), tempfile.mkdtemp(dir=x)\n"
    assert _fixed(tmp_path, TempfilePrefixRule(), body, name="m.py") == (
        "import tempfile\n"
        'a, b = tempfile.mkdtemp(prefix="m-"), tempfile.mkdtemp(prefix="m-", dir=x)\n'
    )


def test_tempfile_direct_import_without_prefix_is_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        TempfilePrefixRule(),
        "from tempfile import TemporaryDirectory, mkstemp as make\n"
        "with TemporaryDirectory() as d:\n    pass\n"
        'make(prefix="x-")\n'
        "make()\n",
    )
    assert [f.line for f in findings] == [2, 5]


def test_tempfile_with_prefix_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        TempfilePrefixRule(),
        'with tempfile.TemporaryDirectory(prefix="x-") as d:\n    pass\n',
    )


def test_tempfile_prefix_fix_handles_all_calls_and_arguments(tmp_path: Path) -> None:
    body = (
        "import tempfile\n"
        "tempfile.TemporaryDirectory()\n"
        'tempfile.NamedTemporaryFile(mode="w")\n'
        'tempfile.mkdtemp("old", dir=base)\n'
        "tempfile.mkstemp(\n    dir=base,\n)\n"
    )
    assert _fixed(tmp_path, TempfilePrefixRule(), body, name="my_module.py") == (
        "import tempfile\n"
        'tempfile.TemporaryDirectory(prefix="my-module-")\n'
        'tempfile.NamedTemporaryFile(prefix="my-module-", mode="w")\n'
        'tempfile.mkdtemp("old", prefix="my-module-", dir=base)\n'
        'tempfile.mkstemp(\n    prefix="my-module-", dir=base,\n)\n'
    )


def test_tempfile_prefix_fix_handles_direct_imports_and_is_idempotent(
    tmp_path: Path,
) -> None:
    body = (
        "from tempfile import NamedTemporaryFile as TempFile, mkstemp\n"
        "TempFile()\n"
        'mkstemp("old")\n'
    )
    fixed = _fixed(tmp_path, TempfilePrefixRule(), body, name="some_file.py")
    assert fixed == (
        "from tempfile import NamedTemporaryFile as TempFile, mkstemp\n"
        'TempFile(prefix="some-file-")\n'
        'mkstemp("old", prefix="some-file-")\n'
    )
    path = tmp_path / "some_file.py"
    path.write_text(fixed, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert TempfilePrefixRule().fix(unit) is None


# -- cwd-final ---------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "_CWD = Path(__file__).resolve().parent\n",
        "ROOT: Final = Path(__file__).resolve().parents[2]\n",
        "def f() -> Path:\n    return Path(__file__).resolve().parent\n",
        "_CWD: Final = Path(__file__).parent\n",
        (
            "_CWD: Final = Path(__file__).resolve().parent\n"
            "_THIS: Final = _CWD / Path(__file__).name\n"
        ),
    ],
)
def test_noncanonical_path_file_is_reported(tmp_path: Path, body: str) -> None:
    assert _findings(tmp_path, CwdFinalRule(), body)


@pytest.mark.parametrize(
    "body",
    [
        '_CWD: Final = Path(__file__).resolve().parent\nDATA = _CWD / "testdata"\n',
        "_THIS: Final = Path(__file__).resolve()\n_CWD: Final = _THIS.parent\n",
    ],
)
def test_canonical_cwd_binding_is_clean(tmp_path: Path, body: str) -> None:
    assert not _findings(tmp_path, CwdFinalRule(), body)


def test_cwd_final_fixes_directory_paths_and_parents(tmp_path: Path) -> None:
    body = (
        "from pathlib import Path\n"
        "from typing import Final\n\n"
        "a = Path(__file__).resolve().parent\n"
        "b = Path(__file__).parent\n"
        "c = Path(__file__).resolve().parents[1]\n"
        "d = Path(__file__).resolve().parents[3]\n"
    )
    fixed = _fixed(tmp_path, CwdFinalRule(), body)
    assert fixed == (
        "from pathlib import Path\n"
        "from typing import Final\n\n"
        "_CWD: Final = Path(__file__).resolve().parent\n\n"
        "a = _CWD\n"
        "b = _CWD\n"
        "c = _CWD.parent\n"
        "d = _CWD.parents[2]\n"
    )


def test_cwd_final_fix_every_shape_on_one_line_each(tmp_path: Path) -> None:
    """One rewrite per ``Path(__file__)``, chosen by walking up its parents."""
    body = (
        "from pathlib import Path\n"
        "from typing import Final\n"
        "\n"
        "_CWD: Final = Path(__file__).resolve().parent\n"
        "a = Path(__file__).resolve().parent, Path(__file__).parent\n"
        "b = Path(__file__).resolve().parents[1]\n"
        "c = Path(__file__).resolve().parents[3]\n"
        "d = Path(__file__).resolve().parents[0]\n"
        "e = Path(__file__).resolve().parents[k]\n"
        "f = Path(__file__) / 'x', 'y' / Path(__file__)\n"
        "g = Path(__file__).resolve().name\n"
        "h = Path(__file__).resolve\n"
        "i = Path(__file__).stem\n"
        "j = Path(__file__).resolve.parent\n"
    )
    assert _fixed(tmp_path, CwdFinalRule(), body) == (
        "from pathlib import Path\n"
        "from typing import Final\n"
        "\n"
        "_THIS: Final = Path(__file__).resolve()\n"
        "\n"
        "_CWD: Final = Path(__file__).resolve().parent\n"
        "a = _CWD, _CWD\n"
        "b = _CWD.parent\n"
        "c = _CWD.parents[2]\n"
        "d = _THIS.parents[0]\n"
        "e = _THIS.parents[k]\n"
        "f = _CWD / 'x', 'y' / _THIS\n"
        "g = _THIS.name\n"
        "h = _THIS\n"
        "i = _THIS.stem\n"
        "j = _CWD\n"
    )


def test_cwd_final_fixes_file_paths_with_this_binding(tmp_path: Path) -> None:
    body = "from pathlib import Path\n\nname = Path(__file__).name\n"
    fixed = _fixed(tmp_path, CwdFinalRule(), body)
    assert fixed == (
        "from pathlib import Path\n"
        "from typing import Final\n\n"
        "_THIS: Final = Path(__file__).resolve()\n"
        "_CWD: Final = _THIS.parent\n\n"
        "name = _THIS.name\n"
    )


def test_cwd_final_fix_is_idempotent_and_parses(tmp_path: Path) -> None:
    body = "from pathlib import Path\n\nvalue = Path(__file__).resolve().parent\n"
    fixed = _fixed(tmp_path, CwdFinalRule(), body)
    assert fixed is not None
    compile(fixed, "mod.py", "exec")
    assert _fixed(tmp_path, CwdFinalRule(), fixed) is None


def test_cwd_final_skips_without_pathlib_import(tmp_path: Path) -> None:
    body = "value = Path(__file__).resolve().parent\n"
    assert _fixed(tmp_path, CwdFinalRule(), body) is None


def test_cwd_final_findings_skip_the_canonical_bindings(tmp_path: Path) -> None:
    body = (
        "from pathlib import Path\n"
        "from typing import Final\n"
        "_THIS: Final = Path(__file__).resolve()\n"
        "_CWD: Final = Path(__file__).resolve().parent\n"
        "x = Path(__file__).parent\n"
        "_OTHER: Final = Path(__file__).resolve().parent\n"
    )
    message = (
        "Path(__file__) outside `_CWD: Final = Path(__file__).resolve().parent`; "
        "bind _CWD once and build subpaths from it"
    )
    assert [(f.line, f.message) for f in _findings(tmp_path, CwdFinalRule(), body)] == [
        (5, message),
        (6, message),
    ]


def test_cwd_final_check_skips_unparsable_file_naming_path(tmp_path: Path) -> None:
    assert _findings(tmp_path, CwdFinalRule(), "x = Path(__file__) +\n") == ()


# -- import-alias ------------------------------------------------------------


def test_import_alias_check_skips_unparsable_file_with_alias(tmp_path: Path) -> None:
    assert _findings(tmp_path, ImportAliasRule(), "import loop.a as b\ndef (\n") == ()


def test_first_party_alias_is_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        ImportAliasRule(),
        "from loop.a import b as c, d as d, e as ee\n"
        "import numpy as np\n"
        "import ops.x as y\n"
        "from . import z as zz\n",
    )
    assert findings == tuple(
        Finding(
            path=tmp_path / "mod.py",
            line=line,
            rule="import-alias",
            message=(
                f"import {name} as {asname}; `{leaf}` is free here, so import it "
                "unaliased (a first-party alias is only for a taken name)"
            ),
        )
        for line, name, asname, leaf in (
            (1, "b", "c", "b"),
            (1, "e", "ee", "e"),
            (3, "ops.x", "y", "x"),
            (4, "z", "zz", "z"),
        )
    )


@pytest.mark.parametrize(
    "body",
    [
        # Two modules share the leaf: one keeps it, the other needs a name.
        "from loop.games.a import kernel\nfrom loop.games.b import kernel as b_kernel\n",
        # Or both take one, so neither reads as the only ``kernel``.
        (
            "from loop.games.a import kernel as a_kernel\n"
            "import loop.games.b.kernel as b_kernel\n"
        ),
        # The module's own public ``render`` owns the leaf.
        "from loop.spine import render as spine_render\ndef render():\n    pass\n",
        "import loop.spine.render as spine_render\nrender = 1\n",
        # A local anywhere in the module takes the leaf.
        "from loop.a import model as a_model\ndef f():\n    model = 1\n    return model\n",
        # So does a builtin: ``from loop.a import input`` would shadow it.
        "from loop.a import input as loop_input\nx = loop_input()\n",
        # An unaliased dotted import binds its ROOT and nothing else, so the
        # root is what takes the leaf -- ``torch``, not ``torch.nn`` or the
        # whole ``torch.nn.functional``.
        "import torch.nn.functional\nfrom loop.models import torch as loop_torch\n",
        # An aliased one binds its asname instead, which is already undotted:
        # ``helper``, not the empty string a separator-only read would give.
        "import pkg.mod as helper\nfrom loop.a import helper as a_helper\n",
    ],
)
def test_first_party_alias_of_a_taken_name_is_clean(tmp_path: Path, body: str) -> None:
    """Aliasing is the only spelling when the plain name already means something."""
    assert _findings(tmp_path, ImportAliasRule(), body) == ()
    assert _fixed(tmp_path, ImportAliasRule(), body) is None


def test_import_alias_fix_unaliases_a_top_level_package(tmp_path: Path) -> None:
    """``from  import ops`` does not parse; one segment has no parent to name."""
    body = "import ops as o\nx = o.y\n"
    assert _fixed(tmp_path, ImportAliasRule(), body) == "import ops\nx = ops.y\n"


def test_dotted_alias_matching_its_leaf_is_clean(tmp_path: Path) -> None:
    assert _findings(tmp_path, ImportAliasRule(), "import loop.a.b as b\n") == ()


def test_import_alias_leaves_unaliased_dotted_imports_alone(tmp_path: Path) -> None:
    """Shortening ``import loop.a.b`` is ``dotted-import``'s; two owners would race."""
    body = "import loop.a.b.c\nvalue = loop.a.b.c.X\n"
    assert _findings(tmp_path, ImportAliasRule(), body) == ()
    assert _fixed(tmp_path, ImportAliasRule(), body) is None


def test_third_party_alias_and_same_name_are_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        ImportAliasRule(),
        "import numpy as np\nimport torch.distributed as dist\n"
        "from collections.abc import Set as AbstractSet\n"
        "import loop.a.b as b\n",
    )


def test_import_alias_fix_drops_alias_and_rewrites_references(tmp_path: Path) -> None:
    body = "from loop.a import b as c\nvalue = c.X + c\n"
    fixed = _fixed(tmp_path, ImportAliasRule(), body)
    assert fixed == "from loop.a import b\nvalue = b.X + b\n"
    compile(fixed, "mod.py", "exec")
    assert _fixed(tmp_path, ImportAliasRule(), fixed) is None


def test_import_alias_fix_leaves_colliding_name(tmp_path: Path) -> None:
    body = "from loop.a.b import c as d\nc = object()\nvalue = d.X\n"
    assert _fixed(tmp_path, ImportAliasRule(), body) is None


# -- dotted-import -----------------------------------------------------------


def _unfixed_reasons(tmp_path: Path, body: str) -> list[tuple[int, str]]:
    """Run ``dotted-import`` over ``body``; return each finding's line and reason."""
    assert _fixed(tmp_path, DottedImportRule(), body) is None
    return [
        (f.line, f.message.partition("(not autofixed: ")[2].removesuffix(")"))
        for f in _findings(tmp_path, DottedImportRule(), body)
    ]


def test_dotted_import_is_reported_whatever_the_party(tmp_path: Path) -> None:
    body = (
        "import json\n"
        "import loop.a\n"
        "import urllib.parse\n"
        "import torch.nn as nn\n"
        "from os import path\n"
        "x = loop.a.X, urllib.parse.quote, json, nn, path\n"
    )
    assert _findings(tmp_path, DottedImportRule(), body) == tuple(
        Finding(path=tmp_path / "mod.py", line=line, rule="dotted-import", message=m)
        for line, m in (
            (
                2,
                (
                    "import loop.a binds loop; write `from loop import a` "
                    "(rerun with --fix dotted-import)"
                ),
            ),
            (
                3,
                (
                    "import urllib.parse binds urllib; write "
                    "`from urllib import parse` (rerun with --fix dotted-import)"
                ),
            ),
        )
    )
    fixed = _fixed(tmp_path, DottedImportRule(), body)
    assert fixed == (
        "import json\n"
        "from loop import a\n"
        "from urllib import parse\n"
        "import torch.nn as nn\n"
        "from os import path\n"
        "x = a.X, parse.quote, json, nn, path\n"
    )
    assert _findings(tmp_path, DottedImportRule(), fixed) == ()
    assert _fixed(tmp_path, DottedImportRule(), fixed) is None


def test_dotted_import_fix_rewrites_references_at_every_depth(tmp_path: Path) -> None:
    body = (
        "import loop.g.d.e\n"
        "x = loop.g.d.e.X\n"
        "def f():\n"
        "    return [loop.g.d.e.Y for _ in loop.g.d.e.Z]\n"
    )
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        "from loop.g.d import e\nx = e.X\ndef f():\n    return [e.Y for _ in e.Z]\n"
    )


def test_dotted_import_fix_imports_a_root_still_read_another_way(
    tmp_path: Path,
) -> None:
    """``ops.bare`` and ``print(ops)`` need ``ops``; ``loop`` has no such read.

    Each root sits beside the other and a fully covered non-root load
    (``other``), so neither root's verdict can leak to the other.
    """
    body = (
        "import ops.p.q.r\n"
        "import loop.g.d.e\n"
        "other = 1\n"
        "x = ops.p.q.r.A, loop.g.d.e.X, other\n"
        "y = ops.bare, print(ops)\n"
    )
    fixed = _fixed(tmp_path, DottedImportRule(), body)
    assert fixed == (
        "import ops\n"
        "from ops.p.q import r\n"
        "from loop.g.d import e\n"
        "other = 1\n"
        "x = r.A, e.X, other\n"
        "y = ops.bare, print(ops)\n"
    )
    assert _fixed(tmp_path, DottedImportRule(), fixed) is None


# Which imports already bind the root decides whether a rewrite needs its own
# `import root`. Each case pins one way an import does or does not count.
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # A KEPT dotted import binds the root too, so nothing is added: the
        # binder is its root (`ops`), not the whole dotted name.
        (
            "import ops.z\nimport ops.r.s\ny = ops.r.s.A, print(ops)\n",
            "import ops.z\nfrom ops.r import s\ny = s.A, print(ops)\n",
        ),
        # Two rewrites in one block share ONE `import ops`: the first becomes a
        # binder for the second.
        (
            "import ops.a.b\nimport ops.c.d\ny = ops.a.b.X, ops.c.d.Y, print(ops)\n",
            (
                "import ops\nfrom ops.a import b\nfrom ops.c import d\n"
                "y = b.X, d.Y, print(ops)\n"
            ),
        ),
        # An ALIASED import of the same root binds its asname, not the root, and
        # an unaliased import of ANOTHER root binds that one -- neither counts.
        (
            (
                "import ops.p.q as pq\nimport json\nimport ops.r.s\n"
                "y = ops.r.s.A, pq, json, print(ops)\n"
            ),
            (
                "import ops.p.q as pq\nimport json\nimport ops\nfrom ops.r import s\n"
                "y = s.A, pq, json, print(ops)\n"
            ),
        ),
        # A binder in ANOTHER scope never ran here, so the module still needs one.
        (
            (
                "import ops.r.s\ndef f():\n    import ops\n    return ops.bare\n"
                "y = ops.r.s.A, print(ops)\n"
            ),
            (
                "import ops\nfrom ops.r import s\ndef f():\n    import ops\n"
                "    return ops.bare\ny = s.A, print(ops)\n"
            ),
        ),
    ],
)
def test_dotted_import_fix_counts_only_the_imports_that_bind_the_root(
    tmp_path: Path,
    body: str,
    expected: str,
) -> None:
    assert _fixed(tmp_path, DottedImportRule(), body) == expected


def test_dotted_import_fix_replaces_a_parenthesized_root_whole(tmp_path: Path) -> None:
    """The read's span starts at the ``(`` its root name sits inside."""
    body = "import a.b\nx = (a).b.c, (\n    a\n).b\n"
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        "from a import b\nx = b.c, b\n"
    )


def test_dotted_import_fix_reuses_a_plain_root_import(tmp_path: Path) -> None:
    body = 'import os\nimport os.path\nx = os.path.join("café", os.sep)\n'
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        'import os\nfrom os import path\nx = path.join("café", os.sep)\n'
    )


def test_dotted_import_fix_rewrites_package_and_submodule_once(tmp_path: Path) -> None:
    body = (
        "import loop.g.pkg\n"
        "import loop.g.pkg.kernel\n"
        "x = loop.g.pkg.kernel.f(loop.g.pkg.Y)\n"
    )
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        "from loop.g import pkg\nfrom loop.g.pkg import kernel\nx = kernel.f(pkg.Y)\n"
    )


def test_dotted_import_fix_frees_a_leaf_another_rewrite_unbinds(
    tmp_path: Path,
) -> None:
    """``c`` is taken until ``import c.d`` becomes ``from c import d``.

    One pass leaves ``loop.a.b.c`` for a second; ``fix`` must reach the
    fixpoint itself, because the dispatcher runs each fixer once.
    """
    body = "import c.d\nimport loop.a.b.c\nx = c.d.Y, loop.a.b.c.X\n"
    fixed = _fixed(tmp_path, DottedImportRule(), body)
    assert fixed == "from c import d\nfrom loop.a.b import c\nx = d.Y, c.X\n"
    assert _fixed(tmp_path, DottedImportRule(), fixed) is None


def test_dotted_import_fix_rewrites_only_where_the_import_binds(
    tmp_path: Path,
) -> None:
    """``g`` reads the module's ``a``; ``f``'s local import never reaches it."""
    body = (
        "import a\n"
        "def f():\n"
        "    import a.b\n"
        "    return a.b.X\n"
        "def g():\n"
        "    return a.b.Y\n"
    )
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        "import a\n"
        "def f():\n"
        "    from a import b\n"
        "    return b.X\n"
        "def g():\n"
        "    return a.b.Y\n"
    )


def test_dotted_import_fix_skips_scopes_that_rebind_the_root(tmp_path: Path) -> None:
    """Parameters shadow ``a``; a class attribute ``a`` does not reach ``m``."""
    body = (
        "import a.b\n"
        "x = a.b.X\n"
        "def f(a):\n"
        "    return a.b.Y\n"
        "g = lambda a=a.b.D: a.b.W\n"
        "class C:\n"
        "    a = 1\n"
        "    def m(self):\n"
        "        return a.b.Z\n"
    )
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        "from a import b\n"
        "x = b.X\n"
        "def f(a):\n"
        "    return a.b.Y\n"
        "g = lambda a=b.D: a.b.W\n"
        "class C:\n"
        "    a = 1\n"
        "    def m(self):\n"
        "        return b.Z\n"
    )


def test_dotted_import_fix_ignores_a_leaf_bound_where_no_read_reaches(
    tmp_path: Path,
) -> None:
    """``spy``'s own ``request`` never meets a rewritten read, so it is no bar."""
    body = (
        "import urllib.request\n"
        "x = urllib.request.urlopen\n"
        "def spy(request):\n"
        "    return request\n"
    )
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        "from urllib import request\n"
        "x = request.urlopen\n"
        "def spy(request):\n"
        "    return request\n"
    )


def test_dotted_import_fix_rewrites_each_function_s_own_import(
    tmp_path: Path,
) -> None:
    body = (
        "def f():\n"
        "    import a.b\n"
        "    return a.b.X\n"
        "def g():\n"
        "    import a.b\n"
        "    return a.b.Y\n"
    )
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        "def f():\n"
        "    from a import b\n"
        "    return b.X\n"
        "def g():\n"
        "    from a import b\n"
        "    return b.Y\n"
    )


def test_dotted_import_fix_keeps_comments_indent_and_guards(tmp_path: Path) -> None:
    body = (
        "if TYPE_CHECKING:\n"
        "    import a.b  # Types only.\n"
        "def f() -> a.b.T:\n"
        "    import c.d\n"
        "    return c.d.X\n"
    )
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        "if TYPE_CHECKING:\n"
        "    from a import (  # Types only.\n"
        "        b,\n"
        "    )\n"
        "def f() -> b.T:\n"
        "    from c import d\n"
        "    return d.X\n"
    )


def test_dotted_import_fix_moves_a_comment_at_the_end_of_the_file(
    tmp_path: Path,
) -> None:
    body = "def f():\n    return a.b.X\nimport a.b  # Last line."
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        "def f():\n    return b.X\nfrom a import (  # Last line.\n    b,\n)"
    )


def test_dotted_import_fix_keeps_a_deferred_import_s_noqa_on_every_statement(
    tmp_path: Path,
) -> None:
    """Ruff's isort wraps a long ``from`` line and strands its noqa on the leaf.

    The comment then suppresses nothing, ruff deletes it, and ``import-hoist``
    hoists the deferred import. After the ``(`` it stays put, and the
    inserted ``import a`` is deferred too, so it needs the same noqa.
    """
    body = "def f():\n    import a.b  # noqa: PLC0415 -- Deferred.\n    return a.b.X, a.Y\n"
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        "def f():\n"
        "    import a  # noqa: PLC0415 -- Deferred.\n"
        "    from a import (  # noqa: PLC0415 -- Deferred.\n"
        "        b,\n"
        "    )\n"
        "    return b.X, a.Y\n"
    )


def test_dotted_import_fix_keeps_a_one_line_block_on_its_line(tmp_path: Path) -> None:
    """A newline after ``if X: import a`` would move the second import out of the if."""
    body = "if X: import a.b\ny = a.b.Z, a.other\n"
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        "if X: import a; from a import b\ny = b.Z, a.other\n"
    )


@pytest.mark.parametrize(
    ("body", "fixed"),
    [
        # The comprehension's ``a`` is its own; ``f``'s ``a.b.Y`` is the import's.
        (
            (
                "import a.b\nx = a.b.X\ndef f(xs):\n    y = a.b.Y\n"
                "    return [a for a in xs]\n"
            ),
            (
                "from a import b\nx = b.X\ndef f(xs):\n    y = b.Y\n"
                "    return [a for a in xs]\n"
            ),
        ),
        (
            "import a.b\nclass C:\n    y = a.b.Y\n    z = {a: 1 for a in range(2)}\n",
            "from a import b\nclass C:\n    y = b.Y\n    z = {a: 1 for a in range(2)}\n",
        ),
        # A walrus in a comprehension binds ``f``'s ``a``, which ``f`` then reads.
        (
            (
                "import a.b\nx = a.b.X\ndef f(xs):\n"
                "    if any((a := v) for v in xs):\n        return a.b.Y\n"
            ),
            (
                "from a import b\nx = b.X\ndef f(xs):\n"
                "    if any((a := v) for v in xs):\n        return a.b.Y\n"
            ),
        ),
    ],
)
def test_dotted_import_fix_scopes_comprehension_targets(
    tmp_path: Path,
    body: str,
    fixed: str,
) -> None:
    assert _fixed(tmp_path, DottedImportRule(), body) == fixed


@pytest.mark.parametrize(
    ("body", "fixed"),
    [
        # Each branch's rewrite needs its own ``import a`` for ``a.z``/``a.d``.
        (
            (
                "def f(flag):\n    if flag:\n        import a.b\n        return a.b.X\n"
                "    import a.c\n    return a.c.Y, a.z\n"
            ),
            (
                "def f(flag):\n    if flag:\n        import a\n        from a import b\n"
                "        return b.X\n    import a\n    from a import c\n"
                "    return c.Y, a.z\n"
            ),
        ),
        (
            (
                "if X:\n    import a.b\n    y = a.b.Y\nelse:\n    import a.c\n"
                "    y = a.c.Y, a.d\n"
            ),
            (
                "if X:\n    import a\n    from a import b\n    y = b.Y\nelse:\n"
                "    import a\n    from a import c\n    y = c.Y, a.d\n"
            ),
        ),
        # An import in the other branch, or one that runs later, binds nothing here.
        (
            "if X:\n    import a.b\n    y = a.b.Y, a.z\nelse:\n    import a.c\n",
            (
                "if X:\n    import a\n    from a import b\n    y = b.Y, a.z\nelse:\n"
                "    import a.c\n"
            ),
        ),
        (
            (
                "def f():\n    import a.b\n    y = a.other\n    x = a.b.X\n"
                "    import a.c\n    return x, y\n"
            ),
            (
                "def f():\n    import a\n    from a import b\n    y = a.other\n"
                "    x = b.X\n    import a.c\n    return x, y\n"
            ),
        ),
        # Only the runtime import can carry ``a`` to ``a.z``; the guarded one stays.
        (
            (
                "if TYPE_CHECKING:\n    import a.b\nimport a.c\ndef f() -> None:\n"
                "    return a.c.g(), a.z, a.b.T\n"
            ),
            (
                "if TYPE_CHECKING:\n    import a.b\nimport a\nfrom a import c\n"
                "def f() -> None:\n    return c.g(), a.z, a.b.T\n"
            ),
        ),
        # Annotations that never run may read a guarded import.
        (
            (
                "from __future__ import annotations\nimport a\nif TYPE_CHECKING:\n"
                "    import a.b\ndef f(x: a.b.T) -> a.b.T:\n    return a.c\n"
            ),
            (
                "from __future__ import annotations\nimport a\nif TYPE_CHECKING:\n"
                "    from a import b\ndef f(x: b.T) -> b.T:\n    return a.c\n"
            ),
        ),
        # Swapping a submodule's attribute, not the submodule, is still seen.
        (
            "import a.b\nx = a.b.X\npatch('a.b.X', fake)\n",
            "from a import b\nx = b.X\npatch('a.b.X', fake)\n",
        ),
        # A loop variable in a sibling function is no F402 and reads no ``b``.
        (
            (
                "def g():\n    import a.b\n    return a.b.X\ndef f(xs):\n"
                "    for b in xs:\n        print(b)\n"
            ),
            (
                "def g():\n    from a import b\n    return b.X\ndef f(xs):\n"
                "    for b in xs:\n        print(b)\n"
            ),
        ),
        # Class keywords and type-parameter bounds run in the enclosing scope.
        (
            "import a.b\nclass C(metaclass=a.b.Meta):\n    pass\n",
            "from a import b\nclass C(metaclass=b.Meta):\n    pass\n",
        ),
        (
            "import a.b\ndef f[T: a.b.X](x: T) -> T:\n    return x\n",
            "from a import b\ndef f[T: b.X](x: T) -> T:\n    return x\n",
        ),
    ],
)
def test_dotted_import_fix_binds_the_root_on_every_path_that_reads_it(
    tmp_path: Path,
    body: str,
    fixed: str,
) -> None:
    """A rewrite holds only where the read can run after the rewritten import."""
    assert _fixed(tmp_path, DottedImportRule(), body) == fixed


def test_dotted_import_suggests_an_alias_for_a_taken_name(tmp_path: Path) -> None:
    (finding,) = _findings(tmp_path, DottedImportRule(), "import pkg.input\n")
    assert finding.message == (
        "import pkg.input binds pkg; write `from pkg import input as "
        "<qualifier>_input` (not autofixed: `input` is a builtin)"
    )


@pytest.mark.parametrize(
    ("body", "reasons"),
    [
        (
            "import loop.a.b.c\nc = object()\nvalue = loop.a.b.c.X\n",
            [(1, "`c` already names something else where it would be read")],
        ),
        (
            # ``except ... as err`` unbinds ``err`` when the handler ends.
            (
                "import a.err\nx = a.err.X\n"
                "try:\n    pass\nexcept OSError as err:\n    pass\n"
            ),
            [(1, "`err` already names something else where it would be read")],
        ),
        (
            # ``f``'s parameter would shadow the rewritten read inside ``f``.
            (
                "import urllib.request\ndef f(request):\n"
                "    return urllib.request.urlopen(request)\n"
            ),
            [(1, "`request` already names something else where it would be read")],
        ),
        (
            # Legal (annotations read the module), but ``pool: pool.Pool`` misleads.
            (
                "import multiprocessing.pool\n"
                "def f(pool: multiprocessing.pool.Pool) -> None:\n    del pool\n"
            ),
            [(1, "`pool` already names something else where it would be read")],
        ),
        (
            # Ruff's F402 rejects a loop variable shadowing an import, in any scope.
            "import a.b\nx = a.b.X\ndef f(xs):\n    for b in xs:\n        del b\n",
            [(1, "`b` already names something else where it would be read")],
        ),
        (
            # Another binder of ``a`` can carry ``a.b`` where this import has not run.
            "import a\nif TYPE_CHECKING:\n    import a.b\ndef f():\n    return a.b.X\n",
            [(3, "a read of `a.b` can run where this import has not")],
        ),
        (
            "import a\nx = a.b.X\nimport a.b\n",
            [(3, "a read of `a.b` can run where this import has not")],
        ),
        (
            # ``f`` would keep reading the module bound at import, not the patch.
            (
                "import a.b\ndef f():\n    return a.b.X\n"
                "def test_f(monkeypatch):\n    monkeypatch.setattr(a, 'b', fake)\n"
            ),
            [
                (
                    1,
                    (
                        "`a.b` is swapped on `a` at runtime (setattr or patch), which "
                        "a bound `b` would not see"
                    ),
                ),
            ],
        ),
        (
            "import a.b\nx = a.b.X\npatch('a.b', fake)\n",
            [
                (
                    1,
                    (
                        "`a.b` is swapped on `a` at runtime (setattr or patch), which "
                        "a bound `b` would not see"
                    ),
                ),
            ],
        ),
        (
            # The fix honours the opt-out instead of rewriting under it.
            (
                "# house-ignore[dotted-import] -- Patched by path elsewhere.\n"
                "import a.b\nx = a.b.X\n"
            ),
            [(2, "it carries a house-ignore[dotted-import] marker")],
        ),
        (
            (
                "import a.b  # house-ignore[digits, dotted-import] -- Patched.\n"
                "x = a.b.X\n"
            ),
            [(1, "it carries a house-ignore[dotted-import] marker")],
        ),
        (
            "import a.b\nx = a.b.X\nmatch v:\n    case a:\n        pass\n",
            [(1, "`a` is also bound another way in this scope")],
        ),
        (
            "import a.b\nx = a.b.X\nmatch v:\n    case [*a]:\n        pass\n",
            [(1, "`a` is also bound another way in this scope")],
        ),
        (
            "import a.b\nx = a.b.X\nmatch v:\n    case {**a}:\n        pass\n",
            [(1, "`a` is also bound another way in this scope")],
        ),
        (
            # A local ``import a.b`` would capture ``f``'s read of the global ``b``.
            "b = 1\ndef f():\n    import a.b\n    return a.b.X, b\n",
            [(3, "`b` already names something else where it would be read")],
        ),
        (
            (
                "import loop.g.a.kernel\nimport loop.g.b.kernel\n"
                "x = loop.g.a.kernel.X\ny = loop.g.b.kernel.Y\n"
            ),
            [
                (1, "another import also ends in `kernel`"),
                (2, "another import also ends in `kernel`"),
            ],
        ),
        ("import pkg.input\nx = pkg.input.X\n", [(1, "`input` is a builtin")]),
        (
            "try:\n    import a.b\nexcept ImportError:\n    a = None\nx = a.b.X\n",
            [(2, "`a` is also bound another way in this scope")],
        ),
        (
            "import a.b\nx = a.b.X\ndef f():\n    global a\n    a = None\n",
            [(1, "`a` is declared global or nonlocal")],
        ),
        (
            "import a.b\na.b = stub\nx = a.b.X\n",
            [(1, "`a.b` itself is assigned or deleted")],
        ),
        (
            "class C:\n    import a.b\n    x = a.b.X\n",
            [(2, "it is in a class body")],
        ),
        (
            "import loop.g.pkg\nimport loop.g.pkg.kernel\n",
            [
                (1, _UNREAD.format("loop.g.pkg")),
                (2, _UNREAD.format("loop.g.pkg.kernel")),
            ],
        ),
    ],
)
def test_dotted_import_leaves_what_it_cannot_rewrite_safely(
    tmp_path: Path,
    body: str,
    reasons: list[tuple[int, str]],
) -> None:
    """Each finding the fix leaves says why, so the author knows what to change."""
    assert _unfixed_reasons(tmp_path, body) == reasons


def test_dotted_import_fix_leaves_an_unread_import_beside_a_rewrite(
    tmp_path: Path,
) -> None:
    body = "import loop.g.pkg\nimport loop.g.pkg.kernel\nx = loop.g.pkg.kernel.X\n"
    fixed = _fixed(tmp_path, DottedImportRule(), body)
    assert fixed == "import loop.g.pkg\nfrom loop.g.pkg import kernel\nx = kernel.X\n"
    assert _unfixed_reasons(tmp_path, fixed) == [
        (1, _UNREAD.format("loop.g.pkg")),
    ]


# -- import-hoist ------------------------------------------------------------


def test_staticmethod_allows_autograd_callbacks(tmp_path: Path) -> None:
    body = (
        "import torch\n\n"
        "class F(torch.autograd.Function):\n"
        "    forward = staticmethod(_forward)\n"
        "    backward = staticmethod(_backward)\n"
    )
    assert not _findings(tmp_path, StaticMethodRule(), body)


def test_staticmethod_exempts_only_autograd_forward_and_backward(
    tmp_path: Path,
) -> None:
    body = (
        "class F(Function):\n"
        "    @staticmethod\n"
        "    def forward(ctx, x): ...\n"
        "    @staticmethod\n"
        "    def backward(ctx, g): ...\n"
        "    @staticmethod\n"
        "    def helper(x): ...\n"
        "class G:\n"
        "    @staticmethod\n"
        "    def forward(x): ...\n"
        "    @staticmethod\n"
        "    async def backward(x): ...\n"
        "    @other\n"
        "    def h(x): ...\n"
        "    a: Callable = staticmethod(f)\n"
        "    b, c = staticmethod(f), 1\n"
    )
    tail = "; use @classmethod (STYLE.md) or a module-level function"
    assert [
        (f.line, f.message) for f in _findings(tmp_path, StaticMethodRule(), body)
    ] == [
        (6, f"@staticmethod on helper{tail}"),
        (9, f"@staticmethod on forward{tail}"),
        (11, f"@staticmethod on backward{tail}"),
        (15, f"@staticmethod on a{tail}"),
    ]


def test_import_hoist_moves_unsuppressed_imports(tmp_path: Path) -> None:
    body = "def f() -> None:\n    import torch\n    torch.zeros(1)\n"
    rule = ImportHoistRule()

    findings = _findings(tmp_path, rule, body)
    assert [finding.line for finding in findings] == [2]
    assert _fixed(tmp_path, rule, body) == (
        "import torch\n\n\ndef f() -> None:\n    torch.zeros(1)\n"
    )


def test_import_hoist_moves_several_and_multiline_imports(tmp_path: Path) -> None:
    """Deletions run bottom-up and a wrapped import moves whole."""
    body = (
        "def f():\n"
        "    from os import (\n"
        "        path,\n"
        "        sep,\n"
        "    )\n"
        "    return path, sep\n"
        "\n"
        "\n"
        "def g():\n"
        "    import sys\n"
        "    return sys\n"
    )
    assert _fixed(tmp_path, ImportHoistRule(), body) == (
        "from os import (\n"
        "    path,\n"
        "    sep,\n"
        ")\n"
        "import sys\n"
        "\n"
        "\n"
        "def f():\n"
        "    return path, sep\n"
        "\n"
        "\n"
        "def g():\n"
        "    return sys\n"
    )


def test_import_hoist_inserts_after_docstring_and_imports(tmp_path: Path) -> None:
    """Moved imports join the existing block; whitespace at the seam is rebuilt."""
    body = (
        '"""Doc."""\n'
        "\n"
        "import os  \n"
        " \n"
        "\n"
        "\t\n"
        "def f():\n"
        "    import sys  \n"
        "    return os, sys\n"
    )
    assert _findings(tmp_path, ImportHoistRule(), body) == (
        Finding(
            path=tmp_path / "mod.py",
            line=8,
            rule="import-hoist",
            message=(
                "import sys is outside module scope; add `# noqa: PLC0415` "
                "only when deferral is intentional"
            ),
        ),
    )
    assert _fixed(tmp_path, ImportHoistRule(), body) == (
        '"""Doc."""\n\nimport os\n\nimport sys\n\n\ndef f():\n    return os, sys\n'
    )


@pytest.mark.parametrize(
    "body",
    [
        "def f(x):\n    if x: import json\n    return 1\n",
        "def f():\n    import json; return json\n",
        "def f():\n    x = 1; import json\n    return json, x\n",
    ],
    ids=["after-colon", "before-statement", "after-statement"],
)
def test_import_hoist_leaves_an_import_sharing_its_line(
    tmp_path: Path,
    body: str,
) -> None:
    """Moving its whole line would move, or drop, the other statement too."""
    rule = ImportHoistRule()

    assert [f.line for f in _findings(tmp_path, rule, body)] == [2]
    assert _fixed(tmp_path, rule, body) is None


def test_import_hoist_leaves_a_dotted_type_checking_import(tmp_path: Path) -> None:
    body = "import typing\n\n\nif typing.TYPE_CHECKING:\n    import json\n"
    rule = ImportHoistRule()

    assert not _findings(tmp_path, rule, body)
    assert _fixed(tmp_path, rule, body) is None


def test_import_hoist_leaves_explicitly_suppressed_imports(tmp_path: Path) -> None:
    body = "def f() -> None:\n    import torch  # noqa: PLC0415\n    torch.zeros(1)\n"
    rule = ImportHoistRule()

    assert not _findings(tmp_path, rule, body)
    assert _fixed(tmp_path, rule, body) is None


def test_import_hoist_leaves_module_optional_imports(tmp_path: Path) -> None:
    body = "try:\n    import torch\nexcept ImportError:\n    torch = None\n"
    rule = ImportHoistRule()

    assert not _findings(tmp_path, rule, body)
    assert _fixed(tmp_path, rule, body) is None


def test_import_hoist_does_not_break_single_statement_try_blocks(
    tmp_path: Path,
) -> None:
    body = (
        "def f() -> None:\n"
        "    try:\n"
        "        import torch\n"
        "    except ImportError:\n"
        "        return\n"
    )
    rule = ImportHoistRule()

    assert _findings(tmp_path, rule, body)
    assert _fixed(tmp_path, rule, body) is None


def test_import_hoist_leaves_multiline_suppressed_imports(tmp_path: Path) -> None:
    body = (
        "def f() -> None:\n"
        "    try:\n"
        "        from torch.utils.tensorboard import (  # noqa: PLC0415 -- Optional.\n"
        "            SummaryWriter,\n"
        "        )\n"
        "    except ImportError:\n"
        "        return\n"
    )
    rule = ImportHoistRule()

    assert not _findings(tmp_path, rule, body)
    assert _fixed(tmp_path, rule, body) is None


# -- nested-def --------------------------------------------------------------


def test_long_nested_def_is_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        NestedDefRule(),
        "def outer() -> None:\n"
        "    def inner() -> int:\n"
        "        a = 1\n"
        "        b = 2\n"
        "        return a + b\n"
        "    def three() -> int:\n"
        "        a = 1\n"
        "        return a\n"
        "async def aouter() -> None:\n"
        "    async def ainner() -> int:\n"
        "        a = 1\n"
        "        b = 2\n"
        "        c = 3\n"
        "        return a + b + c\n",
    )
    assert [(f.line, f.message) for f in findings] == [
        (2, "nested function inner spans 4 lines (> 3); hoist it to module level"),
        (10, "nested function ainner spans 5 lines (> 3); hoist it to module level"),
    ]


def test_escaping_closures_are_clean(tmp_path: Path) -> None:
    """Returned, passed, or decorated: a closure, not a hideable helper."""
    assert not _findings(
        tmp_path,
        NestedDefRule(),
        "def a() -> object:\n"
        "    def inner() -> int:\n"
        "        x = 1\n"
        "        y = 2\n"
        "        return x + y\n"
        "    return inner\n\n"
        "def b() -> None:\n"
        "    def cb() -> int:\n"
        "        x = 1\n"
        "        y = 2\n"
        "        return x + y\n"
        "    register(cb)\n\n"
        "def c() -> None:\n"
        "    @deco\n"
        "    def wrapped() -> int:\n"
        "        x = 1\n"
        "        y = 2\n"
        "        return x + y\n",
    )


def test_short_nested_def_and_nested_class_method_are_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        NestedDefRule(),
        "def outer() -> None:\n"
        "    def inner() -> int:\n"
        "        return 1\n\n"
        "    class K:\n"
        "        def m(self) -> int:\n"
        "            a = 1\n"
        "            b = 2\n"
        "            return a + b\n",
    )


# -- getlogger-name ----------------------------------------------------------


def test_string_logger_name_is_reported(tmp_path: Path) -> None:
    assert _findings(tmp_path, GetLoggerNameRule(), 'log = logging.getLogger("x")\n')


def test_dunder_name_logger_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        GetLoggerNameRule(),
        "log = logging.getLogger(__name__)\n",
    )


def test_getlogger_check_reports_only_logger_bindings(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        GetLoggerNameRule(),
        'import logging\nx: int\ny = 1\nz = make()\nlog = logging.getLogger("x")\n',
    )
    assert [f.line for f in findings] == [5]


# -- nested-try --------------------------------------------------------------


def test_try_inside_try_body_is_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        NestedTryRule(),
        "try:\n    try:\n        pass\n    except A:\n        pass\nexcept B:\n    pass\n"
        "try:\n    x = 1\n    if x:\n        try:\n            pass\n        except C:\n"
        "            pass\nexcept D:\n    pass\n",
    )
    message = (
        "try nested inside a try body; make them sequential (STYLE.md Error Handling)"
    )
    assert [(f.line, f.message) for f in findings] == [(2, message), (11, message)]


def test_try_in_handler_and_sequential_tries_are_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        NestedTryRule(),
        "try:\n    pass\nexcept A:\n    try:\n        pass\n    except B:\n        pass\n"
        "try:\n    pass\nexcept C:\n    pass\n",
    )


def test_try_finally_around_or_inside_a_handler_is_clean(tmp_path: Path) -> None:
    """``finally`` is resource scoping, not a second handler."""
    assert not _findings(
        tmp_path,
        NestedTryRule(),
        "try:\n    try:\n        pass\n    except A:\n        pass\nfinally:\n    pass\n"
        "try:\n    try:\n        pass\n    finally:\n        pass\nexcept B:\n    pass\n",
    )


# -- negative-bool-param -----------------------------------------------------


def test_negative_bool_param_is_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        NegativeBoolParamRule(),
        "def f(*, no_cache: bool = False, dont_x: int = 0): ...\n"
        "async def g(\n    not_ready: bool,\n    without_y: bool,\n    skip_z: bool,\n): ...\n",
    )
    assert [(f.line, f.message) for f in findings] == [
        (
            line,
            f"negative boolean parameter {name}; name the affirmative (True = do it)",
        )
        for line, name in ((1, "no_cache"), (3, "not_ready"), (4, "without_y"))
    ]


def test_affirmative_bool_and_untyped_negative_are_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        NegativeBoolParamRule(),
        "def f(*, use_cache: bool = True, no_cache=None): ...\n",
    )


def test_a_name_ending_in_a_preposition_is_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        FluentNameRule(),
        "def schema_of(kind): ...\n"
        "class C:\n"
        "    async def _label_for(self, parent): ...\n"
        "def lexer_from(path): ...\n",
    )
    assert [(f.line, f.message) for f in findings] == [
        (
            1,
            (
                "schema_of names how a call reads, not what it returns; name the"
                " thing (STYLE.md Code Style)"
            ),
        ),
        (
            3,
            (
                "_label_for names how a call reads, not what it returns; name the"
                " thing (STYLE.md Code Style)"
            ),
        ),
    ]


def test_a_name_a_base_fixes_or_with_a_preposition_inside_is_clean(
    tmp_path: Path,
) -> None:
    assert not _findings(
        tmp_path,
        FluentNameRule(),
        "from typing import override\n"
        "class C(B):\n"
        "    @override\n"
        "    def label_for(self): ...\n"
        "def wait_for_ready(): ...\n"
        "async def _wait_for(hold): ...\n"
        "def test_the_lines_asked_for(): ...\n"
        "def offer(): ...\n"
        "def __for__(self): ...\n",
    )


# -- all-reexport ------------------------------------------------------------


def test_reexported_import_in_all_is_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        AllReexportRule(),
        "import x.y.z\n"
        "import pkg.mod as m\n"
        "from a import b\n"
        "__other__ = ['b']\n"
        "x.__all__ = ['b']\n"
        "__all__ = (\n"
        "    'b',\n"
        "    'c',\n"
        "    'z',\n"
        "    'm',\n"
        "    'y',\n"
        ")\n"
        "def c(): ...\n",
    )
    tail = "; only a package __init__ re-exports (STYLE.md)"
    assert [(f.line, f.message) for f in findings] == [
        (7, f"__all__ re-exports imported 'b'{tail}"),
        (9, f"__all__ re-exports imported 'z'{tail}"),
        (10, f"__all__ re-exports imported 'm'{tail}"),
    ]


def test_init_and_conftest_may_reexport(tmp_path: Path) -> None:
    body = 'from a import b\n\n__all__ = ["b"]\n'
    assert not _findings(tmp_path, AllReexportRule(), body, name="__init__.py")
    assert not _findings(tmp_path, AllReexportRule(), body, name="conftest.py")


# -- init-facade -------------------------------------------------------------


def test_def_in_init_is_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        InitFacadeRule(),
        '"""Pkg."""\n\nfrom .a import b\n\n__all__ = ["b"]\n\n\ndef helper(): ...\n',
        name="__init__.py",
    )
    assert [f.line for f in findings] == [8]


def test_pure_facade_init_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        InitFacadeRule(),
        '"""Pkg."""\n\nfrom .a import b\n\n__all__ = ["b"]\n',
        name="__init__.py",
    )


def test_init_facade_skips_unparsable_init(tmp_path: Path) -> None:
    assert _findings(tmp_path, InitFacadeRule(), "def f(\n", name="__init__.py") == ()


# -- def-order ---------------------------------------------------------------


def test_public_after_private_is_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        DefOrderRule(),
        "def _h(): ...\n\n\ndef api(): ...\n",
    )
    assert [f.line for f in findings] == [1]


def test_public_then_private_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        DefOrderRule(),
        "def api(): ...\n\n\ndef _h(): ...\n",
    )


def test_private_named_in_a_public_signature_may_precede_it(tmp_path: Path) -> None:
    """A signature evaluates at definition time, so the name must exist first."""
    assert not _findings(
        tmp_path,
        DefOrderRule(),
        "class _Bindings: ...\n\n\ndef api(b: _Bindings) -> None: ...\n",
    )


def test_private_used_only_in_a_body_is_no_excuse(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        DefOrderRule(),
        "def _h(): ...\n\n\ndef api() -> None:\n    _h()\n",
    )
    assert [f.line for f in findings] == [1]


def test_private_needed_by_a_constant_or_class_attribute_may_precede(
    tmp_path: Path,
) -> None:
    assert not _findings(
        tmp_path,
        DefOrderRule(),
        "def _build(): ...\n\n\nTABLE = _build()\n\n\n"
        "class _Spec: ...\n\n\nclass K:\n    spec = _Spec()\n\n\n"
        "def _make(): ...\n\n\ndef _bind():\n    reg(_make())\n\n\n_bind()\n",
    )


def test_private_called_by_a_def_under_a_module_if_may_precede(
    tmp_path: Path,
) -> None:
    """A decorator under ``if HAVE_X:`` runs at import; its callee's callees are needed."""
    assert not _findings(
        tmp_path,
        DefOrderRule(),
        "def _pinned(): ...\n\n\n"
        "if HAVE_TRITON:\n\n"
        "    def _configs():\n        return _pinned()\n\n"
        "    @autotune(configs=_configs())\n"
        "    def kernel(): ...\n\n\n"
        "def api(): ...\n",
    )


def test_misplaced_privates_are_moved_to_the_end(tmp_path: Path) -> None:
    fixed = _fixed(
        tmp_path,
        DefOrderRule(),
        '"""Doc."""\n\n\n'
        "# Why h exists.\n"
        "def _h(): ...\n\n\n"
        "def api() -> None:\n    _h()\n\n\n"
        "def _k(): ...\n\n\n"
        "def other() -> None: ...\n\n\n"
        'if __name__ == "__main__":\n    api()\n',
    )
    assert fixed == (
        '"""Doc."""\n\n\n'
        "def api() -> None:\n    _h()\n\n\n"
        "def other() -> None: ...\n\n\n"
        "# Why h exists.\n"
        "def _h(): ...\n\n\n"
        "def _k(): ...\n\n\n"
        'if __name__ == "__main__":\n    api()\n'
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            '"""Doc."""\n\n\ndef _a(): ... \n\n\ndef api() -> None:\n    _a() \n',
            '"""Doc."""\n\n\ndef api() -> None:\n    _a() \n\n\ndef _a(): ... \n',
            id="trailing-whitespace",
        ),
        pytest.param(
            '"""Doc."""\n\n\ndef _a(): return X\n\n\ndef api(): return _a() or X\n',
            '"""Doc."""\n\n\ndef api(): return _a() or X\n\n\ndef _a(): return X\n',
            id="code-ends-in-X",
        ),
    ],
)
def test_misplaced_privates_move_verbatim_without_a_main_guard(
    tmp_path: Path,
    source: str,
    expected: str,
) -> None:
    """With no guard the privates end the file, their text untouched."""
    assert _fixed(tmp_path, DefOrderRule(), source) == expected


def test_type_checking_import_is_not_nested(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text("if TYPE_CHECKING:\n    import a\n", encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert houselint_rules._nested_imports(unit) == ()


# -- module-side-effects -----------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "mpl.use('Agg')\n",
        "for r in routes():\n    app.add(r)\n",
        "os.environ['X'] = '1'\n",
        "if sys.platform == 'darwin':\n    x = 1\n",
        "sys.path.insert(0, 'x')\n",
        "df = pd.read_csv('a.csv')\n",
        "def _go():\n    pass\n_go(1)\n",
        "def _go():\n    pass\nX = _go(load())\n",
    ],
)
def test_module_scope_action_is_reported(tmp_path: Path, body: str) -> None:
    assert _findings(tmp_path, ModuleSideEffectsRule(), body)


def test_import_time_consistency_checks_and_local_derivations_are_clean(
    tmp_path: Path,
) -> None:
    """Asserting two tables agree, or deriving one from another, declares."""
    assert not _findings(
        tmp_path,
        ModuleSideEffectsRule(),
        "assert set(A) == set(B), 'drift'\n"
        "if A <= B:\n    raise RuntimeError('x')\n"
        "if sys.platform == 'darwin':\n    def f(): ...\nelse:\n    def f(): ...\n"
        "try:\n    from x import Y\nexcept ImportError:\n    Y = None\n"
        "def _table():\n    return {}\n"
        "TABLE = _table()\n"
        "def _bind():\n    pass\n"
        "_bind()\n",
    )


def test_declarative_module_scope_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        ModuleSideEffectsRule(),
        '"""Doc."""\n'
        "from __future__ import annotations\n"
        "import logging\n"
        "from typing import TYPE_CHECKING, Final\n"
        "if TYPE_CHECKING:\n    import torch\n"
        "try:\n    import jax\nexcept ImportError:\n    pass\n"
        "logger = logging.getLogger(__name__)\n"
        "_CWD: Final = Path(__file__).resolve().parent\n"
        "MAX = 5\n"
        "NAMES = ('a', 'b')\n"
        "_RE = re.compile('x')\n"
        "_LOCK = threading.Lock()\n"
        "router = APIRouter()\n"
        "type Alias = int | str\n"
        "IDS = [n for n in NAMES]\n"
        '"""Names in order."""\n'
        "_building = ContextVar[bool]('b', default=False)\n"
        "_traces = dict[str, list[str]]()\n"
        "class K: ...\n"
        "def f(): ...\n"
        'if __name__ == "__main__":\n    f()\n',
    )


def test_all_shape_rules_are_registered() -> None:
    names = {r.name for r in houselint.default_rules()}
    assert {"underscore-param", "cli-shape", "comment-sentence", "mkdir-mode"} <= names


# -- underscore-param --------------------------------------------------------


def test_underscore_param_is_reported_and_fixed(tmp_path: Path) -> None:
    body = 'def f(x: int, _y: int) -> int:\n    """Doc."""\n    return x\n'
    assert [f.line for f in _findings(tmp_path, UnderscoreParamRule(), body)] == [1]
    assert _fixed(tmp_path, UnderscoreParamRule(), body) == (
        'def f(x: int, y: int) -> int:\n    """Doc."""\n    del y\n    return x\n'
    )


def test_underscore_param_fix_every_body_shape(tmp_path: Path) -> None:
    """Several params, several defs, docstring-only, wrapped signature, async."""
    body = (
        "def a(_x, y, _z):\n"
        "    return y\n"
        "def b(\n"
        "    _p: int,\n"
        "    q: int,\n"
        ") -> int:\n"
        '    """Only a doc."""\n'
        "async def c(_r, r2):\n"
        "    if r2:\n"
        "        return 1\n"
        "def d(_s, s):\n"
        "    return s\n"
        "def e(__t, _u):\n"
        "    return u\n"
    )
    findings = _findings(tmp_path, UnderscoreParamRule(), body)
    assert [(f.line, f.message) for f in findings] == [
        (
            line,
            (
                f"parameter {name} is underscore-prefixed; keep the name "
                f"{name.lstrip('_')} and `del` it (STYLE.md Deprecation)"
            ),
        )
        for line, name in (
            (1, "_x"),
            (1, "_z"),
            (4, "_p"),
            (8, "_r"),
            (11, "_s"),
            (13, "__t"),
            (13, "_u"),
        )
    ]
    assert _fixed(tmp_path, UnderscoreParamRule(), body) == (
        "def a(x, y, z):\n"
        "    del x, z\n"
        "    return y\n"
        "def b(\n"
        "    p: int,\n"
        "    q: int,\n"
        ") -> int:\n"
        '    """Only a doc."""\n'
        "    del p\n"
        "async def c(r, r2):\n"
        "    del r\n"
        "    if r2:\n"
        "        return 1\n"
        "def d(_s, s):\n"
        "    return s\n"
        "def e(t, _u):\n"
        "    del t\n"
        "    return u\n"
    )


def test_used_underscore_param_is_reported_not_fixed(tmp_path: Path) -> None:
    body = "def f(_y: int) -> int:\n    return _y\n"
    assert _findings(tmp_path, UnderscoreParamRule(), body)
    assert _fixed(tmp_path, UnderscoreParamRule(), body) is None


def test_bare_underscore_and_dunder_params_are_clean(tmp_path: Path) -> None:
    assert not _findings(tmp_path, UnderscoreParamRule(), "def f(_, __x__): ...\n")


@pytest.mark.parametrize(
    "body",
    [
        "class P(Protocol):\n    def f(self, _x: int) -> None: ...\n",
        "def f(_x: int) -> None: pass\n",
        'def f(_x: int) -> None: """Doc."""\n',
        "def f(\n    _x: int,\n): ...\n",
    ],
)
def test_underscore_param_on_a_one_line_def_is_reported_not_fixed(
    tmp_path: Path,
    body: str,
) -> None:
    """A body on the header line has no line for ``del``; the rewrite broke parsing."""
    assert _findings(tmp_path, UnderscoreParamRule(), body)
    assert _fixed(tmp_path, UnderscoreParamRule(), body) is None


def test_triton_jit_kernel_params_are_exempt(tmp_path: Path) -> None:
    """Triton's code generator has no ``del``, so a jitted body cannot honor the rule."""
    body = (
        "@triton.autotune(configs=[], key=['_k'])\n"
        "@triton.jit\n"
        "def kern(ptr, _k: tl.constexpr):\n"
        "    tl.store(ptr, 1)\n"
    )
    assert not _findings(tmp_path, UnderscoreParamRule(), body)
    assert _fixed(tmp_path, UnderscoreParamRule(), body) is None


def test_underscore_param_fix_replaces_a_lone_pass(tmp_path: Path) -> None:
    """The inserted ``del`` is a statement, so a placeholder ``pass`` must go."""
    body = "def f(_y: int) -> None:\n    pass\n"
    assert _fixed(tmp_path, UnderscoreParamRule(), body) == (
        "def f(y: int) -> None:\n    del y\n"
    )
    body = 'def f(_y: int) -> None:\n    """Doc."""\n    pass\n'
    assert _fixed(tmp_path, UnderscoreParamRule(), body) == (
        'def f(y: int) -> None:\n    """Doc."""\n    del y\n'
    )


# -- negated-compare ---------------------------------------------------------


def test_negated_single_compare_is_flipped(tmp_path: Path) -> None:
    body = "if not a <= 0:\n    pass\n"
    assert _findings(tmp_path, NegatedCompareRule(), body)
    assert _fixed(tmp_path, NegatedCompareRule(), body) == "if a > 0:\n    pass\n"


def test_negated_compare_against_a_name_is_reported_not_fixed(tmp_path: Path) -> None:
    """``not a <= b`` on sets is subset; on floats the flip drops NaN rejection."""
    body = "if not a <= b:\n    pass\n"
    assert _findings(tmp_path, NegatedCompareRule(), body)
    assert _fixed(tmp_path, NegatedCompareRule(), body) is None


def test_negated_chain_is_reported_not_fixed(tmp_path: Path) -> None:
    body = "if not 0 <= x <= 1:\n    pass\n"
    assert _findings(tmp_path, NegatedCompareRule(), body)
    assert _fixed(tmp_path, NegatedCompareRule(), body) is None


def test_negated_equality_is_ruffs(tmp_path: Path) -> None:
    assert not _findings(tmp_path, NegatedCompareRule(), "if not a == b:\n    pass\n")


def test_negated_compare_reports_and_flips_several_per_line(tmp_path: Path) -> None:
    body = (
        "x = 1\n"
        "y = (not a < 0, not b >= 2), not c > d\n"
        "z = not 0 <= e <= 1 or not f <= 3\n"
    )
    tail = (
        "; flip the operator (single) or split into `a < lo or a > hi` (chained); "
        "the negated form also rejects NaN, so prepend `math.isnan(a) or` "
        "if NaN can reach here"
    )
    assert [
        (f.line, f.message) for f in _findings(tmp_path, NegatedCompareRule(), body)
    ] == [
        (2, f"negated comparison `not a < 0`{tail}"),
        (2, f"negated comparison `not b >= 2`{tail}"),
        (2, f"negated comparison `not c > d`{tail}"),
        (3, f"negated comparison `not 0 <= e <= 1`{tail}"),
        (3, f"negated comparison `not f <= 3`{tail}"),
    ]
    assert _fixed(tmp_path, NegatedCompareRule(), body) == (
        "x = 1\ny = (a >= 0, b < 2), not c > d\nz = not 0 <= e <= 1 or f > 3\n"
    )


def test_negated_compare_against_negative_int_is_flipped(tmp_path: Path) -> None:
    body = "if not a < -1:\n    pass\n"
    assert _fixed(tmp_path, NegatedCompareRule(), body) == "if a >= -1:\n    pass\n"


# -- negated-ternary ---------------------------------------------------------


def test_negated_ternary_is_swapped(tmp_path: Path) -> None:
    body = "y = foo if not x else bar\n"
    assert _findings(tmp_path, NegatedTernaryRule(), body)
    assert _fixed(tmp_path, NegatedTernaryRule(), body) == "y = bar if x else foo\n"


def test_negated_ternary_reports_and_swaps_several_per_line(tmp_path: Path) -> None:
    body = (
        "x = 1\n"
        "y = (a if not p else b, c if not q.r else d), e if not (s or t) else g\n"
        "z = h if not f() else i\n"
    )
    message = "negated ternary; swap the arms and drop the `not`"
    assert [
        (f.line, f.message) for f in _findings(tmp_path, NegatedTernaryRule(), body)
    ] == [(2, message), (2, message), (3, message)]
    assert _fixed(tmp_path, NegatedTernaryRule(), body) == (
        "x = 1\n"
        "y = (b if p else a, d if q.r else c), e if not (s or t) else g\n"
        "z = i if f() else h\n"
    )


def test_affirmative_ternary_is_clean(tmp_path: Path) -> None:
    assert not _findings(tmp_path, NegatedTernaryRule(), "y = bar if x else foo\n")


# -- super-only --------------------------------------------------------------


def test_super_only_override_is_reported_and_deleted(tmp_path: Path) -> None:
    body = (
        "class B(A):\n"
        "    @override\n"
        "    def f(self, x: int) -> int:\n"
        '        """Doc."""\n'
        "        return super().f(x)\n\n"
        "    def g(self) -> int:\n"
        "        return 1\n"
    )
    assert [f.line for f in _findings(tmp_path, SuperOnlyRule(), body)] == [3]
    assert _fixed(tmp_path, SuperOnlyRule(), body) == (
        "class B(A):\n    def g(self) -> int:\n        return 1\n"
    )


def test_super_only_fix_deletes_several_overrides(tmp_path: Path) -> None:
    """Deletions run bottom-up, async included, and the last one ends the file."""
    body = (
        "class B(A):\n"
        "    @override\n"
        "    def f(self, x):\n"
        "        return super().f(x)\n"
        "\n"
        "    def keep(self):\n"
        "        return 1\n"
        "\n"
        "    async def g(self, y):\n"
        "        return await super().g(y)\n"
        "\n"
        "    def h(self):\n"
        "        return 2\n"
        "\n"
        "    def last(self):\n"
        "        return super().last()\n"
        "\n"
    )
    assert _findings(tmp_path, SuperOnlyRule(), body) == tuple(
        Finding(
            path=tmp_path / "mod.py",
            line=line,
            rule="super-only",
            message=f"{name} only calls super().{name}; delete the override",
        )
        for line, name in ((3, "f"), (9, "g"), (15, "last"))
    )
    assert _fixed(tmp_path, SuperOnlyRule(), body) == (
        "class B(A):\n"
        "    def keep(self):\n"
        "        return 1\n"
        "\n"
        "    def h(self):\n"
        "        return 2\n"
        "\n"
    )
    tail = "class B(A):\n    def k(self):\n        return 1\n\n    def f(self):\n        return super().f()\n"
    assert _fixed(tmp_path, SuperOnlyRule(), tail) == (
        "class B(A):\n    def k(self):\n        return 1\n\n"
    )


@pytest.mark.parametrize(
    "signature",
    ["self, x=2", "self, *, x=2", "self, x: int = 2"],
)
def test_an_override_that_sets_a_default_is_clean(
    tmp_path: Path,
    signature: str,
) -> None:
    """Deleting it would change what a call without ``x`` passes the parent."""
    forwarded = "x=x" if "*" in signature else "x"
    body = (
        f"class B(A):\n    def f({signature}):\n        return super().f({forwarded})\n"
    )
    assert _findings(tmp_path, SuperOnlyRule(), body) == ()
    assert _fixed(tmp_path, SuperOnlyRule(), body) is None


def test_override_that_adds_behavior_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        SuperOnlyRule(),
        "class B(A):\n    def f(self, x: int) -> int:\n        return super().f(x) + 1\n",
    )


def test_override_with_two_statements_is_clean(tmp_path: Path) -> None:
    body = (
        "class B(A):\n"
        "    def f(self, x: int) -> int:\n"
        "        log(x)\n"
        "        return super().f(x)\n"
    )
    assert _findings(tmp_path, SuperOnlyRule(), body) == ()
    assert _fixed(tmp_path, SuperOnlyRule(), body) is None


# -- tensor-value-guard ------------------------------------------------------


def test_item_in_guard_is_reported(tmp_path: Path) -> None:
    body = (
        "if bool((x <= 0).all()):\n"
        "    raise ValueError\n"
        "assert t.item() > 0 and u.any()\n"
        "while w.any():\n"
        "    raise ValueError\n"
        "y = 1 if v.all() else 2\n"
        "assert t.item(0), t.max()\n"
    )
    tail = (
        "() inside a guard synchronizes the device; let out-of-domain input "
        "yield NaN/inf (STYLE.md Error Handling)"
    )
    assert [
        (f.line, f.message) for f in _findings(tmp_path, TensorValueGuardRule(), body)
    ] == [(1, f".all{tail}"), (3, f".any{tail}"), (3, f".item{tail}")]


def test_item_outside_guard_is_clean(tmp_path: Path) -> None:
    assert not _findings(tmp_path, TensorValueGuardRule(), "v = t.item()\n")


def test_a_branch_that_acts_on_values_is_clean(tmp_path: Path) -> None:
    """``if done.any(): restart()`` is the algorithm, not a precondition."""
    assert not _findings(
        tmp_path,
        TensorValueGuardRule(),
        "if bool(done.any()):\n    restart(done)\nif finished.all():\n    break\n",
    )


# -- mkdir-mode --------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        '(config_dir() / "rekursiv-ai").mkdir(parents=True)\n',
        'Path("/opt/scratch/x").mkdir(mode=0o700)\n',
        "try:\n    os.umask(0o077)\n    f()\nfinally:\n    pass\n",
        'Path("out.txt").chmod(0o600)\n',
    ],
)
def test_mode_violations_are_reported(tmp_path: Path, body: str) -> None:
    assert _findings(tmp_path, MkdirModeRule(), body)


def test_mode_conventions_are_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        MkdirModeRule(),
        '(config_dir() / "rekursiv-ai").mkdir(parents=True, mode=0o700)\n'
        'Path("/opt/scratch/x").mkdir(parents=True)\n'
        "old = os.umask(0o077)\ntry:\n    f()\nfinally:\n    os.umask(old)\n"
        "script.chmod(0o755)\n",
    )


# -- returns-named -----------------------------------------------------------


def test_unnamed_returns_entry_is_reported(tmp_path: Path) -> None:
    long_entry = "An entry far longer than forty characters, so it is cut."
    body = (
        "def f() -> int:\n"
        '    """Do.\n\n    Returns:\n      The count.\n\n    """\n'
        "    return 1\n"
        "def g():\n"
        f'    """Do.\n\n    Yields:\n      {long_entry}\n      n: Named.\n\n    """\n'
        "    yield 1\n"
        "def h():\n"
        "    return 1\n"
        "def i():\n"
        '    """Do.\n\n    Args:\n      Unnamed arg.\n\n    """\n'
    )
    tail = "is not `name: description`"
    assert [
        (f.line, f.message) for f in _findings(tmp_path, ReturnsNamedRule(), body)
    ] == [
        (1, f"f: Returns: entry 'The count.' {tail}"),
        (9, f"g: Yields: entry {long_entry[:40]!r} {tail}"),
    ]


def test_private_defs_are_exempt_from_returns_naming(tmp_path: Path) -> None:
    body = 'def _f() -> int:\n    """Do.\n\n    Returns:\n      The count.\n\n    """\n    return 1\n'
    assert _findings(tmp_path, ReturnsNamedRule(), body) == ()
    assert _findings(tmp_path, ReturnsNamedRule(), body.replace("_f", "f")) != ()


def test_named_returns_entry_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        ReturnsNamedRule(),
        'def f() -> int:\n    """Do.\n\n    Returns:\n      count: The count.\n\n    """\n    return 1\n',
    )


def test_returns_entry_continuation_joins_its_entry(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        ReturnsNamedRule(),
        'def f() -> int:\n    """Do.\n\n    Returns:\n      unnamed entry\n'
        '        continued.\n\n    """\n    return 1\n',
    )
    assert [f.message for f in findings] == [
        "f: Returns: entry 'unnamed entry continued.' is not `name: description`",
    ]


# -- args-section ------------------------------------------------------------


_LONG_UNDOCUMENTED: Final = (
    'def f(x: int) -> int:\n    """Do."""\n'
    + "".join(f"    y{i} = x + {i}\n" for i in range(6))
    + "    return y5\n"
)


def test_missing_args_is_reported(tmp_path: Path) -> None:
    findings = _findings(tmp_path, ArgsSectionRule(), _LONG_UNDOCUMENTED)
    assert [f.message for f in findings] == ["f docstring lacks Args:"]


def test_missing_returns_is_reported(tmp_path: Path) -> None:
    findings = _findings(tmp_path, ReturnsSectionRule(), _LONG_UNDOCUMENTED)
    assert [f.message for f in findings] == ["f docstring lacks Returns:"]


def test_args_section_ignores_a_missing_returns(tmp_path: Path) -> None:
    """The two rules split so a tree can be exempt from one alone."""
    src = (
        'def f(x: int) -> int:\n    """Do.\n\n    Args:\n      x: X.\n\n    """\n'
        + "".join(f"    y{i} = x + {i}\n" for i in range(6))
        + "    return y5\n"
    )
    assert not _findings(tmp_path, ArgsSectionRule(), src)
    assert _findings(tmp_path, ReturnsSectionRule(), src)


def test_returns_section_ignores_a_missing_args(tmp_path: Path) -> None:
    src = (
        'def f(x: int) -> int:\n    """Do.\n\n    Returns:\n      y: Y.\n\n    """\n'
        + "".join(f"    y{i} = x + {i}\n" for i in range(6))
        + "    return y5\n"
    )
    assert _findings(tmp_path, ArgsSectionRule(), src)
    assert not _findings(tmp_path, ReturnsSectionRule(), src)


def test_a_generator_wants_yields(tmp_path: Path) -> None:
    src = (
        'def f() -> Iterator[int]:\n    """Do.\n\n    More prose.\n    """\n'
        "    yield 1\n"
    )
    findings = _findings(tmp_path, ReturnsSectionRule(), src)
    assert [f.message for f in findings] == ["f docstring lacks Yields:"]
    assert not _findings(
        tmp_path,
        ReturnsSectionRule(),
        'def f() -> Iterator[int]:\n    """Do.\n\n    Yields:\n      n: N.\n\n    """\n'
        "    yield 1\n",
    )


@pytest.mark.parametrize("rule", [ArgsSectionRule(), ReturnsSectionRule()])
def test_short_public_def_may_carry_a_one_line_docstring(
    tmp_path: Path,
    rule: Rule,
) -> None:
    body = "".join(f"    y{i} = x + {i}\n" for i in range(4))
    assert not _findings(
        tmp_path,
        rule,
        f'def f(x: int) -> int:\n    """Do."""\n{body}    return y3\n',
    )


@pytest.mark.parametrize("rule", [ArgsSectionRule(), ReturnsSectionRule()])
def test_short_def_with_a_multi_line_docstring_is_held_to_sections(
    tmp_path: Path,
    rule: Rule,
) -> None:
    findings = _findings(
        tmp_path,
        rule,
        'def f(x: int) -> int:\n    """Do.\n\n    More prose.\n    """\n    return x\n',
    )
    assert len(findings) == 1


@pytest.mark.parametrize(
    ("rule", "strict"),
    [
        (ArgsSectionRule(), ArgsSectionRule(max_oneliner_body=0)),
        (ReturnsSectionRule(), ReturnsSectionRule(max_oneliner_body=0)),
    ],
)
def test_oneliner_threshold_is_a_setting(
    tmp_path: Path,
    rule: Rule,
    strict: Rule,
) -> None:
    src = 'def f(x: int) -> int:\n    """Do."""\n    return x\n'
    assert not _findings(tmp_path, rule, src)
    assert _findings(tmp_path, strict, src)


@pytest.mark.parametrize("rule", [ArgsSectionRule(), ReturnsSectionRule()])
def test_full_docstring_and_private_def_are_clean(tmp_path: Path, rule: Rule) -> None:
    assert not _findings(
        tmp_path,
        rule,
        'def f(x: int) -> int:\n    """Do.\n\n    Args:\n      x: X.\n\n    Returns:\n      y: Y.\n\n    """\n    return x\n\n'
        'def _g(x: int) -> int:\n    """Do."""\n    return x\n\n'
        'def h() -> None:\n    """Do."""\n',
    )


@pytest.mark.parametrize("decorator", ["overload", "value.setter", "override"])
def test_overload_setter_and_override_defs_are_exempt(
    tmp_path: Path,
    decorator: str,
) -> None:
    undecorated = 'def f(x: int) -> int:\n    """Do.\n\n    More prose.\n    """\n'
    body = f"@{decorator}\n{undecorated}"
    assert _findings(tmp_path, ArgsSectionRule(), body) == ()
    findings = _findings(tmp_path, ArgsSectionRule(), undecorated)
    assert [f.message for f in findings] == ["f docstring lacks Args:"]


# -- private-oneliner --------------------------------------------------------


def test_multiline_private_docstring_is_reported(tmp_path: Path) -> None:
    body = (
        'def _f() -> None:\n    """Do.\n\n    More.\n    """\n'
        'async def _g():\n    """Do.\n    """\n'
        'def __init__(self):\n    """Do.\n\n    More.\n    """\n'
        'def public():\n    """Do.\n\n    More.\n    """\n'
        'def _h():\n    """Do."""\n'
    )
    assert [
        (f.line, f.message) for f in _findings(tmp_path, PrivateOnelinerRule(), body)
    ] == [
        (1, "private _f has a multi-line docstring; one line or none"),
        (6, "private _g has a multi-line docstring; one line or none"),
    ]


def test_blank_padded_private_docstring_is_collapsed(tmp_path: Path) -> None:
    fixed = _fixed(
        tmp_path,
        PrivateOnelinerRule(),
        'def _f() -> None:\n    """Do.\n\n    """\n',
    )
    assert fixed == 'def _f() -> None:\n    """Do."""\n'


def test_raw_private_docstring_keeps_its_prefix(tmp_path: Path) -> None:
    """Dropping the raw prefix turns an escaped tab into a real one (ruff D301)."""
    fixed = _fixed(
        tmp_path,
        PrivateOnelinerRule(),
        'def _f() -> None:\n    r"""Strip a ``<n>\\t`` gutter.\n\n    More.\n    """\n',
    )
    assert fixed == (
        '# More.\ndef _f() -> None:\n    r"""Strip a ``<n>\\t`` gutter."""\n'
    )


def test_private_docstring_prose_becomes_a_comment(tmp_path: Path) -> None:
    fixed = _fixed(
        tmp_path,
        PrivateOnelinerRule(),
        "class K:\n"
        "    @staticmethod\n"
        "    def _f(x: int) -> int:\n"
        '        """Do the thing.\n'
        "\n"
        "        Because the vendor rounds down, and a caller\n"
        "        would otherwise double-round.\n"
        "\n"
        "        Args:\n"
        "          x: X.\n"
        "\n"
        "        Returns:\n"
        "          y: Y.\n"
        '        """\n'
        "        return x\n",
    )
    assert fixed == (
        "class K:\n"
        "    # Because the vendor rounds down, and a caller would otherwise double-round.\n"
        "    @staticmethod\n"
        "    def _f(x: int) -> int:\n"
        '        """Do the thing."""\n'
        "        return x\n"
    )


def test_private_oneliner_fix_rewrites_two_defs_and_wraps_at_width(
    tmp_path: Path,
) -> None:
    """Bottom-up edits keep line numbers valid; widths track the def's indent."""
    words = " ".join(f"w{i}" for i in range(40))
    body = (
        "def _a():\n"
        '    """First.\n'
        "\n"
        "    Reason a.\n"
        '    """\n'
        "class K:\n"
        "    def public(self):\n"
        '        """Keep.\n\n        Two lines.\n        """\n'
        "    @property\n"
        "    def _b(self):\n"
        '        """Second.\n'
        "\n"
        f"        {words}\n"
        "\n"
        "        Tail.\n"
        '        """\n'
    )
    fixed = _fixed(tmp_path, PrivateOnelinerRule(), body)
    wrapped = textwrap.wrap(words, width=88 - 4 - 2)
    assert fixed == (
        "# Reason a.\n"
        "def _a():\n"
        '    """First."""\n'
        "class K:\n"
        "    def public(self):\n"
        '        """Keep.\n\n        Two lines.\n        """\n'
        + "".join(f"    # {line}\n" for line in wrapped)
        + "    #\n"
        "    # Tail.\n"
        "    @property\n"
        "    def _b(self):\n"
        '        """Second."""\n'
    )


def test_one_line_private_docstring_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        PrivateOnelinerRule(),
        'def _f() -> None:\n    """Do."""\n',
    )


def test_private_oneliner_fix_skips_unparsable_file(tmp_path: Path) -> None:
    body = 'def _f(:\n    """A.\n\n    B."""\n'
    assert _fixed(tmp_path, PrivateOnelinerRule(), body) is None


_LONG_HEAD: Final = "Alpha " * 13 + "end"


@pytest.mark.parametrize(
    ("summary", "want"),
    [
        (
            f"{_LONG_HEAD}. Tail words here.",
            f'# Tail words here.\ndef _f():\n    """{_LONG_HEAD}"""\n',
        ),
        (
            f"{_LONG_HEAD}; tail words here",
            f'# Tail words here\ndef _f():\n    """{_LONG_HEAD}."""\n',
        ),
    ],
)
def test_private_docstring_long_summary_splits_at_first_break(
    tmp_path: Path,
    summary: str,
    want: str,
) -> None:
    body = f'def _f():\n    """{summary}\n\n    """\n'
    assert _fixed(tmp_path, PrivateOnelinerRule(), body) == want


def test_private_docstring_drops_indented_section_paragraphs(tmp_path: Path) -> None:
    body = (
        'def _f(x, y):\n    """Do.\n\n'
        "    Args:\n      x: X.\n\n      y: Y.\n\n"
        "    Trailing prose.\n"
        '    """\n'
    )
    assert _fixed(tmp_path, PrivateOnelinerRule(), body) == (
        '# Trailing prose.\ndef _f(x, y):\n    """Do."""\n'
    )


# -- attr-blank-line ---------------------------------------------------------


def test_adjacent_attributes_are_reported_and_separated(tmp_path: Path) -> None:
    body = 'class K:\n    a: int = 1\n    """A."""\n    b: int = 2\n'
    assert [f.line for f in _findings(tmp_path, AttrBlankLineRule(), body)] == [3]
    assert _fixed(tmp_path, AttrBlankLineRule(), body) == (
        'class K:\n    a: int = 1\n    """A."""\n\n    b: int = 2\n'
    )


@pytest.mark.parametrize(
    "body",
    [
        "class K:\n    a: int = 1\n\n    b: int = 2\n",
        "class K:\n    a: int = 1\n    b: int = 2\n",
        'class K:\n    a: int = 1\n    """A."""\n\n    b: int = 2\n    """B."""\n',
    ],
)
def test_undocumented_or_separated_attributes_are_clean(
    tmp_path: Path,
    body: str,
) -> None:
    assert not _findings(tmp_path, AttrBlankLineRule(), body)


# -- attr-doc-form -----------------------------------------------------------


def test_mixed_attribute_doc_forms_are_reported(tmp_path: Path) -> None:
    section = '    """K.\n\n      Attributes:\n        a: A.\n\n    """\n'
    findings = _findings(
        tmp_path,
        AttrDocFormRule(),
        "class K:\n"
        + section
        + '    a: int = 1\n    """A."""\n'
        + "class Plain:\n"
        + section
        + '    """Not after an attribute."""\n'
        + "    def f(self): ...\n"
        + '    """Not after an attribute either."""\n'
        + "class L:\n"
        + section
        + '    b = 2\n    """B."""\n',
    )
    tail = " documents attributes both inline and in Attributes:; pick one form"
    assert [(f.line, f.message) for f in findings] == [
        (1, f"K{tail}"),
        (20, f"L{tail}"),
    ]


def test_single_attribute_doc_form_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        AttrDocFormRule(),
        'class K:\n    """K.\n\n    Attributes:\n      a: A.\n\n    """\n\n    a: int = 1\n',
    )


# -- comment-sentence --------------------------------------------------------


def test_lowercase_and_unpunctuated_comments_are_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        CommentSentenceRule(),
        "# lowercase start.\nx = 1\n# No period\ny = 2\n",
    )
    assert [f.line for f in findings] == [1, 3]


def test_unpunctuated_comments_are_terminated(tmp_path: Path) -> None:
    fixed = _fixed(
        tmp_path,
        CommentSentenceRule(),
        "# A block that runs on\n# and stops here\nx = 1  # trailing comma,\n"
        "# lowercase but punctuated.\n# noqa\n",
    )
    assert fixed == (
        "# A block that runs on\n# and stops here.\nx = 1  # Trailing comma.\n"
        "# Lowercase but punctuated.\n# noqa\n"
    )


@pytest.mark.parametrize(
    "comment",
    ["# FULL_REPLACE", "# BEGIN GENERATED", "# XXX", "# PARTIAL_REPLACE: torch"],
)
def test_an_all_caps_marker_comment_is_not_prose(tmp_path: Path, comment: str) -> None:
    """A tool's marker is matched verbatim; a period appended to it breaks the tool."""
    body = f"{comment}\nx = 1\n"
    assert not _findings(tmp_path, CommentSentenceRule(), body)
    assert _fixed(tmp_path, CommentSentenceRule(), body) is None


def test_an_inline_script_metadata_block_is_not_prose(tmp_path: Path) -> None:
    """``uv run`` refuses a PEP 723 block whose closing ``# ///`` has a period."""
    body = (
        "# /// script\n"
        '# requires-python = ">=3.12"\n'
        '# dependencies = ["hatchling"]\n'
        "# ///\n"
        '"""Doc."""\n'
    )
    assert not _findings(tmp_path, CommentSentenceRule(), body)
    assert _fixed(tmp_path, CommentSentenceRule(), body) is None


def test_an_all_caps_sentence_start_is_still_prose(tmp_path: Path) -> None:
    findings = _findings(tmp_path, CommentSentenceRule(), "# API keys rotate daily\n")
    assert [f.line for f in findings] == [1]


def test_an_all_caps_line_inside_a_paragraph_keeps_it_one_block(tmp_path: Path) -> None:
    """A paragraph whose second line opens ``IGNORED:`` is still one sentence."""
    body = "# Once active, a supplied board is\n# IGNORED: it was segmented early.\n"
    assert not _findings(tmp_path, CommentSentenceRule(), body)


def test_an_all_caps_first_line_does_not_exempt_the_paragraph(tmp_path: Path) -> None:
    """Only a one-line block is a marker; a heading over prose is still prose."""
    body = "# IMPORTANT\n# the hook runs this first\nx = 1\n"
    assert [f.line for f in _findings(tmp_path, CommentSentenceRule(), body)] == [1]


def test_lowercase_comment_starts_are_capitalized(tmp_path: Path) -> None:
    """A plain lowercase first word is a sentence start, inline or not."""
    fixed = _fixed(
        tmp_path,
        CommentSentenceRule(),
        "import asyncpg\n\nspec = 1\n"
        "nav(1)  # plate 1\n"
        "nav(2)  # only one plate pressed.\n"
        "# spec is the parsed form\n"
        "a = 1\n"
        "# asyncpg pools by default\n"
        "b = 2\n"
        "# x.ndim is two here\n",
    )
    assert fixed == (
        "import asyncpg\n\nspec = 1\n"
        "nav(1)  # Plate 1.\n"
        "nav(2)  # Only one plate pressed.\n"
        "# `spec` is the parsed form.\n"
        "a = 1\n"
        "# `asyncpg` pools by default.\n"
        "b = 2\n"
        "# x.ndim is two here.\n"
    )


def test_inline_lowercase_comment_is_reported(tmp_path: Path) -> None:
    findings = _findings(tmp_path, CommentSentenceRule(), "x = 1  # plate 1.\n")
    assert [f.line for f in findings] == [1]


def test_sentence_comments_blocks_and_directives_are_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        CommentSentenceRule(),
        "#!/bin/sh\n"
        "# A block that runs on\n"
        "# and ends here.\n"
        "x = 1  # noqa: E501\n"
        "y = 2  # Trailing inline is fine.\n"
        "# TODO(x): later\n"
        "# https://example.com/a\n"
        "# 3 is the answer.\n",
    )


def test_comment_fix_backticks_def_and_parameter_names(tmp_path: Path) -> None:
    body = (
        "def g(x):\n    # x is ignored\n    return 1\n\n\n# g runs once\nclass K: ...\n"
    )
    assert _fixed(tmp_path, CommentSentenceRule(), body) == (
        "def g(x):\n    # `x` is ignored.\n    return 1\n\n\n"
        "# `g` runs once.\nclass K: ...\n"
    )


def test_comment_fix_every_branch(tmp_path: Path) -> None:
    """Marks after the word survive, CRLF survives, and a clean file is None."""
    body = (
        "spec = 1\n"
        "# spec, the first\r\n"
        "x = 1  # tail;   \n"
        "# NOTE: exempt\n"
        "# Already a sentence.\n"
        "y = 2  # spec: here  \n"
        "z = 3\n"
        "# word word\n"
    )
    assert _fixed(tmp_path, CommentSentenceRule(), body) == (
        "spec = 1\n"
        "# `spec`, the first.\r\n"
        "x = 1  # Tail.   \n"
        "# NOTE: exempt\n"
        "# Already a sentence.\n"
        "y = 2  # `spec`: here.  \n"
        "z = 3\n"
        "# Word word.\n"
    )
    assert _fixed(tmp_path, CommentSentenceRule(), "# Fine.\nx = 1  # Also.\n") is None


def test_comment_fix_on_untokenizable_tail_still_rewrites(tmp_path: Path) -> None:
    body = "x = (\n    # lower c\n"
    assert _fixed(tmp_path, CommentSentenceRule(), body) == "x = (\n    # Lower c.\n"


# -- cli-shape ---------------------------------------------------------------


_POLYGLOT = (
    "#!/bin/sh\n"
    "# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.\n"
    "# fmt: off\n"
    "'''' 2>/dev/null #\n"
    'exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"\n'
    "Doc.\n"
    "'''\n"
    "# fmt: on\n"
)


def test_polyglot_header_drift_and_missing_exec_bit_are_reported(
    tmp_path: Path,
) -> None:
    body = (
        _POLYGLOT.replace("--no-sync ", "")
        + 'from m import main\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
    )
    findings = _findings(tmp_path, CliShapeRule(), body, name="__main__.py")
    msgs = " ".join(f.message for f in findings)
    assert "(a) line 5" in msgs
    assert "vim: ft=python" in msgs
    assert "executable bit" in msgs


def test_half_executable_and_main_shape_are_reported(tmp_path: Path) -> None:
    body = (
        "import argparse, sys\n\n"
        "def main() -> None:\n"
        "    p = argparse.ArgumentParser(description=__doc__)\n"
        '    p.add_argument("--x")\n'
        "    sys.exit(0)\n\n"
        'if __name__ == "__main__":\n    main()\n'
    )
    letters = {f.message[:3] for f in _findings(tmp_path, CliShapeRule(), body)}
    assert {"(b)", "(d)", "(e)", "(f)"} <= letters


def test_conforming_library_module_is_clean(tmp_path: Path) -> None:
    path = tmp_path / "lib.py"
    path.write_text(
        '"""Doc."""\n\nimport argparse\n\n\n'
        "def main() -> int:\n"
        '    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())\n'
        "    _add_arguments(parser)\n"
        "    return 0\n\n\n"
        "def _add_arguments(parser: argparse.ArgumentParser) -> None:\n"
        '    parser.add_argument("--x")\n',
        encoding="utf-8",
    )
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert not CliShapeRule().check(unit)


def test_conforming_polyglot_main_is_clean(tmp_path: Path) -> None:
    path = tmp_path / "__main__.py"
    path.write_text(
        _POLYGLOT
        + 'from m import main\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n# vim: ft=python\n',
        encoding="utf-8",
    )
    path.chmod(0o755)
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert not CliShapeRule().check(unit)


def test_polyglot_exec_search_stops_at_docstring_close(tmp_path: Path) -> None:
    source = (
        "#!/bin/sh\n"
        "# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.\n"
        "# ruff: noqa: N999\n"
        "# fmt: off\n"
        "'''' 2>/dev/null #\n"
        "echo fast\n"
        "'''\n"
        "# vim: ft=python\n"
    )
    unit = Unit(path=tmp_path / "absent.py", source=source, kind="python")
    want = (
        '(a) line 7 must be `exec uv --quiet --project "$(dirname "$0")" run '
        '--frozen --no-sync python3 "$0" "$@"` '
        '(or `... python3 -m <module> "$@"` for a coupled entry)'
    )
    assert [f.message for f in CliShapeRule().check(unit)] == [want]


def test_main_module_without_main_import_or_exit_guard_is_reported(
    tmp_path: Path,
) -> None:
    body = 'from m import run\n\nif __name__ == "__main__":\n    run()\n'
    findings = _findings(tmp_path, CliShapeRule(), body, name="__main__.py")
    assert [f.message for f in findings] == [
        "(c) __main__.py must import `main` exactly once",
        "(c) guard body must be exactly `raise SystemExit(main())`",
    ]


def test_cli_shape_fixes_main_and_guard(tmp_path: Path) -> None:
    body = (
        "def main() -> None:\n"
        "    parser = argparse.ArgumentParser(description=__doc__)\n"
        "    return\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )
    fixed = _fixed(tmp_path, CliShapeRule(), body)
    assert fixed == (
        "def main() -> int:\n"
        '    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())\n'
        "    return 0\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )
    assert _fixed(tmp_path, CliShapeRule(), fixed) is None


def test_cli_shape_fixes_polyglot_header_and_mode(tmp_path: Path) -> None:
    path = tmp_path / "__main__.py"
    body = (
        "#!/bin/sh\n"
        "# ruff: noqa: EXE003, D300, D205, N999\n"
        "wrong\n"
        "old\n"
        "old exec\n"
        "'''Doc.'''\n"
    )
    fixed = _fixed(tmp_path, CliShapeRule(), body, name="__main__.py")
    assert fixed is not None
    assert fixed.startswith(
        "#!/bin/sh\n"
        "# ruff: noqa: EXE003, D300, D205, N999 -- Polyglot shell/Python script.\n"
        "# fmt: off\n"
        "'''' 2>/dev/null #\n"
        'exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"\n',
    )
    assert fixed.endswith("# vim: ft=python\n")
    assert path.stat().st_mode & 0o111


def test_cli_shape_fixes_description_and_add_arguments(tmp_path: Path) -> None:
    body = (
        '"""Doc.\n\nLong description.\n"""\n\n'
        "import argparse\n\n"
        "def main() -> int:\n"
        "    parser = argparse.ArgumentParser(description=__doc__)\n"
        '    parser.add_argument("--one")\n'
        '    parser.add_argument("--two")\n'
        "    return 0\n"
    )
    fixed = _fixed(tmp_path, CliShapeRule(), body)
    assert fixed == (
        '"""Doc.\n\nLong description.\n"""\n\n'
        "import argparse\n\n"
        "def main() -> int:\n"
        '    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())\n'
        "    _add_arguments(parser)\n"
        "    return 0\n\n\n"
        "def _add_arguments(parser: argparse.ArgumentParser) -> None:\n"
        '    parser.add_argument("--one")\n'
        '    parser.add_argument("--two")\n'
    )


def test_cli_shape_fix_skips_return_after_exhaustive_try(tmp_path: Path) -> None:
    body = (
        "def main() -> int:\n"
        "    try:\n"
        "        with ctx():\n"
        "            return run()\n"
        "    except OSError:\n"
        "        return 1\n"
        "    except KeyError:\n"
        "        parser.exit(2)\n"
    )
    assert _fixed(tmp_path, CliShapeRule(), body) is None


@pytest.mark.parametrize(
    "annotation",
    ["NoReturn", "Never", "typing.NoReturn", "typing_extensions.Never"],
)
def test_cli_shape_treats_noreturn_spellings_as_terminal(
    tmp_path: Path,
    annotation: str,
) -> None:
    body = (
        "def main() -> int:\n"
        "    try:\n"
        "        return run()\n"
        "    except BrokenPipeError:\n"
        "        _bail()\n\n\n"
        f"def _bail() -> {annotation}:\n"
        "    raise SystemExit(0)\n"
    )
    assert _findings(tmp_path, CliShapeRule(), body) == ()
    assert _fixed(tmp_path, CliShapeRule(), body) is None


@pytest.mark.parametrize(
    "annotation",
    ["None", "NoReturnX", "never", "a.b.Never.x", "Never | None"],
)
def test_cli_shape_treats_other_annotations_as_returning(
    tmp_path: Path,
    annotation: str,
) -> None:
    body = (
        "def main() -> int:\n"
        "    try:\n"
        "        return run()\n"
        "    except BrokenPipeError:\n"
        "        _bail()\n\n\n"
        f"def _bail() -> {annotation}:\n"
        "    raise SystemExit(0)\n"
    )
    assert [f.line for f in _findings(tmp_path, CliShapeRule(), body)] == [5]


def test_cli_shape_ignores_a_main_method_on_a_protocol(tmp_path: Path) -> None:
    body = "class P(Protocol):\n    def main(self) -> int: ...\n"
    assert _findings(tmp_path, CliShapeRule(), body) == ()
    assert _fixed(tmp_path, CliShapeRule(), body) is None


def test_cli_shape_reports_a_guard_that_drops_mains_exit_code(tmp_path: Path) -> None:
    body = (
        'def main() -> int:\n    return 0\n\n\nif __name__ == "__main__":\n    main()\n'
    )
    assert [
        f.line for f in _findings(tmp_path, CliShapeRule(), body) if "(d)" in f.message
    ] == [6]
    assert _fixed(tmp_path, CliShapeRule(), body) == body.replace(
        "    main()",
        "    raise SystemExit(main())",
    )


def test_cli_shape_reports_and_fixes_a_polyglot_noqa_without_a_reason(
    tmp_path: Path,
) -> None:
    body = (
        "#!/bin/sh\n"
        "# ruff: noqa: EXE003, D300, D205\n"
        "# fmt: off\n"
        "'''' 2>/dev/null #\n"
        'exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"\n'
        "Doc.\n"
        "'''\n"
        "# vim: ft=python\n"
    )
    path = tmp_path / "tool.py"
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert [f.message for f in CliShapeRule().check(unit)] == [
        "(a) line 2 must be `# ruff: noqa: EXE003, D300, D205[, CODE...]`",
    ]
    fixed = CliShapeRule().fix(unit)
    assert fixed == body.replace(
        "D205\n",
        "D205 -- Polyglot shell/Python script.\n",
    )


def test_cli_shape_fix_returns_after_try_that_falls_through(tmp_path: Path) -> None:
    body = (
        "def main() -> int:\n"
        "    try:\n"
        "        return run()\n"
        "    except OSError:\n"
        "        log()\n"
    )
    assert _fixed(tmp_path, CliShapeRule(), body) == body + "    return 0\n"


def test_cli_shape_fix_keeps_extra_noqa_and_comment_lines(tmp_path: Path) -> None:
    body = (
        "#!/bin/sh\n"
        "# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.\n"
        "# ruff: noqa: N999 -- Kebab-case executable;\n"
        "# invoked by name.\n"
        "# fmt: off\n"
        "'''' 2>/dev/null #\n"
        'exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"\n'
        "Doc.\n"
        "'''\n"
        "# vim: ft=python\n"
    )
    assert _fixed(tmp_path, CliShapeRule(), body, name="tool.py") is None


def test_cli_shape_leaves_sys_exit_and_ambiguous_parser_to_check(
    tmp_path: Path,
) -> None:
    body = (
        "def main() -> None:\n"
        "    parser = make_parser()\n"
        '    parser.add_argument("--x")\n'
        "    sys.exit(1)\n"
    )
    assert _fixed(tmp_path, CliShapeRule(), body) is None


# -- suppression-form ------------------------------------------------------


def test_suppression_form_flags_a_bare_noqa(tmp_path: Path) -> None:
    findings = _findings(tmp_path, SuppressionFormRule(), "x = 1  # noqa: E501\n")
    assert [f.line for f in findings] == [1]
    assert "-- Reason." in findings[0].message


def test_suppression_form_accepts_a_reasoned_noqa(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        SuppressionFormRule(),
        "x = 1  # noqa: E501 -- The URL cannot wrap.\n",
    )


def test_suppression_form_flags_pyright_and_ty_without_reason(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        SuppressionFormRule(),
        "a = 1  # pyright: ignore[reportFoo]\n"
        "b = 2  # ty: ignore[invalid-assignment]\n"
        "c = 3  # pyright: ignore[reportFoo] -- Stub lacks the overload.\n",
    )
    assert [f.line for f in findings] == [1, 2]


def test_suppression_form_flags_pyrefly_without_reason(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        SuppressionFormRule(),
        "a = 1  # pyrefly: ignore[bad-return]\n"
        "b = 2  # pyrefly: ignore[bad-return] -- Upstream bug.\n",
    )
    assert [f.line for f in findings] == [1]


def test_suppression_form_flags_mypy_type_ignore(tmp_path: Path) -> None:
    findings = _findings(tmp_path, SuppressionFormRule(), "x = 1  # type: ignore\n")
    assert len(findings) == 1
    assert "mypy" in findings[0].message


def test_suppression_form_accepts_two_reasoned_suppressions_on_one_line(
    tmp_path: Path,
) -> None:
    assert not _findings(
        tmp_path,
        SuppressionFormRule(),
        "x = 1  # noqa: TID251 -- Vendor path.  # house-ignore[xdg-literal] -- Not ours.\n",
    )


def test_suppression_form_ignores_prose_mentioning_noqa(tmp_path: Path) -> None:
    """A docstring that explains the marker is not itself a marker."""
    assert not _findings(
        tmp_path,
        SuppressionFormRule(),
        '"""Write ``# noqa: E501`` to silence the line-length check."""\n',
    )


# -- fmt-directive ---------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "x = [1,   2]  # fmt: skip\n",
        "x = [1,   2]  # fmt:skip\n",
        "x = [1,   2]  # fmt: skip  # -- Hand-aligned.\n",
        "# fmt: off\nx = [1,   2]\n# fmt: on\n",
        "def f() -> None:\n    # fmt: off\n    x = [1,   2]\n",
        "x = [1,   2]  # noqa: E501 -- Wide.  # fmt: skip\n",
    ],
)
def test_fmt_directive_flags_each_formatter_opt_out(tmp_path: Path, body: str) -> None:
    findings = _findings(tmp_path, FmtDirectiveRule(), body)
    assert [f.line for f in findings] == [
        n for n, line in enumerate(body.splitlines(), start=1) if "fmt:" in line
    ]
    assert all("ruff format" in f.message for f in findings)


def test_fmt_directive_exempts_the_polyglot_header(tmp_path: Path) -> None:
    """The ``cli`` skill pins ``# fmt: off`` so ruff keeps the shell preamble."""
    body = (
        "#!/bin/sh\n"
        "# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.\n"
        "# fmt: off\n"
        "'''' 2>/dev/null #\n"
        'exec uv run python3 "$0" "$@"\n'
        "Doc.\n"
        "'''\n"
        "# fmt: on\n"
        "x = 1\n"
    )
    assert not _findings(tmp_path, FmtDirectiveRule(), body)


def test_fmt_directive_flags_an_opt_out_after_the_header(tmp_path: Path) -> None:
    body = (
        "#!/bin/sh\n"
        "# fmt: off\n"
        "'''' 2>/dev/null #\n"
        "'''\n"
        "# fmt: on\n"
        "x = [1,   2]  # fmt: skip\n"
    )
    assert [f.line for f in _findings(tmp_path, FmtDirectiveRule(), body)] == [6]


@pytest.mark.parametrize(
    "body",
    [
        "x = 1\n",
        '"""Write ``# fmt: skip`` to keep a line."""\n',
        "x = '# fmt: off\\n'\n",
        "# The fmt: skip directive is banned here.\n",
    ],
)
def test_fmt_directive_ignores_mentions_and_clean_files(
    tmp_path: Path,
    body: str,
) -> None:
    assert not _findings(tmp_path, FmtDirectiveRule(), body)


def test_fmt_directive_has_no_fix(tmp_path: Path) -> None:
    """Dropping the directive reflows the code; ruff, not houselint, does that."""
    assert _fixed(tmp_path, FmtDirectiveRule(), "x = [1,   2]  # fmt: skip\n") is None


# -- annotation-import -----------------------------------------------------


def test_annotation_import_flags_a_heavy_annotation_only_import(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        AnnotationImportRule(),
        "from __future__ import annotations\n\n"
        "from torch import Tensor\n\n\n"
        "def f(x: Tensor) -> None:\n    del x\n",
    )
    assert [f.line for f in findings] == [3]
    assert "TYPE_CHECKING" in findings[0].message
    assert "1050 ms" in findings[0].message


def test_annotation_import_leaves_a_cheap_annotation_only_import(
    tmp_path: Path,
) -> None:
    """A stdlib name under ``TYPE_CHECKING`` breaks ``get_type_hints`` for nothing."""
    assert not _findings(
        tmp_path,
        AnnotationImportRule(),
        "from __future__ import annotations\n\n"
        "import subprocess\n\n\n"
        "def f(p: subprocess.Popen[bytes]) -> None:\n    del p\n",
    )


def test_annotation_import_leaves_a_runtime_use_alone(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        AnnotationImportRule(),
        "import torch\n\n\n"
        "def f(x: torch.Tensor) -> torch.Tensor:\n    return torch.zeros_like(x)\n",
    )


def test_annotation_import_leaves_a_root_with_any_runtime_local(tmp_path: Path) -> None:
    """Moving ``Tensor`` buys nothing while ``torch`` itself loads eagerly."""
    assert not _findings(
        tmp_path,
        AnnotationImportRule(),
        "import torch\nfrom torch import Tensor\n\n\n"
        "def f(x: Tensor) -> Tensor:\n    return torch.zeros_like(x)\n",
    )


def test_annotation_import_counts_quoted_cast_as_annotation(tmp_path: Path) -> None:
    """``cast("Tensor", x)`` never evaluates ``Tensor``; ``cast(Tensor, x)`` does."""
    quoted = (
        "from typing import cast\n\nfrom torch import Tensor\n\n\n"
        'def f(x: object) -> object:\n    return cast("Tensor", x)\n'
    )
    bare = (
        "from typing import cast\n\nfrom torch import Tensor\n\n\n"
        "def f(x: object) -> object:\n    return cast(Tensor, x)\n"
    )
    assert _findings(tmp_path, AnnotationImportRule(), quoted)
    assert not _findings(tmp_path, AnnotationImportRule(), bare)


def test_annotation_import_ignores_type_checking_block(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        AnnotationImportRule(),
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n    from torch import Tensor\n\n\n"
        "def f(x: Tensor) -> None:\n    del x\n",
    )


def test_annotation_import_threshold_is_a_setting(tmp_path: Path) -> None:
    body = (
        "from __future__ import annotations\n\n"
        "import subprocess\n\n\n"
        "def f(p: subprocess.Popen[bytes]) -> None:\n    del p\n"
    )
    assert not _findings(tmp_path, AnnotationImportRule(), body)
    assert _findings(
        tmp_path,
        AnnotationImportRule(threshold_ms=0, costs_ms={"subprocess": 1}),
        body,
    )


# -- final-inline ----------------------------------------------------------


def test_final_inline_flags_a_single_use_final_literal(tmp_path: Path) -> None:
    (tmp_path / "mod_test.py").write_text("_KEPT\n")
    findings = _findings(
        tmp_path,
        FinalInlineRule(),
        "from typing import Final\n\n_LIMIT: Final = 5\n"
        "_KEPT: Final = 6\n"
        "__DUNDER: Final = 7\n"
        "_UNREAD: Final = 8\n"
        "def f() -> int:\n    return _LIMIT + _KEPT + __DUNDER\n"
        "_LAST: Final = 9\n"
        "g = _LAST\n"
        "_TAIL: Final = 10\n"
        "h = _TAIL\n",
    )
    tail = " read at one site; inline it there (STYLE.md: no module-level convenience constants)"
    assert [(f.line, f.message) for f in findings] == [
        (3, f"'_LIMIT' is a Final literal{tail}"),
        (9, f"'_LAST' is a Final literal{tail}"),
        (11, f"'_TAIL' is a Final literal{tail}"),
    ]


def test_final_inline_leaves_a_twice_read_constant(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        FinalInlineRule(),
        "from typing import Final\n\n_LIMIT: Final = 5\n\n\n"
        "def f() -> int:\n    return _LIMIT + _LIMIT\n",
    )


def test_final_inline_leaves_an_unread_public_constant(tmp_path: Path) -> None:
    """A never-read constant is an export, not a convenience binding."""
    assert not _findings(
        tmp_path,
        FinalInlineRule(),
        "from typing import Final\n\nLIMIT: Final = 5\n",
    )


def test_final_inline_leaves_a_public_constant_read_once(tmp_path: Path) -> None:
    """A public name is importable by another module; one local read proves nothing."""
    assert not _findings(
        tmp_path,
        FinalInlineRule(),
        "from typing import Final\n\nLIMIT: Final = 5\n\n\n"
        "def f() -> int:\n    return LIMIT\n",
    )


def test_final_inline_leaves_a_multi_line_binding(tmp_path: Path) -> None:
    """A binding that needed wrapping is prose or data, not a convenience literal."""
    assert not _findings(
        tmp_path,
        FinalInlineRule(),
        "from typing import Final\n\n"
        '_HELP: Final = (\n    "first line "\n    "second line"\n)\n\n\n'
        "def f() -> str:\n    return _HELP\n",
    )


def test_final_inline_leaves_a_documented_constant(tmp_path: Path) -> None:
    """An attribute docstring names a knob; inlining would orphan the prose."""
    assert not _findings(
        tmp_path,
        FinalInlineRule(),
        "from typing import Final\n\n_LIMIT: Final = 5\n"
        '"""Upper bound on points per batch."""\n\n\n'
        "def f() -> int:\n    return _LIMIT\n",
    )


def test_final_inline_leaves_a_commented_constant(tmp_path: Path) -> None:
    """A ``#`` block above the binding documents the knob as a docstring would."""
    assert not _findings(
        tmp_path,
        FinalInlineRule(),
        "from typing import Final\n\n"
        "# Exceeds the server's own hold so the poll never returns early.\n"
        "_LIMIT: Final = 5\n\n\n"
        "def f() -> int:\n    return _LIMIT\n",
    )


def test_final_inline_leaves_a_constant_its_test_reaches(tmp_path: Path) -> None:
    """The sibling test is a second read site; a monkeypatched knob must stay named."""
    (tmp_path / "mod_test.py").write_text(
        "from ops import mod\n\n\n"
        'def test_f(monkeypatch) -> None:\n    monkeypatch.setattr(mod, "_LIMIT", 1)\n',
        encoding="utf-8",
    )
    assert not _findings(
        tmp_path,
        FinalInlineRule(),
        "from typing import Final\n\n_LIMIT: Final = 5\n\n\n"
        "def f() -> int:\n    return _LIMIT\n",
    )


def test_final_inline_flags_a_single_use_re_compile(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        FinalInlineRule(),
        "import re\n\n_PAT = re.compile(r'x')\n\n\n"
        "def f(s: str) -> bool:\n    return _PAT.search(s) is not None\n",
    )
    assert [f.line for f in findings] == [3]
    assert "re.compile" in findings[0].message


def test_final_inline_leaves_a_non_literal_final(tmp_path: Path) -> None:
    """A computed ``Final`` (a call, a tuple of names) is not an inlineable literal."""
    assert not _findings(
        tmp_path,
        FinalInlineRule(),
        "from typing import Final\n\nimport os\n\n"
        "_HOME: Final = os.getcwd()\n\n\n"
        "def f() -> str:\n    return _HOME\n",
    )


# -- single-use-helper -----------------------------------------------------


def test_single_use_helper_flags_a_one_line_helper_called_once(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        SingleUseHelperRule(),
        "def host_header(host: str, port: int) -> str:\n"
        "    return _netloc(host, port)\n\n\n"
        "def _netloc(hostname: str, port: int) -> str:\n"
        '    """Recombine."""\n'
        '    return f"{hostname}:{port}"\n',
    )
    assert [f.line for f in findings] == [5]
    assert "_netloc" in findings[0].message


def test_single_use_helper_inlines_and_deletes_the_def(tmp_path: Path) -> None:
    """The docstring goes with the def; the expression lands at the call site."""
    fixed = _fixed(
        tmp_path,
        SingleUseHelperRule(),
        "def host_header(host: str, port: int) -> str:\n"
        "    return _netloc(host, port)\n\n\n"
        "def _netloc(hostname: str, port: int) -> str:\n"
        '    """Recombine."""\n'
        '    return f"{hostname}:{port}"\n',
    )
    assert fixed == (
        'def host_header(host: str, port: int) -> str:\n    return f"{host}:{port}"\n'
    )


def test_single_use_helper_parenthesizes_a_compound_argument(tmp_path: Path) -> None:
    """``_double(a + b)`` into ``x * 2`` must read ``(a + b) * 2``, not ``a + b * 2``."""
    fixed = _fixed(
        tmp_path,
        SingleUseHelperRule(),
        "def f(a: int, b: int) -> int:\n"
        "    return _double(a + b)\n\n\n"
        "def _double(x: int) -> int:\n"
        "    return x * 2\n",
    )
    assert fixed == "def f(a: int, b: int) -> int:\n    return (a + b) * 2\n"


def test_single_use_helper_parenthesizes_the_inlined_body(tmp_path: Path) -> None:
    """``1 + _sum(a, b)`` into ``x + y`` must not rebind as ``1 + a + b`` by accident of precedence."""
    fixed = _fixed(
        tmp_path,
        SingleUseHelperRule(),
        "def f(a: int, b: int) -> int:\n"
        "    return 2 * _sum(a, b)\n\n\n"
        "def _sum(x: int, y: int) -> int:\n"
        "    return x + y\n",
    )
    assert fixed == "def f(a: int, b: int) -> int:\n    return 2 * (a + b)\n"


def test_single_use_helper_binds_keywords_and_defaults(tmp_path: Path) -> None:
    fixed = _fixed(
        tmp_path,
        SingleUseHelperRule(),
        "def f(a: int) -> int:\n"
        "    return _scale(a, factor=3)\n\n\n"
        "def _scale(x: int, factor: int = 2, offset: int = 1) -> int:\n"
        "    return x * factor + offset\n",
    )
    assert fixed == "def f(a: int) -> int:\n    return a * 3 + 1\n"


def test_single_use_helper_inlines_a_wrapped_return(tmp_path: Path) -> None:
    """A multi-line return is still one expression; the formatter reflows it."""
    fixed = _fixed(
        tmp_path,
        SingleUseHelperRule(),
        "def f(xs: list[int]) -> int:\n"
        "    return _total(xs)\n\n\n"
        "def _total(items: list[int]) -> int:\n"
        "    return max(\n"
        "        len(items),\n"
        "        1,\n"
        "    )\n",
    )
    assert fixed == "def f(xs: list[int]) -> int:\n    return max(len(xs), 1)\n"


@pytest.mark.parametrize(
    ("helper", "why"),
    [
        (
            "def _is_str(x: object) -> TypeGuard[str]:\n    return isinstance(x, str)\n",
            "TypeGuard narrows only through a call",
        ),
        (
            "def _is_str(x: object) -> TypeIs[str]:\n    return isinstance(x, str)\n",
            "TypeIs narrows only through a call",
        ),
        (
            "@cache\ndef _is_str(x: object) -> bool:\n    return isinstance(x, str)\n",
            "a decorator changes the call",
        ),
        (
            "async def _is_str(x: object) -> bool:\n    return isinstance(x, str)\n",
            "async changes the call protocol",
        ),
        (
            "def _is_str(*xs: object) -> bool:\n    return all(xs)\n",
            "varargs cannot be substituted",
        ),
    ],
)
def test_single_use_helper_leaves_shapes_inlining_would_change(
    tmp_path: Path,
    helper: str,
    why: str,
) -> None:
    del why
    src = (
        "from functools import cache\n"
        "from typing import TypeGuard, TypeIs\n\n\n"
        "def f(x: object) -> bool:\n"
        "    return _is_str(x)\n\n\n" + helper
    )
    assert not _findings(tmp_path, SingleUseHelperRule(), src)
    assert _fixed(tmp_path, SingleUseHelperRule(), src) is None


def test_single_use_helper_leaves_a_name_a_sibling_module_reaches(
    tmp_path: Path,
) -> None:
    """``other.py`` calling ``mod._netloc(...)`` is a second read the module cannot see."""
    (tmp_path / "other.py").write_text(
        "from . import mod\n\n\ndef g() -> str:\n"
        '    return mod._netloc("h")  # noqa: SLF001 -- Shared seam.\n',
        encoding="utf-8",
    )
    assert not _findings(
        tmp_path,
        SingleUseHelperRule(),
        "def f(h: str) -> str:\n    return _netloc(h)\n\n\n"
        "def _netloc(host: str) -> str:\n    return host\n",
    )


def test_single_use_helper_leaves_a_name_a_sibling_imports_by_name(
    tmp_path: Path,
) -> None:
    """``from .mod import _netloc`` is a read spelled without the ``.`` prefix."""
    (tmp_path / "other.py").write_text(
        "from .mod import _netloc\n\n\ndef g() -> str:\n    return _netloc('h')\n",
        encoding="utf-8",
    )
    assert not _findings(
        tmp_path,
        SingleUseHelperRule(),
        "def f(h: str) -> str:\n    return _netloc(h)\n\n\n"
        "def _netloc(host: str) -> str:\n    return host\n",
    )


def test_single_use_helper_ignores_a_bare_mention_in_a_non_test_sibling(
    tmp_path: Path,
) -> None:
    """A same-named private in an unrelated module is not a read of this one."""
    (tmp_path / "other.py").write_text(
        "def _netloc(host: str) -> str:\n    return host\n",
        encoding="utf-8",
    )
    assert _findings(
        tmp_path,
        SingleUseHelperRule(),
        "def f(h: str) -> str:\n    return _netloc(h)\n\n\n"
        "def _netloc(host: str) -> str:\n    return host\n",
    )


def test_single_use_helper_leaves_a_name_the_test_reaches(tmp_path: Path) -> None:
    (tmp_path / "mod_test.py").write_text(
        'def test_it(monkeypatch) -> None:\n    monkeypatch.setattr(mod, "_netloc", lambda *a: "")\n',
        encoding="utf-8",
    )
    assert not _findings(
        tmp_path,
        SingleUseHelperRule(),
        "def f(h: str) -> str:\n    return _netloc(h)\n\n\n"
        "def _netloc(host: str) -> str:\n    return host\n",
    )


def test_single_use_helper_leaves_a_multi_statement_body(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        SingleUseHelperRule(),
        "def f(h: str) -> str:\n    return _g(h)\n\n\n"
        "def _g(host: str) -> str:\n    host = host.strip()\n    return host\n",
    )


def test_single_use_helper_leaves_a_helper_called_twice(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        SingleUseHelperRule(),
        "def f(h: str) -> str:\n    return _g(h) + _g(h)\n\n\n"
        "def _g(host: str) -> str:\n    return host.strip()\n",
    )


@pytest.mark.parametrize(
    "src",
    [
        # Passed as a value: there is no call to substitute into.
        (
            "def f(xs: list[str]) -> list[str]:\n    return sorted(xs, key=_key)\n\n\n"
            "def _key(s: str) -> str:\n    return s.lower()\n"
        ),
        # A parameter read twice: inlining would evaluate the argument twice.
        (
            "def f(xs: list[int]) -> int:\n    return _square(sum(xs))\n\n\n"
            "def _square(x: int) -> int:\n    return x * x\n"
        ),
        # A parameter never read: inlining would drop the argument's evaluation.
        (
            "def f(xs: list[int]) -> int:\n    return _zero(sum(xs))\n\n\n"
            "def _zero(x: int) -> int:\n    return 0\n"
        ),
        # Splatted call: arity cannot be bound statically.
        (
            "def f(xs: list[int]) -> int:\n    return _add(*xs)\n\n\n"
            "def _add(a: int, b: int) -> int:\n    return a + b\n"
        ),
        # The body reads a name the caller also binds locally: capture.
        (
            "def f(items: list[int], n: int) -> int:\n    return _count(items)\n\n\n"
            "def _count(xs: list[int]) -> int:\n    return len(xs) + n\n"
        ),
        # The return annotation narrows to a local Protocol; inlining loses it.
        (
            "from typing import Any, Protocol\n\n\n"
            "class _Tok(Protocol):\n    def encode(self, s: str) -> list[int]: ...\n\n\n"
            "def load(name: str) -> Any:\n    return _get(name)\n\n\n"
            "def _get(name: str) -> _Tok:\n    return make(name)\n"
        ),
        # The return annotation widens a container display; inlined it infers
        # the invariant narrow type and no longer fits the caller's list.
        (
            "def f(rows: list[dict[str, object]], s: str) -> None:\n"
            "    rows.append(_entry(s))\n\n\n"
            "def _entry(s: str) -> dict[str, object]:\n"
            '    return {"name": s, "n": 1}\n'
        ),
        # A suppression on the return line would be dropped.
        (
            "import subprocess\n\n\n"
            "def f(cmd: list[str]) -> int:\n    return _run(cmd)\n\n\n"
            "def _run(cmd: list[str]) -> int:\n"
            "    return subprocess.run(cmd, check=False).returncode  # noqa: S603 -- Trusted.\n"
        ),
        # A rationale block above the def has no mechanical home at the call site.
        (
            "def f(a: int) -> int:\n    return _inc(a)\n\n\n"
            "# Off by one on purpose: the wire protocol is 1-based.\n"
            "def _inc(x: int) -> int:\n    return x + 1\n"
        ),
        # A wrapped string reflows into one unreadable literal.
        (
            "def f(a: str) -> str:\n    return _script(a)\n\n\n"
            "def _script(p: str) -> str:\n"
            '    return (\n        "set -u\\n"\n        f"p={p}\\n"\n        "echo done\\n"\n    )\n'
        ),
        # A literal table the def exists to name is data, not an expression.
        (
            "def f() -> list[str]:\n    return list(_langs())\n\n\n"
            "def _langs() -> tuple[str, ...]:\n"
            '    return (\n        "ada",\n        "agda",\n        "c",\n        "zig",\n    )\n'
        ),
        # An empty container argument is typed by the parameter, not itself.
        (
            "from typing import Any\n\n\n"
            "def f() -> str | None:\n    return _pick({})\n\n\n"
            "def _pick(provider: dict[str, Any]) -> str | None:\n"
            '    return provider.get("project")\n'
        ),
        # A call site inside an f-string cannot host quotes of its own.
        (
            'def f(x: int) -> str:\n    return f"<b>{_tag(x)}</b>"\n\n\n'
            'def _tag(n: int) -> str:\n    return str(n).replace("0", "o")\n'
        ),
    ],
)
def test_single_use_helper_is_silent_where_inlining_is_not_a_pure_rename(
    tmp_path: Path,
    src: str,
) -> None:
    """A finding the fixer cannot honour invites a hand-edit or a suppression."""
    assert not _findings(tmp_path, SingleUseHelperRule(), src)
    assert _fixed(tmp_path, SingleUseHelperRule(), src) is None


def test_single_use_helper_fix_is_idempotent(tmp_path: Path) -> None:
    src = (
        "def f(a: int) -> int:\n    return _inc(a)\n\n\n"
        "def _inc(x: int) -> int:\n    return x + 1\n"
    )
    once = _fixed(tmp_path, SingleUseHelperRule(), src)
    assert once is not None
    assert _fixed(tmp_path, SingleUseHelperRule(), once) is None


# -- forwarder-helper ------------------------------------------------------


_FORWARDER = (
    "def _row(line: str) -> dict[str, object]:\n"
    '    """Read one row."""\n'
    "    return json_loads(line, dict[str, object])\n"
)


def test_forwarder_helper_flags_a_private_def_that_only_calls_another(
    tmp_path: Path,
) -> None:
    findings = _findings(
        tmp_path,
        ForwarderHelperRule(),
        "def f(a: str, b: str) -> object:\n    return [_row(a), _row(b)]\n\n\n"
        + _FORWARDER,
    )
    assert [f.line for f in findings] == [5]
    assert "_row" in findings[0].message


def test_forwarder_helper_inlines_every_call_and_deletes_the_def(
    tmp_path: Path,
) -> None:
    fixed = _fixed(
        tmp_path,
        ForwarderHelperRule(),
        "def f(a: str, b: str) -> object:\n    return [_row(a), _row(b)]\n\n\n"
        + _FORWARDER,
    )
    assert fixed == (
        "def f(a: str, b: str) -> object:\n"
        "    return [json_loads(a, dict[str, object]), json_loads(b, dict[str, object])]\n"
    )


def test_forwarder_helper_substitutes_a_method_call_argument(tmp_path: Path) -> None:
    """``path.read_text()`` is the argument's own call, not a second function."""
    fixed = _fixed(
        tmp_path,
        ForwarderHelperRule(),
        "def f(p: Path, q: Path) -> object:\n    return _load(p), _load(q)\n\n\n"
        "def _load(path: Path) -> State:\n    return json_loads(path.read_text(), State)\n",
    )
    assert fixed == (
        "def f(p: Path, q: Path) -> object:\n"
        "    return json_loads(p.read_text(), State), json_loads(q.read_text(), State)\n"
    )


def test_forwarder_helper_inlines_calls_in_several_functions(tmp_path: Path) -> None:
    fixed = _fixed(
        tmp_path,
        ForwarderHelperRule(),
        "def f(a: int) -> int:\n    return _abs(a)\n\n\n"
        "def g(b: int) -> int:\n    return _abs(b - 1)\n\n\n"
        "def _abs(x: int):\n    return abs(x)\n",
    )
    # A compound argument is parenthesized conservatively; ``ruff format``
    # strips what is redundant, as for ``single-use-helper``.
    assert fixed == (
        "def f(a: int) -> int:\n    return abs(a)\n\n\n"
        "def g(b: int) -> int:\n    return abs((b - 1))\n"
    )


@pytest.mark.parametrize(
    "src",
    [
        # Not a forwarder: the body is an expression, not one call.
        (
            "def f(a: int, b: int) -> int:\n    return _sum(a, b) + _sum(b, a)\n\n\n"
            "def _sum(x: int, y: int) -> int:\n    return x + y\n"
        ),
        # Public and exported: ``__all__`` promises the name to importers.
        (
            '__all__ = ["absolute"]\n\n\n'
            "def f(a: int, b: int) -> int:\n    return absolute(a) + absolute(b)\n\n\n"
            "def absolute(x: int):\n    return abs(x)\n"
        ),
        # Unsafe at one call site: a splat cannot be bound statically.
        (
            "def f(a: int, xs: list[int]) -> int:\n    return _abs(a) + _abs(*xs)\n\n\n"
            "def _abs(x: int):\n    return abs(x)\n"
        ),
        # Passed as a value at one site: there is nothing to substitute into.
        (
            "def f(xs: list[int]) -> list[int]:\n"
            "    return sorted(xs, key=_abs) + [_abs(xs[0])]\n\n\n"
            "def _abs(x: int):\n    return abs(x)\n"
        ),
        # Decorated: the decorator changes what calling it means.
        (
            "from functools import cache\n\n\n"
            "def f(a: int) -> int:\n    return _abs(a) + _abs(a + 1)\n\n\n"
            "@cache\ndef _abs(x: int):\n    return abs(x)\n"
        ),
        # A secret-named keyword: inlined beside a literal it is ruff S106.
        (
            "def f() -> object:\n"
            '    return [_cfg("a"), _cfg("b")]\n\n\n'
            "def _cfg(secret: str) -> object:\n"
            "    return Config(session_secret=secret, max_age=600)\n"
        ),
        # Called from a coroutine: the sync helper is the ASYNC240 boundary.
        (
            "async def f(a: str, b: str) -> bool:\n    return _exists(a) and _exists(b)\n\n\n"
            "def _exists(p: str) -> bool:\n    return Path(p).exists()\n"
        ),
        # The annotation narrows what the call infers: inlining loses it.
        (
            "def f(a: int, b: int) -> object:\n    return [_t(a), _t(b)]\n\n\n"
            "def _t(n: int) -> Tensor:\n    return torch.zeros(n)\n"
        ),
        # The body spans lines: inlining pastes a block at every call.
        (
            "def f(a: int, b: int) -> object:\n    return [_prog(a), _prog(b)]\n\n\n"
            "def _prog(n: int) -> object:\n"
            "    return Program(\n        rows=n,\n        cols=4,\n    )\n"
        ),
        # A cast over a bare parameter: its annotation types the operand.
        (
            'def f() -> object:\n    return [_int("one"), _int("two")]\n\n\n'
            'def _int(value: object) -> int:\n    return cast("int", value)\n'
        ),
        # A dotted-path target: read as a value, never called here.
        (
            'ARM = f"{synthetic_arm.__module__}.synthetic_arm"\n\n\n'
            "def synthetic_arm() -> Arm:\n    return Arm()\n"
        ),
    ],
)
def test_forwarder_helper_is_silent_where_inlining_is_not_a_pure_rename(
    tmp_path: Path,
    src: str,
) -> None:
    assert not _findings(tmp_path, ForwarderHelperRule(), src)
    assert _fixed(tmp_path, ForwarderHelperRule(), src) is None


@pytest.mark.parametrize(
    ("src", "want"),
    [
        # The parameter's own method.
        (
            (
                "def f(a: str, b: str) -> str:\n    return _strip(a) + _strip(b)\n\n\n"
                "def _strip(s: str):\n    return s.strip()\n"
            ),
            "def f(a: str, b: str) -> str:\n    return a.strip() + b.strip()\n",
        ),
        # A cast spells the declared type, quoted or not.
        (
            (
                "def f(z: NpzFile) -> object:\n"
                '    return [_col(z, "a"), _col(z, "b")]\n\n\n'
                "def _col(npz: NpzFile, name: str) -> NDArray:\n"
                '    return cast("NDArray", npz[name])\n'
            ),
            (
                "def f(z: NpzFile) -> object:\n"
                "    return [cast(\"NDArray\", z['a']), cast(\"NDArray\", z['b'])]\n"
            ),
        ),
        # No parameters.
        (
            (
                "def f() -> object:\n    return [_prog(), _prog()]\n\n\n"
                "def _prog() -> Program:\n    return Program(rows=3, cols=4)\n"
            ),
            (
                "def f() -> object:\n"
                "    return [Program(rows=3, cols=4), Program(rows=3, cols=4)]\n"
            ),
        ),
        # An argument that computes.
        (
            (
                "def f(a: int, b: int) -> object:\n    return [_vec(a), _vec(b)]\n\n\n"
                "def _vec(n: int) -> object:\n    return tensor([n, 1, 2], dtype=float64)\n"
            ),
            (
                "def f(a: int, b: int) -> object:\n"
                "    return [tensor([a, 1, 2], dtype=float64), "
                "tensor([b, 1, 2], dtype=float64)]\n"
            ),
        ),
    ],
)
def test_forwarder_helper_inlines_any_one_line_call(
    tmp_path: Path,
    src: str,
    want: str,
) -> None:
    assert _fixed(tmp_path, ForwarderHelperRule(), src) == want


def test_forwarder_helper_inlines_a_public_name_called_once(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    fixed = _fixed(
        tmp_path,
        ForwarderHelperRule(),
        "def f(d: Path) -> Corpus:\n    return read_corpus(d)\n\n\n"
        "def read_corpus(directory: Path) -> Corpus:\n"
        '    return cast(Corpus, torch.load(directory / "corpus.pt"))\n',
    )
    assert fixed == (
        "def f(d: Path) -> Corpus:\n"
        '    return cast(Corpus, torch.load(d / "corpus.pt"))\n'
    )


def test_forwarder_helper_leaves_a_name_the_test_reaches(tmp_path: Path) -> None:
    (tmp_path / "mod_test.py").write_text(
        'def test_it(monkeypatch) -> None:\n    monkeypatch.setattr(mod, "_abs", abs)\n',
        encoding="utf-8",
    )
    src = (
        "def f(a: int, b: int) -> int:\n    return _abs(a) + _abs(b)\n\n\n"
        "def _abs(x: int):\n    return abs(x)\n"
    )
    assert not _findings(tmp_path, ForwarderHelperRule(), src)


def test_forwarder_helper_leaves_a_name_another_package_imports(
    tmp_path: Path,
) -> None:
    """``other_pkg/use.py`` reads ``mod_test._abs``: deleting it breaks that file."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "use.py").write_text(
        "from pkg import mod_test\n\n\nx = mod_test._abs(1)\n",
        encoding="utf-8",
    )
    src = (
        "def f(a: int, b: int) -> int:\n    return _abs(a) + _abs(b)\n\n\n"
        "def _abs(x: int):\n    return abs(x)\n"
    )
    assert not _findings(tmp_path, ForwarderHelperRule(), src, name="pkg/mod_test.py")


def test_forwarder_helper_inlines_a_public_name_only_this_file_calls(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    fixed = _fixed(
        tmp_path,
        ForwarderHelperRule(),
        "def f(a: int, b: int) -> int:\n    return absolute(a) + absolute(b)\n\n\n"
        "def absolute(x: int):\n    return abs(x)\n",
    )
    assert fixed == "def f(a: int, b: int) -> int:\n    return abs(a) + abs(b)\n"


def test_forwarder_helper_leaves_a_public_name_another_module_calls(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "user.py").write_text(
        "from pkg.mod import absolute\n\n\nx = absolute(1)\n",
        encoding="utf-8",
    )
    src = (
        "def f(a: int, b: int) -> int:\n    return absolute(a) + absolute(b)\n\n\n"
        "def absolute(x: int):\n    return abs(x)\n"
    )
    assert not _findings(tmp_path, ForwarderHelperRule(), src, name="pkg/mod.py")


def test_forwarder_helper_leaves_a_package_init(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    src = (
        "def f(a: int, b: int) -> int:\n    return absolute(a) + absolute(b)\n\n\n"
        "def absolute(x: int):\n    return abs(x)\n"
    )
    assert not _findings(tmp_path, ForwarderHelperRule(), src, name="pkg/__init__.py")


def test_forwarder_helper_owns_a_single_call_in_a_test_file(tmp_path: Path) -> None:
    """``single-use-helper`` defers a one-line call to it, so one rule fixes it."""
    src = (
        "def test_f() -> None:\n    assert _dungeon(0)\n\n\n"
        "def _dungeon(i: int) -> DungeonConfig:\n"
        '    return cast("DungeonConfig", CONFIGS.view(np.recarray)[i])\n'
    )
    assert not _findings(tmp_path, SingleUseHelperRule(), src, name="mod_test.py")
    assert _fixed(tmp_path, ForwarderHelperRule(), src, name="mod_test.py") == (
        "def test_f() -> None:\n"
        '    assert cast("DungeonConfig", CONFIGS.view(np.recarray)[0])\n'
    )


def test_forwarder_helper_fix_converges_on_nested_calls(tmp_path: Path) -> None:
    """``_m(_m(x)["a"])`` overlaps itself; each run inlines what it safely can."""
    src = (
        "def f(x: object) -> object:\n    return _m(_m(x)['a'])\n\n\n"
        "def g(y: object) -> object:\n    return _m(y)\n\n\n"
        "def _m(v: object) -> dict[str, object]:\n"
        "    return from_plain(v, dict[str, object])\n"
    )
    once = _fixed(tmp_path, ForwarderHelperRule(), src)
    assert once is not None
    compile(once, "mod.py", "exec")
    twice = _fixed(tmp_path, ForwarderHelperRule(), once)
    assert twice == (
        "def f(x: object) -> object:\n"
        "    return from_plain(from_plain(x, dict[str, object])['a'], dict[str, object])\n\n\n"
        "def g(y: object) -> object:\n    return from_plain(y, dict[str, object])\n"
    )
    assert _fixed(tmp_path, ForwarderHelperRule(), twice) is None


def test_forwarder_helper_fix_is_idempotent(tmp_path: Path) -> None:
    src = (
        "def f(a: int, b: int) -> int:\n    return _abs(a) + _abs(b)\n\n\n"
        "def _abs(x: int):\n    return abs(x)\n"
    )
    once = _fixed(tmp_path, ForwarderHelperRule(), src)
    assert once == "def f(a: int, b: int) -> int:\n    return abs(a) + abs(b)\n"
    assert _fixed(tmp_path, ForwarderHelperRule(), once) is None


# -- single-use-helper and forwarder-helper: bindings ----------------------


@pytest.mark.parametrize(
    "src",
    [
        (
            "def use(x: object) -> list[int]:\n    return f(x)\n\n\n"
            "def f[T](x: object) -> list[T]:\n    return cast(list[T], g(x))\n"
        ),
        (
            "def use(x: object, y: object) -> list[int]:\n"
            "    return f(x) + f(y) + f(x)\n\n\n"
            "def f[T](x: object) -> list[T]:\n    return cast(list[T], g(x))\n"
        ),
        # Only the signature names ``T``; it is still what each call solves.
        (
            "def use(xs: list[int]) -> int:\n    return _first(xs)\n\n\n"
            "def _first[T](xs: list[T]) -> T:\n    return next(iter(xs))\n"
        ),
    ],
    ids=["called-once", "called-several-times", "signature-only"],
)
def test_helper_inlining_leaves_a_def_with_type_parameters(
    tmp_path: Path,
    src: str,
) -> None:
    """``T`` is bound only inside the def: inlined ``list[T]`` is ruff F821."""
    for rule in (SingleUseHelperRule(), ForwarderHelperRule()):
        assert not _findings(tmp_path, rule, src)
        assert _fixed(tmp_path, rule, src) is None


_INC = "n = 1\n\n\ndef _inc(x: int) -> int:\n    return x + n\n\n\n"


@pytest.mark.parametrize(
    "src",
    [
        # A substituted default reads a name the call site rebinds.
        (
            "size = 4\n\n\n"
            "def f(n: int, size: int) -> int:\n    return _fit(n)\n\n\n"
            "def _fit(x: int, cap: int = size) -> int:\n    return min(x, cap)\n"
        ),
        # An enclosing def binds the body's global: the closure captures it.
        _INC + "def f() -> int:\n    n = 100\n\n"
        "    def inner(y: int) -> int:\n        return _inc(y)\n\n"
        "    return inner(1) + n\n",
        # ``nonlocal`` reaches the enclosing def's binding, not the global.
        _INC + "def f() -> int:\n    n = 100\n\n"
        "    def inner(y: int) -> int:\n        nonlocal n\n        return _inc(y)\n\n"
        "    return inner(1)\n",
        # A lambda parameter.
        _INC + "def f(xs: list[int]) -> list[int]:\n"
        "    return sorted(xs, key=lambda n: _inc(n))\n",
        # A comprehension target at module scope, where no def encloses the call.
        _INC + "BUMPED = [_inc(0) for n in range(3)]\n",
        # A class body's own name, read by a statement directly in it.
        _INC + "class A:\n    n = 5\n    bumped = _inc(0)\n",
        # An import inside the caller.
        (
            "import math\n\n\n"
            "def _root(x: float) -> float:\n    return math.sqrt(x)\n\n\n"
            "def f() -> float:\n    import cmath as math\n\n    return _root(4.0)\n"
        ),
        # An ``except ... as`` name.
        _INC + "def f() -> int:\n    try:\n        return 0\n"
        "    except ValueError as n:\n        return _inc(0)\n",
        # A nested def's name.
        _INC + "def f() -> int:\n    def n() -> None: ...\n\n    return _inc(0)\n",
        # A match capture.
        _INC + "def f(v: int) -> int:\n    match v:\n"
        "        case n:\n            return _inc(0)\n",
        # A match mapping's rest.
        _INC + "def f(v: object) -> int:\n    match v:\n"
        "        case {**n}:\n            return _inc(0)\n",
        # The body's generator binds the argument's name: ``x + x``.
        (
            "x = 10\n\n\n"
            "def _shift(xs: list[int], k: int) -> list[int]:\n"
            "    return list(x + k for x in xs)\n\n\n"
            "SHIFTED = _shift([1, 2], x)\n"
        ),
        # The body's lambda binds the argument's name.
        (
            "x = 10\n\n\n"
            "def _adder(k: int) -> Callable[[int], int]:\n"
            "    return lambda x: x + k\n\n\n"
            "ADD = _adder(x)\n"
        ),
        # In a class body, the argument moves into a generator that cannot see
        # the class's ``k``.
        (
            "def _shift(xs: list[int], k: int) -> list[int]:\n"
            "    return list(v + k for v in xs)\n\n\n"
            "class A:\n    k = 2\n    shifted = _shift([1], k)\n"
        ),
        # The body's walrus binds ``y`` in the caller, shadowing the global.
        (
            "y = 100\n\n\n"
            "def _h(v: int) -> int:\n    return (y := v) + 1\n\n\n"
            "def f() -> int:\n    z = _h(3)\n    return z + y\n"
        ),
    ],
    ids=[
        "default",
        "closure",
        "nonlocal",
        "lambda-parameter",
        "module-comprehension",
        "class-body",
        "import",
        "except-as",
        "nested-def",
        "match-capture",
        "match-rest",
        "body-generator",
        "body-lambda",
        "class-body-argument",
        "body-walrus",
    ],
)
def test_helper_inlining_leaves_a_name_it_would_rebind(
    tmp_path: Path,
    src: str,
) -> None:
    """Each name in the inlined text must resolve where it lands as it did before."""
    for rule in (SingleUseHelperRule(), ForwarderHelperRule()):
        assert not _findings(tmp_path, rule, src)
        assert _fixed(tmp_path, rule, src) is None


def test_helper_inlining_ignores_a_class_name_a_method_cannot_see(
    tmp_path: Path,
) -> None:
    """A def in a class body reads ``n`` from the module, before and after."""
    fixed = _fixed(
        tmp_path,
        SingleUseHelperRule(),
        _INC + "class A:\n    n = 5\n\n"
        "    def m(self, y: int) -> int:\n        return _inc(y)\n",
    )
    assert fixed == (
        "n = 1\n\n\nclass A:\n    n = 5\n\n"
        "    def m(self, y: int) -> int:\n        return y + n\n"
    )


def test_helper_inlining_keeps_a_body_generator_that_binds_no_argument(
    tmp_path: Path,
) -> None:
    fixed = _fixed(
        tmp_path,
        SingleUseHelperRule(),
        "def f(items: list[Row]) -> int:\n    return _total(items)\n\n\n"
        "def _total(rows: list[Row]) -> int:\n    return sum(r.n for r in rows)\n",
    )
    assert (
        fixed == "def f(items: list[Row]) -> int:\n    return sum(r.n for r in items)\n"
    )


# -- single-use-helper and forwarder-helper: evaluation --------------------


@pytest.mark.parametrize(
    "src",
    [
        (
            "def f(c: bool) -> int:\n    return _pick(c, load())\n\n\n"
            "def _pick(c: bool, x: int) -> int:\n    return x if c else 0\n"
        ),
        (
            "def f(c: bool) -> bool:\n    return _both(c, load())\n\n\n"
            "def _both(a: bool, b: bool) -> bool:\n    return a and b\n"
        ),
        (
            "def f() -> int:\n    return _total(load(), 3)\n\n\n"
            "def _total(x: int, n: int) -> int:\n    return sum(x for _ in range(n))\n"
        ),
        (
            "def f() -> list[Callable[[int], int]]:\n"
            "    return [_adder(i) for i in range(3)]\n\n\n"
            "def _adder(k: int) -> Callable[[int], int]:\n    return lambda v: v + k\n"
        ),
        (
            "import itertools\n\ncounter = itertools.count()\n\n\n"
            "def f(x: int) -> int:\n    return _tag(x)\n\n\n"
            "def _tag(x: int, n: int = next(counter)) -> int:\n    return x + n\n"
        ),
        (
            "LIMIT = 4\n\n\ndef f(x: int) -> int:\n    return _fit(x)\n\n\n"
            "def _fit(x: int, cap: int = LIMIT) -> int:\n    return min(x, cap)\n"
        ),
        (
            "def f() -> int:\n    return _sub(first(), second())\n\n\n"
            "def _sub(a: int, b: int) -> int:\n    return b - a\n"
        ),
        (
            "def f() -> Callable[[int], int]:\n    return _inc(load())\n\n\n"
            "def _inc(x: int) -> Callable[[int], int]:\n    return lambda x: x + 1\n"
        ),
    ],
    ids=[
        "conditional",
        "short-circuit",
        "generator",
        "closure-in-loop",
        "default-call",
        "default-name",
        "argument-order",
        "shadowed-parameter",
    ],
)
def test_helper_inlining_leaves_an_argument_it_would_evaluate_differently(
    tmp_path: Path,
    src: str,
) -> None:
    """The call ran each argument once, in order, before the body; so must the text."""
    for rule in (SingleUseHelperRule(), ForwarderHelperRule()):
        assert not _findings(tmp_path, rule, src)
        assert _fixed(tmp_path, rule, src) is None


# -- lazy-import-cost ------------------------------------------------------


_ONE_OF_FOUR_DEFS = "".join(
    f"def g{i}() -> int:\n    return {i}\n\n\n" for i in range(3)
)


def test_lazy_import_cost_flags_a_heavy_import_one_helper_needs(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        LazyImportCostRule(),
        "import torch\n\n\n"
        + _ONE_OF_FOUR_DEFS
        + "def f() -> None:\n    torch.zeros(1)\n",
    )
    assert [f.line for f in findings] == [1]
    assert "lazy_import" in findings[0].message
    assert "1 of 4" in findings[0].message


def test_lazy_import_cost_findings_and_boundaries(tmp_path: Path) -> None:
    """One of four reaching torch is exactly the 0.25 fraction: still reported."""
    body = (
        "import torch.nn as tnn, json\n"
        "import yaml\n"
        "import numpy\n"
        "async def a():\n    return tnn\n"
        "def b():\n    return 1\n"
        "def c():\n    return 2\n"
        "def d():\n    return isinstance(1, int)\n"
    )
    findings = _findings(tmp_path, LazyImportCostRule(threshold_ms=10), body)
    costs = houselint_rules._IMPORT_COSTS_MS
    tail = "defs reach it; bind it with `lazy_import` (STYLE.md Code Organization)"
    assert [(f.line, f.message) for f in findings] == [
        (
            1,
            f"'torch.nn' costs ~{costs['torch']} ms to import and only 1 of 4 {tail}",
        ),
        (2, f"'yaml' costs ~{costs['yaml']} ms to import and only 0 of 4 {tail}"),
    ]
    one_def = "import torch\ndef f():\n    return torch\n"
    assert _findings(tmp_path, LazyImportCostRule(), one_def) == ()


def test_lazy_import_cost_flags_a_from_import_the_same_way(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        LazyImportCostRule(),
        "from torch import zeros\n\n\n"
        + _ONE_OF_FOUR_DEFS
        + "def f() -> None:\n    zeros(1)\n",
    )
    assert [f.line for f in findings] == [1]


def test_lazy_import_cost_leaves_a_runtime_type_check_import(
    tmp_path: Path,
) -> None:
    body = (
        "from torch import Tensor\n\n\n"
        + _ONE_OF_FOUR_DEFS
        + "def f(value: object) -> bool:\n    return isinstance(value, Tensor)\n"
    )
    assert not _findings(tmp_path, LazyImportCostRule(), body)


def test_lazy_import_cost_leaves_a_module_whose_purpose_is_the_import(
    tmp_path: Path,
) -> None:
    """Every def touches torch: this IS a torch module; deferring buys nothing."""
    assert not _findings(
        tmp_path,
        LazyImportCostRule(),
        "import torch\n\n\n"
        "def f() -> None:\n    torch.zeros(1)\n\n\n"
        "def g() -> None:\n    torch.ones(1)\n",
    )


def test_lazy_import_cost_leaves_a_module_scope_use(tmp_path: Path) -> None:
    """A base class or decorator needs the real module at import time."""
    assert not _findings(
        tmp_path,
        LazyImportCostRule(),
        "from torch import nn\n\n\n"
        + _ONE_OF_FOUR_DEFS
        + "class M(nn.Module):\n    pass\n",
    )


def test_lazy_import_cost_leaves_a_root_a_sibling_import_implies(
    tmp_path: Path,
) -> None:
    """``import torch`` already loads numpy; deferring numpy saves nothing."""
    body = (
        "import numpy as np\nimport torch\n\n\n"
        + _ONE_OF_FOUR_DEFS
        + "def f() -> None:\n    np.zeros(1)\n\n\n"
        "def h() -> None:\n    torch.zeros(1)\n"
    )
    findings = _findings(tmp_path, LazyImportCostRule(), body)
    assert [f.line for f in findings] == [2]


def test_lazy_import_cost_sees_uses_of_a_try_guarded_import(tmp_path: Path) -> None:
    """``try: import torch / except ImportError`` binds ``torch`` like any import."""
    assert not _findings(
        tmp_path,
        LazyImportCostRule(),
        "try:\n    import torch\nexcept ImportError:\n    torch = None\n\n\n"
        "def f() -> None:\n    torch.zeros(1)\n\n\n"
        "def g() -> None:\n    torch.ones(1)\n",
    )


def test_lazy_import_cost_leaves_a_type_checking_import(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        LazyImportCostRule(),
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import torch\n\n\n"
        + _ONE_OF_FOUR_DEFS
        + "def f(x: torch.Tensor) -> None:\n    del x\n",
    )


def test_lazy_import_cost_leaves_a_cheap_import(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        LazyImportCostRule(),
        "import json\n\n\n"
        + _ONE_OF_FOUR_DEFS
        + "def f() -> str:\n    return json.dumps(1)\n",
    )


def test_lazy_import_cost_thresholds_are_settings(tmp_path: Path) -> None:
    body = (
        "import yaml\n\n\n" + _ONE_OF_FOUR_DEFS + "def f() -> None:\n    yaml.dump(1)\n"
    )
    assert not _findings(tmp_path, LazyImportCostRule(), body)
    assert _findings(tmp_path, LazyImportCostRule(threshold_ms=10), body)
    two_of_four = (
        "import torch\n\n\n"
        "def g0() -> int:\n    return 0\n\n\n"
        "def g1() -> int:\n    return 1\n\n\n"
        "def f() -> None:\n    torch.zeros(1)\n\n\n"
        "def h() -> None:\n    torch.ones(1)\n"
    )
    assert not _findings(tmp_path, LazyImportCostRule(), two_of_four)
    assert _findings(tmp_path, LazyImportCostRule(max_use_fraction=0.5), two_of_four)


# -- assert ------------------------------------------------------------------


def test_assert_is_reported_and_raised(tmp_path: Path) -> None:
    body = "def f(x: int) -> None:\n    assert x > 0\n"
    assert [f.line for f in _findings(tmp_path, AssertRule(), body)] == [2]
    assert _fixed(tmp_path, AssertRule(), body) == (
        "def f(x: int) -> None:\n"
        "    if x <= 0:\n"
        '        raise ValueError("Expected x > 0.")\n'
    )


@pytest.mark.parametrize(
    "read",
    [
        "from_plain(torch.load(path), Corpus)",
        "from_plain(cast(object, torch.load(path)), Corpus)",
        "from_plain(\n    torch.load(path, weights_only=True),\n    dict[str, object],\n)",
    ],
)
def test_treekle_over_a_torch_load_is_reported(tmp_path: Path, read: str) -> None:
    body = f"corpus = {read}\n"
    assert [f.line for f in _findings(tmp_path, TreekleTorchLoadRule(), body)] == [1]
    assert _fixed(tmp_path, TreekleTorchLoadRule(), body) is None


@pytest.mark.parametrize(
    "read",
    [
        "from_plain(state, dict[str, object])",
        "from_plain(state['step']['model'], dict[str, Tensor])",
        "from_plain(state.get('model'), dict[str, Tensor])",
    ],
)
def test_treekle_over_a_payload_bound_to_a_name_is_reported(
    tmp_path: Path,
    read: str,
) -> None:
    body = f"state = torch.load(path, weights_only=True)\nmodel = {read}\n"
    assert [f.line for f in _findings(tmp_path, TreekleTorchLoadRule(), body)] == [2]


def test_a_to_plain_field_of_a_payload_is_not_reported(tmp_path: Path) -> None:
    body = (
        "payload = torch.load(path, weights_only=True)\n"
        "config = from_plain(payload['config'], Model.Config, allow_imports=True)\n"
    )
    assert not _findings(tmp_path, TreekleTorchLoadRule(), body)


def test_a_name_bound_to_a_plain_tree_is_not_reported(tmp_path: Path) -> None:
    body = (
        "state = torch.load(path)\n"
        "config = json.loads(text)\n"
        "model = from_plain(config['model'], Model)\n"
    )
    assert not _findings(tmp_path, TreekleTorchLoadRule(), body)


@pytest.mark.parametrize(
    "read",
    [
        "cast(Corpus, torch.load(path))",
        "from_plain(json.loads(text), Corpus)",
        "from_plain(payload['config'], FPNet.Config)",
    ],
)
def test_a_cast_load_or_a_plain_tree_read_is_not_reported(
    tmp_path: Path,
    read: str,
) -> None:
    body = f"corpus = {read}\n"
    assert not _findings(tmp_path, TreekleTorchLoadRule(), body)


def test_a_marked_assert_is_left_for_its_author(tmp_path: Path) -> None:
    """``house-ignore[assert]`` keeps a check ``-O`` may drop; the fix skips it."""
    body = (
        "def f(x: float) -> None:\n"
        "    assert x >= 0  # house-ignore[assert] -- Suppressible on purpose.\n"
        "    assert x < 9\n"
    )
    assert _fixed(tmp_path, AssertRule(), body) == (
        "def f(x: float) -> None:\n"
        "    assert x >= 0  # house-ignore[assert] -- Suppressible on purpose.\n"
        "    if x >= 9:\n"
        '        raise ValueError("Expected x < 9.")\n'
    )


def test_assert_message_becomes_the_exception_argument(tmp_path: Path) -> None:
    body = 'assert x is not None, f"missing {name}"\n'
    assert _fixed(tmp_path, AssertRule(), body) == (
        'if x is None:\n    raise ValueError(f"missing {name}")\n'
    )


def test_assert_tuple_message_is_kept_as_a_tuple(tmp_path: Path) -> None:
    body = "assert a == b, (a, b)\n"
    assert _fixed(tmp_path, AssertRule(), body) == (
        "if a != b:\n    raise ValueError((a, b))\n"
    )


@pytest.mark.parametrize(
    ("test", "negated"),
    [
        ("a == b", "a != b"),
        ("a != b", "a == b"),
        ("a is b", "a is not b"),
        ("a in b", "a not in b"),
        ("a not in b", "a in b"),
        ("x < 0", "x >= 0"),
        ("x <= 1", "x > 1"),
        ("x > 0", "x <= 0"),
        ("x >= 1", "x < 1"),
        ("not done", "done"),
        ("a and b", "not a or not b"),
        ("a or b", "not a and not b"),
        ("ok", "not ok"),
        ("f(x)", "not f(x)"),
        ("0 < x", "0 >= x"),
        ("x <= 1.5", "x > 1.5"),
        ('p in (d or "")', 'p not in (d or "")'),
        ("a and (b or c)", "not a or (not b and not c)"),
        ("x == (y if c else z)", "x != (y if c else z)"),
    ],
)
def test_assert_negation_shapes(tmp_path: Path, test: str, negated: str) -> None:
    """Ordering flips only against a literal bound (STYLE.md NaN rule); else ``not``."""
    # Preceded by other lines so a segment sliced at the wrong offsets shows.
    fixed = _fixed(tmp_path, AssertRule(), f"x = 1\ny = 2\n\nassert {test}\n")
    assert fixed is not None
    stmt = ast.parse(fixed).body[-1]
    assert isinstance(stmt, ast.If)
    assert ast.unparse(stmt.test) == ast.unparse(ast.parse(negated, mode="eval").body)
    assert not _findings(tmp_path, NegatedCompareRule(), fixed)


@pytest.mark.parametrize(
    "test",
    ["0 <= x <= 1", "a <= b", "ok and a < b"],
)
def test_an_assert_whose_negation_is_a_negated_ordering_is_reported_only(
    tmp_path: Path,
    test: str,
) -> None:
    """``not a <= b`` is what ``negated-compare`` forbids; the author picks the NaN case."""
    body = f"assert {test}\n"
    assert _findings(tmp_path, AssertRule(), body)
    assert _fixed(tmp_path, AssertRule(), body) is None


@pytest.mark.parametrize(
    ("test", "negated"),
    [
        ("not isinstance(x, int)", "isinstance(x, int)"),
        ("type(x) is int", "type(x) is not int"),
        ("callable(f)", "not callable(f)"),
        (
            "isinstance(x, int) or isinstance(x, str)",
            "not isinstance(x, int) and not isinstance(x, str)",
        ),
    ],
)
def test_assert_type_check_raises_type_error(
    tmp_path: Path,
    test: str,
    negated: str,
) -> None:
    """A failed type check is a TypeError (ruff TRY004), not a ValueError."""
    fixed = _fixed(tmp_path, AssertRule(), f"assert {test}, msg\n")
    assert fixed == f"if {negated}:\n    raise TypeError(msg)\n"


def test_assert_isinstance_is_clean(tmp_path: Path) -> None:
    body = (
        "assert isinstance(x, int)\n"
        "assert isinstance(x, int) and isinstance(y, str)\n"
        "assert isinstance(x, int), x\n"
    )
    assert not _findings(tmp_path, AssertRule(), body)
    assert _fixed(tmp_path, AssertRule(), body) is None


def test_assert_multiline_test_is_reflowed(tmp_path: Path) -> None:
    """The wrapping was chosen for ``assert (``; the formatter re-wraps the ``if``."""
    body = "assert (\n    a\n    == b\n), (\n    a,\n    b,\n)\n"
    assert _fixed(tmp_path, AssertRule(), body) == (
        "if a != b:\n    raise ValueError((\n    a,\n    b,\n))\n"
    )
    body = "assert not (\n    x.ready\n    or y.ready\n), msg\n"
    assert _fixed(tmp_path, AssertRule(), body) == (
        "if x.ready or y.ready:\n    raise ValueError(msg)\n"
    )
    body = "assert (\n    a\n    == b\n)\n"
    assert _fixed(tmp_path, AssertRule(), body) == (
        'if a != b:\n    raise ValueError("Expected a == b.")\n'
    )


def test_assert_trailing_comment_moves_to_the_if(tmp_path: Path) -> None:
    body = "assert a is not None  # Set by finalize().\n"
    assert _fixed(tmp_path, AssertRule(), body) == (
        "if a is None:  # Set by finalize().\n"
        '    raise ValueError("Expected a is not None.")\n'
    )
    body = "assert a == b, (\n    a,\n    b,\n)  # Invariant.\n"
    assert _fixed(tmp_path, AssertRule(), body) == (
        "if a != b:  # Invariant.\n    raise ValueError((\n    a,\n    b,\n))\n"
    )


def test_assert_with_an_inner_comment_is_reported_not_fixed(tmp_path: Path) -> None:
    body = "assert (\n    a  # Left.\n    == b\n)\n"
    assert _findings(tmp_path, AssertRule(), body)
    assert _fixed(tmp_path, AssertRule(), body) is None


def test_assert_multiline_message_is_fixed(tmp_path: Path) -> None:
    body = 'assert a == b, (\n    f"{a} "\n    f"{b}"\n)\n'
    fixed = _fixed(tmp_path, AssertRule(), body)
    assert fixed == 'if a != b:\n    raise ValueError(f"{a} "\n    f"{b}")\n'
    ast.parse(fixed)


# -- logging-fstring ---------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "fixed"),
    [
        ('logger.info(f"a {x} b")', 'logger.info("a %s b", x)'),
        ('logger.info(f"{x:.2f}s")', 'logger.info("%.2fs", x)'),
        ('logger.info(f"{x!r}")', 'logger.info("%r", x)'),
        ('logger.info(f"{x:,}")', 'logger.info("%s", f"{x:,}")'),
        ('logger.info(f"{a} {b}", extra=e)', 'logger.info("%s %s", a, b, extra=e)'),
        ('log.warning(f"100% {x}")', 'log.warning("100%% %s", x)'),
        ('self.logger.debug(f"{x}")', 'self.logger.debug("%s", x)'),
        ('logging.info(f"{x}")', 'logging.info("%s", x)'),
        ("logger.exception(f\"{a['k']}\")", "logger.exception(\"%s\", a['k'])"),
        ('logger.info(f"{x:>8}")', 'logger.info("%8s", x)'),
        ('logger.info(f"{x:<8}")', 'logger.info("%-8s", x)'),
        ('logger.info(f"{x:d}")', 'logger.info("%d", x)'),
        ('logger.info(f"{x:.1%}")', 'logger.info("%s", f"{x:.1%}")'),
    ],
)
def test_logging_fstring_is_rewritten(tmp_path: Path, call: str, fixed: str) -> None:
    body = f"import logging\n\n{call}\n"
    assert [f.line for f in _findings(tmp_path, LoggingFstringRule(), body)] == [3]
    assert (
        _fixed(tmp_path, LoggingFstringRule(), body) == f"import logging\n\n{fixed}\n"
    )


def test_logging_fstring_non_ascii_message_splices_at_characters(
    tmp_path: Path,
) -> None:
    """``ast`` columns are UTF-8 bytes; a ``→`` before the call end shifted the cut."""
    body = 'logger.info(f"Converted {n} nn.Linear → Float8Linear")\nreturn model\n'
    assert _fixed(tmp_path, LoggingFstringRule(), body) == (
        'logger.info("Converted %s nn.Linear → Float8Linear", n)\nreturn model\n'
    )


def test_logging_fstring_implicit_concat_is_rewritten(tmp_path: Path) -> None:
    body = 'logger.info(\n    f"a {x} "\n    f"b {y}",\n)\n'
    assert _fixed(tmp_path, LoggingFstringRule(), body) == (
        'logger.info(\n    "a %s b %s", x, y,\n)\n'
    )


def test_logging_fstring_nested_quote_is_reported_not_fixed(tmp_path: Path) -> None:
    """A message that already has an f-string argument list is not one we build."""
    body = 'logger.info(f"{x}", y)\n'
    assert _findings(tmp_path, LoggingFstringRule(), body)
    assert _fixed(tmp_path, LoggingFstringRule(), body) is None


def test_logging_fstring_leaves_non_logging_calls(tmp_path: Path) -> None:
    body = 'print(f"{x}")\nconsole.log(f"{x}")\nlogger.setLevel(f"{x}")\n'
    assert not _findings(tmp_path, LoggingFstringRule(), body)


def test_logging_fstring_leaves_percent_calls(tmp_path: Path) -> None:
    assert not _findings(tmp_path, LoggingFstringRule(), 'logger.info("%s", x)\n')


def test_logging_fstring_receivers_levels_and_quotes(tmp_path: Path) -> None:
    """Single-quoted f-strings, ``.log`` level offsets, and dotted receivers."""
    body = (
        "print(f'{a}')\n"
        "foo.info(f'{a}')\n"
        "a.b.logger.info(f'{a}')\n"
        "self._logger.warning(f'{a}')\n"
        "a.logger.b.info(f'{a}')\n"
        "logger.log(level)\n"
        "logger.log(logging.INFO, f'{a}')\n"
        "logger.log(f'{a}')\n"
    )
    message = (
        "f-string in a logging call formats eagerly; pass a %-format "
        "and arguments (rerun with --fix logging-fstring)"
    )
    assert [
        (f.line, f.message) for f in _findings(tmp_path, LoggingFstringRule(), body)
    ] == [(3, message), (4, message), (7, message)]


def test_logging_fstring_fix_is_idempotent(tmp_path: Path) -> None:
    body = 'logger.info(f"a {x:.1f} {y!r}")\nlogger.info("%s", z)\n'
    once = _fixed(tmp_path, LoggingFstringRule(), body)
    assert once is not None
    ast.parse(once)
    assert _fixed(tmp_path, LoggingFstringRule(), once) is None


def test_assert_fix_is_idempotent_and_parses(tmp_path: Path) -> None:
    body = "def f() -> None:\n    assert a\n    assert isinstance(a, int)\n    assert b, msg\n"
    once = _fixed(tmp_path, AssertRule(), body)
    assert once is not None
    ast.parse(once)
    assert _fixed(tmp_path, AssertRule(), once) is None


# -- coverage branches --------------------------------------------------------


def test_invalid_syntax_units_are_left_alone(tmp_path: Path) -> None:
    body = "class [broken\n"
    for rule in (
        DataclassRule(),
        StaticMethodRule(),
        XdgLiteralRule(),
        UserdirsNamespaceRule(),
        UserdirsWrapperRule(),
        CacheKwargRule(),
        LazyImportBindingRule(),
        TmpLiteralRule(),
        TempfilePrefixRule(),
        CwdFinalRule(),
        ImportAliasRule(),
        ImportHoistRule(),
        DottedImportRule(),
        GetLoggerNameRule(),
        NestedTryRule(),
        NegativeBoolParamRule(),
        AllReexportRule(),
        InitFacadeRule(),
        DefOrderRule(),
        UnderscoreParamRule(),
    ):
        assert not _findings(tmp_path, rule, body)
        assert _fixed(tmp_path, rule, body) is None


def test_dataclass_zero_arg_super_in_a_slotted_class(tmp_path: Path) -> None:
    body = (
        "from dataclasses import dataclass\n"
        "@dataclass(kw_only=True, slots=True)\n"
        "class C:\n"
        "    def f(self):\n"
        "        return super().f()\n"
    )
    assert [
        (f.line, f.message) for f in _findings(tmp_path, DataclassRule(), body)
    ] == [
        (
            5,
            (
                "zero-arg super() in slotted dataclass C; use super(C, self) -- "
                "slots=True rebuilds the class and the zero-arg form raises below "
                "Python 3.14"
            ),
        ),
    ]


def test_dataclass_config_and_staticmethod_edge_shapes(tmp_path: Path) -> None:
    assert _findings(
        tmp_path,
        DataclassRule(),
        "from dataclasses import dataclass\n@dataclass\nclass C: ...\n",
    )
    assert _findings(
        tmp_path,
        StaticMethodRule(),
        "class C:\n    x = staticmethod(f)\n",
    )
    assert _findings(
        tmp_path,
        StaticMethodRule(),
        "class C:\n    @staticmethod\n    def f(): ...\n",
    )
    assert _findings(
        tmp_path,
        ConfigGlobalsRule(),
        "DEFAULT = 3\nclass C(Fig[int]):\n    x: int = DEFAULT\n",
    )


def test_digits_notebook_malformed_json_is_unchanged(tmp_path: Path) -> None:
    assert _fixed(tmp_path, DigitsRule(), "{bad 12345", name="x.ipynb") is None


@pytest.mark.parametrize(
    "notebook",
    [
        [12_345],
        {"metadata": 12_345},
        {"cells": 12_345},
        {"cells": [12_345]},
        {"cells": [{"cell_type": 1, "source": ["x = 12345\n"]}]},
        {"cells": [{"cell_type": "code", "source": 12_345}]},
    ],
)
def test_digits_leaves_a_malformed_notebook_alone(
    tmp_path: Path,
    notebook: object,
) -> None:
    body = json.dumps(notebook)
    assert _fixed(tmp_path, DigitsRule(), body, name="x.ipynb") is None


def test_digits_groups_a_notebook_cell_whose_source_is_one_string(
    tmp_path: Path,
) -> None:
    """Nbformat's ``multiline_string`` is a string or a list of strings."""
    notebook = json.dumps(
        {
            "cells": [
                12_345,
                {"cell_type": "code", "source": "x = 12345\ny = 1\n"},
            ],
        },
    )

    fixed = _fixed(tmp_path, DigitsRule(), notebook, name="n.ipynb")

    assert fixed is not None
    assert json.loads(fixed)["cells"] == [
        12_345,
        {"cell_type": "code", "source": ["x = 12_345\n", "y = 1\n"]},
    ]


@pytest.mark.parametrize("name", ["mod.py", "mod.pyi"])
def test_digits_groups_a_long_literal_in_code(tmp_path: Path, name: str) -> None:
    assert (
        _fixed(tmp_path, DigitsRule(), "x = 12345\ny = 1234\n", name=name)
        == "x = 12_345\ny = 1234\n"
    )


def test_digits_threshold_is_the_min_digits_setting(tmp_path: Path) -> None:
    rule = DigitsRule(min_digits=6)

    assert _fixed(tmp_path, rule, "x = 12345\n") is None
    assert _fixed(tmp_path, rule, "x = 123456\n") == "x = 123_456\n"


def test_digits_groups_a_notebook_code_cell(tmp_path: Path) -> None:
    notebook = json.dumps(
        {"cells": [{"cell_type": "code", "source": ["x = 12345\n"]}]},
    )

    fixed = _fixed(tmp_path, DigitsRule(), notebook, name="n.ipynb")

    assert fixed is not None
    assert json.loads(fixed)["cells"][0]["source"] == ["x = 12_345\n"]


def test_digits_rewrites_a_notebook_as_nbformat_writes_it(tmp_path: Path) -> None:
    """One-space indent and raw UTF-8, as Jupyter saves; non-string lines drop."""
    notebook = json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["s = 'caf\u00e9'\n", 3, "x = 12345\n"],
                },
            ],
        },
    )

    assert _fixed(tmp_path, DigitsRule(), notebook, name="n.ipynb") == (
        '{\n "cells": [\n  {\n   "cell_type": "code",\n   "source": [\n'
        '    "s = \'caf\u00e9\'\\n",\n    "x = 12_345\\n"\n   ]\n  }\n ]\n}\n'
    )


def test_digits_leaves_a_grouped_notebook_alone(tmp_path: Path) -> None:
    notebook = json.dumps(
        {"cells": [{"cell_type": "code", "source": ["x = 12_345 + 99999.5\n"]}]},
    )

    assert _fixed(tmp_path, DigitsRule(), notebook, name="n.ipynb") is None


def test_digits_reports_one_finding_per_file(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text("x = 12345\ny = 67890\n")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)

    assert DigitsRule(min_digits=5).check(unit) == (
        Finding(
            path=path,
            line=1,
            rule="digits",
            message=(
                "integer literal of >= 5 digits without underscore separators "
                "(rerun with --fix digits)"
            ),
        ),
    )


def test_xdg_docstring_is_clean(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        XdgLiteralRule(),
        '"""Path.home() is documented here."""\n',
    )


def test_userdirs_namespace_and_wrapper_nonliteral_paths(tmp_path: Path) -> None:
    assert _findings(tmp_path, UserdirsNamespaceRule(), "x = state_dir() / name\n")
    assert _findings(
        tmp_path,
        UserdirsWrapperRule(),
        'ROOT = state_dir() / "rekursiv-ai"\n',
    )


def test_cache_local_api_is_not_reported(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        CacheKwargRule(),
        "def load(*, cache_dir): ...\nload(cache_dir=cache_dir)\n",
    )


def test_tempfile_fix_handles_direct_import_with_keyword(tmp_path: Path) -> None:
    assert (
        _fixed(
            tmp_path,
            TempfilePrefixRule(),
            "from tempfile import mkstemp\nmkstemp(dir=base)\n",
            name="sample.py",
        )
        == 'from tempfile import mkstemp\nmkstemp(prefix="sample-", dir=base)\n'
    )


def test_import_alias_fix_preserves_relative_and_plain_imports(tmp_path: Path) -> None:
    assert (
        _fixed(
            tmp_path,
            ImportAliasRule(),
            "from .mod import x as y\nimport json\n",
        )
        is None
    )


def test_init_facade_allows_type_checking_block(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        InitFacadeRule(),
        "if TYPE_CHECKING:\n    from .x import Y\n",
        name="__init__.py",
    )


def test_module_side_effects_reports_unhandled_try(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        ModuleSideEffectsRule(),
        "try:\n    work()\nexcept Exception:\n    pass\n",
    )
    assert [f.message for f in findings] == [
        "try at module scope; move it into a function",
    ]


def test_underscore_param_empty_and_varargs_are_handled(tmp_path: Path) -> None:
    assert not _findings(tmp_path, UnderscoreParamRule(), "def f(_, __x__): ...\n")
    assert (
        _fixed(
            tmp_path,
            UnderscoreParamRule(),
            "def f(_x: int, /, *args):\n    return 1\n",
        )
        == "def f(x: int, /, *args):\n    del x\n    return 1\n"
    )


def test_negated_ternary_multiline_is_reported_not_fixed(tmp_path: Path) -> None:
    body = "x = (a if not cond else\n b)\n"
    assert _findings(tmp_path, NegatedTernaryRule(), body)
    assert _fixed(tmp_path, NegatedTernaryRule(), body) is None


def test_super_only_nonforwarding_and_varargs_are_clean(tmp_path: Path) -> None:
    body = "class C(A):\n    def f(self, *args, **kwargs):\n        return super().f(args, **kwargs)\n"
    assert not _findings(tmp_path, SuperOnlyRule(), body)
    assert _fixed(tmp_path, SuperOnlyRule(), body) is None


def test_returns_named_ignores_other_sections_and_empty_docs(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        ReturnsNamedRule(),
        'def f() -> int:\n    """Do.\n\n    Args:\n      x: X.\n\n    """\n    return 1\n',
    )


def test_args_sections_include_vararg_and_kwarg(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        ArgsSectionRule(),
        'def f(*args, **kwargs):\n    """Do.\n\n    More.\n    """\n',
    )
    assert [f.message for f in findings] == ["f docstring lacks Args:"]


def test_private_oneliner_empty_docstring_and_section_continuation(
    tmp_path: Path,
) -> None:
    assert (
        _fixed(
            tmp_path,
            PrivateOnelinerRule(),
            'def _f():\n    """\n\n    """\n',
        )
        == 'def _f():\n    """"""\n'
    )
    assert (
        _fixed(
            tmp_path,
            PrivateOnelinerRule(),
            'def _f():\n    """Summary.\n\n    Why this exists and remains useful.\n\n    Args:\n      x: X.\n      continued text.\n    """\n',
        )
        is not None
    )


def test_attr_blank_line_skips_undocumented_attribute(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        AttrBlankLineRule(),
        "class K:\n    a = 1\n    b = 2\n",
    )


def test_annotation_import_skips_runtime_only_name(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        AnnotationImportRule(),
        "import torch\n\ndef f(x: torch.Tensor):\n    return torch.zeros(1)\n",
    )


def test_lazy_import_settings_and_suppression(tmp_path: Path) -> None:
    body = "import torch  # house-ignore[lazy-import-cost] -- Local.\n\n\ndef f():\n    torch.zeros(1)\n\ndef g():\n    return 1\n"
    assert not _findings(tmp_path, LazyImportCostRule(), body)
    assert not _findings(
        tmp_path,
        LazyImportCostRule(costs_ms={"torch": 0}),
        body,
    )


def test_cli_description_t201_and_main_module_shapes(tmp_path: Path) -> None:
    body = (
        '"""Doc."""\nimport argparse\n\n'
        "def main():\n"
        "    p = argparse.ArgumentParser(description=__doc__)\n"
        "    doc = __doc__\n"
        "    p = argparse.ArgumentParser(description=inspect.cleandoc(__doc__))\n"
        "    return 0\n"
    )
    msgs = [f.message for f in _findings(tmp_path, CliShapeRule(), body)]
    assert any("(f) description=" in m for m in msgs)
    assert any("cleandoc" in m for m in msgs)
    assert any("local `doc" in m for m in msgs)
    poly = (
        "#!/bin/sh\n# ruff: noqa: EXE003, D300, D205 -- Why.\n" + _POLYGLOT_HEADER_TEXT
    )
    assert _findings(tmp_path, CliShapeRule(), poly, name="__main__.py")


_POLYGLOT_HEADER_TEXT = "# fmt: off\n'''' 2>/dev/null #\nexec uv bad\n"


def test_cli_t201_header_and_inline_rules(tmp_path: Path) -> None:
    one = "#!/bin/sh\n# ruff: noqa: EXE003, D300, D205, T201 -- Why.\nprint(1)\n"
    assert any(
        "fewer than 2" in f.message for f in _findings(tmp_path, CliShapeRule(), one)
    )
    two = one + "print(2)  # noqa: T201\n"
    assert any(
        "file-level" in f.message for f in _findings(tmp_path, CliShapeRule(), two)
    )


def test_assert_negation_handles_negative_literals_and_parentheses(
    tmp_path: Path,
) -> None:
    assert _fixed(tmp_path, AssertRule(), "assert x < -1\n") == (
        'if x >= -1:\n    raise ValueError("Expected x < -1.")\n'
    )
    assert _fixed(tmp_path, AssertRule(), "assert (x + y)\n") == (
        'if not (x + y):\n    raise ValueError("Expected x + y.")\n'
    )


def test_logging_format_fallback_and_percent_escape(tmp_path: Path) -> None:
    body = 'logger.info(f"{x:,} {y:=8} 100%")\n'
    fixed = _fixed(tmp_path, LoggingFstringRule(), body)
    assert fixed == 'logger.info("%s %s 100%%", f"{x:,}", f"{y:=8}")\n'


def test_helpers_cover_bound_names_and_comment_blocks(tmp_path: Path) -> None:
    body = "# lower name\nvalue = 1\n# second line\n"
    fixed = _fixed(tmp_path, CommentSentenceRule(), body)
    assert fixed == "# Lower name.\nvalue = 1\n# Second line.\n"


def test_marker_syntax_errors_short_circuit_rules(tmp_path: Path) -> None:
    cases: list[tuple[Rule, str]] = [
        (StaticMethodRule(), "staticmethod(\n"),
        (XdgLiteralRule(), "Path.home(\n"),
        (UserdirsNamespaceRule(), "state_dir() /\n"),
        (UserdirsWrapperRule(), "state_dir() /\n"),
        (CacheKwargRule(), "f(cache_dir=\n"),
        (LazyImportBindingRule(), "lazy_import(\n"),
        (TmpLiteralRule(), '"/tmp/\n'),
        (TempfilePrefixRule(), "tempfile.\n"),
        (CwdFinalRule(), "Path(__file__\n"),
        (ImportAliasRule(), "import x as\n"),
        (ImportHoistRule(), "def f(:\n import x\n"),
        (DottedImportRule(), "import x.y\nx.y(\n"),
        (GetLoggerNameRule(), "getLogger(\n"),
        (NestedTryRule(), "try:\n  try:\n"),
        (NegativeBoolParamRule(), "def f(no_cache: bool\n"),
        (FluentNameRule(), "def schema_of(\n"),
        (AllReexportRule(), "__all__ = [\n"),
        (InitFacadeRule(), "def f(\n"),
        (DefOrderRule(), "def _x(\n"),
        (UnderscoreParamRule(), "def f(_x\n"),
    ]
    for rule, body in cases:
        assert not _findings(tmp_path, rule, body)
        assert _fixed(tmp_path, rule, body) is None


def test_edge_branches_in_rules_are_real_findings(tmp_path: Path) -> None:
    assert not _findings(tmp_path, GetLoggerNameRule(), "x = 1\n")
    assert not _findings(
        tmp_path,
        ReturnsNamedRule(),
        'def f() -> int:\n    """Do.\n\n    Args:\n      x: X.\n\n    """\n    return 1\n',
    )
    assert not _findings(
        tmp_path,
        AnnotationImportRule(),
        "from torch import Tensor\n\ndef f():\n    return 1\n",
    )
    assert not _findings(
        tmp_path,
        LazyImportCostRule(),
        "import torch\n\ndef f():\n    torch.zeros(1)\n\ndef g():\n    return 1\n",
    )
    assert not _findings(
        tmp_path,
        PrivateOnelinerRule(),
        "def public():\n    return 1\n",
    )
    assert not _findings(
        tmp_path,
        AttrBlankLineRule(),
        "class K:\n    x = 1\n    y = 2\n",
    )


def test_cli_header_and_description_edge_branches(tmp_path: Path) -> None:
    body = (
        "#!/bin/sh\n# ruff: noqa: EXE003\n# shell note\n"
        "# fmt: off\n'''' 2>/dev/null #\nexec uv \\\n"
        ' --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" \\\n'
        ' "$@"\n# vim: ft=python\n'
    )
    findings = _findings(tmp_path, CliShapeRule(), body, name="__main__.py")
    assert any("line 2" in finding.message for finding in findings)
    assert _findings(
        tmp_path,
        CliShapeRule(),
        "import argparse\n\n"
        "def main():\n"
        "    p = argparse.ArgumentParser(description=__doc__, add_help=True)\n"
        "    return 0\n",
    )


def test_logger_and_returns_edge_branches(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        GetLoggerNameRule(),
        "value = 1\nlog = 1\nlog = factory()\n",
    )
    assert not _findings(
        tmp_path,
        ReturnsNamedRule(),
        "def f() -> int:\n    # Returns: prose marker\n    return 1\n",
    )


def test_single_use_helper_overlapping_edits_are_deferred(tmp_path: Path) -> None:
    body = (
        "def f(x: int) -> int:\n    return _a(x)\n\n\n"
        "def _a(y: int) -> int:\n    return _b(y)\n\n\n"
        "def _b(z: int) -> int:\n    return z + 1\n"
    )
    fixed = _fixed(tmp_path, SingleUseHelperRule(), body)
    assert fixed is not None
    assert "_b(x)" in fixed


def test_triton_jit_name_and_super_async_shapes(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        UnderscoreParamRule(),
        "@jit\ndef f(_x):\n    return 1\n",
    )
    assert _findings(
        tmp_path,
        SuperOnlyRule(),
        "class C(A):\n    async def f(self, x):\n        await super().f(x)\n",
    )


def test_mkdir_atomic_userdirs_mode_is_reported(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        MkdirModeRule(),
        'atomic_write_bytes(state_dir() / "rekursiv-ai" / "x", data)\n'
        "try:\n"
        "    old = os.umask(0o077)\n"
        "    f()\n"
        "finally:\n"
        "    cleanup()\n"
        "try:\n"
        "    old = os.umask(0o077)\n"
        "except OSError:\n"
        "    pass\n"
        "finally:\n"
        "    os.umask(old)\n",
    )
    assert [(f.line, f.message) for f in findings] == [
        (1, "atomic_write_bytes under a userdirs path without file_mode=0o600"),
        (3, "os.umask without a finally that restores the previous mask"),
    ]


def test_attr_doc_check_skips_non_docstring_pair(tmp_path: Path) -> None:
    assert not _findings(
        tmp_path,
        AttrBlankLineRule(),
        "class K:\n    x = 1\n    y = 2\n    z = 3\n",
    )


def test_comment_block_yields_final_standalone_comment(tmp_path: Path) -> None:
    fixed = _fixed(tmp_path, CommentSentenceRule(), "# final comment\n")
    assert fixed == "# Final comment.\n"


def test_config_globals_checks_non_annotation_class_members(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        ConfigGlobalsRule(),
        "DEFAULT = 3\n"
        "class C(Fig[int]):\n"
        "    x = 1\n"
        "    w: int = 2\n"
        "    y: int = DEFAULT\n"
        "    z: int = mod.LIMIT\n"
        "class Plain:\n"
        "    v: int = DEFAULT\n",
    )
    assert [f.line for f in findings] == [1, 5, 6]
    assert [f.message for f in findings[1:]] == [
        f"Config field {field!r} defaults to module constant {const!r}; inline "
        "the literal so the default lives in the Config (or suppress: "
        "house-ignore[globals] -- Reason.)"
        for field, const in (("y", "DEFAULT"), ("z", "LIMIT"))
    ]


def test_tempfile_prefix_check_skips_prefixed_and_unknown_calls(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        TempfilePrefixRule(),
        'tempfile.mkdtemp(prefix="ok-")\ntempfile.other()\n',
    )
    assert findings == ()


def test_tempfile_prefix_fix_skips_non_tempfile_calls(tmp_path: Path) -> None:
    fixed = _fixed(
        tmp_path,
        TempfilePrefixRule(),
        'tempfile.mkdtemp()\nprint("tempfile.mkdtemp()")\n',
    )
    assert fixed == 'tempfile.mkdtemp(prefix="mod-")\nprint("tempfile.mkdtemp()")\n'


def test_tempfile_prefix_fix_recognizes_aliased_direct_import(tmp_path: Path) -> None:
    fixed = _fixed(
        tmp_path,
        TempfilePrefixRule(),
        "from tempfile import mkstemp as make_temp\nmake_temp()\n",
        name="alias.py",
    )
    assert fixed == (
        'from tempfile import mkstemp as make_temp\nmake_temp(prefix="alias-")\n'
    )


def test_private_docstring_fix_skips_public_and_non_doc_defs(tmp_path: Path) -> None:
    assert (
        _fixed(
            tmp_path,
            PrivateOnelinerRule(),
            "def public():\n    return 1\n\ndef _private():\n    return 2\n",
        )
        is None
    )


def test_returns_named_checks_only_returns_sections(tmp_path: Path) -> None:
    findings = _findings(
        tmp_path,
        ReturnsNamedRule(),
        'def f() -> int:\n    """Do.\n\n    Args:\n      x: X.\n\n    Returns:\n      unnamed entry\n\n    """\n    return 1\n',
    )
    assert [f.message for f in findings] == [
        "f: Returns: entry 'unnamed entry' is not `name: description`",
    ]


def test_private_docstring_fix_wraps_long_summary_and_multiple_prose(
    tmp_path: Path,
) -> None:
    fixed = _fixed(
        tmp_path,
        PrivateOnelinerRule(),
        'def _f():\n    """A very long summary that exceeds the available width and. '
        'continues beyond it.\n\n    First rationale paragraph.\n\n    Second rationale paragraph.\n    """\n',
    )
    assert fixed == (
        "# First rationale paragraph.\n#\n# Second rationale paragraph.\n"
        'def _f():\n    """A very long summary that exceeds the available width and. '
        'continues beyond it."""\n'
    )


def test_cli_main_module_shapes_report_each_bad_shape(tmp_path: Path) -> None:
    body = (
        '"""Doc."""\nfrom x import other\nfrom y import main\n'
        'x = 1\nif __name__ == "__main__":\n    main()\n'
    )
    messages = [
        finding.message
        for finding in _findings(tmp_path, CliShapeRule(), body, name="__main__.py")
    ]
    assert any("more than" in message for message in messages)


def test_cli_description_ignores_non_description_keyword(tmp_path: Path) -> None:
    body = (
        "import argparse\n\ndef main() -> int:\n"
        '    argparse.ArgumentParser(description=__doc__, prog="x")\n'
        "    return 0\n"
    )
    findings = _findings(tmp_path, CliShapeRule(), body)
    assert any("description=" in finding.message for finding in findings)


def test_assert_fix_handles_tuple_and_boolean_parentheses(tmp_path: Path) -> None:
    fixed = _fixed(tmp_path, AssertRule(), "assert (a, b)\n")
    assert fixed == 'if not ((a, b)):\n    raise ValueError("Expected (a, b).")\n'


def test_logging_fstring_skips_non_formatted_parts(tmp_path: Path) -> None:
    fixed = _fixed(tmp_path, LoggingFstringRule(), 'logger.info(f"plain {x!s}")\n')
    assert fixed == 'logger.info("plain %s", x)\n'


def test_helpers_cover_dotted_and_attribute_names(tmp_path: Path) -> None:
    fixed = _fixed(
        tmp_path,
        CommentSentenceRule(),
        "import package\n# package is used\nvalue = package.attr\n",
    )
    assert fixed == "import package\n# `package` is used.\nvalue = package.attr\n"


# -- fixers must never write broken or wrong Python ---------------------------


def test_underscore_param_del_goes_above_a_decorated_first_statement(
    tmp_path: Path,
) -> None:
    """The ``del`` landed between ``@deco`` and its ``def``: a SyntaxError."""
    body = "def outer(_x):\n    @deco\n    def inner():\n        return 1\n    return inner\n"
    assert _fixed(tmp_path, UnderscoreParamRule(), body) == (
        "def outer(x):\n    del x\n    @deco\n    def inner():\n"
        "        return 1\n    return inner\n"
    )


def test_super_only_fix_keeps_a_class_it_would_empty(tmp_path: Path) -> None:
    """Deleting the only method left ``class A(B):`` with no body."""
    body = "class A(B):\n    def f(self):\n        return super().f()\n"
    assert _findings(tmp_path, SuperOnlyRule(), body)
    assert _fixed(tmp_path, SuperOnlyRule(), body) is None


def test_assert_fix_skips_an_assert_sharing_its_line(tmp_path: Path) -> None:
    """A one-line ``if x: assert y`` was rewritten to a SyntaxError."""
    for body in ("if x: assert y\n", "assert y; z = 1\nprint(z)\n"):
        assert _findings(tmp_path, AssertRule(), body)
        assert _fixed(tmp_path, AssertRule(), body) is None, body


def test_cli_shape_bare_return_rewrite_touches_only_the_return(
    tmp_path: Path,
) -> None:
    """``return`` inside ``returned`` became ``return 0ed``."""
    body = "def main() -> int:\n    if returned: return\n    return 0\n"
    assert _fixed(tmp_path, CliShapeRule(), body) == (
        "def main() -> int:\n    if returned: return 0\n    return 0\n"
    )


def test_cli_shape_moved_arguments_keep_mains_trailing_return(
    tmp_path: Path,
) -> None:
    """``return 0`` was inserted at a pre-move line, inside ``_add_arguments``."""
    body = (
        "import argparse\n"
        "def main() -> int:\n"
        "    p = argparse.ArgumentParser()\n"
        '    p.add_argument("--a")\n'
        '    p.add_argument("--b")\n'
        "    run(p)\n"
    )
    assert _fixed(tmp_path, CliShapeRule(), body) == (
        "import argparse\n"
        "def main() -> int:\n"
        "    p = argparse.ArgumentParser()\n"
        "    _add_arguments(p)\n"
        "    run(p)\n"
        "    return 0\n"
        "\n\n"
        "def _add_arguments(p: argparse.ArgumentParser) -> None:\n"
        '    p.add_argument("--a")\n'
        '    p.add_argument("--b")\n'
    )


def test_cli_shape_polyglot_fix_keeps_the_docstring_summary(tmp_path: Path) -> None:
    """A header missing its noqa line had its docstring summary overwritten."""
    body = (
        "#!/bin/sh\n# fmt: off\n'''' 2>/dev/null #\n"
        'exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"\n'
        "Summary line of the tool.\n'''\n# fmt: on\n# vim: ft=python\n"
    )
    fixed = _fixed(tmp_path, CliShapeRule(), body)
    assert fixed == body.replace(
        "#!/bin/sh\n",
        "#!/bin/sh\n# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.\n",
    )


def test_tempfile_prefix_fix_on_a_wrapped_call_with_a_trailing_comma(
    tmp_path: Path,
) -> None:
    """``, prefix=`` went after the trailing comma: a SyntaxError."""
    body = "import tempfile\nd = tempfile.mkdtemp(\n    dir,\n)\n"
    assert _fixed(tmp_path, TempfilePrefixRule(), body) == (
        'import tempfile\nd = tempfile.mkdtemp(\n    dir, prefix="mod-",\n)\n'
    )


@pytest.mark.parametrize(
    ("line", "fixed"),
    [
        ("v = a if not c else (b if d else e)", "v = (b if d else e) if c else a"),
        ("v = a if not c else lambda: b", "v = (lambda: b) if c else a"),
        # Right-associative: the unparenthesized tail already groups this way.
        ("v = (a if e else f) if not c else b", "v = b if c else a if e else f"),
    ],
)
def test_negated_ternary_fix_keeps_arm_grouping(
    tmp_path: Path,
    line: str,
    fixed: str,
) -> None:
    """A low-precedence else-arm moved to the front regrouped the whole ternary."""
    assert _fixed(tmp_path, NegatedTernaryRule(), line + "\n") == fixed + "\n"


def test_negated_compare_fix_keeps_a_low_precedence_operand(tmp_path: Path) -> None:
    """``not (a or b) < 0`` became ``a or b >= 0``, which compares only ``b``."""
    body = "v = not (a or b) < 0\n"
    assert _fixed(tmp_path, NegatedCompareRule(), body) == "v = (a or b) >= 0\n"


def test_single_use_helper_survives_an_unparsable_sibling(tmp_path: Path) -> None:
    """A sibling with a syntax error crashed the whole hook with a traceback.

    Its imports cannot be read, so its mention of the name counts as a reach.
    """
    (tmp_path / "broken.py").write_text("_helper(\n", encoding="utf-8")
    body = "def _helper(x):\n    return x + 1\n\n\ndef f():\n    return _helper(2)\n"
    assert _findings(tmp_path, SingleUseHelperRule(), body) == ()
    (tmp_path / "broken.py").write_text("unrelated(\n", encoding="utf-8")
    assert [f.line for f in _findings(tmp_path, SingleUseHelperRule(), body)] == [1]


# -- mutation coverage: cli-shape --------------------------------------------

_EXEC: Final = 'exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"'
_HEAD: Final = "#!/bin/sh\n# ruff: noqa: EXE003, D300, D205 -- W.\n"
_PINNED: Final = "# fmt: off\n'''' 2>/dev/null #\n"
_EXEC_MSG: Final = (
    f'must be `{_EXEC}` (or `... python3 -m <module> "$@"` for a coupled entry)'
)


def _cli(
    tmp_path: Path,
    body: str,
    *,
    mode: int = 0o755,
) -> tuple[list[tuple[int, str]], str | None]:
    """Run CliShapeRule check and fix over ``body`` saved as an executable tool."""
    path = tmp_path / "tool.py"
    path.write_text(body, encoding="utf-8")
    path.chmod(mode)
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    rule = CliShapeRule()
    return [(f.line, f.message) for f in rule.check(unit)], rule.fix(unit)


def test_cli_header_short_file_is_repaired_to_the_full_template(tmp_path: Path) -> None:
    findings, fixed = _cli(tmp_path, _HEAD + _PINNED)
    assert findings == [
        (1, "(a) line 2 must be `# ruff: noqa: EXE003, D300, D205[, CODE...]`"),
        (1, f"(a) line 5 {_EXEC_MSG}"),
        (1, "(a) last line must be `# vim: ft=python`"),
    ]
    assert fixed == _HEAD + _PINNED + _EXEC + "\n# vim: ft=python\n"


def test_cli_header_wrong_pinned_lines_are_named_and_rewritten(tmp_path: Path) -> None:
    body = _HEAD + "# fmt: offx\n'''' x\n" + _EXEC + "\n'''\n# vim: ft=python\n"
    findings, fixed = _cli(tmp_path, body)
    assert findings == [
        (1, "(a) line 3 must be `# fmt: off`"),
        (1, "(a) line 4 must be `'''' 2>/dev/null #`"),
    ]
    assert fixed == _HEAD + _PINNED + _EXEC + "\n'''\n# vim: ft=python\n"


@pytest.mark.parametrize(
    "prelude",
    [
        "# ruff: noqa: N999 -- K.\n# more\n",
        "",
    ],
)
@pytest.mark.parametrize(
    "exec_line",
    [
        _EXEC,
        _EXEC.replace('"$0" "$@"', '-m pkg.mod "$@"'),
        _EXEC.replace(" run ", " \\\nrun "),
    ],
)
def test_cli_header_accepts_every_canonical_shape(
    tmp_path: Path,
    prelude: str,
    exec_line: str,
) -> None:
    """Extra noqa lines, ``-m``, and a backslash-continued exec are all canonical."""
    body = (
        _HEAD
        + prelude
        + _PINNED
        + "[ -x v ] && exec v\n"
        + exec_line
        + "\n'''\n# vim: ft=python\n"
    )
    assert _cli(tmp_path, body) == ([], None)


def test_cli_header_names_the_line_a_wrapped_exec_ends_on(tmp_path: Path) -> None:
    wrapped = 'exec uv --quiet --project "$(dirname "$0")" \\\n  run --frozen python3 "$0" "$@"\n'
    findings, _ = _cli(tmp_path, _HEAD + _PINNED + wrapped + "'''\n# vim: ft=python\n")
    assert findings == [(1, f"(a) line 6 {_EXEC_MSG}")]


def test_cli_header_skips_a_shell_comment_line(tmp_path: Path) -> None:
    body = _HEAD + "# shell note\n" + _PINNED + _EXEC + "\n'''\n# vim: ft=python\n"
    assert _cli(tmp_path, body) == ([], None)


def test_cli_header_missing_exec_bit_is_reported_and_set(tmp_path: Path) -> None:
    body = _HEAD + _PINNED + _EXEC + "\n'''\n# vim: ft=python\n"
    findings, _ = _cli(tmp_path, body, mode=0o640)
    assert findings == [(1, "(a) polyglot file lacks the executable bit; chmod +x")]
    assert (tmp_path / "tool.py").stat().st_mode & 0o777 == 0o751


def test_cli_trailer_with_trailing_space_is_replaced_not_duplicated(
    tmp_path: Path,
) -> None:
    body = _HEAD + _PINNED + _EXEC + "\n'''\n"
    findings, fixed = _cli(tmp_path, body + "# vim: ft=python \n")
    assert findings == [(1, "(a) last line must be `# vim: ft=python`")]
    assert fixed == body + "# vim: ft=python\n"


def test_cli_header_noqa_keeps_extra_codes_and_drops_duplicates(tmp_path: Path) -> None:
    body = "#!/bin/sh\n# ruff: noqa: N999, EXE003, T201\nx\ny\nz\n"
    _, fixed = _cli(tmp_path, body)
    assert fixed == (
        "#!/bin/sh\n"
        "# ruff: noqa: EXE003, D300, D205, N999, T201 -- Polyglot shell/Python script.\n"
        + _PINNED
        + _EXEC
        + "\nx\ny\nz\n# vim: ft=python\n"
    )


def test_cli_header_keeps_crlf_line_endings() -> None:
    body = (_HEAD + _PINNED + _EXEC + "\n'''\n").replace("\n", "\r\n")
    lines = [str(line) for line in body.splitlines(keepends=True)]
    houselint_rules._fix_polyglot_header(lines)
    assert "".join(lines) == body


def test_cli_header_on_a_one_line_file_is_left_alone() -> None:
    lines = ["#!/bin/sh\n"]
    houselint_rules._fix_polyglot_header(lines)
    assert lines == ["#!/bin/sh\n"]


def test_cli_header_two_line_file_gets_the_pinned_lines() -> None:
    lines = ["#!/bin/sh\n", "# ruff: noqa: EXE003, D300, D205 -- W.\n"]
    houselint_rules._fix_polyglot_header(lines)
    assert "".join(lines) == _HEAD + _PINNED + _EXEC + "\n"


def test_cli_t201_placement_follows_the_print_count(tmp_path: Path) -> None:
    file_level = _HEAD + "# ruff: noqa: T201 -- P.\n" + _PINNED + _EXEC + "\n'''\n"
    assert _cli(
        tmp_path,
        file_level + "console.print(1)\nprint(2)\n# vim: ft=python\n",
    )[0] == [
        (1, "(g) file-level T201 with fewer than 2 prints; use inline noqa"),
    ]
    inline = _HEAD + _PINNED + _EXEC + "\n'''\nx = 1\n"
    two = "print(1)  # noqa: T201\nprint(2)  # noqa: T201\n# vim: ft=python\n"
    assert _cli(tmp_path, inline + two)[0] == [
        (8, "(g) inline T201 with 2+ prints; use the file-level header"),
    ]
    assert (
        _cli(tmp_path, "x = 1\nprint(1)  # noqa: T201\nprint(2)  # noqa: T201\n")[0]
        == []
    )
    late = "x\n" * 8 + "# ruff: noqa: T201\nprint(1)\n"
    assert _cli(tmp_path, late)[0] == []


def test_cli_main_def_reports_each_violation_on_its_line(tmp_path: Path) -> None:
    body = "def main():\n    exit(1)\n    if a:\n        return\n    sys.exit(2)\n"
    assert _cli(tmp_path, body)[0] == [
        (1, "(d) main must be annotated `-> int`"),
        (2, "(d) sys.exit inside main; return the code instead"),
        (4, "(d) bare return in main; return an exit code"),
        (5, "(d) sys.exit inside main; return the code instead"),
    ]


def test_cli_main_def_reports_the_line_main_falls_off(tmp_path: Path) -> None:
    body = "def main() -> int:\n    x = (\n        1\n    )\n"
    findings, fixed = _cli(tmp_path, body)
    assert findings == [(4, "(d) main can fall off its end; finish with `return 0`")]
    assert fixed == body + "    return 0\n"


@pytest.mark.parametrize(
    ("header", "fixed"),
    [
        ("def main() -> str:\n", "def main() -> int:\n"),
        ("def main(\n    x=1,\n):\n", "def main(\n    x=1,\n) -> int:\n"),
        ("def main():  # noqa\n", "def main() -> int:  # noqa\n"),
        ("def main():\r\n", "def main() -> int:\r\n"),
    ],
)
def test_cli_main_annotation_is_rewritten_on_its_header_line(
    tmp_path: Path,
    header: str,
    fixed: str,
) -> None:
    body = header + "    return 0\n"
    path = tmp_path / "tool.py"
    path.write_bytes(body.encode())
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert CliShapeRule().fix(unit) == fixed + "    return 0\n"


def test_cli_description_reports_each_idiom_violation(tmp_path: Path) -> None:
    body = (
        "import argparse, inspect\n\n"
        "def main() -> int:\n"
        "    argparse.ArgumentParser(description=__doc__)\n"
        "    doc = __doc__.strip()\n"
        "    other = __doc__\n"
        "    argparse.ArgumentParser(description=inspect.cleandoc(__doc__))\n"
        '    argparse.ArgumentParser(description=(__doc__ or "").strip())\n'
        "    return 0\n"
    )
    assert _cli(tmp_path, body)[0] == [
        (4, '(f) description= must be `(__doc__ or "").strip()`'),
        (7, '(f) description= must be `(__doc__ or "").strip()`'),
        (7, "(f) inspect.cleandoc on __doc__; use the slice idiom"),
        (5, "(f) local `doc = __doc__...`; inline the idiom"),
    ]


_POLYGLOT_DESCRIPTION = '__doc__.split("\\n", 2)[2] if __doc__ else None'


@pytest.mark.parametrize(
    ("doc", "expected"),
    [(None, None), ("exec uv ...\n\nProse.\n", "Prose.\n")],
    ids=["python-OO", "docstring"],
)
def test_the_polyglot_description_fix_is_total(
    tmp_path: Path,
    doc: str | None,
    expected: str | None,
) -> None:
    """``python -OO`` strips ``__doc__`` to None; the fixed line must not raise."""
    body = (
        _HEAD + _PINNED + _EXEC + "\n'''\nimport argparse\n\n"
        "def main() -> int:\n"
        "    argparse.ArgumentParser(description=__doc__)\n"
        "    return 0\n# vim: ft=python\n"
    )
    fixed = _cli(tmp_path, body)[1]
    assert fixed is not None
    [call] = [
        node
        for node in ast.walk(ast.parse(fixed))
        if isinstance(node, ast.Call)
        and houselint_rules._call_text(node).endswith("ArgumentParser")
    ]
    [keyword] = call.keywords
    expression = ast.Expression(body=keyword.value)
    namespace: dict[str, object] = {"__doc__": doc}
    assert eval(compile(expression, "<fix>", "eval"), namespace) == expected  # noqa: S307 -- Evaluates the rule's own fixed expression on a fixed namespace.


def test_cli_polyglot_description_uses_the_slice_idiom(tmp_path: Path) -> None:
    body = (
        _HEAD + _PINNED + _EXEC + "\n'''\nimport argparse\n\n"
        "def main() -> int:\n"
        "    argparse.ArgumentParser(description=__doc__)\n"
        "    return 0\n# vim: ft=python\n"
    )
    findings, fixed = _cli(tmp_path, body)
    want = _POLYGLOT_DESCRIPTION
    assert findings == [(10, f"(f) description= must be `{want}`")]
    assert fixed == body.replace("description=__doc__", f"description={want}")


@pytest.mark.parametrize(
    "description",
    [
        '__doc__.split("\\n", 2)[2]',
        '(__doc__ or "").split("\\n", 2)[2]',
        "__doc__.strip()",
        '__doc__.split("\\n\\n", 1)[0] if __doc__ else ""',
        '"x" + __doc__',
    ],
)
def test_cli_any_module_doc_description_off_the_idiom_is_fixed(
    tmp_path: Path,
    description: str,
) -> None:
    """The check matches the docstring's use, not a ``(__doc__`` spelling prefix."""
    body = (
        _HEAD + _PINNED + _EXEC + "\n'''\nimport argparse\n\n"
        "def main() -> int:\n"
        f"    argparse.ArgumentParser(description={description})\n"
        "    return 0\n# vim: ft=python\n"
    )
    findings, fixed = _cli(tmp_path, body)
    want = _POLYGLOT_DESCRIPTION
    assert findings == [(10, f"(f) description= must be `{want}`")]
    assert fixed == body.replace(f"description={description}", f"description={want}")


def test_cli_another_objects_docstring_is_not_the_modules(tmp_path: Path) -> None:
    body = (
        "import argparse\n\ndef main() -> int:\n"
        "    argparse.ArgumentParser(description=cls.__doc__)\n"
        "    return 0\n"
    )
    assert not [m for _, m in _cli(tmp_path, body)[0] if m.startswith("(f)")]


def test_cli_add_arguments_moves_a_wrapped_call_and_keeps_the_tail(
    tmp_path: Path,
) -> None:
    body = (
        "import argparse\n\n"
        "def main() -> int:\n"
        "    p = argparse.ArgumentParser()\n"
        "    p.add_argument(\n"
        '        "--a",\n'
        "    )\n"
        "    other.add_argument('--z')\n"
        "    return run(p)\n"
        "\nx = 1\n"
    )
    assert _cli(tmp_path, body)[1] == (
        "import argparse\n\n"
        "def main() -> int:\n"
        "    p = argparse.ArgumentParser()\n"
        "    _add_arguments(p)\n"
        "    other.add_argument('--z')\n"
        "    return run(p)\n"
        "\n\n"
        "def _add_arguments(p: argparse.ArgumentParser) -> None:\n"
        "    p.add_argument(\n"
        '        "--a",\n'
        "    )\n"
        "\nx = 1\n"
    )


def test_cli_add_arguments_without_a_parser_is_left_alone(tmp_path: Path) -> None:
    body = "def main() -> int:\n    p.add_argument('--a')\n    return 0\n"
    assert _cli(tmp_path, body)[1] is None


@pytest.mark.parametrize(
    ("tail", "falls"),
    [
        ("    raise SystemExit(1)\n", False),
        ("    os._exit(1)\n", False),
        ("    parser.exit(1)\n", False),
        ("    log.exit_handler(1)\n", True),
        ("    if a:\n        return 1\n    else:\n        return 2\n", False),
        ("    if a:\n        return 1\n", True),
        ("    with c:\n        return 1\n", False),
        ("    with c:\n        x = 1\n", True),
        ("    try:\n        return 1\n    finally:\n        return 2\n", False),
        ("    try:\n        x = 1\n    finally:\n        y = 2\n", True),
        (
            "    try:\n        x = 1\n    except E:\n        return 1\n    else:\n        return 2\n",
            False,
        ),
        ("    try:\n        return 1\n    except E:\n        log()\n", True),
        ("    for x in y:\n        return 1\n", True),
    ],
)
def test_falls_through_follows_each_compound_shape(
    tail: str,
    *,
    falls: bool,
) -> None:
    tree = ast.parse("def main() -> int:\n" + tail)
    main = tree.body[0]
    assert isinstance(main, ast.FunctionDef)
    assert houselint_rules._falls_through(main.body, noreturn=frozenset()) is falls


def test_falls_through_honours_this_modules_noreturn_defs() -> None:
    tree = ast.parse("def main() -> int:\n    bail()\n")
    main = tree.body[0]
    assert isinstance(main, ast.FunctionDef)
    assert not houselint_rules._falls_through(main.body, noreturn=frozenset({"bail"}))
    assert houselint_rules._falls_through(main.body, noreturn=frozenset({"other"}))


def test_cli_header_fix_handles_noqa_crlf_and_extra_noqa_lines() -> None:
    lines = [
        "#!/bin/sh\r\n",
        "# ruff: noqa: EXE003, D300, D205, D300, N999\r\n",
        "# ruff: noqa: T201 -- P.\r\n",
        "# fmt: off\r\n",
        "'''' 2>/dev/null #\r\n",
        _EXEC + "\r\n",
    ]
    houselint_rules._fix_polyglot_header(lines)
    assert (
        lines[1]
        == "# ruff: noqa: EXE003, D300, D205, D300, N999 -- Polyglot shell/Python script.\n"
    )
    assert lines[2:] == [
        "# ruff: noqa: T201 -- P.\r\n",
        "# fmt: off\r\n",
        "'''' 2>/dev/null #\r\n",
        _EXEC + "\r\n",
    ]


def test_cli_header_fix_drops_the_pinned_codes_from_the_extras() -> None:
    lines = ["#!/bin/sh\n", "# ruff: noqa: D205, N999, D300, EXE003\n"]
    houselint_rules._fix_polyglot_header(lines)
    assert (
        lines[1]
        == "# ruff: noqa: EXE003, D300, D205, N999 -- Polyglot shell/Python script.\n"
    )


def test_cli_header_fix_stops_the_exec_search_at_the_docstring() -> None:
    """A docstring close before any ``exec uv`` means the exec line is missing."""
    lines = [
        "#!/bin/sh\n",
        "# ruff: noqa: EXE003, D300, D205 -- W.\n",
        "# fmt: off\n",
        "'''' 2>/dev/null #\n",
        "echo fast\n",
        "'''\n",
        _EXEC + "\n",
    ]
    houselint_rules._fix_polyglot_header(lines)
    assert lines[4] == _EXEC + "\n"
    assert lines[5:] == ["echo fast\n", "'''\n", _EXEC + "\n"]


def test_cli_header_fix_keeps_a_custom_exec_two_lines_down() -> None:
    lines = [
        "#!/bin/sh\n",
        "# ruff: noqa: EXE003, D300, D205 -- W.\n",
        "# fmt: off\n",
        "'''' 2>/dev/null #\n",
        "[ -x v ] && exec v\n",
        "exec uv run x\n",
        "'''\n",
    ]
    before = list(lines)
    houselint_rules._fix_polyglot_header(lines)
    assert lines == before


def test_cli_header_fix_skips_comment_lines_after_extra_noqa() -> None:
    lines = [
        "#!/bin/sh\n",
        "# ruff: noqa: EXE003, D300, D205 -- W.\n",
        "# ruff: noqa: N999 -- K.\n",
        "# ruff: noqa: T201 -- P.\n",
        "# continued reason\n",
        "x\n",
        "y\n",
    ]
    houselint_rules._fix_polyglot_header(lines)
    assert lines[5:] == [
        "# fmt: off\n",
        "'''' 2>/dev/null #\n",
        _EXEC + "\n",
        "x\n",
        "y\n",
    ]


def test_cli_header_check_skips_extra_noqa_and_comment_lines(tmp_path: Path) -> None:
    body = (
        _HEAD
        + "# ruff: noqa: N999 -- K.\n# ruff: noqa: E501 -- L.\n# continued\n"
        + "# fmt: off\n'''' x\n"
        + _EXEC
        + "\n'''\n# vim: ft=python\n"
    )
    assert _cli(tmp_path, body)[0] == [(1, "(a) line 7 must be `'''' 2>/dev/null #`")]


@pytest.mark.parametrize("length", [4, 5])
def test_cli_header_check_needs_five_lines(tmp_path: Path, length: int) -> None:
    lines = [_HEAD, "# fmt: off\n", "'''' 2>/dev/null #\n", _EXEC + "\n"][: length - 1]
    findings, _ = _cli(tmp_path, "".join(lines))
    noqa = (1, "(a) line 2 must be `# ruff: noqa: EXE003, D300, D205[, CODE...]`")
    assert (noqa in findings) is (length < 5)


def test_cli_header_check_joins_a_continued_exec_without_its_backslash(
    tmp_path: Path,
) -> None:
    exec_lines = (
        'exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync \\\n'
        '  python3 "$0" "$@"\n'
    )
    body = _HEAD + _PINNED + exec_lines + "'''\n# vim: ft=python\n"
    assert _cli(tmp_path, body)[0] == []


def test_cli_header_fix_drops_the_continuation_of_a_replaced_exec(
    tmp_path: Path,
) -> None:
    """Replacing a backslash-continued exec left its ``uv`` line dangling below."""
    exec_lines = (
        'exec env PYTHONPATH="$(cd "$(dirname "$0")/../../../.." && pwd)'
        '${PYTHONPATH:+:$PYTHONPATH}" \\\n'
        '    uv --quiet --project "$(dirname "$0")" run --frozen --no-sync '
        'python3 "$0" "$@"\n'
    )
    tail = "Score the best checkpoint on the ETTh1 test split.\n'''\n# vim: ft=python\n"
    _, fixed = _cli(tmp_path, _HEAD + _PINNED + exec_lines + tail)
    assert fixed == _HEAD + _PINNED + _EXEC + "\n" + tail


@pytest.mark.parametrize(
    "tail",
    [
        "Summary line of the tool.\n'''\n# vim: ft=python\n",
        "'''\nx = 1\n# vim: ft=python\n",
    ],
)
def test_cli_header_fix_inserts_a_missing_exec_above_the_docstring(
    tmp_path: Path,
    tail: str,
) -> None:
    """A missing exec overwrote the summary or an empty docstring's close."""
    _, fixed = _cli(tmp_path, _HEAD + _PINNED + tail)
    assert fixed == _HEAD + _PINNED + _EXEC + "\n" + tail


def test_cli_header_fix_inserts_a_missing_pinned_line(tmp_path: Path) -> None:
    """A missing ``# fmt: off`` shifted each pinned line onto the one below it."""
    exec_line = _EXEC.replace('"$0" "$@"', '-m pkg.tool "$@"')
    tail = "\nSummary line of the tool.\n'''\n# vim: ft=python\n"
    _, fixed = _cli(tmp_path, _HEAD + "'''' 2>/dev/null #\n" + exec_line + tail)
    assert fixed == _HEAD + _PINNED + exec_line + tail


def test_cli_header_fix_replaces_a_launcher_below_a_shell_prelude() -> None:
    """The exec slot's fast path was overwritten and the old launcher kept."""
    lines = [
        "#!/bin/sh\n",
        "# ruff: noqa: EXE003, D300, D205 -- W.\n",
        "# fmt: off\n",
        "'''' 2>/dev/null #\n",
        "[ -x v ] && exec v\n",
        'exec python3 "$0" "$@"\n',
        "Summary.\n",
        "'''\n",
    ]
    houselint_rules._fix_polyglot_header(lines)
    assert lines[4:] == ["[ -x v ] && exec v\n", _EXEC + "\n", "Summary.\n", "'''\n"]


def test_cli_trailer_terminates_the_last_line_only() -> None:
    lines = ["a\n", "b"]
    houselint_rules._fix_polyglot_trailer(lines)
    assert lines == ["a\n", "b\n", "# vim: ft=python\n"]
    lines = ["a", "b\n"]
    houselint_rules._fix_polyglot_trailer(lines)
    assert lines == ["a", "b\n", "# vim: ft=python\n"]


def test_cli_main_module_reports_extra_statements(tmp_path: Path) -> None:
    body = 'from m import main\nx = 1\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
    path = tmp_path / "__main__.py"
    path.write_text(body, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert [(f.line, f.message) for f in CliShapeRule().check(unit)] == [
        (1, "(c) __main__.py holds more than `from X import main` + the guard"),
    ]


def test_cli_guard_dropping_mains_code_names_the_fix(tmp_path: Path) -> None:
    body = (
        'def main() -> int:\n    return 0\n\n\nif __name__ == "__main__":\n    main()\n'
    )
    assert (6, "(d) guard drops main's exit code; `raise SystemExit(main())`") in (
        _cli(tmp_path, body)[0]
    )


@pytest.mark.parametrize(("prints", "flagged"), [(1, True), (2, False), (3, False)])
def test_cli_t201_file_level_needs_two_prints(
    tmp_path: Path,
    prints: int,
    *,
    flagged: bool,
) -> None:
    body = (
        _HEAD
        + "# ruff: noqa: T201 -- P.\n"
        + _PINNED
        + _EXEC
        + "\n'''\n"
        + "print(1)\n" * prints
        + "# vim: ft=python\n"
    )
    file_level = (1, "(g) file-level T201 with fewer than 2 prints; use inline noqa")
    assert (file_level in _cli(tmp_path, body)[0]) is flagged


def test_cli_fix_skips_a_main_that_calls_bare_exit(tmp_path: Path) -> None:
    body = "def main():\n    exit(1)\n"
    assert _cli(tmp_path, body)[1] is None


def test_cli_fix_rewrites_the_polyglot_description(tmp_path: Path) -> None:
    body = (
        _HEAD + _PINNED + _EXEC + "\n'''\nimport argparse\n\n"
        "def main() -> int:\n"
        '    argparse.ArgumentParser(description=(__doc__ or "").strip())\n'
        "    return 0\n# vim: ft=python\n"
    )
    findings, fixed = _cli(tmp_path, body)
    want = _POLYGLOT_DESCRIPTION
    assert findings == [(10, f"(f) description= must be `{want}`")]
    assert fixed == body.replace('(__doc__ or "").strip()', want)


def test_cli_main_annotation_scan_stops_at_the_body(tmp_path: Path) -> None:
    """Only header lines are rewritten; a body line ending in ``:`` is not."""
    body = "def main(\n    a,\n    b,\n):\n    if a:\n        return 1\n    return 0\n"
    assert _cli(tmp_path, body)[1] == body.replace("\n):\n", "\n) -> int:\n")


def test_cli_main_annotation_on_a_decorated_main(tmp_path: Path) -> None:
    body = "@wrap\ndef main():\n    return 0\n"
    assert _cli(tmp_path, body)[1] == "@wrap\ndef main() -> int:\n    return 0\n"


def test_cli_main_annotation_keeps_a_colon_in_a_default(tmp_path: Path) -> None:
    body = 'def main(sep=":"):\n    return 0\n'
    assert _cli(tmp_path, body)[1] == 'def main(sep=":") -> int:\n    return 0\n'


def test_cli_main_annotation_rewrites_the_first_arrow_only(tmp_path: Path) -> None:
    body = "def main(f: Callable[[], str] = g) -> str:\n    return 0\n"
    assert _cli(tmp_path, body)[1] == body.replace(") -> str:", ") -> int:")


def test_cli_add_arguments_indents_at_mains_body(tmp_path: Path) -> None:
    body = (
        "import argparse\n\n"
        "def main() -> int:\n"
        "        p = argparse.ArgumentParser()\n"
        "        p.add_argument('--a')\n"
        "        return 0\n"
    )
    assert _cli(tmp_path, body)[1] == (
        "import argparse\n\n"
        "def main() -> int:\n"
        "        p = argparse.ArgumentParser()\n"
        "        _add_arguments(p)\n"
        "        return 0\n"
        "\n\n"
        "def _add_arguments(p: argparse.ArgumentParser) -> None:\n"
        "        p.add_argument('--a')\n"
    )


def test_cli_add_arguments_needs_an_argumentparser_binding(tmp_path: Path) -> None:
    body = (
        "def main() -> int:\n    p = make()\n    p.add_argument('--a')\n    return 0\n"
    )
    assert _cli(tmp_path, body)[1] is None


def test_falls_through_threads_noreturn_into_every_branch() -> None:
    bodies = [
        "    if a:\n        bail()\n    else:\n        bail()\n",
        "    with c:\n        bail()\n",
        "    try:\n        x = 1\n    finally:\n        bail()\n",
        "    try:\n        bail()\n    except E:\n        bail()\n",
    ]
    for body in bodies:
        main = ast.parse("def main() -> int:\n" + body).body[0]
        assert isinstance(main, ast.FunctionDef)
        assert not houselint_rules._falls_through(
            main.body,
            noreturn=frozenset({"bail"}),
        )


def test_falls_through_matches_exit_on_the_last_attribute() -> None:
    main = ast.parse("def main() -> int:\n    exit_codes.log(1)\n").body[0]
    assert isinstance(main, ast.FunctionDef)
    assert houselint_rules._falls_through(main.body, noreturn=frozenset())


# -- mutation coverage: mkdir-mode, side effects, globals, returns, digits -------


def _messages(tmp_path: Path, rule: Rule, body: str) -> list[tuple[int, str]]:
    """Return ``(line, message)`` for every finding of ``rule`` over ``body``."""
    return [(f.line, f.message) for f in _findings(tmp_path, rule, body)]


def test_mkdir_mode_reports_each_violation_by_name(tmp_path: Path) -> None:
    body = (
        "from codedoctor.lib.userdirs import config_dir\n"
        "config_dir().mkdir()\n"
        "config_dir().mkdir(mode=0o700)\n"
        'Path("/opt/scratch/x").mkdir(mode=0o755)\n'
        'Path("/opt/scratch/x").mkdir()\n'
        'atomic_write_bytes(config_dir() / "s", b"")\n'
        'atomic_write_bytes(config_dir() / "s", b"", file_mode=0o600)\n'
        'atomic_write_bytes(config_dir() / "s", b"", mode=0o600)\n'
        'atomic_write_bytes(Path("x"), b"")\n'
        "atomic_write_bytes()\n"
        'Path("k").chmod(0o600)\n'
        'Path("k").chmod(0o700)\n'
        'Path("k").chmod(0o400)\n'
        'Path("k").chmod(0o640)\n'
        'Path("k").chmod(0o100)\n'
        'Path("k").chmod(0o000)\n'
        "config_dir().chmod(0o600)\n"
        'Path("k").chmod()\n'
        'Path("k").chmod(MODE)\n'
        'os.chmod("k", 0o600)\n'
    )
    mkdir = "mkdir under a userdirs path without mode=0o700"
    atomic = "atomic_write_bytes under a userdirs path without file_mode=0o600"
    scratch = "mode= on a /opt/scratch path; the provisioned umask governs"
    chmod = "chmod to a private mode on a non-userdirs path"
    assert _messages(tmp_path, MkdirModeRule(), body) == [
        (2, mkdir),
        (4, scratch),
        (6, atomic),
        (8, atomic),
        (11, chmod),
        (12, chmod),
        (13, chmod),
    ]


def test_mkdir_mode_matches_the_method_on_a_dotted_receiver(tmp_path: Path) -> None:
    body = "import codedoctor.lib.userdirs as u\nu.config_dir().joinpath('a').mkdir()\n"
    assert _messages(tmp_path, MkdirModeRule(), body) == [
        (2, "mkdir under a userdirs path without mode=0o700"),
    ]


def test_module_side_effects_exempts_dunder_all_and_loggers(tmp_path: Path) -> None:
    body = (
        "__all__ = f()\nlogger = f()\nlog = f()\n_log = f()\n"
        "logs = f()\n__version__ = f()\n"
    )
    assert _messages(tmp_path, ModuleSideEffectsRule(), body) == [
        (5, "assignment from 'f()' at module scope; move it into a function"),
        (6, "assignment from 'f()' at module scope; move it into a function"),
    ]


@pytest.mark.parametrize(
    ("annotation", "exempt"),
    [
        ("Final", True),
        ("typing.Final[int]", True),
        ("ClassVar[int]", True),
        ("TypeAlias", True),
        ("x.TypeAlias", True),
        ("Finalx", False),
        ("Final.x", False),
        ("int", False),
    ],
)
def test_module_side_effects_exempts_declarative_annotations(
    tmp_path: Path,
    annotation: str,
    *,
    exempt: bool,
) -> None:
    findings = _messages(tmp_path, ModuleSideEffectsRule(), f"X: {annotation} = f()\n")
    assert (findings == []) is exempt


def test_module_side_effects_truncates_long_values_at_40(tmp_path: Path) -> None:
    value = "f(" + "a" * 50 + ")"
    expr = "g(" + "b" * 50 + ")"
    assert _messages(tmp_path, ModuleSideEffectsRule(), f"X = {value}\n{expr}\n") == [
        (1, f"assignment from {value[:40]!r} at module scope; move it into a function"),
        (2, f"expression {expr[:40]!r} at module scope; move it into a function"),
    ]


def test_module_side_effects_names_each_statement_kind(tmp_path: Path) -> None:
    body = (
        "obj.attr = 1\n"
        "if flag:\n    x = 1\n"
        "while x:\n    pass\n"
        "del y\n"
        "with c:\n    pass\n"
    )
    suffix = " at module scope; move it into a function"
    assert _messages(tmp_path, ModuleSideEffectsRule(), body) == [
        (1, "attribute/subscript assignment" + suffix),
        (2, "runtime if" + suffix),
        (4, "while" + suffix),
        (6, "delete" + suffix),
        (7, "with" + suffix),
    ]


def test_module_side_effects_allows_an_attribute_docstring_once(tmp_path: Path) -> None:
    body = 'X = 1\n"""Doc for X."""\n"""Second."""\n'
    assert _messages(tmp_path, ModuleSideEffectsRule(), body) == [
        (3, "expression \"'Second.'\" at module scope; move it into a function"),
    ]


def test_module_side_effects_module_docstring_is_not_an_attribute_doc(
    tmp_path: Path,
) -> None:
    assert _messages(tmp_path, ModuleSideEffectsRule(), '"""Mod."""\n"""Two."""\n') == [
        (2, "expression \"'Two.'\" at module scope; move it into a function"),
    ]


def test_module_side_effects_allows_a_local_builder_of_declarative_args(
    tmp_path: Path,
) -> None:
    body = (
        "def _build(a, b):\n    return a\n\n"
        "X = _build(1, b=2)\n"
        "Y = _build(f(), b=2)\n"
        "Z = _build(1, b=f())\n"
    )
    suffix = " at module scope; move it into a function"
    assert _messages(tmp_path, ModuleSideEffectsRule(), body) == [
        (5, "assignment from '_build(f(), b=2)'" + suffix),
        (6, "assignment from '_build(1, b=f())'" + suffix),
    ]


@pytest.mark.parametrize(
    ("head", "declarative"),
    [
        ("Widget()", True),
        ("_Widget()", True),
        ("__Widget()", True),
        ("pkg.mod.Widget()", True),
        ("widget()", False),
        ("_widget()", False),
        ("Widget.make()", False),
        ("frozenset({1})", True),
    ],
)
def test_module_side_effects_treats_class_calls_as_declarative(
    tmp_path: Path,
    head: str,
    *,
    declarative: bool,
) -> None:
    findings = _messages(tmp_path, ModuleSideEffectsRule(), f"X = {head}\n")
    assert (findings == []) is declarative


def test_globals_knob_exemptions(tmp_path: Path) -> None:
    body = (
        '__all__ = ["a"]\n__version__ = "1"\n__x = 3\nx__ = 4\n'
        "logger = 1\nlog = 2\n_log = 3\nlogs = 4\n"
        "T: TypeAlias = int\nF: Final = 5\nG: typing.Final = 6\n"
        'H: "Final" = 7\nI = int | None\nJ = cast(int, 8)\nK = 9\nL: int\n'
        "(a, b) = (1, 2)\nM = N = 3\n"
    )
    names = [
        message.split("'")[1]
        for _, message in _messages(tmp_path, ConfigGlobalsRule(), body)
    ]
    assert names == ["__x", "x__", "logs", "H", "K"]


@pytest.mark.parametrize(
    ("returns", "doc", "flagged"),
    [
        ("None", "Args:\n      x: X.", False),
        ("NoReturn", "Args:\n      x: X.", False),
        ("Never", "Args:\n      x: X.", False),
        ("int", "Args:\n      x: X.", True),
        ("int", "Yields:\n      x: X.", False),
        ("int", "Returns:\n      x: X.", False),
        ("None | int", "Args:\n      x: X.", True),
    ],
)
def test_returns_section_exempts_non_returning_annotations(
    tmp_path: Path,
    returns: str,
    doc: str,
    *,
    flagged: bool,
) -> None:
    body_lines = "\n".join(["    x = 1"] * 6)
    body = f'def f() -> {returns}:\n    """F.\n\n    {doc}\n\n    """\n{body_lines}\n'
    findings = _messages(tmp_path, ReturnsSectionRule(), body)
    assert findings == ([(1, "f docstring lacks Returns:")] if flagged else [])


def test_returns_section_exempts_a_property(tmp_path: Path) -> None:
    body_lines = "\n".join(["        x = 1"] * 6)
    body = (
        "class K:\n    @property\n    def f(self) -> int:\n"
        f'        """F.\n\n        Args:\n          x: X.\n\n        """\n{body_lines}\n'
    )
    assert _messages(tmp_path, ReturnsSectionRule(), body) == []


def test_returns_section_wants_yields_for_a_generator(tmp_path: Path) -> None:
    body_lines = "\n".join(["    x = 1"] * 5)
    body = (
        'def g():\n    """G.\n\n    Args:\n      x: X.\n\n    """\n'
        f"{body_lines}\n    yield 1\n"
    )
    assert _messages(tmp_path, ReturnsSectionRule(), body) == [
        (1, "g docstring lacks Yields:"),
    ]


@pytest.mark.parametrize(
    "literal",
    [
        "0x12345",
        "0X12345",
        "0o12345",
        "0O12345",
        "0b10101",
        "0B10101",
        "1_2345",
        "12345.0",
        "1e12345",
        "1E10000",
        "12345j",
        "12345J",
    ],
)
def test_group_leaves_non_decimal_and_non_integer_literals(literal: str) -> None:
    assert houselint_rules._group(literal) == literal


@pytest.mark.parametrize(
    ("literal", "grouped"),
    [("12345", "12_345"), ("123456", "123_456"), ("1234567", "1_234_567")],
)
def test_group_inserts_separators_every_three_digits(
    literal: str,
    grouped: str,
) -> None:
    assert houselint_rules._group(literal) == grouped


def test_cli_trailer_on_a_single_unterminated_line() -> None:
    lines = ["x = 1"]
    houselint_rules._fix_polyglot_trailer(lines)
    assert lines == ["x = 1\n", "# vim: ft=python\n"]


def test_cli_header_fix_keeps_an_exec_line_right_after_the_pinned_lines() -> None:
    """The ``exec uv`` search starts right after the two pinned lines."""
    lines = [
        "#!/bin/sh\n",
        "# ruff: noqa: EXE003, D300, D205 -- W.\n",
        "# fmt: off\n",
        "'''' 2>/dev/null #\n",
        "exec uv run custom\n",
        "'''\n",
    ]
    before = list(lines)
    houselint_rules._fix_polyglot_header(lines)
    assert lines == before


def test_cli_header_fix_rewrites_a_pinned_line_with_trailing_space() -> None:
    lines = [
        "#!/bin/sh\n",
        "# ruff: noqa: EXE003, D300, D205 -- W.\n",
        "# fmt: off  \n",
        "'''' 2>/dev/null #\n",
        _EXEC + "\n",
    ]
    houselint_rules._fix_polyglot_header(lines)
    assert lines[2] == "# fmt: off  \n"


def test_cli_header_check_reads_a_header_of_only_comments(tmp_path: Path) -> None:
    findings, _ = _cli(tmp_path, _HEAD + "# a\n# b\n")
    assert findings[:2] == [
        (1, "(a) line 2 must be `# ruff: noqa: EXE003, D300, D205[, CODE...]`"),
        (1, "(a) line 5 must be `# fmt: off`"),
    ]


def test_cli_header_check_continuation_may_trail_spaces(tmp_path: Path) -> None:
    exec_lines = (
        'exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync \\  \n'
        '  python3 "$0" "$@"\n'
    )
    findings, _ = _cli(
        tmp_path,
        _HEAD + _PINNED + exec_lines + "'''\n# vim: ft=python\n",
    )
    assert findings == []


def test_cli_main_annotation_rewrites_the_last_arrow(tmp_path: Path) -> None:
    """``-> str`` also appears inside a default string; the annotation is last."""
    body = 'def main(x="-> str") -> str:\n    return 0\n'
    assert _cli(tmp_path, body)[1] == 'def main(x="-> str") -> int:\n    return 0\n'


def test_cli_main_annotation_leaves_a_respelled_annotation(tmp_path: Path) -> None:
    """``unparse`` normalizes quotes; an unmatched spelling is not rewritten."""
    body = 'def main() -> "str":\n    return 0\n'
    assert _cli(tmp_path, body)[1] is None


def test_cli_main_annotation_starts_at_mains_own_line(tmp_path: Path) -> None:
    body = "x = {\n    'a': 1,\n}\ndef main():\n    return 0\n"
    assert _cli(tmp_path, body)[1] == body.replace("def main():", "def main() -> int:")


def test_cli_main_annotation_stops_before_the_body(tmp_path: Path) -> None:
    body = "def main(): return 0\nd = {1: 2}\n"
    assert _cli(tmp_path, body)[1] is None


def test_falls_through_reads_the_callees_last_attribute() -> None:
    main = ast.parse("def main() -> int:\n    os.path.exit(1)\n").body[0]
    assert isinstance(main, ast.FunctionDef)
    assert not houselint_rules._falls_through(main.body, noreturn=frozenset())


@pytest.mark.parametrize(
    ("source", "end"),
    [
        ("", 0),
        ('"""Doc."""\n', 1),
        ('"""Doc."""\nimport a\nfrom b import (\n    c,\n)\nx = 1\nimport d\n', 5),
        ("import a\n", 1),
        ("x = 1\nimport a\n", 0),
        ("1\nimport a\n", 0),
    ],
)
def test_module_import_end_is_the_line_after_the_leading_imports(
    source: str,
    end: int,
) -> None:
    assert houselint_rules._module_import_end(ast.parse(source)) == end


@pytest.mark.parametrize(
    ("signature", "call", "bound"),
    [
        ("a, b=2", "f(1)", {"a": "1", "b": "2"}),
        ("a, b=2", "f(1, 3)", {"a": "1", "b": "3"}),
        ("a, b", "f(1)", None),
        ("a", "f(1, 2)", None),
        ("a", "f(a=1)", {"a": "1"}),
        ("a", "f(1, a=2)", None),
        ("a", "f(b=1)", None),
        ("a, /", "f(a=1)", None),
        ("*, k=3", "f()", {"k": "3"}),
        ("*, k", "f()", None),
        ("a, /, b=1", "f(1, b=x + y)", {"a": "1", "b": "x + y"}),
        ("a, *, k=1", "f(1, k=2)", {"a": "1", "k": "2"}),
        ("a=1, b=2", "f()", {"a": "1", "b": "2"}),
        # A default ran once, with the def; copied to the call it runs each time.
        ("a, b=next(c)", "f(1)", None),
        ("a, b=[]", "f(1)", None),
        ("a, b=LIMIT", "f(1)", None),
        ("*, k=-1", "f()", None),
        ("a, b=next(c)", "f(1, 2)", {"a": "1", "b": "2"}),
    ],
)
def test_bind_arguments_maps_each_parameter_or_refuses(
    signature: str,
    call: str,
    bound: dict[str, str] | None,
) -> None:
    node = ast.parse(f"def f({signature}): pass").body[0]
    assert isinstance(node, ast.FunctionDef)
    expr = ast.parse(call, mode="eval").body
    assert isinstance(expr, ast.Call)
    got = houselint_rules._bind_arguments(node, expr)
    texts = None if got is None else {name: ast.unparse(e) for name, e in got.items()}
    assert texts == bound


@pytest.mark.parametrize(
    ("text", "summary", "rest"),
    [
        ("Short.", "Short.", []),
        (
            "A long first sentence here. Then the second one is here.",
            "A long first sentence here",
            ["Then the second one", "is here."],
        ),
        (
            "A long first clause here; then another clause follows.",
            "A long first clause here.",
            ["Then another clause", "follows."],
        ),
        (
            "A long single sentence without any break at all",
            "A long single sentence without any break at all",
            [],
        ),
        (
            (
                "Sum.\n\nArgs:\n    x: X.\n\nReturns:\n    y: Y.\n\n"
                "Rationale paragraph that is fairly long and wraps.\n\nAnother."
            ),
            "Sum.",
            [
                "Rationale paragraph",
                "that is fairly long",
                "and wraps.",
                "",
                "Another.",
            ],
        ),
        ("  Lone.  ", "Lone.", []),
    ],
)
def test_split_private_docstring_wraps_summary_and_rationale(
    text: str,
    summary: str,
    rest: list[str],
) -> None:
    assert houselint_rules._split_private_docstring(
        text,
        summary_width=20 if "Sum." not in text else 40,
        comment_width=20,
    ) == (summary, rest)


@pytest.mark.parametrize(
    ("source", "needs_this", "expected"),
    [
        (
            "x = 1\n",
            False,
            (
                "from typing import Final\n\n"
                "_CWD: Final = Path(__file__).resolve().parent\n\nx = 1\n"
            ),
        ),
        (
            "import os \nX = 1\n",
            False,
            (
                "import os \nfrom typing import Final\n\n"
                "_CWD: Final = Path(__file__).resolve().parent\n\nX = 1\n"
            ),
        ),
        (
            "from a import (\n    b,\n)\nx = 1\n",
            False,
            (
                "from a import (\n    b,\n)\nfrom typing import Final\n\n"
                "_CWD: Final = Path(__file__).resolve().parent\n\nx = 1\n"
            ),
        ),
        (
            "import X\n\n  # c\nX = 1\n",
            False,
            (
                "import X\nfrom typing import Final\n\n"
                "_CWD: Final = Path(__file__).resolve().parent\n\n  # c\nX = 1\n"
            ),
        ),
        (
            "from typing import Final\nimport os\n_THIS: Final = 1\n",
            True,
            (
                "from typing import Final\nimport os\n\n"
                "_CWD: Final = _THIS.parent\n\n_THIS: Final = 1\n"
            ),
        ),
        (
            "from typing import Final\nimport os\n_CWD: Final = 1\n",
            True,
            (
                "from typing import Final\nimport os\n\n"
                "_THIS: Final = Path(__file__).resolve()\n\n_CWD: Final = 1\n"
            ),
        ),
        (
            "from typing import Final\n_CWD: Final = 1\n",
            False,
            "from typing import Final\n_CWD: Final = 1\n",
        ),
        (
            "from x import Final\nx = 1\n",
            False,
            (
                "from x import Final\nfrom typing import Final\n\n"
                "_CWD: Final = Path(__file__).resolve().parent\n\nx = 1\n"
            ),
        ),
        (
            "from typing import Final \nx = 1\n",
            False,
            (
                "from typing import Final \n\n"
                "_CWD: Final = Path(__file__).resolve().parent\n\nx = 1\n"
            ),
        ),
        (
            "from typing import Final\nimport X\nx = 1\n",
            False,
            (
                "from typing import Final\nimport X\n\n"
                "_CWD: Final = Path(__file__).resolve().parent\n\nx = 1\n"
            ),
        ),
        (
            "from typing import Final\nimport os\n_this: Final = 1\n",
            True,
            (
                "from typing import Final\nimport os\n\n"
                "_THIS: Final = Path(__file__).resolve()\n_CWD: Final = _THIS.parent\n\n"
                "_this: Final = 1\n"
            ),
        ),
    ],
)
def test_insert_cwd_bindings_places_missing_bindings_after_the_imports(
    source: str,
    *,
    needs_this: bool,
    expected: str,
) -> None:
    assert houselint_rules._insert_cwd_bindings(source, needs_this=needs_this) == (
        expected
    )


@pytest.mark.parametrize(
    ("source", "span"),
    [
        ("# a\n# b\ndef f():\n    pass\n", (0, 4)),
        ("x = 1\n# a\n@d\n@e\ndef f():\n    pass\n", (1, 6)),
        ("def f():\n    pass\n", (0, 2)),
        ("x = 1\n  # a\nclass C:\n    pass\n", (1, 4)),
        ("x = 1\n\n# a\n# b\n# c\ndef f():\n    pass\n", (2, 7)),
        ("x = 1 # a\ndef f():\n    pass\n", (1, 3)),
        ("# a\nx = 1\n", (0, 2)),
        ("def f():\n    pass\n# z\n", (0, 2)),
    ],
)
def test_def_span_takes_decorators_and_the_comment_block_above(
    source: str,
    span: tuple[int, int],
) -> None:
    stmt = ast.parse(source).body[-1]
    assert houselint_rules._def_span(stmt, source.splitlines()) == span


@pytest.mark.parametrize(
    ("lines", "lineno", "commented"),
    [
        (["# a", "x = 1"], 2, True),
        (["  # a", "x = 1"], 2, True),
        (["# a", "y = 2", "x = 1"], 3, True),
        (["# a", "", "x = 1"], 3, False),
        (["x = 1"], 1, False),
        (["y", "z", "x"], 3, False),
        (["# a", "y", "z", "x"], 4, True),
        (["# a", "", "y", "x"], 4, False),
        (["y #", "x"], 2, False),
        (["# a", "y", "z", "w", "x"], 5, True),
    ],
)
def test_has_leading_comment_scans_the_run_above(
    lines: list[str],
    lineno: int,
    *,
    commented: bool,
) -> None:
    assert houselint_rules._has_leading_comment(lines, lineno) is commented


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('logger.info(f"a {x:>{w}}")\n', 'logger.info("a %s", f"{x:>{w}}")\n'),
        ('logger.info(f"a {x:{w}.{p}f}")\n', 'logger.info("a %s", f"{x:{w}.{p}f}")\n'),
        ('logger.info(f"a {x:>8}")\n', 'logger.info("a %8s", x)\n'),
        ('logger.info(f"a {x:<8}")\n', 'logger.info("a %-8s", x)\n'),
        ('logger.info(f"a {x:.2f}")\n', 'logger.info("a %.2f", x)\n'),
        ('logger.info(f"a {x!r:,}")\n', 'logger.info("a %s", f"{x!r:,}")\n'),
        ('logger.info(f"a {x:,}")\n', 'logger.info("a %s", f"{x:,}")\n'),
        ('logger.info(f"a {x!r} 5%")\n', 'logger.info("a %r 5%%", x)\n'),
        (
            'logger.info(f"{x!a} {y!s} {z!r:>4} {w}")\n',
            'logger.info("%a %s %s %s", x, y, f"{z!r:>4}", w)\n',
        ),
    ],
)
def test_logging_fstring_fix_carries_each_spec(
    tmp_path: Path,
    source: str,
    expected: str,
) -> None:
    head = "import logging\nlogger = logging.getLogger(__name__)\n"
    assert _fixed(tmp_path, LoggingFstringRule(), head + source) == head + expected


@pytest.mark.parametrize(
    ("test", "expected"),
    [
        ("not (a\n and b)", "a and b"),
        ("(a\n or b)", "not a and not b"),
        ("(a and\n b > 1)", "not a or b <= 1"),
        ("x < 1", "x >= 1"),
        ("1 < x", "1 >= x"),
        ("x < y", "not x < y"),
        ("not  x", "x"),
        ("a and (b or c)", "not a or (not b and not c)"),
        ("a or  b", "not a and not b"),
        ("x + 1", "not (x + 1)"),
        ("f( x )", "not f( x )"),
    ],
)
def test_negated_spells_not_test(test: str, expected: str) -> None:
    source = f"({test})"
    node = ast.parse(source, mode="eval").body
    assert houselint_rules._negated(node, source) == expected


def test_segment_falls_back_to_unparse_without_a_span() -> None:
    node = ast.Name(id="x")
    assert houselint_rules._segment("y", node) == "x"


def _inline(tmp_path: Path, helper: str, call: str) -> str | None:
    """Inline the single ``_h`` call in ``call`` using the def in ``helper``."""
    path = tmp_path / "mod.py"
    path.write_text(
        f"def f(a, k, x):\n    return {call}\n\n\n{helper}",
        encoding="utf-8",
    )
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert unit.tree is not None
    node = unit.tree.body[-1]
    assert isinstance(node, ast.FunctionDef)
    site = next(
        n
        for n in unit.nodes(ast.Call)
        if isinstance(n.func, ast.Name) and n.func.id == "_h"
    )
    return houselint_rules._inline_call(node, site, unit)


@pytest.mark.parametrize(
    ("call", "inlined"),
    [
        ("_h(*a)", None),
        ("_h(**k)", None),
        ("_h(a, **k)", None),
        ("_h({})", None),
        ("_h([])", None),
        ("_h(())", None),
        ("_h(set())", "set() + 1"),
        ("_h([1])", "[1] + 1"),
        ("_h((1,))", "(1,) + 1"),
        ("_h({1: 2})", "{1: 2} + 1"),
        ("_h(v=a)", "a + 1"),
        ("g(_h(a))", "a + 1"),
        ("2 * _h(a)", "(a + 1)"),
        ("[_h(a)]", "a + 1"),
        ("x[_h(a)]", "a + 1"),
        ("-_h(a)", "(a + 1)"),
    ],
)
def test_inline_call_binds_or_refuses(
    tmp_path: Path,
    call: str,
    inlined: str | None,
) -> None:
    assert _inline(tmp_path, "def _h(v):\n    return v + 1\n", call) == inlined


@pytest.mark.parametrize(
    ("helper", "inlined"),
    [
        ("def _h(v):\n    return v  # why\n", None),
        ("def _h(v):  # why\n    return v\n", None),
        ("# why\ndef _h(v):\n    return v\n", None),
        ('def _h(v):\n    return ("a"\n    "b", v)\n', None),
        ('def _h(v):\n    return f"""{v}\n"""\n', None),
        ('def _h(v):\n    return [\n        "a", v,\n    ]\n', "['a', a]"),
        ('def _h(v):\n    return [\n        "a",\n        v,\n    ]\n', None),
        ("def _h(v):\n    return max(\n        v,\n        1,\n    )\n", "max(a, 1)"),
        ("def _h(v):\n    return v if c else 1\n", "(a if c else 1)"),
        ("def _h(v):\n    return lambda w=v: w\n", "(lambda w=a: w)"),
        # The walrus would bind ``w`` in the caller.
        ("def _h(v):\n    return (w := v)\n", None),
    ],
)
def test_inline_call_handles_comments_strings_and_reflow(
    tmp_path: Path,
    helper: str,
    inlined: str | None,
) -> None:
    assert _inline(tmp_path, helper, "_h(a)") == inlined


@pytest.mark.parametrize(
    ("helper", "call", "inlined"),
    [
        # A branch of a conditional runs never or once; the call ran it once.
        ("def _h(c, v):\n    return v if c else 0\n", "_h(a, g())", None),
        ("def _h(c, v):\n    return 0 if c else v\n", "_h(a, g())", None),
        ("def _h(c, v):\n    return c and v\n", "_h(a, g())", None),
        ("def _h(c, v):\n    return c or v\n", "_h(a, g())", None),
        ("def _h(lo, v, hi):\n    return lo < v < hi\n", "_h(a, k, g())", None),
        # The test, the first operand, and the first comparison always run.
        ("def _h(c):\n    return 1 if c else 0\n", "_h(g())", "(1 if g() else 0)"),
        ("def _h(c, v):\n    return c and v\n", "_h(g(), a)", "g() and a"),
        (
            "def _h(lo, v, hi):\n    return lo < v < hi\n",
            "_h(g(), h(), a)",
            "g() < h() < a",
        ),
        # A lambda runs its body when called; a comprehension once per element.
        ("def _h(v):\n    return lambda: v\n", "_h(g())", None),
        ("def _h(v):\n    return len([v for _ in range(3)])\n", "_h(g())", None),
        ("def _h(v):\n    return sum(v for _ in range(3))\n", "_h(g())", None),
        ("def _h(v):\n    return len([w for w in range(3) if v])\n", "_h(g())", None),
        (
            "def _h(v):\n    return len([w for u in range(3) for w in v])\n",
            "_h(g())",
            None,
        ),
        # The outermost iterable runs once, before any element.
        (
            "def _h(v):\n    return len([int(w) for w in v])\n",
            "_h(g())",
            "len([int(w) for w in g()])",
        ),
        # A name or a constant is the same value wherever it is read...
        ("def _h(c, v):\n    return v if c else 0\n", "_h(a, k)", "(k if a else 0)"),
        (
            "def _h(v):\n    return sum(v for _ in range(3))\n",
            "_h(k)",
            "sum(k for _ in range(3))",
        ),
        ("def _h(v):\n    return lambda: v\n", "_h(1)", "(lambda: 1)"),
        # ...except a lambda reads the name when called: a closure built in a
        # loop sees the loop's last value.
        ("def _h(v):\n    return lambda: v\n", "_h(k)", None),
        # Arguments run in call order, before any call the body makes.
        ("def _h(v, w):\n    return w - v\n", "_h(g(), h())", None),
        ("def _h(v):\n    return g() + v\n", "_h(h())", None),
        ("def _h(v):\n    return ([g(w) for w in ws], v)\n", "_h(h())", None),
        ("def _h(v, w):\n    return (v, w)\n", "_h(g(), h())", "(g(), h())"),
        ("def _h(v):\n    return g(v) + 1\n", "_h(h())", "g(h()) + 1"),
        ("def _h(v):\n    return (lambda: h(), v)\n", "_h(g())", "(lambda: h(), g())"),
        # A default ran once, when the def did.
        ("def _h(v, n=next(c)):\n    return v + n\n", "_h(a)", None),
        ("def _h(v, n=1):\n    return v + n\n", "_h(a)", "a + 1"),
        # A name is the parameter only where the def's own scope binds it.
        ("def _h(v):\n    return lambda v: v\n", "_h(1)", None),
        ("def _h(v):\n    return len([v for v in range(3)])\n", "_h(a)", None),
        ("def _h(v):\n    return (v := v + 1)\n", "_h(a)", None),
    ],
)
def test_inline_call_evaluates_each_argument_as_the_call_did(
    tmp_path: Path,
    helper: str,
    call: str,
    inlined: str | None,
) -> None:
    """The call evaluated each argument once, in order, before the body ran."""
    assert _inline(tmp_path, helper, call) == inlined


@pytest.mark.parametrize(
    ("expr", "atomic"),
    [
        ("x", "x"),
        ("1", "1"),
        ("a.b", "a.b"),
        ("f()", "f()"),
        ("a[1]", "a[1]"),
        ("a[1, 2]", "a[1, 2]"),
        ("[1]", "[1]"),
        ("(1, 2)", "(1, 2)"),
        ("{1}", "{1}"),
        ("{1: 2}", "{1: 2}"),
        ('f"{x}"', "f'{x}'"),
        ("a + b", "(a + b)"),
        ("-a", "(-a)"),
        ("a if b else c", "(a if b else c)"),
    ],
)
def test_atomic_parenthesizes_loose_expressions(expr: str, atomic: str) -> None:
    node = ast.parse(expr, mode="eval").body
    assert houselint_rules._atomic(node) == atomic


@pytest.mark.parametrize(
    ("source", "usage"),
    [
        ("x: A = b\n", {"A": (1, 0), "b": (0, 1), "x": (0, 1)}),
        ("b\n", {"b": (0, 1)}),
        ("def f(a: A) -> R:\n    return b\n", {"A": (1, 0), "R": (1, 0), "b": (0, 1)}),
        ('x: "A[B]"\n', {"A": (1, 0), "B": (1, 0), "x": (0, 1)}),
        ('cast("T", v)\n', {"T": (1, 0), "v": (0, 1)}),
        ("if TYPE_CHECKING:\n    import a\nelse:\n    c\n", {"c": (0, 1)}),
    ],
)
def test_name_usage_tallies_annotation_and_runtime_uses(
    source: str,
    usage: dict[str, tuple[int, int]],
) -> None:
    got = houselint_rules._name_usage(ast.parse(source))
    assert {k: (v.annotation, v.runtime) for k, v in got.items()} == usage


@pytest.mark.parametrize(
    ("test", "expected"),
    [
        ("TYPE_CHECKING", True),
        ("typing.TYPE_CHECKING", True),
        ("type_checking", False),
        ("x.type_checking", False),
        ("TYPE_CHECKING()", False),
        ("TYPE_CHECKING.x", False),
    ],
)
def test_is_type_checking_test(test: str, *, expected: bool) -> None:
    node = ast.parse(test, mode="eval").body
    assert houselint_rules._is_type_checking_test(node) is expected


@pytest.mark.parametrize(
    ("source", "imports"),
    [
        ("import a\nfrom b import c\n", ["import a", "from b import c"]),
        (
            "try:\n    import a\nexcept E:\n    import b\nelse:\n    import c\nfinally:\n    import d\n",
            ["import a", "import c", "import d"],
        ),
        ("if x:\n    import a\nelse:\n    import b\n", ["import a", "import b"]),
        ("if TYPE_CHECKING:\n    import a\nelse:\n    import b\n", []),
        ("if x:\n    y = 1\n", []),
        ("def f():\n    import a\n", []),
    ],
)
def test_eager_imports_skip_type_checking_blocks(
    source: str,
    imports: list[str],
) -> None:
    got = houselint_rules._eager_imports(ast.parse(source))
    assert [ast.unparse(node) for node in got] == imports


@pytest.mark.parametrize(
    ("source", "modules"),
    [
        ("import a.b, c", ["a.b", "c"]),
        ("from a.b import c", ["a.b"]),
        ("from . import c", []),
        ("from .a import c", []),
    ],
)
def test_imported_modules_skip_relative_imports(
    source: str,
    modules: list[str],
) -> None:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.Import | ast.ImportFrom)
    assert list(houselint_rules._imported_modules(node)) == modules


@pytest.mark.parametrize(
    ("source", "names"),
    [
        ("import a.b, c as d", ["a", "d"]),
        ("from a import *", []),
        ("from a import b as c, d", ["c", "d"]),
    ],
)
def test_import_locals_binds_roots_and_aliases(source: str, names: list[str]) -> None:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.Import | ast.ImportFrom)
    assert list(houselint_rules._import_locals(node)) == names


def test_names_reached_by_siblings(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "mod_test.py").write_text('"_a"\n', encoding="utf-8")
    (tmp_path / "b.py").write_text("m._b\n_e\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("from mod import _c\n_f = 1\n", encoding="utf-8")
    (tmp_path / "d.py").write_text("_d (\n", encoding="utf-8")
    (tmp_path / "e.py").write_bytes(b"\xff_g\n")
    (tmp_path / "f.txt").write_text("_h\n", encoding="utf-8")
    names = {"_a", "_b", "_c", "_d", "_e", "_f", "_g", "_h", "_i"}
    assert houselint_rules._names_reached_by_siblings(tmp_path / "mod.py", names) == {
        "_a",
        "_b",
        "_c",
        "_d",
    }


def test_names_reached_by_siblings_reads_every_sibling(tmp_path: Path) -> None:
    for name in ("a.py", "b_test.py", "c.py", "mod.py", "z.py"):
        (tmp_path / name).write_text("_n\n", encoding="utf-8")
    (tmp_path / "b_test.py").write_text("_a\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("x._b\n", encoding="utf-8")
    (tmp_path / "z.py").write_text("from q import _c\n", encoding="utf-8")
    assert houselint_rules._names_reached_by_siblings(
        tmp_path / "mod.py",
        {"_a", "_b", "_c"},
    ) == {"_a", "_b", "_c"}


def test_names_reached_by_siblings_skips_itself(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("from x import _a\n", encoding="utf-8")
    (tmp_path / "mod_test.py").write_text("\n", encoding="utf-8")
    assert houselint_rules._names_reached_by_siblings(tmp_path / "mod.py", {"_a"}) == (
        frozenset()
    )


@pytest.mark.parametrize(
    ("source", "forwarding"),
    [
        ("def f(self, a):\n    return super().f(a)\n", True),
        ("def f(self, a):\n    super().f(a)\n", True),
        ("async def f(self, a):\n    return await super().f(a)\n", True),
        ("def f(self, a):\n    return super().f(a=a)\n", True),
        ("def f(self, *a, **k):\n    return super().f(*a, **k)\n", True),
        ("def f(self, *a):\n    return super().f(a)\n", False),
        ("def f(self, **k):\n    return super().f(k=k)\n", False),
        ("def f(self, a):\n    return super().f(a=b)\n", False),
        ("def f(self, a):\n    return super().g(a)\n", False),
        ("def f(self, a):\n    return sup().f(a)\n", False),
        ("def f(self, a):\n    return x.f(a)\n", False),
        ("def f(self, a):\n    x = super().f(a)\n", False),
        ("def f(self, a):\n    super().f(a)\n    return 1\n", False),
        ('def f(self, a):\n    """D."""\n    return super().f(a)\n', True),
    ],
)
def test_is_super_only(source: str, *, forwarding: bool) -> None:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    assert houselint_rules._is_super_only(node) is forwarding


@pytest.mark.parametrize(
    ("test", "expected"),
    [
        ("isinstance(x, int)", True),
        ("issubclass(x, int)", True),
        ("callable(x)", True),
        ("len(x)", False),
        ("not isinstance(x, int)", True),
        ("isinstance(x, int) and callable(x)", True),
        ("isinstance(x, int) and x", False),
        ("type(x) is int", True),
        ("int is type(x)", True),
        ("len(x) is int", False),
        ("x is type", False),
        ("type(x) is int is y", False),
        ("x", False),
    ],
)
def test_is_type_test(test: str, *, expected: bool) -> None:
    node = ast.parse(test, mode="eval").body
    assert houselint_rules._is_type_test(node) is expected


def _dataclass_class(
    source: str,
) -> tuple[ast.ClassDef, houselint_rules._DataclassBindings]:
    """Parse ``source`` and return its last class with its dataclass bindings."""
    tree = ast.parse(source)
    node = tree.body[-1]
    assert isinstance(node, ast.ClassDef)
    return node, houselint_rules._dataclass_bindings(tree)


@pytest.mark.parametrize(
    ("decorators", "missing", "slotted"),
    [
        ("@dataclass", ("kw_only", "slots"), False),
        ("@dataclass(kw_only=True, slots=True)", (), True),
        ("@dataclass(kw_only=True)", ("slots",), False),
        ("@dataclass(slots=1, kw_only=False)", ("kw_only", "slots"), False),
        ("@dataclass(slots=x)", ("kw_only", "slots"), False),
        ("@dataclass(**o)", ("kw_only", "slots"), False),
        ("@other(slots=True)", (), False),
        ("@other", (), False),
        ("@dc.dataclass(slots=True)", ("kw_only",), True),
        ("@x.dataclass(slots=True)", (), False),
        ("@other()\n@dataclass(slots=True)", ("kw_only",), True),
        ("@dataclass(frozen=True, slots=True)", ("kw_only",), True),
        ("@dataclass(slots=True)()", (), False),
    ],
)
def test_dataclass_kwargs_and_slots(
    decorators: str,
    missing: tuple[str, ...],
    *,
    slotted: bool,
) -> None:
    source = (
        "from dataclasses import dataclass\nimport dataclasses as dc\n"
        f"{decorators}\nclass C:\n    pass\n"
    )
    node, bound = _dataclass_class(source)
    got = tuple(
        key
        for dec in node.decorator_list
        for key in houselint_rules._missing_kwargs(dec, ("kw_only", "slots"), bound)
    )
    assert got == missing
    assert houselint_rules._is_slotted(node, bound) is slotted


def test_dataclass_bindings_collect_each_spelling() -> None:
    tree = ast.parse(
        "from dataclasses import dataclass as dc, field\nimport dataclasses, os\n"
        "import dataclasses as d2\nif x:\n    from dataclasses import dataclass\n",
    )
    assert houselint_rules._dataclass_bindings(
        tree,
    ) == houselint_rules._DataclassBindings(
        direct=frozenset({"dc", "dataclass"}),
        modules=frozenset({"dataclasses", "d2"}),
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a_test.py", True),
        ("test_a.py", True),
        ("test_a.pyi", False),
        ("a_test.pyi", False),
        ("Test_a.py", False),
        ("tests.py", False),
        ("a.py", False),
        ("test_.py", True),
    ],
)
def test_is_test_file(tmp_path: Path, name: str, *, expected: bool) -> None:
    assert houselint_rules._is_test_file(tmp_path / name) is expected


@pytest.mark.parametrize(
    ("source", "lines"),
    [
        ("def f():\n    pass\n", 1),
        ('def f():\n    """D."""\n', 0),
        ('def f():\n    """D."""\n    x = 1\n    y = (\n        2\n    )\n', 4),
        ("def f():\n    x = 1\n    if x:\n        y = 2\n", 3),
    ],
)
def test_body_lines_spans_the_body_without_its_docstring(
    source: str,
    lines: int,
) -> None:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    assert houselint_rules._body_lines(node) == lines


@pytest.mark.parametrize(
    ("source", "rule", "ignored"),
    [
        ("import a  # house-ignore[x]\n", "x", True),
        ("import a  # house-ignore[y]\n", "x", False),
        ("from a import (  # house-ignore[x]\n    b,\n)\n", "x", True),
        ("from a import (\n    b,\n)  # house-ignore[x]\n", "x", True),
        ("# house-ignore[x]\nimport a\n", "x", False),
        ("import a\n# house-ignore[x]\n", "x", False),
    ],
)
def test_house_ignore_reads_only_the_statement_lines(
    tmp_path: Path,
    source: str,
    rule: str,
    *,
    ignored: bool,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    node = next(iter(unit.nodes(ast.Import, ast.ImportFrom)))
    assert houselint_rules._house_ignore(unit, node, rule) is ignored


@pytest.mark.parametrize(
    ("source", "suppressed"),
    [
        ("def f():\n    import a  # noqa\n", True),
        ("def f():\n    import a  # NOQA: PLC0415\n", True),
        ("def f():\n    import a  # noqa: E501\n", False),
        ("def f():\n    from a import (  # noqa: PLC0415\n        b,\n    )\n", True),
        ("def f():\n    from a import (\n        b,\n    )  # noqa: PLC0415\n", True),
        ("def f():  # noqa\n    import a\n", False),
        ("def f():\n    import a\n    # noqa\n", False),
        ("def f():\n    import a  # x: y\n", False),
        ("def f():\n    x = 1  # noqa\n    import a\n", False),
    ],
)
def test_noqa_plc0415_reads_only_the_statement_lines(
    tmp_path: Path,
    source: str,
    *,
    suppressed: bool,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    node = next(iter(unit.nodes(ast.Import, ast.ImportFrom)))
    assert houselint_rules._noqa_plc0415(unit, node) is suppressed


@pytest.mark.parametrize(
    ("source", "nested"),
    [
        ("import a\n", []),
        ("def f():\n    import a\n", [2]),
        ('if __name__ == "__main__":\n    import a\n', []),
        ("if TYPE_CHECKING:\n    import a\n", []),
        ("if typing.TYPE_CHECKING:\n    import a\n", []),
        ("if x:\n    import a\n", [2]),
        ("try:\n    import a\nexcept ImportError:\n    pass\n", []),
        ("try:\n    import a\nexcept:\n    pass\n", []),
        ("try:\n    import a\nexcept (OSError, ImportError):\n    pass\n", []),
        ("try:\n    import a\nexcept OSError:\n    pass\n", [2]),
        ("try:\n    import a\nexcept (OSError,):\n    pass\n", [2]),
        ("try:\n    import a\nexcept x.ImportError:\n    pass\n", [2]),
        (
            "def f():\n    try:\n        import a\n    except ImportError:\n        pass\n",
            [3],
        ),
        (
            "class C:\n    try:\n        import a\n    except ImportError:\n        pass\n",
            [3],
        ),
        (
            "if x:\n    try:\n        import a\n    except ImportError:\n        pass\n",
            [],
        ),
    ],
)
def test_nested_imports_exempt_guards_and_optional_imports(
    tmp_path: Path,
    source: str,
    nested: list[int],
) -> None:
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert [node.lineno for node in houselint_rules._nested_imports(unit)] == nested


@pytest.mark.parametrize(
    ("value", "knob"),
    [
        ("1", True),
        ("'a'", True),
        ("1 | 2", False),
        ("int | None", False),
        ("Mapping[str, int]", False),
        ("cast(int, 1)", False),
        ("re.compile('a')", False),
        ("(1, 2)", True),
    ],
)
def test_is_knob_exempts_type_expressions(value: str, *, knob: bool) -> None:
    stmt = ast.parse(f"X = {value}\n").body[0]
    assert isinstance(stmt, ast.Assign)
    assert houselint_rules._is_knob(stmt) is knob


@pytest.mark.parametrize(
    ("source", "knob"),
    [
        ("__all__ = 1\n", False),
        ("__x__ = 1\n", False),
        ("__x = 1\n", True),
        ("X: TypeAlias = 1\n", False),
        ("X: typealias = 1\n", True),
        ("X: t.TypeAlias = 1\n", True),
        ("X: int = 1\n", True),
        ("X: int\n", False),
        ("a.b = 1\n", False),
    ],
)
def test_is_knob_exempts_dunders_and_aliases(source: str, *, knob: bool) -> None:
    stmt = ast.parse(source).body[0]
    assert isinstance(stmt, ast.Assign | ast.AnnAssign)
    assert houselint_rules._is_knob(stmt) is knob


@pytest.mark.parametrize(
    ("value", "constant"),
    [
        ("DEFAULT_X", "DEFAULT_X"),
        ("mod.DEFAULT_X", "DEFAULT_X"),
        ("Default", None),
        ("_1", None),
        ("X1", "X1"),
        ("f()", None),
    ],
)
def test_constant_default_names_an_all_caps_reference(
    value: str,
    constant: str | None,
) -> None:
    field = ast.parse(f"x: int = {value}\n").body[0]
    assert isinstance(field, ast.AnnAssign)
    assert houselint_rules._constant_default(field) == constant


@pytest.mark.parametrize(
    ("source", "binding"),
    [
        ("_X: Final = 1\n", ("_X", "Final literal")),
        ("_X: Final = f()\n", None),
        ("_X: int = 1\n", None),
        ("_X: Final\n", None),
        ("a.b: Final = 1\n", None),
        ("_R = re.compile('a')\n", ("_R", "single-use re.compile")),
        ("_R = x.compile('a')\n", None),
        ("_R = re.match('a')\n", None),
        ("_R = compile('a')\n", None),
        ("_R = _S = re.compile('a')\n", None),
        ("a.b = re.compile('a')\n", None),
        ("x\n", None),
    ],
)
def test_inlineable_binding(source: str, binding: tuple[str, str] | None) -> None:
    stmt = ast.parse(source).body[0]
    assert houselint_rules._inlineable_binding(stmt) == binding


@pytest.mark.parametrize(
    ("text", "exempt"),
    [
        ("noqa: E501", True),
        ("pyrefly: ignore[bad-return] -- upstream bug.", True),
        ("https://example.com/x", True),
        ("http://example.com", True),
        ("HTTPS://example.com", False),
        ("see https://example.com", False),
        ("https://a b", False),
        ("-- Section --", True),
        ("=====", True),
        ("prose here", False),
    ],
)
def test_is_exempt_comment(text: str, *, exempt: bool) -> None:
    assert houselint_rules._is_exempt_comment(text) is exempt


@pytest.mark.parametrize(
    ("source", "text"),
    [("# a", "a"), ("#  a  ", "a"), ("##a", "a"), ("#X#", "X#")],
)
def test_comment_text_strips_hashes_and_space(source: str, text: str) -> None:
    tok = next(t for t in _tokenize(source) if t.string[:1] == "#")
    assert houselint_rules._comment_text(tok) == text


@pytest.mark.parametrize(
    ("source", "blocks"),
    [
        ("# a\n# b\nx = 1\n# c\n", [["# a", "# b"], ["# c"]]),
        ("# a\nx = 1\n# b\n", [["# a"], ["# b"]]),
        ("# a\n\n# b\n", [["# a"], ["# b"]]),
        ("# a\n  # b\n", [["# a"], ["# b"]]),
        ("x = 1  # a\ny = 2  # b\n", [["# a"], ["# b"]]),
        ("# a\n# noqa\n", [["# a"], ["# noqa"]]),
        ("def f():\n    # a\n    # b\n    pass\n", [["# a", "# b"]]),
    ],
)
def test_comment_blocks_group_adjacent_full_line_comments(
    source: str,
    blocks: list[list[str]],
) -> None:
    got = houselint_rules._comment_blocks(tuple(_tokenize(source)))
    assert [[tok.string for tok in block] for block in got] == blocks


def test_bound_names_cover_every_binding_kind(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text(
        "import a.b\nfrom c import d as e\nclass K:\n    pass\n"
        "def f(p):\n    return q.r\nasync def g():\n    pass\n",
        encoding="utf-8",
    )
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert houselint_rules._bound_names(unit) == {
        "a",
        "e",
        "K",
        "f",
        "g",
        "p",
        "q",
        "r",
    }


@pytest.mark.parametrize(
    ("chain", "longest"),
    [
        (("a", "b", "c"), ("a", "b")),
        (("a",), ("a",)),
        (("z", "b"), None),
        (("a", "x"), ("a",)),
    ],
)
def test_longest_import_prefers_the_longest_prefix(
    chain: tuple[str, ...],
    longest: tuple[str, ...] | None,
) -> None:
    dotted = {("a",), ("a", "b")}
    assert houselint_rules._longest_import(chain, dotted=dotted) == longest


def test_import_alias_fix_splits_each_statement_at_its_indent(
    tmp_path: Path,
) -> None:
    source = "def f():\n    from loop.a import b as c, d\n    return c, d\n"
    assert _fixed(tmp_path, ImportAliasRule(), source) == (
        "def f():\n    from loop.a import b\n    from loop.a import d\n"
        "    return b, d\n"
    )


@pytest.mark.parametrize(
    ("source", "fixed"),
    [
        (
            "if x:\n\timport loop.a.b.c, os\n\tloop.a.b.c.f()\n",
            "if x:\n\tfrom loop.a.b import c\n\timport os\n\tc.f()\n",
        ),
        (
            "import os, loop.a.b.c\nloop.a.b.c.f()\n",
            "import os\nfrom loop.a.b import c\nc.f()\n",
        ),
        (
            "if x:\n\timport a.b  # Note.\n\ta.b.f()\n",
            "if x:\n\tfrom a import (  # Note.\n\t    b,\n\t)\n\tb.f()\n",
        ),
    ],
)
def test_dotted_import_fix_splits_each_statement_at_its_indent(
    tmp_path: Path,
    source: str,
    fixed: str,
) -> None:
    assert _fixed(tmp_path, DottedImportRule(), source) == fixed


@pytest.mark.parametrize(
    ("name", "underscored"),
    [("_x", True), ("_", False), ("__x__", False), ("__x", True), ("x", False)],
)
def test_is_underscored(name: str, *, underscored: bool) -> None:
    assert houselint_rules._is_underscored(name) is underscored


@pytest.mark.parametrize(
    ("decorator", "kernel"),
    [
        ("@triton.jit", True),
        ("@triton.jit()", True),
        ("@jit", True),
        ("@jit(x=1)", True),
        ("@triton.njit", False),
        ("@other", False),
        ("@x[0]", False),
    ],
)
def test_is_triton_kernel(decorator: str, *, kernel: bool) -> None:
    node = ast.parse(f"{decorator}\ndef f():\n    pass\n").body[0]
    assert isinstance(node, ast.FunctionDef)
    assert houselint_rules._is_triton_kernel(node) is kernel


@pytest.mark.parametrize(
    ("decorators", "name", "has"),
    [
        ("@a.override", "override", True),
        ("@override", "override", True),
        ("@overrides", "override", False),
        ("@other\n@override", "override", True),
        ("@other", "override", False),
    ],
)
def test_has_decorator_matches_the_last_segment(
    decorators: str,
    name: str,
    *,
    has: bool,
) -> None:
    node = ast.parse(f"{decorators}\ndef f():\n    pass\n").body[0]
    assert isinstance(node, ast.FunctionDef)
    assert houselint_rules._has_decorator(node, name) is has


@pytest.mark.parametrize(
    ("source", "escapes"),
    [
        ("def o():\n    def i():\n        pass\n    i()\n", False),
        ("def o():\n    def i():\n        pass\n    return i\n", True),
        ("def o():\n    @d\n    def i():\n        pass\n    i()\n", True),
        ("def o():\n    def i():\n        pass\n    f(i)\n", True),
        ("def o():\n    def i():\n        pass\n    i()\n    x.i\n", False),
        ("def o():\n    def i():\n        pass\n    j()\n", False),
    ],
)
def test_escapes_when_used_as_a_value(source: str, *, escapes: bool) -> None:
    outer = ast.parse(source).body[0]
    assert isinstance(outer, ast.FunctionDef)
    inner = outer.body[0]
    assert isinstance(inner, ast.FunctionDef)
    assert houselint_rules._escapes(inner, outer) is escapes


def test_direct_nested_defs_keep_walking_past_a_class() -> None:
    outer = ast.parse(
        "def o():\n    def a():\n        pass\n    class K:\n        pass\n",
    ).body[0]
    assert isinstance(outer, ast.FunctionDef)
    assert [d.name for d in houselint_rules._direct_nested_defs(outer)] == ["a"]


def test_direct_nested_defs_skip_classes_but_not_later_defs() -> None:
    outer = ast.parse(
        "def o():\n    class K:\n        def m(self):\n            pass\n"
        "    if x:\n        def a():\n            pass\n"
        "    def b():\n        def c():\n            pass\n",
    ).body[0]
    assert isinstance(outer, ast.FunctionDef)
    names = sorted(d.name for d in houselint_rules._direct_nested_defs(outer))
    assert names == ["a", "b"]


@pytest.mark.parametrize(
    ("source", "misplaced"),
    [
        ("def _a():\n    pass\ndef b():\n    pass\n", [("_a", "b")]),
        ("def b():\n    pass\ndef _a():\n    pass\n", []),
        ("def _a():\n    pass\ndef _b():\n    pass\n", []),
        ("def _a():\n    pass\nx = _a()\ndef b():\n    pass\n", []),
        (
            "def _a():\n    pass\ndef _c():\n    pass\nclass B:\n    pass\n",
            [("_a", "B"), ("_c", "B")],
        ),
        ("x = 1\ndef _a():\n    pass\ndef b():\n    pass\n", [("_a", "b")]),
        (
            "def b():\n    pass\ndef _a():\n    pass\ndef c():\n    pass\n",
            [("_a", "c")],
        ),
    ],
)
def test_misplaced_privates(source: str, misplaced: list[tuple[str, str]]) -> None:
    got = houselint_rules._misplaced_privates(ast.parse(source))
    assert [(node.name, public) for node, public in got] == misplaced


def test_mkdir_mode_private_chmod_needs_an_owner_bit(tmp_path: Path) -> None:
    body = "".join(
        f'Path("k").chmod({mode})\n'
        for mode in ("0o000", "0o100", "0o200", "0o400", "0o500", "0o700", "0o604")
    )
    assert _messages(tmp_path, MkdirModeRule(), body) == [
        (3, "chmod to a private mode on a non-userdirs path"),
        (4, "chmod to a private mode on a non-userdirs path"),
        (5, "chmod to a private mode on a non-userdirs path"),
        (6, "chmod to a private mode on a non-userdirs path"),
    ]


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["x = 1  \n"], ["x = 1  \n", "# vim: ft=python\n"]),
        (["a  X"], ["a  X\n", "# vim: ft=python\n"]),
    ],
)
def test_cli_trailer_strips_only_newlines(
    lines: list[str],
    expected: list[str],
) -> None:
    houselint_rules._fix_polyglot_trailer(lines)
    assert lines == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "# see:\ndef main():\n    pass\n",
            "# see:\ndef main() -> int:\n    pass\n    return 0\n",
        ),
        (
            "def main():  # noqa: D103\n    if x:\n        return\n",
            "def main():  # noqa: D103\n    if x:\n        return 0\n    return 0\n",
        ),
    ],
)
def test_fix_main_returns_annotates_only_the_signature(
    source: str,
    expected: str,
) -> None:
    tree = ast.parse(source)
    main = tree.body[-1]
    assert isinstance(main, ast.FunctionDef)
    lines = source.splitlines(keepends=True)
    houselint_rules._fix_main_returns(main, lines, noreturn=frozenset())
    assert "".join(lines) == expected


def test_module_side_effects_strips_only_underscores_from_a_class_head(
    tmp_path: Path,
) -> None:
    body = "X = Xwidget()\nY = _Widget()\n"
    assert _messages(tmp_path, ModuleSideEffectsRule(), body) == []


@pytest.mark.parametrize(
    ("expr", "matches"),
    [
        ("Path(__file__)", True),
        ("foo(__file__)", False),
        ("Path(__file__, x)", False),
        ("Path(__file__, a=1)", False),
        ("Path(x)", False),
        ("Path(f.__file__)", False),
    ],
)
def test_is_file_path_call(expr: str, *, matches: bool) -> None:
    node = ast.parse(expr, mode="eval").body
    assert houselint_rules._is_file_path_call(node) is matches


@pytest.mark.parametrize(
    ("source", "loads"),
    [
        ("x = a\ndef f(p=b) -> c:\n    d\n", {"a", "b", "c"}),
        (
            "x = a\nclass K(b):\n    y = c\n    def m(self):\n        d\n",
            {"a", "b", "c"},
        ),
    ],
)
def test_module_scope_loads_skip_only_function_bodies(
    source: str,
    loads: set[str],
) -> None:
    assert houselint_rules._module_scope_loads(ast.parse(source)) == loads


@pytest.mark.parametrize(
    ("annotation", "final"),
    [
        ("Final", True),
        ("Final[int]", True),
        ("typing.Final[int]", True),
        ("int", False),
        ("list[Final]", False),
    ],
)
def test_annotation_is_final(annotation: str, *, final: bool) -> None:
    node = ast.parse(annotation, mode="eval").body
    assert houselint_rules._annotation_is_final(node) is final


@pytest.mark.parametrize(
    ("base", "name"),
    [("a.B[int]", "B"), ("B", "B"), ("B[int]", "B"), ("f()", ""), ("f()[0]", "")],
)
def test_base_name_is_the_head_name(base: str, name: str) -> None:
    node = ast.parse(base, mode="eval").body
    assert houselint_rules._base_name(node) == name


@pytest.mark.parametrize(
    ("expr", "text"),
    [("a.b.c()", "a.b.c"), ("f", "f"), ("f()()", ""), ("x[0].y", ".y"), ("1", "")],
)
def test_call_text_spells_the_dotted_callee(expr: str, text: str) -> None:
    node = ast.parse(expr, mode="eval").body
    assert houselint_rules._call_text(node) == text


# -- hlr_g01_test ----------------------------------------------------------


def _footer_messages(tmp_path: Path, source: str) -> list[tuple[int, str]]:
    path = tmp_path / "sample_test.py"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    return [
        (finding.line, finding.message) for finding in PytestFooterRule().check(unit)
    ]


def _footer_fixed(
    tmp_path: Path,
    source: str,
    *,
    name: str = "sample_test.py",
) -> str | None:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    return PytestFooterRule().fix(unit)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "x = 1\nfrom codedoctor.lib.testing " + "import test_main\n",
            [
                (
                    2,
                    (
                        "footer imports test_main from the package facade; "
                        "use `from codedoctor.lib.testing.runner import test_main`"
                    ),
                ),
                (
                    2,
                    (
                        "test file lacks the standard footer "
                        "(STYLE.md Testing); rerun with --fix test-footer"
                    ),
                ),
            ],
        ),
        (
            "x = 1\n",
            [
                (
                    1,
                    (
                        "test file lacks the standard footer "
                        "(STYLE.md Testing); rerun with --fix test-footer"
                    ),
                ),
            ],
        ),
        *[
            (
                _FOOTER + tail,
                [
                    (
                        4,
                        (
                            "test file lacks the standard footer "
                            "(STYLE.md Testing); rerun with --fix test-footer"
                        ),
                    ),
                ],
            )
            for tail in ("   ", "XX")
        ],
    ],
)
def test_footer_findings_have_exact_lines_and_messages(
    tmp_path: Path,
    source: str,
    expected: list[tuple[int, str]],
) -> None:
    assert _footer_messages(tmp_path, source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x = 1\n", "x = 1\n\n\n" + _FOOTER),
        ("x = 1   \n", "x = 1   \n\n\n" + _FOOTER),
        ("x = 1XX", "x = 1XX\n\n\n" + _FOOTER),
        (
            'x = 1\nif __name__ == "__main__":\n    test_main(__file__)\ny = 2\n',
            "x = 1\ny = 2\n\n\n" + _FOOTER,
        ),
    ],
)
def test_footer_fix_returns_exact_source(
    tmp_path: Path,
    source: str,
    expected: str,
) -> None:
    assert _footer_fixed(tmp_path, source) == expected


def test_footer_fix_skips_clean_test_and_non_test_files(tmp_path: Path) -> None:
    assert _footer_fixed(tmp_path, _FOOTER) is None
    assert _footer_fixed(tmp_path, "x = 1\n", name="sample.py") is None


# -- hlr_g02_test ----------------------------------------------------------


def test_single_use_helper_scans_past_non_function_statements() -> None:
    unit = Unit(
        path=Path("mod.py"),
        source="VALUE = 1\ndef _helper(x):\n    return x + 1\ndef run():\n    return _helper(2)\n",
        kind="python",
    )
    assert [
        helper.node.name for helper in houselint_rules._single_use_helpers(unit)
    ] == [
        "_helper",
    ]


def test_single_use_helper_name_filter_and_signature_exclusions() -> None:
    source = (
        "def deco(fn):\n    return fn\n"
        "@deco\ndef _decorated(x):\n    return x\n"
        "def _varargs(*xs):\n    return xs\n"
        "def _kwargs(**xs):\n    return xs\n"
        "def _single(x):\n    return x\n"
        "def run():\n"
        "    _decorated(1)\n    _varargs(1)\n    _kwargs(x=1)\n"
        "    return _single(1)\n"
    )
    ast.parse(source)
    unit = Unit(path=Path("mod.py"), source=source, kind="python")
    assert [
        helper.node.name for helper in houselint_rules._single_use_helpers(unit)
    ] == [
        "_single",
    ]


def test_single_call_returns_none_when_name_is_absent() -> None:
    unit = Unit(path=Path("mod.py"), source="def run():\n    return 1\n", kind="python")
    assert houselint_rules._single_call(unit, "_missing") is None


def test_single_call_rejects_name_not_used_as_direct_callee() -> None:
    unit = Unit(
        path=Path("mod.py"),
        source="def _helper():\n    return 1\nvalue = _helper\n",
        kind="python",
    )
    assert houselint_rules._single_call(unit, "_helper") is None


def test_single_use_helper_fix_skips_bad_site_and_continues(tmp_path: Path) -> None:
    source = (
        "def f(a, items):\n    _unsafe(*items)\n    return _safe(a)\n\n\n"
        "def _unsafe(x):\n    return x + 1\n\n\n"
        "def _safe(x):\n    return x * 2\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    unit = Unit(path=path, source=source, kind="python")
    fixed = houselint_rules.SingleUseHelperRule().fix(unit)
    assert (
        fixed
        == "def f(a, items):\n    _unsafe(*items)\n    return a * 2\n\n\ndef _unsafe(x):\n    return x + 1\n"
    )


def test_single_use_helper_fix_uses_exact_output_and_preserves_trailing_newline(
    tmp_path: Path,
) -> None:
    source = (
        "def f(x):\n    return _twice(x)\n\n\ndef _twice(y):\n    return y * 2\n\n\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    unit = Unit(path=path, source=source, kind="python")
    assert houselint_rules.SingleUseHelperRule().fix(unit) == (
        "def f(x):\n    return x * 2\n"
    )


def test_single_use_helper_skips_each_ineligible_signature_and_body(
    tmp_path: Path,
) -> None:
    source = (
        "from typing import TypeGuard, TypeIs\n"
        "def __dunder__(x):\n    return x\n"
        "def _guard(x) -> TypeGuard[int]:\n    return True\n"
        "def _type(x) -> TypeIs[int]:\n    return True\n"
        "def _vararg(*xs):\n    return xs\n"
        "def _kwarg(**xs):\n    return xs\n"
        "def _multiple(x):\n    x += 1\n    return x\n"
        "def _after(x):\n    return x + 1\n"
        "def run(x):\n"
        "    __dunder__(x)\n    _guard(x)\n    _type(x)\n    _vararg(x)\n"
        "    _kwarg(x=x)\n    _multiple(x)\n    return _after(x)\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    helpers = houselint_rules._single_use_helpers(
        Unit(path=path, source=source, kind="python"),
    )
    assert [helper.node.name for helper in helpers] == ["_after"]


def test_single_use_helper_fix_keeps_later_independent_edit(tmp_path: Path) -> None:
    source = (
        "def f(x):\n    return _twice(x)\n\n\n"
        "def _twice(y):\n    return y * 2\n\n\n"
        "def g(x):\n    return _plus(x)\n\n\n"
        "def _plus(y):\n    return y + 1\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    unit = Unit(path=path, source=source, kind="python")
    assert houselint_rules.SingleUseHelperRule().fix(unit) == (
        "def f(x):\n    return x * 2\n\n\ndef g(x):\n    return x + 1\n"
    )


def test_single_call_ignores_other_names_before_requested_name() -> None:
    unit = Unit(
        path=Path("mod.py"),
        source="def _first():\n    return 1\ndef run():\n    return _second()\n",
        kind="python",
    )
    call = houselint_rules._single_call(unit, "_second")
    assert call is not None
    assert ast.unparse(call) == "_second()"


def test_single_use_helper_prefilter_needs_definition_marker() -> None:
    unit = Unit(
        path=Path("mod.py"),
        source="def  _helper(x):\n    return x + 1\ndef run():\n    return _helper(2)\n",
        kind="python",
    )
    assert houselint_rules._single_use_helpers(unit) == []


def test_single_use_helper_rejects_empty_body_after_docstring() -> None:
    source = (
        'def _empty():\n    """Only a docstring."""\n'
        "def _value():\n    return 1\n"
        "def run():\n    _empty()\n    return _value()\n"
    )
    unit = Unit(path=Path("mod.py"), source=source, kind="python")
    assert [
        helper.node.name for helper in houselint_rules._single_use_helpers(unit)
    ] == [
        "_value",
    ]


def test_single_use_helper_fix_preserves_trailing_spaces(tmp_path: Path) -> None:
    source = "def f():\n    return _one()\n\n\ndef _one():\n    return 1\n# tail   \n"
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    assert (
        houselint_rules.SingleUseHelperRule().fix(
            Unit(path=path, source=source, kind="python"),
        )
        == "def f():\n    return 1\n\n\n# tail   \n"
    )


def test_single_use_helper_fix_preserves_trailing_marker(tmp_path: Path) -> None:
    source = "def f():\n    return _one()\n\n\ndef _one():\n    return 1\n# X\n"
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    assert (
        houselint_rules.SingleUseHelperRule().fix(
            Unit(path=path, source=source, kind="python"),
        )
        == "def f():\n    return 1\n\n\n# X\n"
    )


def test_single_use_helper_fix_allows_abutting_edits(tmp_path: Path) -> None:
    source = "def _b():\n    return 2\ndef _a():\n    return 1\n_b()\nprint(_a())\n"
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    assert (
        houselint_rules.SingleUseHelperRule().fix(
            Unit(path=path, source=source, kind="python"),
        )
        == "2\nprint(1)\n"
    )


def test_single_use_helper_fix_continues_after_overlapping_edits(
    tmp_path: Path,
) -> None:
    source = (
        "def run(x):\n    return _outer(x)\n\n\n"
        "def _outer(x):\n    return _inner(x) - 1\n\n\n"
        "def _inner(y):\n    return y * 2\n\n\n"
        "def use():\n    return _last(1)\n\n\n"
        "def _last(z):\n    return z + 1\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    assert houselint_rules.SingleUseHelperRule().fix(
        Unit(path=path, source=source, kind="python"),
    ) == (
        "def run(x):\n    return _inner(x) - 1\n\n\n"
        "def _inner(y):\n    return y * 2\n\n\n"
        "def use():\n    return 1 + 1\n"
    )


# -- hlr_g03_test ----------------------------------------------------------


def test_private_docstring_splits_at_the_first_sentence() -> None:
    text = "First sentence is deliberately long. Second sentence. Third sentence."
    assert houselint_rules._split_private_docstring(
        text,
        summary_width=20,
        comment_width=88,
    ) == (
        "First sentence is deliberately long",
        ["Second sentence. Third sentence."],
    )


def test_private_docstring_skips_sections_and_keeps_following_prose() -> None:
    text = (
        "Summary.\n\n"
        "Args:\n"
        "  x: ignored\n"
        "\n"
        "  continuation ignored\n\n"
        "Rationale survives.\n"
        "\n"
        "  Indented prose survives after rationale.\n"
    )
    assert houselint_rules._split_private_docstring(
        text,
        summary_width=88,
        comment_width=88,
    ) == (
        "Summary.",
        [
            "Rationale survives.",
            "",
            "Indented prose survives after rationale.",
        ],
    )


def test_private_docstring_section_detection_uses_the_whole_head_line() -> None:
    text = "Summary.\n\nArgs: explanatory prose\n\nRationale survives."
    assert houselint_rules._split_private_docstring(
        text,
        summary_width=88,
        comment_width=88,
    ) == ("Summary.", ["Args: explanatory prose", "", "Rationale survives."])


def test_private_docstring_section_detection_uses_the_first_line() -> None:
    text = "Summary.\n\nOpening prose\nArgs:\n  x: ignored\n\nRationale survives."
    assert houselint_rules._split_private_docstring(
        text,
        summary_width=88,
        comment_width=88,
    ) == (
        "Summary.",
        ["Opening prose Args: x: ignored", "", "Rationale survives."],
    )


def test_private_oneliner_wraps_summary_at_its_actual_column(tmp_path: Path) -> None:
    summary = "A sentence that has enough words to overflow the summary width."
    source = f'def _f():\n    """{summary}\n\n    """\n'
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert PrivateOnelinerRule().fix(unit) == (
        'def _f():\n    """A sentence that has enough words to overflow the summary width."""\n'
    )


def test_private_docstring_keeps_indented_prose_before_a_later_section() -> None:
    text = "Summary.\n\n    Rationale.\n\nLater prose."
    assert houselint_rules._split_private_docstring(
        text,
        summary_width=88,
        comment_width=88,
    ) == ("Summary.", ["Rationale.", "", "Later prose."])


def test_private_docstring_section_detection_ignores_later_lines() -> None:
    text = "Summary.\n\nArgs:\n  x: ignored\n  y: ignored\n\nRationale."
    assert houselint_rules._split_private_docstring(
        text,
        summary_width=88,
        comment_width=88,
    ) == ("Summary.", ["Rationale."])


def test_private_oneliner_comment_width_boundaries(tmp_path: Path) -> None:
    rationale_86 = "a " * 42 + "ab"
    rationale_87 = "a " * 42 + "abc"
    source = (
        'def _a():\n    """Summary.\n\n'
        f'    {rationale_86}\n    """\n'
        'def _b():\n    """Summary.\n\n'
        f'    {rationale_87}\n    """\n'
    )
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert PrivateOnelinerRule().fix(unit) == (
        f"# {rationale_86}\n"
        'def _a():\n    """Summary."""\n'
        f"# {'a ' * 41}a\n# abc\n"
        'def _b():\n    """Summary."""\n'
    )


def test_private_oneliner_summary_width_tracks_docstring_column(tmp_path: Path) -> None:
    summary = "A" * 68 + ". Tail."
    source = f'def _outer():\n    def _f():\n        """{summary}\n\n        """\n'
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert PrivateOnelinerRule().fix(unit) == (
        f'def _outer():\n    # Tail.\n    def _f():\n        """{"A" * 68}"""\n'
    )


def test_private_oneliner_uses_docstring_line_indentation(tmp_path: Path) -> None:
    source = 'def _f():\n    """Summary.\n\n        Rationale.\n  """\n'
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert PrivateOnelinerRule().fix(unit) == (
        '# Rationale.\ndef _f():\n    """Summary."""\n'
    )


# -- hlr_g04_test ----------------------------------------------------------


def test_assert_check_message_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text("assert ready\n", encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert [
        (finding.line, finding.message) for finding in AssertRule().check(unit)
    ] == [
        (
            1,
            (
                "assert used as a runtime check; `-O` drops it. Raise "
                "ValueError instead (rerun with --fix assert), or narrow "
                "with isinstance"
            ),
        ),
    ]


def test_assert_fix_continues_after_a_narrowing_assert(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    source = (
        "assert isinstance(value, int)\n"
        "assert ready\n"
        "assert isinstance(other, str)\n"
        "assert done\n"
    )
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert AssertRule().fix(unit) == (
        "assert isinstance(value, int)\n"
        "if not ready:\n"
        '    raise ValueError("Expected ready.")\n'
        "assert isinstance(other, str)\n"
        "if not done:\n"
        '    raise ValueError("Expected done.")\n'
    )


def test_assert_fix_skips_unrewritable_sites_and_keeps_later_sites(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mod.py"
    source = (
        "header = 1\n"
        "assert ready; keep = 2\n"
        "assert (\n"
        "    ready  # explanatory\n"
        "    and valid\n"
        ")\n"
        "assert later\n"
    )
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert AssertRule().fix(unit) == (
        "header = 1\n"
        "assert ready; keep = 2\n"
        "assert (\n"
        "    ready  # explanatory\n"
        "    and valid\n"
        ")\n"
        "if not later:\n"
        '    raise ValueError("Expected later.")\n'
    )


def test_assert_fix_uses_the_original_text_for_single_line_tests(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mod.py"
    source = "header = 1\nassert (left + right)\n"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert AssertRule().fix(unit) == (
        "header = 1\n"
        'if not (left + right):\n    raise ValueError("Expected left + right.")\n'
    )


def test_assert_fix_quotes_expected_messages_with_embedded_quotes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text('assert value == "ready"\n', encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert AssertRule().fix(unit) == (
        "if value != 'ready':\n    raise ValueError('Expected value == \"ready\".')\n"
    )


# -- hlr_g05_test ----------------------------------------------------------


def _annotation_messages(
    tmp_path: Path,
    body: str,
    rule: AnnotationImportRule,
) -> list[tuple[int, str]]:
    path = tmp_path / "mod.py"
    path.write_text(body, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    return [(finding.line, finding.message) for finding in rule.check(unit)]


def test_annotation_import_keeps_the_root_for_runtime_imports(tmp_path: Path) -> None:
    body = (
        "import torch.nn as nn\nfrom torch import Tensor\n\n"
        "def f(value: Tensor) -> None:\n    nn.Module\n"
    )
    assert (
        _annotation_messages(
            tmp_path,
            body,
            AnnotationImportRule(threshold_ms=50, costs_ms={"torch": 1050}),
        )
        == []
    )


def test_annotation_import_uses_the_module_root_for_costs(tmp_path: Path) -> None:
    body = (
        "from torch.nn import Module\n\ndef f(value: Module) -> None:\n    del value\n"
    )
    assert (
        _annotation_messages(
            tmp_path,
            body,
            AnnotationImportRule(
                threshold_ms=50,
                costs_ms={"torch": 0, "torch.nn": 100},
            ),
        )
        == []
    )


def test_annotation_import_missing_cost_uses_zero(tmp_path: Path) -> None:
    body = "from unknown_module import Type\n\ndef f(value: Type) -> None:\n    del value\n"
    assert (
        _annotation_messages(
            tmp_path,
            body,
            AnnotationImportRule(threshold_ms=1, costs_ms={}),
        )
        == []
    )


def test_annotation_import_reports_cost_equal_to_threshold(tmp_path: Path) -> None:
    body = "from torch import Tensor\n\ndef f(value: Tensor) -> None:\n    del value\n"
    assert _annotation_messages(
        tmp_path,
        body,
        AnnotationImportRule(threshold_ms=50, costs_ms={"torch": 50}),
    ) == [
        (
            1,
            (
                "'torch.Tensor' costs ~50 ms to import and is used only in "
                "1 annotation(s); import it under `if TYPE_CHECKING:` "
                "(STYLE.md Code Organization)"
            ),
        ),
    ]


def test_annotation_import_continues_after_cheap_import(tmp_path: Path) -> None:
    body = (
        "import json\nfrom torch import Tensor\n\n"
        "def f(value: Tensor) -> None:\n    del value\n"
    )
    assert _annotation_messages(
        tmp_path,
        body,
        AnnotationImportRule(threshold_ms=50, costs_ms={"torch": 1050}),
    ) == [
        (
            2,
            (
                "'torch.Tensor' costs ~1050 ms to import and is used only in "
                "1 annotation(s); import it under `if TYPE_CHECKING:` "
                "(STYLE.md Code Organization)"
            ),
        ),
    ]


def test_annotation_import_continues_after_unused_import(tmp_path: Path) -> None:
    body = (
        "from unknown_module import Unused\nfrom torch import Tensor\n\n"
        "def f(value: Tensor) -> None:\n    del value\n"
    )
    assert _annotation_messages(
        tmp_path,
        body,
        AnnotationImportRule(
            threshold_ms=50,
            costs_ms={"unknown_module": 100, "torch": 1050},
        ),
    ) == [
        (
            2,
            (
                "'torch.Tensor' costs ~1050 ms to import and is used only in "
                "1 annotation(s); import it under `if TYPE_CHECKING:` "
                "(STYLE.md Code Organization)"
            ),
        ),
    ]


def test_annotation_import_skips_an_unparseable_import_file(tmp_path: Path) -> None:
    body = "import \\\n"
    path = tmp_path / "mod.py"
    path.write_text(body, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert unit.tree is None
    assert AnnotationImportRule().check(unit) == ()


def test_name_usage_separates_annotation_and_runtime_contexts() -> None:
    tree = ast.parse(
        "value: A = make(B)\n"
        "def f(arg: C = default(D)) -> E:\n"
        "    return cast('F', G)\n"
        "if TYPE_CHECKING:\n"
        "    import H\n"
        "else:\n"
        "    I\n",
    )
    usage = houselint_rules._name_usage(tree)
    assert {
        name: (counts.annotation, counts.runtime) for name, counts in usage.items()
    } == {
        "A": (1, 0),
        "make": (0, 1),
        "B": (0, 1),
        "value": (0, 1),
        "C": (1, 0),
        "default": (0, 1),
        "D": (0, 1),
        "E": (1, 0),
        "F": (1, 0),
        "G": (0, 1),
        "I": (0, 1),
    }


def test_usage_walk_adds_repeated_name_counts() -> None:
    tree = ast.parse("x + x")
    walker = houselint_rules._UsageWalk()
    walker.visit(tree)
    assert walker.runtime == {"x": 2}
    assert walker.annotation == {}


def test_type_checking_else_preserves_annotation_context() -> None:
    tree = ast.parse("if TYPE_CHECKING:\n    pass\nelse:\n    fallback\n")
    walker = houselint_rules._UsageWalk()
    walker.visit(tree, in_annotation=True)
    assert walker.annotation == {"fallback": 1}
    assert walker.runtime == {}


def test_annotation_import_requires_an_annotation_use(tmp_path: Path) -> None:
    body = "from torch import Tensor\n\ndef f() -> None:\n    del Tensor\n"
    assert (
        _annotation_messages(
            tmp_path,
            body,
            AnnotationImportRule(threshold_ms=50, costs_ms={"torch": 1050}),
        )
        == []
    )


# -- hlr_g06_test ----------------------------------------------------------


def test_runtime_imports_distinguishes_roots_aliases_and_wildcards() -> None:
    tree = ast.parse(
        "import typing_extensions.foo.bar\n"
        "import foo.typing_extensions\n"
        "import package.submodule\n"
        "from package.typing_extensions import value\n"
        "from package.submodule import *\n",
    )
    assert houselint_rules._runtime_imports(tree) == {
        "foo": (2, "foo.typing_extensions"),
        "package": (3, "package.submodule"),
        "value": (4, "package.typing_extensions.value"),
    }


def test_runtime_imports_skips_only_annotation_aliases() -> None:
    tree = ast.parse("import typing_extensions, package\n")
    assert houselint_rules._runtime_imports(tree) == {"package": (1, "package")}


def test_runtime_imports_skips_annotation_module_roots() -> None:
    tree = ast.parse(
        "from typing_extensions.foo.bar import annotation\n"
        "from package.typing_extensions import value\n",
    )
    assert houselint_rules._runtime_imports(tree) == {
        "value": (2, "package.typing_extensions.value"),
    }


def test_runtime_imports_continues_after_wildcard_alias() -> None:
    tree = ast.parse("from package import *\n")
    node = tree.body[0]
    assert isinstance(node, ast.ImportFrom)
    node.names.append(ast.alias(name="value"))
    assert houselint_rules._runtime_imports(tree) == {"value": (1, "package.value")}


def test_import_alias_claims_keep_each_module_path_segment(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text("from loop.pkg import item as renamed\n", encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    tree = unit.tree
    assert tree is not None
    node = tree.body[0]
    assert isinstance(node, ast.ImportFrom)
    claims = ImportAliasRule()._claims([node], set())
    assert claims == {node.names[0]: ("loop", "pkg", "item")}


def test_import_alias_claims_empty_module_without_inventing_root() -> None:
    node = ast.ImportFrom(
        module=None,
        names=[ast.alias(name="item", asname="renamed")],
        level=1,
    )
    rule = ImportAliasRule(first_party=("XXXX",))
    assert rule._claims([node], set()) == {node.names[0]: ("", "item")}


def test_import_alias_first_party_does_not_invent_missing_module() -> None:
    node = ast.ImportFrom(module=None, names=[ast.alias(name="item")], level=0)
    rule = ImportAliasRule(first_party=("XXXX",))
    assert not rule._is_first_party(node, node.names[0])


def test_import_alias_first_party_uses_first_module_segment() -> None:
    node = ast.ImportFrom(
        module="other.loop.module",
        names=[ast.alias(name="item")],
        level=0,
    )
    rule = ImportAliasRule(first_party=("other.loop",))
    assert not rule._is_first_party(node, node.names[0])


def test_import_alias_uses_top_level_first_party_package(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text("import loop.subpackage.module as local\n", encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    rule = ImportAliasRule()
    tree = unit.tree
    assert tree is not None
    node = tree.body[0]
    assert isinstance(node, ast.Import)
    assert rule._is_first_party(node, node.names[0])


@pytest.mark.parametrize(
    "reference",
    [
        "loop.pkg.submodule.answer",
        "(loop).pkg.submodule.answer",
        "(loop.pkg).submodule.answer",
    ],
)
def test_dotted_import_fix_rewrites_the_complete_dotted_reference(
    tmp_path: Path,
    reference: str,
) -> None:
    """The whole ``loop.pkg.submodule`` prefix goes, parentheses and all."""
    body = f"import loop.pkg.submodule\nvalue = {reference}\n"
    assert _fixed(tmp_path, DottedImportRule(), body) == (
        "from loop.pkg import submodule\nvalue = submodule.answer\n"
    )


# -- hlr_g07_test ----------------------------------------------------------


def _cost_messages(
    tmp_path: Path,
    rule: LazyImportCostRule,
    source: str,
) -> list[tuple[int, str]]:
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    return [(finding.line, finding.message) for finding in rule.check(unit)]


def test_regroup_continues_after_short_and_nondecimal_tokens() -> None:
    assert (
        houselint_rules._regroup(
            "a = 1234 + 12345 + 0x12345\nb = 67890\n",
            tuple(_tokenize("a = 1234 + 12345 + 0x12345\nb = 67890\n")),
            5,
        )
        == "a = 1234 + 12_345 + 0x12345\nb = 67_890\n"
    )


def test_regroup_does_not_rewrite_underscored_numbers() -> None:
    assert (
        houselint_rules._regroup(
            "a = 12_345\nb = 67890\n",
            tuple(_tokenize("a = 12_345\nb = 67890\n")),
            5,
        )
        == "a = 12_345\nb = 67_890\n"
    )


def test_lazy_import_cost_fallback_cost_is_zero(tmp_path: Path) -> None:
    source = "import unknown\n\ndef f(): ...\n\ndef g(): ...\n"
    rule = LazyImportCostRule(threshold_ms=1, costs_ms={})
    assert _cost_messages(tmp_path, rule, source) == []


def test_lazy_import_cost_includes_threshold_boundary(tmp_path: Path) -> None:
    source = "import unknown\n\ndef f(): ...\n\ndef g(): ...\n"
    rule = LazyImportCostRule(threshold_ms=10, costs_ms={"unknown": 10})
    assert _cost_messages(tmp_path, rule, source) == [
        (
            1,
            (
                "'unknown' costs ~10 ms to import and only 0 of 2 defs reach it; "
                "bind it with `lazy_import` (STYLE.md Code Organization)"
            ),
        ),
    ]


def test_lazy_import_cost_reports_when_use_fraction_is_exactly_boundary(
    tmp_path: Path,
) -> None:
    source = "import torch\n\ndef a():\n    return torch\n\ndef b(): ...\n"
    assert _cost_messages(
        tmp_path,
        LazyImportCostRule(
            threshold_ms=10,
            costs_ms={"torch": 10},
            max_use_fraction=0.5,
        ),
        source,
    ) == [
        (
            1,
            (
                "'torch' costs ~10 ms to import and only 1 of 2 defs reach it; "
                "bind it with `lazy_import` (STYLE.md Code Organization)"
            ),
        ),
    ]


def test_lazy_import_cost_continues_after_a_main_path_import(tmp_path: Path) -> None:
    source = (
        "import torch\nimport unknown\n\n"
        "def a(): return torch\n\ndef b(): return torch\n"
        "def c(): ...\n\ndef d(): ...\n"
    )
    assert [
        line
        for line, _ in _cost_messages(
            tmp_path,
            LazyImportCostRule(threshold_ms=10, costs_ms={"torch": 10, "unknown": 10}),
            source,
        )
    ] == [2]


def test_lazy_import_cost_reports_two_defs_at_configured_boundary(
    tmp_path: Path,
) -> None:
    source = "import unknown\n\ndef a(): ...\n\ndef b(): ...\n"
    assert _cost_messages(
        tmp_path,
        LazyImportCostRule(
            threshold_ms=10,
            costs_ms={"unknown": 10},
            max_use_fraction=0.5,
        ),
        source,
    )


def test_lazy_import_cost_suppression_names_rule(tmp_path: Path) -> None:
    source = (
        "import unknown  # house-ignore[lazy-import-cost] -- Kept eager.\n\n"
        "def a(): ...\n\ndef b(): ...\n"
    )
    assert (
        _cost_messages(
            tmp_path,
            LazyImportCostRule(threshold_ms=10, costs_ms={"unknown": 10}),
            source,
        )
        == []
    )


def test_lazy_import_cost_needs_two_defs(tmp_path: Path) -> None:
    source = "import unknown\ndef only(): ...\n"
    rule = LazyImportCostRule(threshold_ms=10, costs_ms={"unknown": 10})
    assert _cost_messages(tmp_path, rule, source) == []


# -- hlr_g08_test ----------------------------------------------------------


@pytest.mark.parametrize(
    ("rule", "source", "expected"),
    [
        (
            CommentSentenceRule(),
            "x = 1  # plate is here\n",
            "x = 1  # Plate is here.\n",
        ),
        (
            CommentSentenceRule(),
            "# starts X\n",
            "# Starts X.\n",
        ),
        (
            CommentSentenceRule(),
            "plate = 1  # plate is here\n",
            "plate = 1  # `plate` is here.\n",
        ),
        (
            LoggingFstringRule(),
            'logger.info(f"value {x}\\\\path")\n',
            'logger.info("value %s\\\\path", x)\n',
        ),
        (
            LoggingFstringRule(),
            "logger.info(f'say {x} \"ok\"')\n",
            'logger.info("say %s \\"ok\\"", x)\n',
        ),
    ],
)
def test_fixes_preserve_exact_text(
    tmp_path: Path,
    rule: CommentSentenceRule | LoggingFstringRule,
    source: str,
    expected: str,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert rule.fix(unit) == expected


def test_logging_fstring_skips_a_call_with_extra_args_and_continues(
    tmp_path: Path,
) -> None:
    source = 'logger.info(f"first {x}", other)\nlogger.info(f"second {y}")\n'
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert LoggingFstringRule().fix(unit) == (
        'logger.info(f"first {x}", other)\nlogger.info("second %s", y)\n'
    )


def test_logging_fstring_edits_multiple_calls_from_right_to_left(
    tmp_path: Path,
) -> None:
    source = 'logger.info(f"first {x}")\nlogger.info(f"second {y}")\n'
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert LoggingFstringRule().fix(unit) == (
        'logger.info("first %s", x)\nlogger.info("second %s", y)\n'
    )


# -- hlr_g09_test ----------------------------------------------------------


def test_literal_expression_requires_both_binary_operands() -> None:
    literals = ["1 + 2", "1 + NAME", "NAME + 1", "NAME + OTHER"]
    assert [
        _is_literal_value(ast.parse(expr, mode="eval").body) for expr in literals
    ] == [True, False, False, False]


def test_literal_collections_must_be_nonempty_and_all_literal() -> None:

    assert not _is_literal_value(ast.parse("[]", mode="eval").body)
    assert not _is_literal_value(ast.parse("[1, NAME]", mode="eval").body)
    assert _is_literal_value(ast.parse("[1, 2]", mode="eval").body)


def test_target_name_rejects_non_name_annotated_targets() -> None:
    name = ast.parse("x: int = 1").body[0]
    attribute = ast.parse("obj.x: int = 1").body[0]
    assert isinstance(name, ast.AnnAssign)
    assert isinstance(attribute, ast.AnnAssign)
    assert _target_name(name) == "x"
    assert _target_name(attribute) is None


def test_pytest_naming_ignores_non_python_and_other_test_names(tmp_path: Path) -> None:
    rule = PytestNamingRule()
    for path in (tmp_path / "test_case.txt", tmp_path / "module.py"):
        assert rule.check(Unit(path=path, source="x = 1\n", kind="python")) == ()


def test_pytest_naming_reports_exact_name_and_finding_line(tmp_path: Path) -> None:
    path = tmp_path / "test_widget.py"
    finding = PytestNamingRule().check(Unit(path=path, source="x = 1\n", kind="python"))
    assert [(item.line, item.message) for item in finding] == [
        (1, "test file named test_*.py; rename to widget_test.py (STYLE.md Testing)"),
    ]


def test_pytest_naming_exempts_tools_tests_path(tmp_path: Path) -> None:
    path = tmp_path / "tools" / "tests" / "test_widget.py"
    assert (
        PytestNamingRule().check(Unit(path=path, source="x = 1\n", kind="python")) == ()
    )


def test_pytest_naming_does_not_exempt_other_tests_under_tools(tmp_path: Path) -> None:
    path = tmp_path / "tools" / "other" / "test_widget.py"
    finding = PytestNamingRule().check(Unit(path=path, source="x = 1\n", kind="python"))
    assert [(item.line, item.message) for item in finding] == [
        (1, "test file named test_*.py; rename to widget_test.py (STYLE.md Testing)"),
    ]


# -- hlr_g10_test ----------------------------------------------------------


def test_comment_sentence_reports_each_block_after_exempt_and_marker_blocks(
    tmp_path: Path,
) -> None:
    source = (
        "# noqa: E501\n"
        "value = 1\n"
        "# FULL_REPLACE\n"
        "value = 2\n"
        "# lowercase start\n"
        "value = 3\n"
        "# Another unpunctuated block\n"
        "# continuation\n"
    )
    assert _messages(tmp_path, CommentSentenceRule(), source) == [
        (
            5,
            "comment starts lowercase and lacks terminal punctuation (STYLE.md Comments)",
        ),
        (7, "comment lacks terminal punctuation (STYLE.md Comments)"),
    ]


def test_comment_sentence_exact_combined_message(tmp_path: Path) -> None:
    assert _messages(tmp_path, CommentSentenceRule(), "# lower start\n") == [
        (
            1,
            "comment starts lowercase and lacks terminal punctuation (STYLE.md Comments)",
        ),
    ]


def test_emdash_reports_exact_message_for_every_occurrence(tmp_path: Path) -> None:
    dash = chr(0x2014)
    assert _messages(
        tmp_path,
        EmDashRule(),
        f"a = 1  # x {dash} y\nb = '{dash}'\n",
    ) == [
        (1, "em dash (U+2014); write `--` (rerun with --fix emdash)"),
        (2, "em dash (U+2014); write `--` (rerun with --fix emdash)"),
    ]


def test_getlogger_reports_each_invalid_binding_and_skips_valid_calls(
    tmp_path: Path,
) -> None:
    source = (
        "good = logging.getLogger(__name__)\n"
        "bad = logging.getLogger('first')\n"
        "also_bad: object = getLogger()\n"
        "later = logging.getLogger('last')\n"
        "def local():\n    return logging.getLogger('local')\n"
    )
    message = "module logger is not getLogger(__name__); loggers are per-module"
    assert _messages(tmp_path, GetLoggerNameRule(), source) == [
        (2, message),
        (3, message),
        (4, message),
    ]


# -- hlr_g11_test ----------------------------------------------------------


def test_args_section_continues_after_skipped_defs(tmp_path: Path) -> None:
    source = (
        "def undocumented(x): ...\n"
        "@value.setter\n"
        "def inherited(x):\n"
        '    """Do.\n\n    More prose.\n    """\n'
        "def short(x):\n"
        '    """Do."""\n'
        "    return x\n"
        "def documented_later(x):\n"
        '    """Do.\n\n    More prose.\n    """\n'
    )
    assert _messages(tmp_path, ArgsSectionRule(), source) == [
        (11, "documented_later docstring lacks Args:"),
    ]


def test_args_section_includes_each_variadic_parameter(tmp_path: Path) -> None:
    source = (
        'def f(*args):\n    """Do.\n\n    More prose.\n    """\n'
        'def g(**kwargs):\n    """Do.\n\n    More prose.\n    """\n'
        'def h(*args, **kwargs):\n    """Do.\n\n    More prose.\n    """\n'
    )
    assert _messages(tmp_path, ArgsSectionRule(), source) == [
        (1, "f docstring lacks Args:"),
        (6, "g docstring lacks Args:"),
        (11, "h docstring lacks Args:"),
    ]


def test_args_section_ignores_implicit_parameters(tmp_path: Path) -> None:
    source = (
        "class C:\n"
        "    def f(self):\n"
        '        """Do.\n\n        More prose.\n        """\n'
        "    @classmethod\n"
        "    def g(cls):\n"
        '        """Do.\n\n        More prose.\n        """\n'
    )
    assert _messages(tmp_path, ArgsSectionRule(), source) == []


def test_args_section_recognizes_indented_args_section(tmp_path: Path) -> None:
    source = 'def f(value):\n    """Do.\n\n    Args:\n      value: Input.\n    """\n'
    assert _messages(tmp_path, ArgsSectionRule(), source) == []


def test_args_section_excludes_only_self_and_cls(tmp_path: Path) -> None:
    source = (
        "class C:\n"
        "    def f(self, x):\n"
        '        """Do.\n\n        More prose.\n        """\n'
        "    @classmethod\n"
        "    def g(cls, y):\n"
        '        """Do.\n\n        More prose.\n        """\n'
    )
    assert _messages(tmp_path, ArgsSectionRule(), source) == [
        (2, "f docstring lacks Args:"),
        (8, "g docstring lacks Args:"),
    ]


def test_returns_named_continues_after_defs_without_docstrings(tmp_path: Path) -> None:
    source = (
        "def undocumented(): ...\n"
        'def later():\n    """Do.\n\n    Returns:\n      unnamed result\n\n    """\n'
        "    return 1\n"
    )
    assert _messages(tmp_path, ReturnsNamedRule(), source) == [
        (2, "later: Returns: entry 'unnamed result' is not `name: description`"),
    ]


def test_returns_named_uses_cleaned_docstring_indentation(tmp_path: Path) -> None:
    source = (
        "def f():\n"
        '    """Do.\n\n'
        "        Returns:\n"
        "          unnamed result\n\n"
        '        """\n'
        "    return 1\n"
    )
    assert _messages(tmp_path, ReturnsNamedRule(), source) == [
        (1, "f: Returns: entry 'unnamed result' is not `name: description`"),
    ]


# -- hlr_g12_test ----------------------------------------------------------


def _unit(tmp_path: Path, source: str, *, name: str = "mod.py") -> Unit:
    """Build a Unit from source saved below ``tmp_path``."""
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    return unit


def test_sections_scans_past_non_section_lines() -> None:
    assert list(_sections("noise\nArgs:\n  x: value\n")) == [
        ("Args", ["x: value"]),
    ]


def test_sections_does_not_skip_a_section_after_a_noise_line() -> None:
    assert list(_sections("noise\nReturns:\n  value: result\n")) == [
        ("Returns", ["value: result"]),
    ]


def test_sections_stops_when_the_next_entry_is_unindented() -> None:
    assert list(_sections("Args:\n  x: value\nunindented\n")) == [
        ("Args", ["x: value"]),
    ]


def test_sections_stops_at_blank_line_before_next_section() -> None:
    assert list(_sections("Args:\n  x: value\n\nReturns:\n  value: result\n")) == [
        ("Args", ["x: value"]),
        ("Returns", ["value: result"]),
    ]


def test_sections_keeps_non_continuation_indent_as_a_new_entry() -> None:
    assert list(_sections("Args:\n  x: value\n  continued\n")) == [
        ("Args", ["x: value", "continued"]),
    ]


def test_sections_only_joins_four_space_continuations() -> None:
    assert list(_sections("Args:\n  x: value\n    details\n  y: other\n")) == [
        ("Args", ["x: value details", "y: other"]),
    ]


def test_sections_includes_a_section_on_the_first_line() -> None:
    assert list(_sections("Args:\n  x: value\n")) == [("Args", ["x: value"])]


def test_attr_doc_form_ignores_an_attributes_comment_without_a_class_doc(
    tmp_path: Path,
) -> None:
    unit = _unit(
        tmp_path,
        '# Attributes:\nclass C:\n    value = 1\n    """Value."""\n',
    )
    assert AttrDocFormRule().check(unit) == ()


def test_attr_doc_form_uses_empty_docstring_for_missing_class_doc(
    tmp_path: Path,
) -> None:
    unit = _unit(
        tmp_path,
        '# Attributes:\nclass C:\n    value = 1\n    """Value."""\n',
    )
    assert AttrDocFormRule().check(unit) == ()


def test_init_facade_reports_a_later_docstring_as_a_statement(tmp_path: Path) -> None:
    unit = _unit(
        tmp_path,
        '"""Package."""\nfrom .module import value\n"""Stray."""\n',
        name="__init__.py",
    )
    findings = InitFacadeRule().check(unit)
    assert [(finding.line, finding.message) for finding in findings] == [
        (
            3,
            "Expr in __init__.py; a package facade holds only docstring, imports, __all__ (STYLE.md)",
        ),
    ]


def test_init_facade_allows_name_targets_and_version_assignment(tmp_path: Path) -> None:
    unit = _unit(
        tmp_path,
        '__version__ = "1"\n__all__ = []\n',
        name="__init__.py",
    )
    assert InitFacadeRule().check(unit) == ()


def test_init_facade_reports_non_name_assignment_target(tmp_path: Path) -> None:
    unit = _unit(tmp_path, "module.value = 1\n", name="__init__.py")
    findings = InitFacadeRule().check(unit)
    assert [(finding.line, finding.message) for finding in findings] == [
        (
            1,
            "Assign in __init__.py; a package facade holds only docstring, imports, __all__ (STYLE.md)",
        ),
    ]


def test_init_facade_continues_after_type_checking_block(tmp_path: Path) -> None:
    unit = _unit(
        tmp_path,
        "if TYPE_CHECKING:\n    from .types import T\n\ndef helper(): ...\n",
        name="__init__.py",
    )
    findings = InitFacadeRule().check(unit)
    assert [(finding.line, finding.message) for finding in findings] == [
        (
            4,
            "FunctionDef in __init__.py; a package facade holds only docstring, imports, __all__ (STYLE.md)",
        ),
    ]


# -- hlr_g13_test ----------------------------------------------------------


def test_attr_blank_line_scans_after_unrelated_class_statements(tmp_path: Path) -> None:
    source = '''\
class Example:
    first = 1
    first is None
    second = 2
    """Second attribute."""
    following = 3
'''
    unit = _unit(tmp_path, source)

    findings = AttrBlankLineRule().check(unit)

    assert [(finding.line, finding.message) for finding in findings] == [
        (5, "attribute docstring not followed by a blank line"),
    ]


def test_attr_blank_line_does_not_stop_after_undocumented_attribute(
    tmp_path: Path,
) -> None:
    source = '''\
class Example:
    first = 1
    """First attribute."""
    second = 2
    second is None
    third = 3
    """Third attribute."""
    following = 4
'''
    unit = _unit(tmp_path, source)

    findings = AttrBlankLineRule().check(unit)

    assert [(finding.line, finding.message) for finding in findings] == [
        (3, "attribute docstring not followed by a blank line"),
        (7, "attribute docstring not followed by a blank line"),
    ]


def test_attr_blank_line_uses_multiline_docstring_end(tmp_path: Path) -> None:
    source = '''\
class Example:
    value = 1
    """First line.

    More detail.
    """
    following = 2
'''
    unit = _unit(tmp_path, source)

    findings = AttrBlankLineRule().check(unit)

    assert [(finding.line, finding.message) for finding in findings] == [
        (6, "attribute docstring not followed by a blank line"),
    ]


def test_attr_blank_line_fix_inserts_each_blank_line_exactly(tmp_path: Path) -> None:
    source = '''\
class Example:
    first = 1
    """First attribute."""
    middle = 2
    middle is None
    last = 3
    """Last attribute."""
    following = 4
'''

    fixed = houselint_rules.AttrBlankLineRule().fix(_unit(tmp_path, source))

    assert (
        fixed
        == '''\
class Example:
    first = 1
    """First attribute."""

    middle = 2
    middle is None
    last = 3
    """Last attribute."""

    following = 4
'''
    )


def test_attr_blank_line_check_message_is_exact(tmp_path: Path) -> None:
    unit = _unit(
        tmp_path,
        'class Example:\n    value = 1\n    """Value."""\n    following = 2\n',
    )

    findings = AttrBlankLineRule().check(unit)

    assert [finding.message for finding in findings] == [
        "attribute docstring not followed by a blank line",
    ]


def test_staticmethod_fix_rewrites_decorator_to_classmethod(tmp_path: Path) -> None:
    source = """\
class Example:
    @staticmethod  # @staticmethod
    def value(item: int) -> int:
        return item
"""

    fixed = StaticMethodRule().fix(_unit(tmp_path, source))

    assert (
        fixed
        == """\
class Example:
    @classmethod  # @staticmethod
    def value(cls, item: int) -> int:
        return item
"""
    )


def test_staticmethod_fix_rewrites_bindings_bottom_up(tmp_path: Path) -> None:
    source = """\
def first(item: int) -> int:
    return item


def second(item: int) -> int:
    return item


class Example:
    first = staticmethod(first)
    second = staticmethod(second)
"""

    fixed = StaticMethodRule().fix(_unit(tmp_path, source))

    assert (
        fixed
        == """\
def first(item: int) -> int:
    return item


def second(item: int) -> int:
    return item


class Example:
    @classmethod
    def first(cls, item: int) -> int:
        return first(item)
    @classmethod
    def second(cls, item: int) -> int:
        return second(item)
"""
    )


def test_staticmethod_fix_replaces_multiline_binding(tmp_path: Path) -> None:
    source = """\\
def value(item: int) -> int:
    return item


class Example:
    value = staticmethod(
        value,
    )
"""

    fixed = StaticMethodRule().fix(_unit(tmp_path, source))

    assert (
        fixed
        == """\\
def value(item: int) -> int:
    return item


class Example:
    @classmethod
    def value(cls, item: int) -> int:
        return value(item)
"""
    )


def test_staticmethod_fix_replaces_multiline_binding_before_later_binding(
    tmp_path: Path,
) -> None:
    source = """\\
def f(x):
    return x


class A:
    g = staticmethod(
        f
    )
    h = staticmethod(f)
"""

    fixed = StaticMethodRule().fix(_unit(tmp_path, source))

    assert (
        fixed
        == """\\
def f(x):
    return x


class A:
    @classmethod
    def g(cls, x):
        return f(x)
    @classmethod
    def h(cls, x):
        return f(x)
"""
    )


def test_staticmethod_fix_forwards_every_parameter_kind(tmp_path: Path) -> None:
    """The forwarder copies the bound def's whole signature and passes each name."""
    source = (
        "def price[T](a: int, /, b: int = 1, *rest: int, k: int = 2, **kw: int) -> T:\n"
        '    """Sum everything."""\n'
        "    return a\n\n\n"
        "async def fetch(x):\n    return x\n\n\n"
        "def plain(y):\n    return y\n\n\n"
        "class A:\n"
        "    cost = staticmethod(price)\n"
        "    get = staticmethod(fetch)\n"
        "    same = staticmethod(plain)\n"
    )

    fixed = StaticMethodRule().fix(_unit(tmp_path, source))

    assert fixed == source.replace(
        "    cost = staticmethod(price)\n"
        "    get = staticmethod(fetch)\n"
        "    same = staticmethod(plain)\n",
        "    @classmethod\n"
        "    def cost[T](cls, a: int, /, b: int=1, *rest: int, k: int=2, **kw: int)"
        " -> T:\n"
        '        """Sum everything."""\n'
        "        return price(a, b, *rest, k=k, **kw)\n"
        "    @classmethod\n"
        "    async def get(cls, x):\n"
        "        return await fetch(x)\n"
        "    @classmethod\n"
        "    def same(cls, y):\n"
        "        return plain(y)\n",
    )


@pytest.mark.parametrize(
    "binding",
    [
        "a = b = staticmethod(f)",
        "a = staticmethod(f, g)",
        "a = staticmethod(f=f)",
        "a = staticmethod(f())",
        "a = staticmethod(missing)",
        "a = staticmethod(decorated)",
        "a = staticmethod(takes_cls)",
        "a = staticmethod(kwonly_cls)",
        "a = staticmethod(star_cls)",
        "a = staticmethod(kw_cls)",
    ],
)
def test_staticmethod_fix_leaves_each_binding_it_cannot_forward(
    tmp_path: Path,
    binding: str,
) -> None:
    source = (
        "def f():\n    return 1\n\n\n"
        "def g():\n    return 1\n\n\n"
        "@cache\ndef decorated():\n    return 1\n\n\n"
        "def takes_cls(cls):\n    return 1\n\n\n"
        "def kwonly_cls(*, cls):\n    return 1\n\n\n"
        "def star_cls(*cls):\n    return 1\n\n\n"
        "def kw_cls(**cls):\n    return 1\n\n\n"
        f"class A:\n    {binding}\n"
    )

    assert StaticMethodRule().fix(_unit(tmp_path, source)) is None


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("def f(a):", "def f(cls, a):"),
        ("def f(a, /):", "def f(cls, a, /):"),
        ("def f(cls):", "def f(cls):"),
        ("def f(cls, /):", "def f(cls, /):"),
        ("def f():", "def f(cls):"),
        ("def f(*a):", "def f(cls, *a):"),
        ("def f( ):", "def f(cls ):"),
        ("def f() -> tuple[int, (int)]:", "def f(cls) -> tuple[int, (int)]:"),
        (
            "def f(\n        *,\n        a: int,\n    ):",
            "def f(cls,\n        *,\n        a: int,\n    ):",
        ),
        ("def f(\n    ):", "def f(cls\n    ):"),
    ],
)
def test_staticmethod_fix_inserts_cls_once(
    tmp_path: Path,
    header: str,
    expected: str,
) -> None:
    source = (
        f"class A:\n    x = (1)\n    @staticmethod\n    {header}\n        return 1\n"
    )

    assert StaticMethodRule().fix(_unit(tmp_path, source)) == (
        f"class A:\n    x = (1)\n    @classmethod\n    {expected}\n        return 1\n"
    )


def test_insert_cls_starts_at_the_def_line_not_the_line_above() -> None:
    """The ``(`` search begins at the ``def``, so a call above it is not touched."""
    source = "x = f(1)\ndef g(*a):\n    return a\n"
    node = ast.parse(source).body[1]
    assert isinstance(node, ast.FunctionDef)
    lines = [str(line) for line in source.splitlines(keepends=True)]

    houselint_rules._insert_cls(lines, node)

    assert "".join(lines) == "x = f(1)\ndef g(cls, *a):\n    return a\n"


def test_staticmethod_forwarder_keeps_only_a_string_docstring(tmp_path: Path) -> None:
    """A leading bare expression that is not a string is body, not docstring."""
    source = (
        "def f(x):\n    x\n    return x\n\n\n"
        "def g(y):\n    1\n    return y\n\n\n"
        "class A:\n    a = staticmethod(f)\n    b = staticmethod(g)\n"
    )

    assert StaticMethodRule().fix(_unit(tmp_path, source)) == source.replace(
        "    a = staticmethod(f)\n    b = staticmethod(g)\n",
        "    @classmethod\n    def a(cls, x):\n        return f(x)\n"
        "    @classmethod\n    def b(cls, y):\n        return g(y)\n",
    )


def test_staticmethod_fix_keeps_unrewritable_binding_unchanged(
    tmp_path: Path,
) -> None:
    source = """\
class Example:
    value = staticmethod(external)
"""

    assert StaticMethodRule().fix(_unit(tmp_path, source)) is None


# -- hlr_g14_test ----------------------------------------------------------


@pytest.mark.parametrize(
    ("lookup", "expected"),
    [
        ('os.environ.get("HOME")', "$HOME lookup; go through codedoctor.lib.userdirs"),
        ('environ.get("HOME")', "$HOME lookup; go through codedoctor.lib.userdirs"),
        ('os.getenv("HOME")', "$HOME lookup; go through codedoctor.lib.userdirs"),
        ('getenv("HOME")', "$HOME lookup; go through codedoctor.lib.userdirs"),
    ],
)
def test_home_lookup_spellings_are_reported_exactly(
    tmp_path: Path,
    lookup: str,
    expected: str,
) -> None:
    assert _messages(tmp_path, XdgLiteralRule(), f"value = {lookup}\n") == [
        (1, expected),
    ]


def test_async_function_docstring_is_exempt_but_later_expression_is_not(
    tmp_path: Path,
) -> None:
    assert _messages(
        tmp_path,
        XdgLiteralRule(),
        "async def f():\n"
        '    """XDG_CONFIG_HOME is described, not executed."""\n'
        '    "XDG_CONFIG_HOME is not a docstring here"\n',
    ) == [
        (
            3,
            (
                "literal 'XDG_CONFIG_HOME is not a docstring here'; "
                "go through codedoctor.lib.userdirs"
            ),
        ),
    ]


def test_underscore_parameter_message_strips_only_leading_underscores(
    tmp_path: Path,
) -> None:
    assert _messages(
        tmp_path,
        UnderscoreParamRule(),
        "def f(__name: int, _tail_: int, _Xname: int):\n    return 1\n",
    ) == [
        (
            1,
            (
                "parameter _Xname is underscore-prefixed; keep the name "
                "Xname and `del` it (STYLE.md Deprecation)"
            ),
        ),
        (
            1,
            (
                "parameter __name is underscore-prefixed; keep the name "
                "name and `del` it (STYLE.md Deprecation)"
            ),
        ),
        (
            1,
            (
                "parameter _tail_ is underscore-prefixed; keep the name "
                "tail_ and `del` it (STYLE.md Deprecation)"
            ),
        ),
    ]


def test_underscore_fix_skips_ineligible_def_and_fixes_later_def(
    tmp_path: Path,
) -> None:
    assert (
        _fixed(
            tmp_path,
            UnderscoreParamRule(),
            "@jit\ndef kernel(_x):\n    return 1\ndef f(_y):\n    return 1\n"
            "def g(_Xname, name):\n    return 1\n",
        )
        == "@jit\ndef kernel(_x):\n    return 1\n"
        "def f(y):\n    del y\n    return 1\n"
        "def g(Xname, name):\n    del Xname\n    return 1\n"
    )


def test_underscore_fix_strips_all_leading_underscores_consistently(
    tmp_path: Path,
) -> None:
    assert (
        _fixed(
            tmp_path,
            UnderscoreParamRule(),
            "def f(__x, _y, _Xz):\n    return 1\n",
        )
        == "def f(x, y, Xz):\n    del x, y, Xz\n    return 1\n"
    )


# -- hlr_g15_test ----------------------------------------------------------


def test_tempfile_direct_import_checks_every_site(tmp_path: Path) -> None:
    assert _messages(
        tmp_path,
        TempfilePrefixRule(),
        "from tempfile import mkdtemp as make, mkstemp\nprint(1)\nmake()\nmkstemp()\n",
    ) == [
        (3, "make without prefix=; name the temporary"),
        (4, "mkstemp without prefix=; name the temporary"),
    ]


def test_cache_kwarg_keeps_scanning_after_a_local_api_call(tmp_path: Path) -> None:
    assert _messages(
        tmp_path,
        CacheKwargRule(),
        "def local(*, cache_dir): ...\nlocal(cache_dir=path)\nload(cache_dir=path)\n",
    ) == [
        (
            3,
            (
                "explicit cache_dir= overrides the provisioned cache; let the library "
                "read its env var (AGENTS.md Provisioned Paths)"
            ),
        ),
    ]


def test_zero_arg_super_ignores_nested_class_calls() -> None:
    tree = ast.parse(
        "class Outer:\n"
        "    def outer(self):\n"
        "        super()\n"
        "    class Inner:\n"
        "        def inner(self):\n"
        "            super()\n",
    )
    outer = tree.body[0]
    assert isinstance(outer, ast.ClassDef)
    assert houselint_rules._zero_arg_supers(outer) == (3,)


def test_dataclass_message_lists_each_missing_kwarg(tmp_path: Path) -> None:
    assert _messages(
        tmp_path,
        DataclassRule(),
        "from dataclasses import dataclass\n@dataclass\nclass C: ...\n",
    ) == [(2, "@dataclass missing kw_only=True, slots=True on C")]


def test_suppression_finding_uses_comment_line_and_exact_message(
    tmp_path: Path,
) -> None:
    assert _messages(
        tmp_path,
        SuppressionFormRule(),
        "value = 1\n# type: ignore[problem]\n",
    ) == [
        (
            2,
            (
                "`type: ignore` is a mypy directive; neither ty nor basedpyright reads "
                "it -- remove it, or use `ty: ignore[code]` / `pyright: ignore[code]`"
            ),
        ),
    ]


def test_wrong_parser_descriptions_continue_past_nonparser_call() -> None:
    unit = Unit(
        path=Path("mod.py"),
        source=(
            "other.call(description=__doc__)\n"
            "argparse.ArgumentParser(description=__doc__)\n"
        ),
        kind="python",
    )
    assert [
        call.lineno
        for call, _ in houselint_rules._wrong_descriptions(
            unit,
            polyglot=False,
        )
    ] == [2]


# -- hlr_g16_test ----------------------------------------------------------


def test_config_globals_checks_class_after_unrelated_class(tmp_path: Path) -> None:
    assert _messages(
        tmp_path,
        ConfigGlobalsRule(),
        "class Ordinary: ...\nDEFAULT = 3\nclass C(Fig[int]):\n    value: int = DEFAULT\n",
    ) == [
        (
            2,
            (
                "module-level tunable value 'DEFAULT'; move it into the call site. "
                "This is a global and globals invite confusion. Meanwhile functions' "
                "kwargs (with this as a default) should be bearing the load of this "
                "value thus being a stateless path for user respecfication. "
                "(If you have to repeat this default value, that's less bad than "
                "global.) If you're using configgle then its particularly silly to "
                "use a global -- the configgle Config _is_ the container of defaults. "
            ),
        ),
        (
            4,
            (
                "Config field 'value' defaults to module constant 'DEFAULT'; inline "
                "the literal so the default lives in the Config (or suppress: "
                "house-ignore[globals] -- Reason.)"
            ),
        ),
    ]


def test_config_globals_checks_assignment_after_module_docstring(
    tmp_path: Path,
) -> None:
    assert (
        _messages(
            tmp_path,
            ConfigGlobalsRule(),
            '"""Module doc."""\nVALUE = 3\n',
        )[0][0]
        == 2
    )


def test_cwd_final_fix_requires_the_exact_path_import(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text(
        "from pathlib import PurePath\nfrom elsewhere import Path\n"
        "value = Path(__file__).resolve().parent\n",
        encoding="utf-8",
    )
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert CwdFinalRule().fix(unit) is None


def test_final_inline_reports_single_use_without_sibling_test(
    tmp_path: Path,
) -> None:
    assert _messages(
        tmp_path,
        FinalInlineRule(),
        "_: Final = 5\ndef f() -> int:\n    return _\n",
    ) == [
        (
            1,
            (
                "'_' is a Final literal read at one site; inline it there "
                "(STYLE.md: no module-level convenience constants)"
            ),
        ),
    ]


def test_final_inline_does_not_match_a_sibling_name_substring(
    tmp_path: Path,
) -> None:
    (tmp_path / "mod_test.py").write_text("XXXX\n", encoding="utf-8")
    assert _messages(
        tmp_path,
        FinalInlineRule(),
        "from typing import Final\n_A_XXXX: Final = 5\n\n\n"
        "def f() -> int:\n    return _A_XXXX\n",
    ) == [
        (
            2,
            (
                "'_A_XXXX' is a Final literal read at one site; inline it there "
                "(STYLE.md: no module-level convenience constants)"
            ),
        ),
    ]


def test_lazy_import_binding_rejects_alias_from_different_module(
    tmp_path: Path,
) -> None:
    assert (
        _messages(
            tmp_path,
            LazyImportBindingRule(),
            "from wrapt import other as li\n# lazy_import\n",
        )
        == []
    )


def test_mkdir_mode_prefilter_sees_atomic_write_calls(tmp_path: Path) -> None:
    assert _messages(
        tmp_path,
        MkdirModeRule(),
        'atomic_write_bytes(config_dir() / "rekursiv-ai" / "secret", b"x")\n',
    ) == [
        (1, "atomic_write_bytes under a userdirs path without file_mode=0o600"),
    ]


def test_module_side_effects_checks_the_first_string_expression(
    tmp_path: Path,
) -> None:
    assert _messages(
        tmp_path,
        ModuleSideEffectsRule(),
        '"""Module doc."""\n"""Second string."""\n',
    ) == [
        (2, "expression \"'Second string.'\" at module scope; move it into a function"),
    ]


def test_nested_try_continues_past_try_without_handlers(tmp_path: Path) -> None:
    assert _messages(
        tmp_path,
        NestedTryRule(),
        "try:\n    pass\nfinally:\n    pass\n"
        "try:\n    try:\n        pass\n    except ValueError:\n        pass\n"
        "except Exception:\n    pass\n",
    ) == [
        (
            6,
            "try nested inside a try body; make them sequential (STYLE.md Error Handling)",
        ),
    ]


def test_tensor_guard_continues_past_nonvalidation_if(tmp_path: Path) -> None:
    assert _messages(
        tmp_path,
        TensorValueGuardRule(),
        "if done.any():\n    restart()\nif x.item() > 0:\n    raise ValueError\n",
    ) == [
        (
            3,
            (
                ".item() inside a guard synchronizes the device; let out-of-domain "
                "input yield NaN/inf (STYLE.md Error Handling)"
            ),
        ),
    ]


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
