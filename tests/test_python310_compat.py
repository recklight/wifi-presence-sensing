"""Python 3.10 checks that do not depend on the interpreter running them.

The package declares ``requires-python = ">=3.10"``, but a test run on 3.11 or
later passes even when a module uses newer syntax or library names.  So the
syntax check compiles every module with :func:`ast.parse` pinned to the 3.10
grammar, and the library check scans executable code (not comments or
strings) for standard-library names that only exist in 3.11 or later and for
the NumPy aliases removed in 2.0.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

_PROSE_TOKENS = frozenset({"COMMENT", "STRING", "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"})

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_STDLIB: dict[str, str] = {
    "tomllib": "pyyaml or another reader (3.11+)",
    "typing.Self": "the class name as a string (3.11+)",
    "LiteralString": "str (3.11+)",
    "assert_never": "raise AssertionError (3.11+)",
    "typing.Unpack": "explicit parameters (3.11+)",
    "TypeVarTuple": "a plain TypeVar (3.11+)",
    "StrEnum": "class Foo(str, Enum) (3.11+)",
    "ReprEnum": "Enum (3.11+)",
    "datetime.UTC": "datetime.timezone.utc (3.11+)",
    "TaskGroup": "asyncio.gather (3.11+)",
    "ExceptionGroup": "a single exception (3.11+)",
    "contextlib.chdir": "os.chdir in try/finally (3.11+)",
    "file_digest": "hash the file in chunks (3.11+)",
    "math.cbrt": "x ** (1/3) (3.11+)",
    "getmembers_static": "inspect.getmembers (3.11+)",
    "typing.override": "a plain method (3.12+)",
    "itertools.batched": "an explicit slice loop (3.12+)",
    "ReadOnly": "a plain annotation (3.13+)",
    "copy.replace": "dataclasses.replace (3.13+)",
}

FORBIDDEN_NUMPY: dict[str, str] = {
    "np.float_": "np.float64",
    "np.int0": "np.intp",
    "np.NaN": "np.nan",
    "np.Inf": "np.inf",
    "np.product": "np.prod",
    "np.alltrue": "np.all",
    "np.round_": "np.round",
    "np.msort": "np.sort",
    "np.trapz": "an explicit sum (renamed in NumPy 2.0)",
}


def code_only(source: str) -> str:
    """Strip comments and string literals so documentation is not mistaken for code."""
    pieces: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if tokenize.tok_name.get(token.type, "") in _PROSE_TOKENS:
            continue
        pieces.append(token.string)
    return "\n".join(pieces)


def python_files() -> list[Path]:
    files: list[Path] = []
    for folder in ("src", "tests", "examples"):
        files.extend(sorted((PROJECT_ROOT / folder).rglob("*.py")))
    return files


def files_to_scan() -> list[Path]:
    """Every module except this one, which spells out the forbidden names on purpose."""
    this_file = Path(__file__).resolve()
    return [path for path in python_files() if path.resolve() != this_file]


def test_there_is_something_to_check():
    assert len(python_files()) > 30
    assert len(files_to_scan()) == len(python_files()) - 1


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_every_module_parses_under_the_python_310_grammar(path: Path):
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path), feature_version=(3, 10))
    except SyntaxError as exc:
        pytest.fail(f"{path.name} does not parse as Python 3.10: {exc.msg} (line {exc.lineno})")


@pytest.mark.parametrize("path", files_to_scan(), ids=lambda p: p.name)
def test_no_standard_library_name_newer_than_310(path: Path):
    source = code_only(path.read_text(encoding="utf-8"))
    offences = [
        f"{name} -> {fix}"
        for name, fix in FORBIDDEN_STDLIB.items()
        if re.search(rf"(?<![\w]){re.escape(name)}\b", source)
    ]
    assert not offences, f"{path.name}: " + "; ".join(offences)


@pytest.mark.parametrize("path", files_to_scan(), ids=lambda p: p.name)
def test_no_numpy_alias_removed_in_2_0(path: Path):
    source = code_only(path.read_text(encoding="utf-8"))
    offences = [
        f"{name} -> {fix}"
        for name, fix in FORBIDDEN_NUMPY.items()
        if re.search(rf"(?<![\w.]){re.escape(name)}\b", source)
    ]
    assert not offences, f"{path.name}: " + "; ".join(offences)


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_no_pep695_generics_or_type_aliases(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        assert getattr(node, "type_params", []) == [], f"{path.name}: PEP 695 type parameters"
        assert type(node).__name__ != "TypeAlias", f"{path.name}: 'type X = ...' statement"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_every_module_defers_annotation_evaluation(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    future = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
    ]
    assert future, f"{path.name} lacks 'from __future__ import annotations'"


@pytest.mark.parametrize("path", sorted((PROJECT_ROOT / "src").rglob("*.py")), ids=lambda p: p.name)
def test_no_frozen_slotted_dataclasses(path: Path):
    """Frozen dataclasses with slots do not unpickle on early 3.10 releases.

    Several of them cross process boundaries through joblib, so the
    combination is avoided altogether.
    """
    source = path.read_text(encoding="utf-8")
    assert not re.search(r"@dataclass\([^)]*frozen=True[^)]*slots=True", source)
    assert not re.search(r"@dataclass\([^)]*slots=True[^)]*frozen=True", source)


def test_pyproject_targets_python_310_with_reachable_floors():
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10"' in text
    assert 'target-version = "py310"' in text
    for requirement in (
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "matplotlib>=3.6",
        "pandas>=1.5",
    ):
        assert requirement in text, requirement
    assert "numpy>=2" not in text
    assert 'authors = [{ name = "RL" }]' in text


def test_package_version_matches_the_release():
    import wifi_presence_sensing

    assert wifi_presence_sensing.__version__ == "0.1.0"
