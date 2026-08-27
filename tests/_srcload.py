"""
Load a single function out of a source file WITHOUT importing the module.

The training scripts in this repository import torch, deepspeed, trl and
transformers at module scope. Importing them just to unit-test one pure
function would require the whole heavy stack — and a GPU for some of it.

Instead we parse the file with `ast`, pull out the one function (or the one
method of a class) under test, and exec only that node. The test therefore
runs against the ACTUAL SHIPPED SOURCE rather than a copy that can silently
drift out of sync with it.

Stdlib only — no dependencies.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_function(
    source_path: str | Path,
    function_name: str,
    class_name: Optional[str] = None,
    extra_globals: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Extract one function from a Python file and return it as a callable.

    Args:
        source_path: Path to the .py file, absolute or relative to the repo root
        function_name: Name of the function (or method) to extract
        class_name: If given, look for the function inside this class
        extra_globals: Names to inject into the function's global namespace

    Returns:
        The extracted function object

    Raises:
        FileNotFoundError: if the source file does not exist
        LookupError: if the function cannot be found
    """
    path = Path(source_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    search_body = tree.body
    if class_name is not None:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                search_body = node.body
                break
        else:
            raise LookupError(f"Class {class_name!r} not found in {path}")

    for node in search_body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            # Drop decorators (@staticmethod etc.) — we want a plain callable.
            node.decorator_list = []
            namespace: Dict[str, Any] = {"__name__": "_extracted"}
            if extra_globals:
                namespace.update(extra_globals)
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, filename=str(path), mode="exec"), namespace)  # noqa: S102
            return namespace[function_name]

    where = f"class {class_name}" if class_name else "module scope"
    raise LookupError(f"Function {function_name!r} not found at {where} in {path}")


def source_contains(source_path: str | Path, needle: str) -> bool:
    """True if the given text appears anywhere in the source file."""
    path = Path(source_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return needle in path.read_text(encoding="utf-8")


def source_code_contains(source_path: str | Path, needle: str) -> bool:
    """
    True if `needle` appears in EXECUTABLE code, ignoring comments and strings.

    Necessary because these files deliberately describe the bugs they used to
    have ("the previous self.model.base_model access was ..."). A naive text
    search would match that prose and report the bug as still present.
    """
    import io
    import tokenize

    path = Path(source_path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    source = path.read_text(encoding="utf-8")
    kept = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    except (tokenize.TokenError, IndentationError):
        # Fall back to a line filter if tokenizing fails.
        kept = [
            line for line in source.splitlines()
            if not line.strip().startswith("#")
        ]
    return needle in " ".join(kept)


class Results:
    """Minimal pass/fail tracker so the tests need no test framework."""

    def __init__(self, title: str):
        self.title = title
        self.passed = 0
        self.failed = 0
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")

    def check(self, condition: bool, description: str, detail: str = "") -> bool:
        if condition:
            self.passed += 1
            print(f"  PASS  {description}")
        else:
            self.failed += 1
            print(f"  FAIL  {description}")
            if detail:
                print(f"        {detail}")
        return condition

    def finish(self) -> int:
        total = self.passed + self.failed
        print(f"\n  {self.passed}/{total} checks passed")
        if self.failed:
            print(f"  {self.failed} FAILED")
        return 1 if self.failed else 0
