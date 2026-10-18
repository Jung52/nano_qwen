"""The original all-row MTP prefill head, with current decode/rollback intact."""

import ast
from contextlib import contextmanager
from pathlib import Path
import subprocess
from unittest.mock import patch

from nano_qwen.engine import model_runner


REFERENCE = "f443cb3a5fad6c596dd0d80e49c7ee83b194f523"


def prefill_reference():
    root = Path(__file__).resolve().parents[1]
    path = "src/nano_qwen/engine/model_runner.py"
    source = subprocess.check_output(["git", "show", f"{REFERENCE}:{path}"], cwd=root, text=True)
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "ModelRunner")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_run_mtp_prefill")
    namespace = dict(vars(model_runner))
    exec(compile(ast.Module(body=[method], type_ignores=[]), f"{REFERENCE}:{path}", "exec"), namespace)
    return namespace[method.name]


@contextmanager
def all_row_head(runner, method):
    with patch.object(runner, "_run_mtp_prefill", method.__get__(runner)):
        yield
