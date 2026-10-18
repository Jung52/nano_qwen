"""Load the exact pre-batching rollback path for controlled A/B diagnostics."""

import ast
from contextlib import contextmanager
import hashlib
import subprocess
from unittest.mock import patch

from nano_qwen.engine import model_runner
from nano_qwen.layers import gated_delta_net


REFERENCE = "f443cb3a5fad6c596dd0d80e49c7ee83b194f523"


def reference_methods(root, revision=REFERENCE):
    """Compile only the two reference methods, using their production globals."""
    methods, hashes = {}, {}
    for module, cls, name in (
        (model_runner, "ModelRunner", "sample_tokens"),
        (gated_delta_net, "GatedDeltaNet", "rollback_speculative_state"),
    ):
        path = "src/" + module.__name__.replace(".", "/") + ".py"
        source = subprocess.check_output(
            ["git", "show", f"{revision}:{path}"], cwd=root, text=True,
        )
        tree = ast.parse(source)
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
        method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
        namespace = dict(vars(module))
        exec(compile(ast.Module(body=[method], type_ignores=[]), f"{revision}:{path}", "exec"), namespace)
        methods[name] = namespace[name]
        hashes[path] = hashlib.sha256(source.encode()).hexdigest()
    return methods, hashes


@contextmanager
def legacy_rollback(runner, methods):
    """Keep the same resident engine/graphs/tuner; switch only host rollback code."""
    with patch.object(runner, "sample_tokens", methods["sample_tokens"].__get__(runner)), \
         patch.object(gated_delta_net.GatedDeltaNet, "rollback_speculative_state", methods["rollback_speculative_state"]):
        yield
