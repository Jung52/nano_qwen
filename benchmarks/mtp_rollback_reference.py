"""Load the exact pre-batching rollback path for controlled A/B diagnostics."""

import ast
from copy import copy
from types import SimpleNamespace

from nano_qwen.engine.execution import RequestOutput
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
    """Legacy state oracle with an explicit benchmark-only transport adapter.

    The pinned sampler predates independent draft rows and compact logits.
    Shadow sequences restore its old indexing without changing live history.
    Adapter allocation costs make this unsuitable for fresh timing attribution;
    reproduce historical timings at their recorded source revision instead.
    """
    sample = methods["sample_tokens"].__get__(runner)

    def legacy_sample():
        logits, temperatures, seqs, prefill = runner._pending
        shadows = []
        for seq in seqs:
            shadow = copy(seq)
            shadow.token_ids = list(seq.token_ids)
            if seq.is_speculative:
                shadow.append_token(seq.draft_token)
            shadows.append(shadow)
        plan = getattr(runner, "_sampling_plan", None)
        if prefill and not all(s.is_speculative for s in seqs) and plan is not None:
            # Only selected rows are consumed by the legacy tail selection.
            full = logits.new_zeros((sum(s.num_scheduled_tokens for s in seqs), logits.shape[-1]))
            if plan.row_indices:
                full[list(plan.row_indices)] = logits
            logits = full
        runner._pending = (logits, temperatures, shadows, prefill)
        runner._sampling_plan = None
        output = sample()

        def results():
            rows = output.get_output()
            result = []
            for seq, row in zip(seqs, rows):
                partial = not seq.is_speculative and seq.is_prefill and (
                    seq.num_cached_tokens+seq.num_scheduled_tokens < seq.num_tokens)
                tokens = () if partial else tuple(t for t in row[:2] if t != -1)
                accepted = int(seq.is_speculative and row[0] == seq.draft_token)
                result.append(RequestOutput(tokens, accepted, None if partial or row[2] == -1 else row[2],
                                            len(tokens) if seq.is_speculative else seq.num_scheduled_tokens))
            return result
        return SimpleNamespace(get_results=results, get_output=output.get_output)

    with patch.object(runner, "sample_tokens", legacy_sample), \
         patch.object(gated_delta_net.GatedDeltaNet, "rollback_speculative_state", methods["rollback_speculative_state"]):
        yield
