from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nano_qwen.utils.context import get_context, reset_context, set_context


if TYPE_CHECKING:
    from .model_runner import ModelRunner


class CudaGraphManager:
    """Own decode and piecewise-prefill CUDA graph capture/replay."""

    def __init__(self, runner: "ModelRunner") -> None:
        self.runner = runner

        self.decode_graph_sizes: list[int] = []
        self.decode_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.decode_graph_pool = None
        self.decode_graph_vars: dict[str, torch.Tensor] | None = None

        self.prefill_graph_sizes: list[int] = []
        self.prefill_graphs: dict[int, dict[int, dict]] = {}
        self.prefill_graph_pool = None
        self.piecewise_buffers: dict | None = None
        self._piecewise_compiled: dict[tuple[int, str], callable] = {}

    @property
    def model(self):
        return self.runner.model

    @property
    def config(self):
        return self.runner.config

    def capture(self) -> None:
        self._capture_decode()
        if self.runner.use_prefill_cudagraph:
            self.capture_prefill()

    def _capture_decode(self) -> None:
        config = self.config
        hf_config = config.hf_config
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (
            config.max_model_len + self.runner.block_size - 1
        ) // self.runner.block_size
        device = torch.cuda.current_device()
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=device)
        positions = torch.zeros(max_bs, dtype=torch.int64, device=device)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device=device)
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        block_tables = torch.zeros(
            max_bs,
            max_num_blocks,
            dtype=torch.int32,
            device=device,
        )
        state_indices = torch.zeros(max_bs, dtype=torch.int64, device=device)
        outputs = torch.zeros(
            max_bs,
            hf_config.hidden_size,
            device=device,
        )

        self.decode_graph_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.decode_graphs = {}
        self.decode_graph_pool = None

        for bs in reversed(self.decode_graph_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                state_indices=state_indices[:bs],
            )
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            torch.cuda.current_stream().wait_stream(warmup_stream)

            with torch.cuda.graph(graph, self.decode_graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.decode_graph_pool is None:
                self.decode_graph_pool = graph.pool()
            self.decode_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.decode_graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "state_indices": state_indices,
            "outputs": outputs,
        }

    def capture_prefill(
        self,
        buckets: list[int] | None = None,
    ) -> None:
        max_prefill_tokens = self.config.max_num_batched_tokens
        if buckets is None:
            buckets = [
                size
                for size in (128, 256, 512, 768, 1024, 1536, 2048)
                if size <= max_prefill_tokens
            ]
            if max_prefill_tokens > 0 and max_prefill_tokens not in buckets:
                buckets.append(max_prefill_tokens)

        self.prefill_graph_sizes = list(buckets)
        self.clear_prefill()
        if not self.runner.use_prefill_cudagraph or not buckets:
            return

        self.prefill_graph_pool = torch.cuda.graph_pool_handle()
        self.piecewise_buffers = self._allocate_piecewise_buffers(
            max(buckets),
            self.config.hf_config.hidden_size,
        )

        dynamo_config = torch._dynamo.config
        old_recompile_limit = dynamo_config.recompile_limit
        dynamo_config.recompile_limit = max(
            1024,
            len(self.model.model.layers) * len(buckets) * 4,
        )
        try:
            for graph_size in buckets:
                bucket_entries = {}
                for layer_idx, layer in enumerate(self.model.model.layers):
                    bucket_entries[layer_idx] = self.capture_prefill_layer_segments(
                        layer,
                        graph_size,
                        has_residual=layer_idx > 0,
                    )
                self.prefill_graphs[graph_size] = bucket_entries
                torch.cuda.synchronize()
        finally:
            dynamo_config.recompile_limit = old_recompile_limit

    def run_prefill(self, input_ids: torch.Tensor, positions: torch.Tensor):
        context = get_context()
        if context.prefill_slices is None:
            raise RuntimeError("prefill graph requires Context.prefill_slices")

        num_tokens = input_ids.size(0)
        graph_size = next(
            (size for size in self.prefill_graph_sizes if size >= num_tokens),
            None,
        )
        if graph_size is None or graph_size not in self.prefill_graphs:
            return self.model.compute_logits(self.model(input_ids, positions))

        entries = self.prefill_graphs[graph_size]
        hidden = self.model.model.embed_tokens(input_ids)
        residual = None
        for layer_idx, layer in enumerate(self.model.model.layers):
            entry = entries[layer_idx]

            pre = entry["pre"]
            pre["hidden_in"].zero_()
            pre["hidden_in"][:num_tokens].copy_(hidden[:num_tokens])
            if residual is not None:
                pre["residual_in"].zero_()
                pre["residual_in"][:num_tokens].copy_(residual[:num_tokens])
            pre["graph"].replay()
            attention_pre = pre["outputs"]
            layer_residual = pre["residual_out"][:graph_size]
            attention_pre = tuple(
                tensor[:num_tokens] if tensor.dim() == 2 else tensor[:, :num_tokens]
                for tensor in attention_pre
            )
            attention_output = layer.forward_attention_core(
                attention_pre,
                positions,
                context.prefill_slices,
            )

            post = entry["post"]
            post["hidden_in"].zero_()
            post["hidden_in"][:num_tokens].copy_(attention_output)
            post["residual_in"].copy_(layer_residual)
            post["graph"].replay()
            hidden = post["hidden_out"]
            residual = post["residual_out"]

        hidden, _ = self.model.model.norm(
            hidden[:num_tokens],
            residual[:num_tokens],
        )
        return self.model.compute_logits(hidden)

    def run_decode(self, input_ids: torch.Tensor, positions: torch.Tensor):
        bs = input_ids.size(0)
        if bs not in self.decode_graphs:
            return self.model.compute_logits(self.model(input_ids, positions))

        context = get_context()
        graph = self.decode_graphs[bs]
        graph_vars = self.decode_graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"].fill_(-1)
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = (
            context.block_tables
        )
        graph_vars["state_indices"].zero_()
        graph_vars["state_indices"][:bs] = context.state_indices
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def clear_prefill(self) -> None:
        self.prefill_graphs.clear()
        self.piecewise_buffers = None
        self._piecewise_compiled.clear()
        self.prefill_graph_pool = None

    def clear(self) -> None:
        self.prefill_graphs.clear()
        self.piecewise_buffers = None
        self._piecewise_compiled.clear()
        self.prefill_graph_pool = None
        self.decode_graphs.clear()
        self.decode_graph_vars = None
        self.decode_graph_pool = None

    def _allocate_piecewise_buffers(
        self,
        max_tokens: int,
        hidden_size: int,
    ) -> dict:
        model_weight = self.model.model.embed_tokens.weight
        shape = (max_tokens, hidden_size)
        return {
            "pre_hidden": torch.zeros(
                shape, dtype=model_weight.dtype, device=model_weight.device,
            ),
            "pre_residual": torch.zeros(
                shape, dtype=model_weight.dtype, device=model_weight.device,
            ),
            "pre_residual_out": torch.zeros(
                shape, dtype=model_weight.dtype, device=model_weight.device,
            ),
            "post_hidden": torch.zeros(
                shape, dtype=model_weight.dtype, device=model_weight.device,
            ),
            "post_residual": torch.zeros(
                shape, dtype=model_weight.dtype, device=model_weight.device,
            ),
            "post_hidden_out": torch.zeros(
                shape, dtype=model_weight.dtype, device=model_weight.device,
            ),
            "post_residual_out": torch.zeros(
                shape, dtype=model_weight.dtype, device=model_weight.device,
            ),
            "pre_out": {},
        }

    def _piecewise_callable(self, layer, kind: str):
        cache_key = (id(layer), kind)
        if cache_key not in self._piecewise_compiled:
            fn = (
                layer.forward_piecewise_pre
                if kind == "pre"
                else layer.forward_output
            )
            self._piecewise_compiled[cache_key] = torch.compile(
                fn,
                dynamic=False,
            )
        return self._piecewise_compiled[cache_key]

    def _allocate_pre_out(
        self,
        layer_idx: int,
        outputs: tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        if layer_idx not in self.piecewise_buffers["pre_out"]:
            self.piecewise_buffers["pre_out"][layer_idx] = [
                output.new_zeros(
                    (max(self.prefill_graph_sizes), *output.shape[1:]),
                    device=output.device,
                )
                for output in outputs
            ]
        return self.piecewise_buffers["pre_out"][layer_idx]

    def _copy_pre_outputs(self, layer_idx: int, outputs, residual):
        static_outputs = self._allocate_pre_out(layer_idx, outputs)
        for dst, src in zip(static_outputs, outputs):
            dst[:src.shape[0]].copy_(src)
        static_residual = self.piecewise_buffers["pre_residual_out"]
        static_residual[:residual.shape[0]].copy_(residual)
        return static_outputs, static_residual

    def _copy_post_outputs(self, outputs):
        hidden, residual = outputs
        self.piecewise_buffers["post_hidden_out"][:hidden.shape[0]].copy_(hidden)
        self.piecewise_buffers["post_residual_out"][:residual.shape[0]].copy_(
            residual
        )
        return (
            self.piecewise_buffers["post_hidden_out"],
            self.piecewise_buffers["post_residual_out"],
        )

    @torch.inference_mode()
    def capture_prefill_layer_segments(
        self,
        layer,
        graph_size: int,
        has_residual: bool,
    ):
        buffers = self.piecewise_buffers
        layer_idx = next(
            index
            for index, item in enumerate(self.model.model.layers)
            if item is layer
        )
        pre_hidden = buffers["pre_hidden"][:graph_size]
        pre_residual = (
            buffers["pre_residual"][:graph_size] if has_residual else None
        )
        pre_fn = self._piecewise_callable(layer, "pre")
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                pre_outputs, _ = pre_fn(pre_hidden, pre_residual)
        self._allocate_pre_out(layer_idx, pre_outputs)
        torch.cuda.synchronize()

        pre_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(pre_graph, self.prefill_graph_pool):
            pre_outputs, pre_layer_residual = pre_fn(pre_hidden, pre_residual)
            static_pre_outputs, static_pre_residual = self._copy_pre_outputs(
                layer_idx,
                pre_outputs,
                pre_layer_residual,
            )
        torch.cuda.synchronize()

        post_hidden = buffers["post_hidden"][:graph_size]
        post_residual = buffers["post_residual"][:graph_size]
        post_fn = self._piecewise_callable(layer, "post")
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                post_outputs = post_fn(post_hidden, post_residual)
        torch.cuda.synchronize()

        post_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(post_graph, self.prefill_graph_pool):
            post_outputs = post_fn(post_hidden, post_residual)
            static_post_hidden, static_post_residual = self._copy_post_outputs(
                post_outputs
            )
        torch.cuda.synchronize()

        return {
            "pre": {
                "graph": pre_graph,
                "hidden_in": pre_hidden,
                "residual_in": pre_residual,
                "outputs": static_pre_outputs,
                "residual_out": static_pre_residual,
            },
            "post": {
                "graph": post_graph,
                "hidden_in": post_hidden,
                "residual_in": post_residual,
                "hidden_out": static_post_hidden,
                "residual_out": static_post_residual,
            },
        }
