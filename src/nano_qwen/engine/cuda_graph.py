from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nano_qwen.utils.context import get_context, reset_context, set_context


if TYPE_CHECKING:
    from .model_runner import ModelRunner


def decode_capture_sizes(max_batch: int) -> list[int]:
    # The final exact size also covers nonstandard capacities such as 3/5/17.
    candidates = [1, 2, 4, 8] + list(range(16, max_batch + 1, 16))
    return sorted({n for n in candidates if n <= max_batch} | {max_batch})


class CudaGraphManager:
    """Own decode and piecewise-prefill CUDA graph capture/replay."""

    def __init__(self, runner: "ModelRunner") -> None:
        self.runner = runner

        self.decode_graph_sizes: list[int] = []
        self.decode_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.decode_graph_pool = None
        self.decode_graph_vars: dict[str, torch.Tensor] | None = None

        self.verify_graphs: dict[int, tuple] = {}
        self.mtp_decode_graphs: dict[int, tuple] = {}
        self.mtp_cache_graphs: dict[int, tuple] = {}
        self.verify_graph: torch.cuda.CUDAGraph | None = None
        self.verify_graph_vars: dict[str, torch.Tensor] | None = None
        self.verify_graph_mode: str | None = None
        self.mtp_decode_graph: torch.cuda.CUDAGraph | None = None
        self.mtp_decode_graph_vars: dict[str, torch.Tensor] | None = None
        self.mtp_cache_graph: torch.cuda.CUDAGraph | None = None
        self.mtp_cache_graph_vars: dict[str, torch.Tensor] | None = None

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
        if self.runner.mtp is not None:
            self._capture_verify()
            self._capture_mtp_decode()
            self._capture_mtp_decode(cache_only=True)
        if self.runner.use_prefill_cudagraph:
            self.capture_prefill()

    def _capture_verify(self) -> None:
        """Exact-size graphs: no padding request can update a live state slot."""
        self.verify_graphs.clear()
        for batch in range(1, self.config.max_num_seqs + 1):
            graph, buffers = self._capture_verify_batch(batch)
            self.verify_graphs[batch] = graph, buffers
        self.verify_graph, self.verify_graph_vars = self.verify_graphs[1]
        self.verify_graph_mode = self.config.mtp_verify_mode

    def _capture_verify_batch(self, batch: int):
        max_blocks = (
            self.config.max_model_len + self.runner.block_size - 1
        ) // self.runner.block_size
        device = torch.cuda.current_device()
        input_ids = torch.zeros(2 * batch, dtype=torch.int64, device=device)
        positions = torch.arange(2, dtype=torch.int64, device=device).repeat(batch)
        slot_mapping = (torch.arange(batch, device=device)[:, None] * self.runner.block_size + torch.arange(2, device=device)).to(torch.int32).flatten()
        block_tables = torch.zeros((batch, max_blocks), dtype=torch.int32, device=device)
        block_tables[:, 0] = torch.arange(batch, device=device)
        cu_seqlens_q = torch.arange(0, 2 * batch + 1, 2, dtype=torch.int32, device=device)
        cu_seqlens_k = cu_seqlens_q.clone()
        context_lens = torch.full((batch,), 2, dtype=torch.int32, device=device)
        state_indices = torch.arange(batch, dtype=torch.int64, device=device)
        outputs = torch.empty(
            (2 * batch, self.config.hf_config.hidden_size),
            dtype=self.model.model.embed_tokens.weight.dtype,
            device=device,
        )

        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=2,
            max_seqlen_k=2,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            state_indices=state_indices,
            prefill_slices=[(2*i, 2*i + 2) for i in range(batch)],
            is_verify=True,
        )
        try:
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    outputs.copy_(self.runner.forward_verify(input_ids, positions))
            torch.cuda.current_stream().wait_stream(warmup_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs.copy_(self.runner.forward_verify(input_ids, positions))
            buffers = {
                "input_ids": input_ids,
                "positions": positions,
                "slot_mapping": slot_mapping,
                "block_tables": block_tables,
                "cu_seqlens_k": cu_seqlens_k,
                "context_lens": context_lens,
                "state_indices": state_indices,
                "outputs": outputs,
            }
            return graph, buffers
        finally:
            reset_context()
            for layer in self.runner.gdn_layers:
                layer.reset_state(list(range(batch)))

    def run_verify(self, input_ids: torch.Tensor, positions: torch.Tensor):
        if self.verify_graph_mode != self.config.mtp_verify_mode:
            raise RuntimeError("MTP verify mode changed; recapture its CUDA Graph first")
        context = get_context()
        graph, buffers = self.verify_graphs[input_ids.numel() // 2]
        buffers["input_ids"].copy_(input_ids)
        buffers["positions"].copy_(positions)
        buffers["slot_mapping"].copy_(context.slot_mapping)
        buffers["block_tables"].fill_(-1)
        buffers["block_tables"][:, :context.block_tables.size(1)].copy_(
            context.block_tables
        )
        buffers["cu_seqlens_k"].copy_(context.cu_seqlens_k)
        buffers["state_indices"].copy_(context.state_indices)
        buffers["context_lens"].copy_(context.context_lens)
        graph.replay()
        hidden = buffers["outputs"]
        self.runner._target_hidden = hidden
        return self.runner.compute_verify_logits(hidden)

    def _capture_mtp_decode(self, cache_only: bool = False) -> None:
        bank = self.mtp_cache_graphs if cache_only else self.mtp_decode_graphs
        bank.clear()
        for batch in range(1, self.config.max_num_seqs + 1):
            bank[batch] = self._capture_mtp_decode_batch(batch, cache_only)
        if cache_only:
            self.mtp_cache_graph, self.mtp_cache_graph_vars = bank[1]
        else:
            self.mtp_decode_graph, self.mtp_decode_graph_vars = bank[1]

    def _capture_mtp_decode_batch(self, batch: int, cache_only: bool):
        max_blocks = (
            self.config.max_model_len + self.runner.block_size - 1
        ) // self.runner.block_size
        device = torch.cuda.current_device()
        input_ids = torch.zeros(batch, dtype=torch.int64, device=device)
        positions = torch.ones(batch, dtype=torch.int64, device=device)
        target_hidden = torch.zeros(
            (batch, self.config.hf_config.hidden_size),
            dtype=self.model.model.embed_tokens.weight.dtype, device=device,
        )
        # RoPE position 1 is stored in MTP cache row zero.
        slot_mapping = torch.arange(batch, dtype=torch.int32, device=device) * self.runner.block_size
        context_lens = torch.ones(batch, dtype=torch.int32, device=device)
        block_tables = torch.zeros((batch, max_blocks), dtype=torch.int32, device=device)
        block_tables[:, 0] = torch.arange(batch, device=device)
        drafts = torch.empty(batch, dtype=torch.int64, device=device)

        set_context(
            False, slot_mapping=slot_mapping, context_lens=context_lens,
            block_tables=block_tables,
        )
        try:
            def forward():
                if cache_only:
                    self.runner.mtp.cache_only(input_ids, positions, target_hidden,
                                               self.model.model.embed_tokens)
                    return
                hidden = self.runner.mtp(
                    input_ids, positions, target_hidden,
                    self.model.model.embed_tokens,
                )
                drafts.copy_(self.model.compute_logits(hidden).argmax(dim=-1))

            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(warmup_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            buffers = {
                "input_ids": input_ids, "positions": positions,
                "target_hidden": target_hidden, "slot_mapping": slot_mapping,
                "context_lens": context_lens, "block_tables": block_tables,
                "drafts": drafts,
            }
            return graph, buffers
        finally:
            reset_context()

    def run_mtp_decode(self, input_ids, positions, target_hidden, slot_mapping,
                       context_lens, block_tables):
        graph, buffers = self.mtp_decode_graphs[input_ids.numel()]
        self._update_mtp_buffers(buffers, input_ids, positions, target_hidden,
                                 slot_mapping, context_lens, block_tables)
        graph.replay()
        return buffers["drafts"]

    def run_mtp_cache_only(self, input_ids, positions, target_hidden, slot_mapping,
                           context_lens, block_tables):
        graph, buffers = self.mtp_cache_graphs[input_ids.numel()]
        self._update_mtp_buffers(buffers, input_ids, positions,
                                 target_hidden, slot_mapping, context_lens, block_tables)
        graph.replay()

    @staticmethod
    def _update_mtp_buffers(buffers, input_ids, positions, target_hidden,
                            slot_mapping, context_lens, block_tables):
        buffers["input_ids"].copy_(input_ids)
        buffers["positions"].copy_(positions)
        buffers["target_hidden"].copy_(target_hidden)
        buffers["slot_mapping"].copy_(slot_mapping)
        buffers["context_lens"].copy_(context_lens)
        buffers["block_tables"].fill_(-1)
        buffers["block_tables"][:, :block_tables.size(1)].copy_(block_tables)

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
        # In-place GDN updates require distinct live slots during warmup and
        # capture too. Replay overwrites this same buffer with request slots.
        state_indices = torch.arange(max_bs, dtype=torch.int64, device=device)
        outputs = torch.zeros(
            max_bs,
            hf_config.hidden_size,
            device=device,
        )

        self.decode_graph_sizes = decode_capture_sizes(max_bs)
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
            hidden = self.model.model(input_ids, positions)
            self.runner._target_hidden = hidden
            return self.model.compute_logits(hidden)

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
        self.runner._target_hidden = hidden
        return self.model.compute_logits(hidden)

    def run_decode(self, input_ids: torch.Tensor, positions: torch.Tensor):
        bs = input_ids.size(0)
        if bs not in self.decode_graphs:
            hidden = self.model.model(input_ids, positions)
            self.runner._target_hidden = hidden
            return self.model.compute_logits(hidden)

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
        hidden = graph_vars["outputs"][:bs]
        self.runner._target_hidden = hidden
        return self.model.compute_logits(hidden)

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
        self.verify_graphs.clear()
        self.mtp_decode_graphs.clear()
        self.mtp_cache_graphs.clear()
        self.verify_graph = None
        self.verify_graph_vars = None
        self.verify_graph_mode = None
        self.mtp_decode_graph = None
        self.mtp_decode_graph_vars = None
        self.mtp_cache_graph = None
        self.mtp_cache_graph_vars = None

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
            if self.config.quant_config is not None:
                # Inductor fuses BF16 normalization/casts before FP8 input
                # quantization, changing rounding and subsequent FP8 bins.
                # Capture the original operations so eager and graph use the
                # same numerical path; CUDA Graph replay remains enabled.
                self._piecewise_compiled[cache_key] = fn
            else:
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
            # Each layer consumes its projections before the next layer runs
            # on the same stream. Reuse compatible storage rather than retain
            # a full M-sized QKV/z copy for every layer in every captured graph.
            signature = tuple((tuple(output.shape[1:]), output.dtype, output.device)
                              for output in outputs)
            shared = self.piecewise_buffers.setdefault("pre_out_pool", {})
            if signature not in shared:
                shared[signature] = [
                    output.new_zeros(
                        (max(self.prefill_graph_sizes), *output.shape[1:]),
                        device=output.device,
                    )
                    for output in outputs
                ]
            self.piecewise_buffers["pre_out"][layer_idx] = shared[signature]
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
