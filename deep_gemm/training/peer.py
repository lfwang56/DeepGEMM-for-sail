"""Native peer-read/peer-write Tensor Cell GEMMs for PPU MoE training."""

import math

import torch
import torch.distributed as dist

import deep_gemm as dg
from .bf16 import _validate_accumulators
from .native import NativeMoEPlan, _launch
from .routed_rows import allocate_route_windows, set_route_window


def _peer(operation, tensors, dims, compile_only=False):
    dg._C._moe_training_peer_launch(operation, tensors, dims, compile_only)


class PeerMoEPlan(NativeMoEPlan):
    """Bounded BF16 training with dispatch reads and return writes inside GEMM.

    Requires a direct peer domain, including cross-host ICN within a supernode.
    PCCL/RDMA remains available through NativeMoEPlan for other topologies.
    """

    transport = "peer"

    def __init__(self, *args, gemm_configs=None, fuse_dispatch=True, fuse_return=True,
                 active_rows=True, fuse_local_grad=True, fuse_activation_backward=False,
                 vectorized_copy=True, dispatch_cache_n=0, reuse_packed_dy=False,
                 dispatch_cache_aiu=False, staged_return=False, dispatch_cache_ctas=2,
                 dispatch_producer=False, dispatch_producer_tiles=1, producer_local_aiu=False,
                 producer_input_aiu=False, dispatch_pipeline=False, pipeline_comm_ctas=2,
                 pipeline_statistics=False, return_pipeline=False, pipeline_serialize=False,
                 inline_dispatch=False, cache_return_addresses=False, ffn_fusion=False,
                 expert_local_grad=False, ring_rows=None, return_vector_bytes=4,
                 dag_fusion=False, dag_input_ctas=4, dag_return_ctas=4,
                 dag_serialize=False, dag_trace=False, dag_shared_pool=False, dag_input_prefetch=False,
                 dag_backward=False, dag_direct_return=False,
                 input_copy_ilp=1, dispatch_assist_ctas=0, dispatch_external=False,
                 fuse_fc1_activation=False, **kwargs):
        if type(dispatch_assist_ctas) is not int or not 0 <= dispatch_assist_ctas <= 156:
            raise ValueError("dispatch_assist_ctas must be an integer in [0,156]")
        if not isinstance(dispatch_external, bool):
            raise TypeError("dispatch_external must be bool")
        if dispatch_external and (not dispatch_assist_ctas or inline_dispatch):
            raise ValueError("external dispatch requires positive assistants without inline owners")
        self.dispatch_assist_ctas = dispatch_assist_ctas
        self.dispatch_external = dispatch_external
        if dispatch_assist_ctas:
            inline_dispatch = not dispatch_external
            dispatch_pipeline = True
        if kwargs.get("overlap_expert_dy", False) and (not dispatch_producer or not expert_local_grad
                or ring_rows is None or dag_fusion or dag_backward or ffn_fusion
                or dispatch_pipeline or return_pipeline):
            raise ValueError("DY prefetch requires the producer expert-gradient routed-window schedule")
        if kwargs.get("wgrad_overlap", "none") != "none" and (dag_backward or ffn_fusion):
            raise ValueError("wgrad stream overlap requires the separate backward GEMM schedule")
        if kwargs.get("wgrad_overlap") == "both" and not fuse_activation_backward:
            raise ValueError("DW2 overlap requires the fused dHidden epilogue to preserve forward Hidden")
        if type(input_copy_ilp) is not int or input_copy_ilp not in (1, 2, 4, 8):
            raise ValueError("input_copy_ilp must be1,2,4or8")
        if input_copy_ilp != 1 and not (dag_fusion or inline_dispatch or dispatch_external or (dispatch_producer and producer_input_aiu)):
            raise ValueError("input copy ILP requires the forward DAG or producer input staging")
        self.input_copy_ilp = input_copy_ilp
        if not isinstance(fuse_fc1_activation, bool):
            raise TypeError("fuse_fc1_activation must be bool")
        g8 = kwargs.get("gate_up_layout", "split") == "g8"
        if g8 and (self.transport != "peer" or ffn_fusion or (dag_fusion and not (dag_shared_pool and fuse_fc1_activation)) or (dispatch_pipeline and not inline_dispatch and not dispatch_external)
                   or return_pipeline):
            raise ValueError("G8 supports direct/producer peer paths or the shared task pool with FC1 activation fusion")
        if fuse_fc1_activation and (not g8 or not expert_local_grad or not fuse_dispatch or not fuse_return
                                   or dispatch_cache_n or ffn_fusion):
            raise ValueError("FC1 activation fusion requires G8, bilateral fusion and expert-local gradients")
        self.fuse_fc1_activation = fuse_fc1_activation
        if not all(isinstance(value, bool) for value in (dag_fusion, dag_serialize, dag_trace, dag_shared_pool,
                                                       dag_input_prefetch, dag_backward, dag_direct_return)):
            raise TypeError("DAG switches must be bool")
        if any(type(value) is not int or value < 1 for value in (dag_input_ctas, dag_return_ctas)):
            raise ValueError("DAG communication CTA counts must be positive integers")
        if dag_fusion and (not expert_local_grad or not fuse_dispatch or not fuse_return
                           or ffn_fusion or dispatch_producer or dispatch_pipeline or return_pipeline
                           or inline_dispatch or dispatch_cache_n or staged_return or return_vector_bytes != 4):
            raise ValueError("DAG fusion requires direct bilateral fusion and expert-local gradients without other pipelines")
        self.dag_fusion = dag_fusion
        if dag_backward and (not g8 or not dag_shared_pool or not expert_local_grad
                             or not fuse_activation_backward or not fuse_dispatch or not fuse_return
                             or ring_rows is None or ffn_fusion or dispatch_cache_n or dispatch_pipeline
                             or return_pipeline or inline_dispatch or staged_return or return_vector_bytes != 4):
            raise ValueError("backward DAG requires G8 shared-pool routed windows and fused activation gradients")
        if dag_direct_return and not (dag_shared_pool and (dag_fusion or dag_backward)):
            raise ValueError("DAG direct return requires an active shared task pool")
        self.dag_backward, self.dag_direct_return = dag_backward, dag_direct_return
        self.dag_input_ctas, self.dag_return_ctas = dag_input_ctas, dag_return_ctas
        self.dag_serialize, self.dag_trace = dag_serialize, dag_trace
        self.dag_shared_pool = dag_shared_pool
        self.dag_input_prefetch = dag_input_prefetch
        if dag_input_prefetch and not dag_shared_pool:
            raise ValueError("input prefetch credits require the shared DAG task pool")
        if dag_shared_pool and not (dag_fusion or dag_backward):
            raise ValueError("shared DAG task pools require forward or backward DAG execution")
        if type(return_vector_bytes) is not int or return_vector_bytes not in (4, 8):
            raise ValueError("return_vector_bytes must be 4 or 8")
        if return_vector_bytes != 4 and (not fuse_return or return_pipeline or staged_return or ffn_fusion):
            raise ValueError("eight-byte return stores require direct fused return without staging/resident/FFN modes")
        self.return_vector_bytes = return_vector_bytes
        block_m = kwargs.get("block_m", 128)
        if block_m != 128 and (self.transport != "peer" or ffn_fusion or (dispatch_pipeline and not inline_dispatch and not dispatch_external)
                                or return_pipeline or staged_return or dispatch_cache_n):
            raise ValueError("BM256 currently supports direct/producer peer GEMMs without resident/cache/staging modes")
        if "packed_rows" in kwargs:
            raise ValueError("use ring_rows to enable routed-row scheduling")
        if ring_rows is not None:
            if not expert_local_grad or ffn_fusion or (dispatch_pipeline and not inline_dispatch and not dispatch_external) or return_pipeline:
                raise ValueError("routed-row windows require expert-local gradients without a resident pipeline")
        self.ring_rows = ring_rows
        if kwargs.get("compute", "bf16") != "bf16":
            raise ValueError("peer-fused GEMMs currently use BF16 compute")
        if not isinstance(fuse_dispatch, bool) or not isinstance(fuse_return, bool):
            raise TypeError("peer fusion switches must be bool")
        self.fuse_dispatch, self.fuse_return = fuse_dispatch, fuse_return
        if not isinstance(expert_local_grad, bool):
            raise TypeError("expert_local_grad must be bool")
        if expert_local_grad and (ffn_fusion or not reuse_packed_dy or not fuse_local_grad):
            raise ValueError("expert-local gradients require packed DY reuse and local-gradient fusion without whole-FFN fusion")
        self.expert_local_grad = expert_local_grad
        if not isinstance(ffn_fusion, bool):
            raise TypeError("ffn_fusion must be bool")
        if ffn_fusion and (not fuse_dispatch or not fuse_return or not fuse_activation_backward
                           or not reuse_packed_dy or dispatch_producer or dispatch_cache_n
                           or dispatch_pipeline or return_pipeline or inline_dispatch
                           or staged_return or cache_return_addresses):
            raise ValueError("FFN fusion requires direct bilateral fusion, packed DY reuse and fused activation backward")
        self.ffn_fusion = ffn_fusion
        if not isinstance(dispatch_producer, bool):
            raise TypeError("dispatch_producer must be bool")
        if dispatch_producer and (not fuse_dispatch or self.transport != "peer" or dispatch_cache_n):
            raise ValueError("producer dispatch requires fused peer dispatch without a private N cache")
        if type(dispatch_producer_tiles) is not int or dispatch_producer_tiles not in (1, 2, 4):
            raise ValueError("dispatch_producer_tiles must be 1, 2 or 4")
        self.dispatch_producer, self.dispatch_producer_tiles = dispatch_producer, dispatch_producer_tiles
        if not isinstance(producer_local_aiu, bool):
            raise TypeError("producer_local_aiu must be bool")
        if producer_local_aiu and not dispatch_producer:
            raise ValueError("local AIU input requires producer dispatch")
        self.producer_local_aiu = producer_local_aiu
        if not isinstance(producer_input_aiu, bool):
            raise TypeError("producer_input_aiu must be bool")
        if producer_input_aiu and (not dispatch_producer or dispatch_producer_tiles != 1):
            raise ValueError("producer input AIU requires exactly one producer N tile")
        self.producer_input_aiu = producer_input_aiu
        if not isinstance(inline_dispatch, bool) or not isinstance(cache_return_addresses, bool):
            raise TypeError("inline dispatch and cached-return switches must be bool")
        if inline_dispatch:
            dispatch_pipeline = True
        self.inline_dispatch = inline_dispatch
        self.cache_return_addresses = cache_return_addresses
        if not isinstance(dispatch_pipeline, bool) or not isinstance(pipeline_statistics, bool):
            raise TypeError("pipeline switches must be bool")
        if dispatch_pipeline and (not fuse_dispatch or self.transport != "peer"
                                  or dispatch_producer or dispatch_cache_n):
            raise ValueError("dispatch pipeline requires fused peer dispatch without producer/cache modes")
        if type(pipeline_comm_ctas) is not int or pipeline_comm_ctas < 1:
            raise ValueError("pipeline_comm_ctas must be a positive integer")
        self.dispatch_pipeline = dispatch_pipeline
        self.pipeline_comm_ctas = pipeline_comm_ctas
        self.collect_pipeline_statistics = pipeline_statistics
        if not isinstance(pipeline_serialize, bool):
            raise TypeError("pipeline_serialize must be bool")
        self.pipeline_serialize = pipeline_serialize
        if inline_dispatch and pipeline_serialize:
            raise ValueError("inline input owners require tile readiness; global dispatch serialization is unsupported")
        if not isinstance(return_pipeline, bool):
            raise TypeError("return_pipeline must be bool")
        if return_pipeline and (not fuse_return or self.transport != "peer" or staged_return):
            raise ValueError("return pipeline requires fused peer return without staged-return mode")
        self.return_pipeline = return_pipeline
        if cache_return_addresses and (not fuse_return or return_pipeline or staged_return):
            raise ValueError("cached return addresses require direct fused return without a return pipeline or staging")
        if not isinstance(active_rows, bool) or not isinstance(fuse_local_grad, bool):
            raise TypeError("peer pointwise switches must be bool")
        self.active_rows, self.fuse_local_grad = active_rows, fuse_local_grad
        if not isinstance(fuse_activation_backward, bool):
            raise TypeError("fuse_activation_backward must be bool")
        self.fuse_activation_backward = fuse_activation_backward
        if not isinstance(vectorized_copy, bool):
            raise TypeError("vectorized_copy must be bool")
        self.vectorized_copy = vectorized_copy
        if type(dispatch_cache_n) is not int or dispatch_cache_n not in (0, 2, 4, 8, 16):
            raise ValueError("dispatch_cache_n must be 0, 2, 4, 8 or 16")
        if dispatch_cache_n and (not fuse_dispatch or self.transport != "peer"):
            raise ValueError("dispatch caching requires the peer fused-dispatch path")
        self.dispatch_cache_n = dispatch_cache_n
        if type(dispatch_cache_ctas) is not int or dispatch_cache_ctas not in (1, 2, 4):
            raise ValueError("dispatch_cache_ctas must be 1, 2 or 4")
        self.dispatch_cache_ctas = dispatch_cache_ctas
        if not isinstance(dispatch_cache_aiu, bool):
            raise TypeError("dispatch_cache_aiu must be bool")
        if dispatch_cache_aiu and not dispatch_cache_n:
            raise ValueError("AIU cache reads require dispatch_cache_n > 0")
        self.dispatch_cache_aiu = dispatch_cache_aiu
        if not isinstance(staged_return, bool):
            raise TypeError("staged_return must be bool")
        self.staged_return = staged_return and fuse_return
        self.staged_return_row_padding = 16 if self.staged_return else 0
        if not isinstance(reuse_packed_dy, bool):
            raise TypeError("reuse_packed_dy must be bool")
        if reuse_packed_dy and self.transport != "peer":
            raise ValueError("packed gradient reuse is a peer-plan option")
        self.reuse_packed_dy = reuse_packed_dy and fuse_dispatch
        super().__init__(*args, packed_rows=ring_rows, **kwargs)
        if self.overlap_expert_dy:
            self.packed_raw_dy = torch.empty_like(self.packed_dy)
            self.buffers["packed_raw_dy"] = self.packed_raw_dy
        if self.dispatch_assist_ctas:
            self._input_stream = torch.cuda.Stream(device=self.device)
            self.input_assisted_tiles = torch.zeros((), device=self.device, dtype=torch.int32)
            self.buffers["input_assisted_tiles"] = self.input_assisted_tiles
        if self.group is not None:
            recipe = (self.fuse_dispatch, self.fuse_return, self.ffn_fusion, self.expert_local_grad,
                      self.dag_fusion, self.dag_backward, self.dag_direct_return, self.dag_shared_pool,
                      self.dag_input_prefetch, self.dag_serialize, self.input_copy_ilp,
                      self.inline_dispatch, self.dispatch_assist_ctas, self.dispatch_external)
            recipes = [None] * self.size
            dist.all_gather_object(recipes, recipe, group=self.group)
            if any(other != recipe for other in recipes):
                raise ValueError("EP ranks must agree on the peer gradient and return protocol")
        if gemm_configs is None and self.block_m == 256:
            gemm_configs = {role: (256, 64, 64, 64, 2) for role in ("fc1", "fc2", "dhidden", "dx")}
        self.gemm_configs = {} if gemm_configs is None else {name: tuple(value) for name, value in gemm_configs.items()}
        if set(self.gemm_configs) - {"fc1", "fc2", "dhidden", "dx"}:
            raise ValueError("unknown peer GEMM configuration role")
        if ffn_fusion and len({self.gemm_configs.get(role, (256, 64, 64, 64, 2))
                               for role in ("fc1", "fc2", "dhidden", "dx")}) != 1:
            raise ValueError("FFN fusion currently requires matching four GEMM configurations")
        if self.size > 64:
            raise ValueError("peer training supports at most 64 ranks per domain")
        c, h, k = self.chunk_tokens, self.h, self.k
        descriptions = (("local_x", (c, h), torch.bfloat16),
                        ("local_ids", (c, k), torch.int64),
                        ("local_values", (c * k, h), torch.bfloat16),
                        ("local_dx", (c * k, h), torch.bfloat16),
                        ("local_dy", (c * k, h), torch.bfloat16),
                        ("barrier_state", (4,), torch.int32))
        if self.expert_local_grad:
            descriptions += (("raw_dy", (c, h), torch.bfloat16),
                             ("raw_scores", (c, k), torch.float32),
                             ("local_dscores", (c, k), torch.float32))
        layout, size = {}, 0
        for name, shape, dtype in descriptions:
            size = (size + 15) // 16 * 16
            count = math.prod(shape)
            element = 2 if dtype == torch.bfloat16 else 8 if dtype == torch.int64 else 4
            layout[name] = (size, count * element, shape, dtype)
            size += count * element
        self.symm_handle = None
        if self.size == 1:
            self.payload = torch.empty(size, device=self.device, dtype=torch.uint8)
            self.peer_buffers = [self.payload]
        else:
            from torch.distributed import _symmetric_memory as sm

            self.payload = sm.empty((size,), dtype=torch.uint8, device=self.device)
            self.symm_handle = sm.rendezvous(self.payload, self.group)
            direct = getattr(self.symm_handle, "world_within_direct_access", None)
            direct = direct() if callable(direct) else direct
            # The native PPU Torch 2.10 CUDA backend imports/maps every peer
            # allocation through the driver, but omits the newer topology flag.
            # Accept that direct-buffer backend after successful rendezvous;
            # do not infer direct pointers for an unknown/RDMA memory backend.
            if direct is None and sm.get_backend(self.device) != "CUDA":
                raise RuntimeError("symmetric-memory backend cannot supply verified direct buffers; use NativeMoEPlan")
            if direct is not None and not direct:
                raise RuntimeError("peer training requires an EP group within one direct-access domain")
            self.peer_buffers = [self.symm_handle.get_buffer(rank, (size,), torch.uint8) for rank in range(self.size)]
        self.payload.zero_()
        self.pointer_tables = {}
        for name, (offset, length, shape, dtype) in layout.items():
            view = self.payload[offset:offset + length].view(dtype).reshape(shape)
            setattr(self, name, view)
            self.buffers[name] = view
            addresses = [buffer.data_ptr() + offset for buffer in self.peer_buffers]
            encoded = [value if value < 2**63 else value - 2**64 for value in addresses]
            table = torch.tensor(encoded, device=self.device, dtype=torch.int64)
            self.pointer_tables[name] = table
            self.buffers[name + "_peers"] = table
        for name in ("gathered_x", "gathered_ids", "route_values", "group_ids"):
            self.buffers.pop(name)
            delattr(self, name)
        self.source_rows = torch.empty(self.rows, device=self.device, dtype=torch.int32)
        self.identity_rows = torch.arange(self.rows, device=self.device, dtype=torch.int32)
        self.work_info = torch.empty((self.rows // self.block_m, 4), device=self.device, dtype=torch.int32)
        self.num_blocks = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.buffers.update(source_rows=self.source_rows, identity_rows=self.identity_rows,
                            work_info=self.work_info, num_blocks=self.num_blocks)
        self.route_window_count = 1
        if self.ring_rows is not None:
            allocate_route_windows(self)
        self.dag_controls = None
        self._dag_slot = 0
        if dag_fusion or dag_backward:
            config = self.gemm_configs.get("fc1" if dag_fusion else "dhidden", (256, 64, 64, 64, 2))
            roles = (("fc1", "fc2") if dag_fusion else ()) + (("dhidden", "dx") if dag_backward else ())
            if any(self.gemm_configs.get(role, config) != config for role in roles):
                raise ValueError("GEMMs in the same DAG plan require matching tiles")
            bn, bk, wm, wn, stages = config
            if bn not in (128, 256) or bk != 64 or wm != 64 or wn != 64 or stages not in (2, 3, 4):
                raise ValueError("unsupported DAG GEMM tile")
            if self.i % bn or self.h % bn:
                raise ValueError("DAG hidden and intermediate dimensions must be divisible by BN")
            self.dag_config = config
            tiles, pairs, n2 = self.rows // self.block_m, self.i // bn, self.h // bn
            windows = math.ceil(self.max_tokens / self.chunk_tokens) * self.route_window_count
            self.dag_forward_slots = windows if dag_fusion else 0
            self.dag_slots = ((2 if dag_fusion else 0) + (1 if dag_backward else 0)) * windows
            sample_capacity = 8192 if dag_trace else 1
            descriptions = (("state", (18,), torch.int32),
                            ("input_ready", (tiles * (1 + 2*pairs if dag_shared_pool else 1),), torch.int32),
                            ("pair_done", (tiles * pairs,), torch.int32),
                            ("activated_pairs", (tiles,), torch.int32),
                            ("queue", (tiles * n2,), torch.int32),
                            ("output_ready", (tiles * n2 * (2 if dag_shared_pool else 1),), torch.int32),
                            ("monitor", (self.dag_slots, 12), torch.int64),
                            ("samples", (self.dag_slots, sample_capacity, 6), torch.int64))
            self.dag_controls = []
            for name, shape, dtype in descriptions:
                value = torch.zeros(shape, device=self.device, dtype=dtype)
                setattr(self, "dag_" + name, value)
                self.buffers["dag_" + name] = value
                self.dag_controls.append(value)
        self.dispatch_scratch = None
        self.pipeline_ready = self.return_pipeline_ready = self.pipeline_statistics = None
        self.return_pipeline_statistics = None
        self.dispatch_pipeline_ctas = self.return_pipeline_ctas = 0
        if dispatch_pipeline or return_pipeline:
            ctas = dg.get_num_sms() - 1
            if ctas < 1:
                raise ValueError("pipeline needs at least two CUs; occupancy is checked during preparation")
            if dispatch_pipeline:
                self.pipeline_ready = torch.zeros(self.rows // self.block_m + 1, device=self.device, dtype=torch.int32)
                self.buffers["pipeline_ready"] = self.pipeline_ready
            if return_pipeline:
                minimum_bn = min(self.gemm_configs.get(role, (128,))[0] for role in ("fc2", "dx"))
                ntiles = (self.h + minimum_bn - 1) // minimum_bn
                self.return_pipeline_ready = torch.zeros(self.rows // 128 * ntiles + 1,
                                                        device=self.device, dtype=torch.int32)
                self.buffers["return_pipeline_ready"] = self.return_pipeline_ready
        if self.dispatch_cache_n:
            cus = torch.cuda.get_device_properties(self.device).multi_processor_count
            self.dispatch_scratch = torch.empty((cus * self.dispatch_cache_ctas * 128, self.h),
                                                device=self.device, dtype=torch.bfloat16)
            self.buffers["dispatch_scratch"] = self.dispatch_scratch
        self.gate_up.zero_()
        self.hidden.zero_()
        self._prepared = False
        torch.cuda.synchronize(self.device)
        if self.group is not None:
            dist.barrier(group=self.group)

    def _barrier(self, tag, compile_only=False):
        _peer(5, [self.pointer_tables["barrier_state"], self.barrier_state, self.status],
              [self.size, self.rank, 180 * 2_000_000_000, tag], compile_only)

    def _peer_gemm(self, a, weight, output, source_rows, *, read=False, write=False,
                   source="local_x", destination="local_values", compile_only=False, pack_input=False, role=None,
                   first_n_tile=0, num_n_tiles=0, local_input_aiu=False, producer_input_aiu=False,
                   dispatch_pipeline=False, keep_gate_up=True):
        source_stride = self.chunk_tokens if source == "local_x" else self.chunk_tokens * self.k
        if role is None:
            role = ("fc1" if source == "local_x" else "dhidden") if read else ("dx" if destination == "local_dx" else "fc2")
        return_pipeline = write and self.return_pipeline
        fc1_activation = role == "fc1" and self.fuse_fc1_activation
        interleaved = self.g8_interleaved and (fc1_activation or (role == "dhidden" and self.fuse_activation_backward))
        if return_pipeline and not compile_only:
            self.return_pipeline_ready.zero_()
        grid = dg._C._moe_training_peer_gemm(
            a, weight, output, self.pointer_tables[source], self.pointer_tables[destination],
            source_rows, self.permutation, self.work_info, self.num_blocks,
            source_stride, self.chunk_tokens * self.k, read, write, compile_only,
            pack_input, self.gemm_configs.get(role, ()),
            self.hidden if fc1_activation else self.gate_up if role == "dhidden" and self.fuse_activation_backward else None,
            self.dispatch_cache_n if read else 0, self.dispatch_scratch if read else None,
            self.dispatch_cache_aiu if read else False,
            self.staged_return if write else False,
            self.dispatch_cache_ctas,
            first_n_tile, num_n_tiles, local_input_aiu, producer_input_aiu,
            self.return_pipeline_ready if return_pipeline else self.pipeline_ready if dispatch_pipeline else None,
            self.status if dispatch_pipeline or return_pipeline else None,
            (self.return_pipeline_statistics if return_pipeline else self.pipeline_statistics if dispatch_pipeline else None)
                if not compile_only else None,
            0 if dispatch_pipeline and (self.inline_dispatch or self.dispatch_external) else self.pipeline_comm_ctas,
            self.pipeline_serialize,
            self.cache_return_addresses if write else False,
            self.block_m,
            self.return_vector_bytes if write else 4,
            interleaved, fc1_activation, keep_gate_up if fc1_activation else True,
            self.input_copy_ilp if producer_input_aiu or (dispatch_pipeline and self.inline_dispatch) else 1,
            dispatch_pipeline and self.dispatch_external,
        )
        if compile_only and (dispatch_pipeline or return_pipeline):
            grid_name = "return_pipeline_ctas" if return_pipeline else "dispatch_pipeline_ctas"
            setattr(self, grid_name, max(grid, getattr(self, grid_name)))
            if self.collect_pipeline_statistics:
                name = "return_pipeline_statistics" if return_pipeline else "pipeline_statistics"
                previous = getattr(self, name)
                if previous is None or previous.size(0) < grid:
                    statistics = torch.zeros((grid, 4), device=self.device, dtype=torch.int64)
                    setattr(self, name, statistics)
                    self.buffers[name] = statistics

    def _fc1(self, w1, *, pack_input=False, compile_only=False, keep_gate_up=True, prefetch_dy=False):
        if self.dispatch_pipeline:
            if not compile_only:
                self.pipeline_ready.zero_()
            if self.dispatch_assist_ctas and not compile_only:
                self.input_assisted_tiles.zero_()
                self._input_stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(self._input_stream):
                    for tensor in (self.pointer_tables["local_x"], self.source_rows, self.packed_x,
                                   self.work_info, self.num_blocks, self.pipeline_ready, self.status,
                                   self.input_assisted_tiles):
                        tensor.record_stream(self._input_stream)
                    self._assist_input()
            self._peer_gemm(self.packed_x, w1, self.gate_up, self.source_rows,
                            role="fc1", compile_only=compile_only, dispatch_pipeline=True, keep_gate_up=keep_gate_up)
            if self.dispatch_assist_ctas and not compile_only:
                torch.cuda.current_stream(self.device).wait_stream(self._input_stream)
        elif self.dispatch_producer:
            bn = self.gemm_configs.get("fc1", (128, 64, 64, 64, 3))[0]
            ntiles = (2 * self.i + bn - 1) // bn
            produced = min(ntiles, self.dispatch_producer_tiles)
            # The first GEMM slice dispatches X and computes real FC1 columns.
            # Stream ordering then publishes packed X to a local GEMM over
            # the remaining columns. No cross-CTA wait or dispatch-only kernel.
            self._peer_gemm(self.packed_x, w1, self.gate_up, self.source_rows,
                            read=True, pack_input=True, role="fc1", compile_only=compile_only,
                            first_n_tile=0, num_n_tiles=produced,
                            producer_input_aiu=self.producer_input_aiu, keep_gate_up=keep_gate_up)
            # X dispatch is complete. Move raw DY while the remaining FC1
            # columns and FC2 replay compute, avoiding competition with X.
            if prefetch_dy and not compile_only:
                self._start_gradient_prefetch()
            if produced < ntiles:
                self._peer_gemm(self.packed_x, w1, self.gate_up, self.identity_rows,
                                role="fc1", compile_only=compile_only, first_n_tile=produced,
                                local_input_aiu=self.producer_local_aiu, keep_gate_up=keep_gate_up)
        else:
            self._peer_gemm(self.packed_x, w1, self.gate_up,
                            self.source_rows if self.fuse_dispatch else self.identity_rows,
                            read=self.fuse_dispatch, pack_input=pack_input and self.fuse_dispatch,
                            role="fc1", compile_only=compile_only, keep_gate_up=keep_gate_up)

    def _assist_input(self, compile_only=False):
        _peer(16, [self.pointer_tables["local_x"], self.source_rows, self.packed_x,
                   self.work_info, self.num_blocks, self.pipeline_ready, self.status,
                   self.input_assisted_tiles],
              [self.rows, self.h, self.chunk_tokens, self.size, self.block_m,
               self.dispatch_assist_ctas, self.input_copy_ilp], compile_only)

    @torch.no_grad()
    def prepare(self, w1, w2):
        """Compile all peer GEMM variants before entering device peer barriers."""
        with self._execution():
            if not self._prepared:
                error = None
                try:
                    self._prepare(w1, w2)
                except Exception as exception:
                    error = exception
                self._finish_preparation(error)

    def _prepare(self, w1, w2):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up PeerMoEPlan before graph capture")
        self._prepare_pointwise()
        if self.dispatch_assist_ctas:
            self._assist_input(compile_only=True)
        w1, w2 = w1.to(torch.bfloat16), w2.to(torch.bfloat16)
        self._prepare_weights(w1, w2, backward=True)
        if self.ffn_fusion:
            self._ffn(w1, w2, compile_only=True)
            self._ffn(self.w2_t, self.w1_t, backward=True, compile_only=True)
        if self.dag_fusion:
            self._dag(w1, w2, return_output=True, compile_only=True)
            self._dag(w1, w2, return_output=False, compile_only=True)
        if self.dag_backward:
            self._dag(self.w2_t, self.w1_t, backward=True, compile_only=True)
        if self.expert_local_grad:
            self._expert_gradients(compile_only=True)
        if self.overlap_expert_dy:
            self._prefetch_raw_dy(compile_only=True)
        if self.ring_rows is not None:
            set_route_window(self, 0, compile_only=True)
        self._barrier(0, compile_only=True)
        _peer(0, [self.pointer_tables["local_ids"], self.counts, self.route_experts, self.route_rows, self.status],
              [self.routes, self.e, self.rank, self.size, self.chunk_tokens * self.k], True)
        _peer(1, [self.permutation, self.source_rows], [self.rows, self.size * self.chunk_tokens], True)
        _peer(2, [self.route_experts, self.route_rows, self.offsets, self.permutation, self.source_rows, self.status],
              [self.routes, self.rows, self.k], True)
        _peer(3, [self.offsets, self.lengths, self.work_info, self.num_blocks], [self.e, self.rows, self.block_m], True)
        _peer(4, [self.pointer_tables["local_x"], self.permutation, self.packed_x, self.status],
              [self.rows, self.h, self.chunk_tokens, self.k, self.size], True)
        if not self.fuse_dispatch:
            self._stage_input(compile_only=True)
        if self.active_rows:
            self._activation(compile_only=True)
            self._activation(backward=True, compile_only=True)
            self._pack_dy(compile_only=True)
        if self.fuse_local_grad:
            empty_dy = torch.empty((0, self.h), device=self.device, dtype=torch.bfloat16)
            empty_scores = torch.empty((0, self.k), device=self.device, dtype=torch.float32)
            self._local_gradients(empty_dy, empty_scores, empty_scores, 0, compile_only=True)
        self._fc1(w1, compile_only=True)
        if self.fuse_fc1_activation:
            self._fc1(w1, compile_only=True, keep_gate_up=False)
        self._peer_gemm(self.hidden, w2, self.packed_output, self.identity_rows,
                        write=self.fuse_return, compile_only=True)
        read_dy = self.fuse_dispatch and not self.reuse_packed_dy
        self._peer_gemm(self.packed_dy, self.w2_t, self.hidden,
                        self.permutation if read_dy else self.identity_rows,
                        read=read_dy, source="local_dy", compile_only=True, role="dhidden")
        self._peer_gemm(self.gate_up, self.w1_t, self.packed_output, self.identity_rows,
                        write=self.fuse_return, destination="local_dx", compile_only=True)
        if not self.fuse_return:
            self._scatter_peer("local_values", compile_only=True)
        self.lengths.zero_()
        dg.k_grouped_bf16_gemm_tn_device(self.gate_up, self.packed_x,
                                       torch.zeros_like(w1, dtype=torch.float32), self.lengths, self.rows)
        dg.k_grouped_bf16_gemm_tn_device(self.packed_dy, self.hidden,
                                       torch.zeros_like(w2, dtype=torch.float32), self.lengths, self.rows)
        torch.cuda.synchronize(self.device)

    def _route_chunk(self, x, ids, valid_tokens, start, zero_values=True):
        _launch(13, [x, ids, self.local_x, self.local_ids, valid_tokens, self.status],
                [x.shape[0], self.chunk_tokens, self.h, self.k, start])
        if zero_values:
            self.local_values.zero_()
        self.local_dx.zero_()
        self._barrier(start * 4)
        self.counts.zero_()
        _peer(0, [self.pointer_tables["local_ids"], self.counts, self.route_experts, self.route_rows, self.status],
              [self.routes, self.e, self.rank, self.size, self.chunk_tokens * self.k])
        offsets = self.offsets if self.ring_rows is None else self.global_offsets
        lengths = self.lengths if self.ring_rows is None else self.global_lengths
        permutation = self.permutation if self.ring_rows is None else self.global_permutation
        source_rows = self.source_rows if self.ring_rows is None else self.global_source_rows
        capacity = self.rows if self.ring_rows is None else self.full_row_capacity
        _launch(1, [self.counts, offsets, lengths], [self.e, self.alignment])
        _peer(1, [permutation, source_rows], [capacity, self.size * self.chunk_tokens])
        _peer(2, [self.route_experts, self.route_rows, offsets, permutation, source_rows, self.status],
              [self.routes, capacity, self.k])
        if self.ring_rows is not None:
            return
        _peer(3, [self.offsets, self.lengths, self.work_info, self.num_blocks], [self.e, self.rows, self.block_m])

    def _chunk_forward(self, w1, w2, pack_input=False, return_output=True):
        if self.ring_rows is not None:
            if not return_output:
                raise RuntimeError("backward routed-row windows own their replay schedule")
            for start in range(0, self.full_row_capacity, self.rows):
                self._set_window(start)
                self._chunk_forward_single(w1, w2, pack_input=pack_input, synchronize_output=False)
            self._barrier(1)
            return
        self._chunk_forward_single(w1, w2, pack_input=pack_input, return_output=return_output)

    def _set_window(self, start):
        set_route_window(self, start)
        _peer(3, [self.offsets, self.lengths, self.work_info, self.num_blocks], [self.e, self.rows, self.block_m])

    def _chunk_forward_single(self, w1, w2, pack_input=False, return_output=True, synchronize_output=True,
                              prefetch_dy=False):
        if self.dag_fusion:
            self._dag(w1, w2, return_output=return_output)
            if return_output and synchronize_output:
                self._barrier(1)
            return
        if self.ffn_fusion:
            self._ffn(w1, w2)
            self._barrier(1)
            return
        if not self.fuse_dispatch:
            self._stage_input()
        self._fc1(w1, pack_input=pack_input, keep_gate_up=not return_output, prefetch_dy=prefetch_dy)
        if not self.fuse_fc1_activation:
            self._activation()
        self._peer_gemm(self.hidden, w2, self.packed_output, self.identity_rows,
                        write=self.fuse_return and return_output)
        if not return_output:
            return
        if not self.fuse_return:
            self._scatter_peer("local_values")
        if synchronize_output:
            self._barrier(1)

    def _dag(self, w1, w2, return_output=True, compile_only=False, backward=False):
        slot = 0 if compile_only else self._dag_slot
        if not compile_only:
            if slot >= self.dag_slots:
                raise RuntimeError("DAG trace slot schedule exceeds the fixed plan")
            self._dag_slot += 1
            for value in self.dag_controls[:6]:
                value.zero_()
            if self.dag_trace:
                self.dag_monitor[slot].zero_()
        controls = [*self.dag_controls[:6], self.dag_monitor[slot], self.dag_samples[slot], self.status]
        tensors = [self.packed_dy if backward else self.packed_x,
                   w1, self.gate_up, self.hidden, w2, self.packed_output,
                   self.pointer_tables["raw_dy" if backward else "local_x"],
                   self.pointer_tables["local_dx" if backward else "local_values"],
                   self.source_rows, self.permutation, self.work_info, self.num_blocks]
        if backward:
            tensors.extend([self.pointer_tables["raw_scores"], self.pointer_tables["local_dscores"]])
        self.dag_grid = dg._C._moe_training_peer_dag(
            tensors,
            controls, self.chunk_tokens, self.chunk_tokens * self.k, self.block_m,
            self.dag_config, self.dag_input_ctas, self.dag_return_ctas, return_output,
            self.dag_serialize, self.dag_trace, compile_only, self.dag_shared_pool,
            self.fuse_fc1_activation and not backward, self.dag_input_prefetch,
            backward, self.dag_direct_return, 1 if backward else self.input_copy_ilp)

    def _ffn(self, first_weight, second_weight, backward=False, compile_only=False):
        return dg._C._moe_training_peer_ffn(
            [self.packed_dy if backward else self.packed_x, first_weight,
             self.gate_up, self.hidden, second_weight, self.packed_output,
             self.pointer_tables["local_x"],
             self.pointer_tables["local_dx" if backward else "local_values"],
             self.source_rows, self.permutation, self.work_info, self.num_blocks],
            self.chunk_tokens, self.chunk_tokens * self.k, backward,
            self.gemm_configs.get("fc1", (256, 64, 64, 64, 2)), compile_only)

    def _scatter_peer(self, destination, compile_only=False):
        _peer(12 if self.vectorized_copy else 6,
              [self.packed_output, self.permutation, self.pointer_tables[destination], self.status, self.num_blocks],
              [self.rows, self.h, self.chunk_tokens * self.k, self.size, self.block_m], compile_only)

    def _stage_input(self, compile_only=False):
        _peer(11 if self.vectorized_copy else 7,
              [self.pointer_tables["local_x"], self.permutation, self.packed_x, self.status, self.num_blocks],
              [self.rows, self.h, self.chunk_tokens, self.k, self.size, self.block_m], compile_only)

    def _activation(self, backward=False, compile_only=False):
        if self.active_rows:
            _peer(9 if backward else 8, [self.gate_up, self.hidden, self.num_blocks, self.status],
                  [self.rows, self.i, self.block_m] + ([8] if self.g8_interleaved else []), compile_only)
        elif not compile_only:
            tensors = [self.gate_up, self.hidden] + ([self.gate_up] if backward else [])
            _launch(8 if backward else 7, tensors, [self.rows, self.i] + ([8] if self.g8_interleaved else []))

    def _pack_dy(self, compile_only=False):
        tensors = [self.pointer_tables["local_dy"], self.permutation, self.packed_dy, self.status]
        if self.active_rows:
            tensors.append(self.num_blocks)
        operation = (11 if self.vectorized_copy else 7) if self.active_rows else 4
        _peer(operation, tensors,
              [self.rows, self.h, self.chunk_tokens * self.k, 1, self.size]
              + ([self.block_m] if self.active_rows else []), compile_only)

    def _local_gradients(self, dy, scores, dscores, start, compile_only=False):
        _peer(10, [self.local_values, dy, scores, dscores, self.local_dy],
              [self.chunk_tokens, self.h, self.k, dy.shape[0], start], compile_only)

    def _expert_gradients(self, compile_only=False):
        tensors = [self.pointer_tables["raw_dy"], self.pointer_tables["raw_scores"],
                   self.pointer_tables["local_dscores"], self.permutation,
                   self.packed_output, self.packed_dy, self.num_blocks, self.status]
        if self.overlap_expert_dy:
            if not compile_only:
                self._join_gradient_prefetch()
            tensors.append(self.packed_raw_dy)
        _peer(13, tensors,
              [self.rows, self.h, self.chunk_tokens, self.k, self.size, self.block_m], compile_only)

    def _prefetch_raw_dy(self, compile_only=False):
        _peer(15, [self.pointer_tables["raw_dy"], self.permutation, self.packed_raw_dy,
                   self.num_blocks, self.status],
              [self.rows, self.h, self.chunk_tokens, self.k, self.size, self.block_m], compile_only)

    def _start_gradient_prefetch(self):
        self._gradient_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._gradient_stream):
            for tensor in (self.pointer_tables["raw_dy"], self.permutation, self.packed_raw_dy,
                           self.num_blocks, self.status):
                tensor.record_stream(self._gradient_stream)
            self._prefetch_raw_dy()
        self._gradient_prefetch_pending = True

    def _publish_gradient_chunk(self, dy, scores, start):
        count = max(0, min(self.chunk_tokens, dy.shape[0] - start))
        self.raw_dy.zero_()
        self.raw_scores.zero_()
        self.local_dscores.zero_()
        self.raw_dy[:count].copy_(dy[start:start + count])
        self.raw_scores[:count].copy_(scores[start:start + count])

    def _backward_windows(self, dy, scores, w1, w2, dw1, dw2, dx, dscores, start):
        for window in range(0, self.full_row_capacity, self.rows):
            self._set_window(window)
            self._chunk_forward_single(w1, w2, pack_input=True, return_output=False,
                                       prefetch_dy=self.overlap_expert_dy)
            if self.dag_backward:
                self._dag(self.w2_t, self.w1_t, backward=True)
                # The fused dHidden epilogue writes only gate/up gradients,
                # preserving forward hidden values for full DW2 accumulation.
                dg.k_grouped_bf16_gemm_tn_device(self.packed_dy, self.hidden, dw2, self.lengths, self.rows)
                dg.k_grouped_bf16_gemm_tn_device(self.gate_up, self.packed_x, dw1, self.lengths, self.rows)
            else:
                self._expert_gradients()
                self._wgrad(self.packed_dy, self.hidden, dw2,
                            asynchronous=self.wgrad_overlap == "both")
                self._peer_gemm(self.packed_dy, self.w2_t, self.hidden, self.identity_rows, role="dhidden")
                if not self.fuse_activation_backward:
                    self._activation(backward=True)
                self._wgrad(self.gate_up, self.packed_x, dw1,
                            asynchronous=self.wgrad_overlap != "none")
                self._peer_gemm(self.gate_up, self.w1_t, self.packed_output, self.identity_rows,
                                write=self.fuse_return, destination="local_dx")
                if not self.fuse_return:
                    self._scatter_peer("local_dx")
                # Both wgrads read window metadata/activations. Join before
                # any next-window overwrite and before entering rank barriers.
                self._join_wgrad()
        self._barrier(start * 4 + 3)
        _launch(6, [self.local_dx, scores, dx],
                [self.chunk_tokens, self.h, self.k, dy.shape[0], start, 0])
        count = max(0, min(self.chunk_tokens, dy.shape[0] - start))
        dscores[start:start + count].copy_(self.local_dscores[:count])

    @torch.no_grad()
    def forward(self, x, ids, scores, w1, w2, valid_tokens):
        with self._execution():
            self._dag_slot = 0
            self._validate(x, ids, scores, w1, w2, valid_tokens)
            self.prepare(w1, w2)
            return super().forward(x, ids, scores, w1, w2, valid_tokens)

    @torch.no_grad()
    def backward(self, dy, x, ids, scores, w1, w2, valid_tokens, accumulators=None):
        self._validate(x, ids, scores, w1, w2, valid_tokens)
        _validate_accumulators(accumulators, w1, w2)
        if dy.shape != x.shape or dy.device != x.device or dy.dtype != x.dtype or not dy.is_contiguous():
            raise ValueError("dy must be contiguous and match x")
        with self._execution():
            if self.dag_fusion or self.dag_backward:
                self._dag_slot = self.dag_forward_slots
            self.prepare(w1, w2)
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.reset_errors()
            self._prepare_weights(w1, w2, backward=True)
            dx, dscores = torch.empty_like(x), torch.empty_like(scores)
            dw1, dw2 = ((torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32))
                        if accumulators is None else accumulators)
            for start in range(0, self.max_tokens, self.chunk_tokens):
                if self.expert_local_grad:
                    self._publish_gradient_chunk(dy, scores, start)
                self._route_chunk(x, ids, valid_tokens, start, zero_values=not self.expert_local_grad)
                if self.ring_rows is not None:
                    self._backward_windows(dy, scores, w1, w2, dw1, dw2, dx, dscores, start)
                    continue
                self._chunk_forward(w1, w2, pack_input=True, return_output=not self.expert_local_grad)
                dims = [self.chunk_tokens, self.h, self.k, x.shape[0], start]
                if self.expert_local_grad:
                    self._expert_gradients()
                elif self.fuse_local_grad:
                    self._local_gradients(dy, scores, dscores, start)
                else:
                    _launch(12, [self.local_values, dy, dscores], dims)
                    _launch(11, [dy, scores, self.local_dy], dims)
                if not self.expert_local_grad:
                    self._barrier(start * 4 + 2)
                    self._pack_dy()
                dg.k_grouped_bf16_gemm_tn_device(self.packed_dy, self.hidden, dw2, self.lengths, self.rows)
                if self.ffn_fusion:
                    self._ffn(self.w2_t, self.w1_t, backward=True)
                    dg.k_grouped_bf16_gemm_tn_device(self.gate_up, self.packed_x, dw1, self.lengths, self.rows)
                    self._barrier(start * 4 + 3)
                    _launch(6, [self.local_dx, scores, dx], dims + [0])
                    continue
                # DW2 requires packed_dy above. Reuse that mandatory staging
                # result instead of fetching the same peer gradients again.
                read_dy = self.fuse_dispatch and not self.reuse_packed_dy
                self._peer_gemm(self.packed_dy, self.w2_t, self.hidden,
                                self.permutation if read_dy else self.identity_rows,
                                read=read_dy, source="local_dy", role="dhidden")
                if not self.fuse_activation_backward:
                    self._activation(backward=True)
                dg.k_grouped_bf16_gemm_tn_device(self.gate_up, self.packed_x, dw1, self.lengths, self.rows)
                self._peer_gemm(self.gate_up, self.w1_t, self.packed_output, self.identity_rows,
                                write=self.fuse_return, destination="local_dx")
                if not self.fuse_return:
                    self._scatter_peer("local_dx")
                self._barrier(start * 4 + 3)
                _launch(6, [self.local_dx, scores, dx], dims + [0])
                if self.expert_local_grad:
                    count = max(0, min(self.chunk_tokens, x.shape[0] - start))
                    dscores[start:start + count].copy_(self.local_dscores[:count])
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return dx, dw1, dw2, dscores
