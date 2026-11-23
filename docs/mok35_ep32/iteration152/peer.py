"""Native peer-read/peer-write Tensor Cell GEMMs for PPU MoE training."""

import math
import os

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
                 input_copy_ilp=1, dispatch_assist_ctas=0, dispatch_external=False, dispatch_external_serial=False,
                 fuse_fc1_activation=False, dgrad_compute_cus=0, fuse_replay_gradients=False, fuse_dhidden_scores=False, transpose_free_dgrad=False,
                 input_helper_threads=None, input_helper_block_m=None, input_helper_block_n=None,
                 fp8_weight_quantization=None, fp8_input_pipeline=None,
                 fp8_input_ctas=None, fp8_input_serialize=None, fp8_input_threads=None,
                 fp8_input_cache_scales=None, reuse_forward_context=None, deferred_dw_compute_cus=None, **kwargs):
        self._shared_fp8_weights_ready=False
        self._pending_last_dw1=None
        self._defer_last_dw1_enabled=False
        # Optional CTA helper recipe; environment values allow the same large
        # reference and capacity harnesses to exercise the exact candidate.
        def helper_value(value, key, default):
            value = int(os.environ.get(key, str(default))) if value is None else value
            if type(value) is not int:
                raise TypeError(key + " must be an integer")
            return value
        self.deferred_dw_compute_cus=helper_value(deferred_dw_compute_cus,'DG_MOE_TAIL_DW_COMPUTE_CUS',0)
        target_device=kwargs.get('device',None)
        max_cus=torch.cuda.get_device_properties(target_device).multi_processor_count
        if not 0<=self.deferred_dw_compute_cus<=max_cus:
            raise ValueError('tail DW compute budget is outside the device CU range')
        input_helper_threads = helper_value(input_helper_threads, "DG_MOE_INPUT_HELPER_THREADS", 0)
        if input_helper_threads not in (0, 32, 64):
            raise ValueError("input helper threads must be0,32or64")
        self.input_helper_threads = input_helper_threads
        self.input_helper_block_m = self.input_helper_block_n = 0
        if input_helper_threads:
            hm = helper_value(input_helper_block_m, "DG_MOE_INPUT_HELPER_BLOCK_M", 128)
            hn = helper_value(input_helper_block_n, "DG_MOE_INPUT_HELPER_BLOCK_N", 512)
            metadata_m = kwargs.get("block_m", 128)
            if (hm not in (128, 256) or hn not in (256, 512) or hm > metadata_m
                    or metadata_m % hm or (hn == 512 and hm != 128)):
                raise ValueError("unsupported input-helper compute/metadata geometry")
            if (not dispatch_producer or not producer_input_aiu or not producer_local_aiu
                    or not fuse_fc1_activation or not expert_local_grad or not reuse_packed_dy
                    or not fuse_dispatch or not fuse_return or ring_rows is None
                    or kwargs.get("gate_up_layout", "split") != "g8"
                    or dag_fusion or dag_backward or ffn_fusion or dispatch_pipeline
                    or return_pipeline or dispatch_assist_ctas or dispatch_external
                    or staged_return or dispatch_cache_n or kwargs.get("overlap_expert_dy", False)):
                raise ValueError("input helpers require the G8 producer bilateral routed-window recipe")
            self.input_helper_block_m, self.input_helper_block_n = hm, hn
        if not isinstance(transpose_free_dgrad, bool):
            raise TypeError("transpose_free_dgrad must be bool")
        if transpose_free_dgrad and (not expert_local_grad or not reuse_packed_dy
                or not fuse_dispatch or not fuse_return or not fuse_activation_backward
                or ring_rows is None or dag_fusion or dag_backward or ffn_fusion
                or dispatch_pipeline or return_pipeline or staged_return or dispatch_cache_n):
            raise ValueError("transpose-free dGrad requires routed bilateral expert-local gradients")
        self.transpose_free_dgrad = transpose_free_dgrad
        if not isinstance(fuse_dhidden_scores, bool):
            raise TypeError("fuse_dhidden_scores must be bool")
        if fuse_dhidden_scores and (not expert_local_grad or not reuse_packed_dy
                or not fuse_dispatch or not fuse_return or not fuse_activation_backward
                or not fuse_fc1_activation or kwargs.get('gate_up_layout', 'split') != 'g8'
                or ring_rows is None or dag_fusion or dag_backward or ffn_fusion
                or dispatch_pipeline or return_pipeline or staged_return or fuse_replay_gradients
                or kwargs.get('overlap_expert_dy', False) or dispatch_cache_n):
            raise ValueError("dHidden scores require G8 bilateral routed expert gradients with fused activations")
        if fuse_dhidden_scores and (gemm_configs is None or not gemm_configs.get('dhidden')):
            raise ValueError("dHidden scores require an explicit dHidden GEMM configuration")
        self.fuse_dhidden_scores = fuse_dhidden_scores
        if not isinstance(fuse_replay_gradients, bool):
            raise TypeError("fuse_replay_gradients must be bool")
        if fuse_replay_gradients and (not expert_local_grad or not reuse_packed_dy
                or not fuse_dispatch or not fuse_return or not fuse_activation_backward
                or ring_rows is None or dag_fusion or dag_backward or ffn_fusion
                or return_pipeline or staged_return or kwargs.get('overlap_expert_dy', False)):
            raise ValueError("replay gradients require routed bilateral expert-local gradients without DY prefetch or DAG modes")
        if fuse_replay_gradients and (gemm_configs is None or not gemm_configs.get('fc2')):
            raise ValueError("replay gradients require an explicit FC2 GEMM configuration")
        self.fuse_replay_gradients = fuse_replay_gradients
        if type(dgrad_compute_cus) is not int or dgrad_compute_cus < 0:
            raise ValueError("dgrad_compute_cus must be a nonnegative integer")
        self.dgrad_compute_cus = dgrad_compute_cus
        if type(dispatch_assist_ctas) is not int or not 0 <= dispatch_assist_ctas <= 156:
            raise ValueError("dispatch_assist_ctas must be an integer in [0,156]")
        if not isinstance(dispatch_external, bool):
            raise TypeError("dispatch_external must be bool")
        if dispatch_external and (not dispatch_assist_ctas or inline_dispatch):
            raise ValueError("external dispatch requires positive assistants without inline owners")
        self.dispatch_assist_ctas = dispatch_assist_ctas
        if not isinstance(dispatch_external_serial, bool):
            raise TypeError("dispatch_external_serial must be bool")
        if dispatch_external_serial and not dispatch_external:
            raise ValueError("serialized external dispatch requires external input consumers")
        self.dispatch_external = dispatch_external
        self.dispatch_external_serial = dispatch_external_serial
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
        self.fp8_peer_compute = kwargs.get("compute", "bf16") == "fp8"
        def fp8_flag(value, key):
            if value is None:
                raw = os.environ.get(key, "0")
                if raw not in ("0", "1"):
                    raise ValueError(key + " must be0or1")
                value = raw == "1"
            if type(value) is not bool:
                raise TypeError(key + " must be bool")
            return value
        self.fp8_input_pipeline = fp8_flag(fp8_input_pipeline, "DG_MOE_FP8_INPUT_PIPELINE")
        self.fp8_bf16_input_pipeline = fp8_flag(None, "DG_MOE_FP8_BF16_INPUT_PIPELINE")
        self.fp8_input_co_resident = fp8_flag(None, "DG_MOE_FP8_INPUT_CO_RESIDENT")
        self.fp8_input_elastic_workers = fp8_flag(None, "DG_MOE_FP8_INPUT_ELASTIC_WORKERS")
        self.fp8_input_slices = helper_value(None, "DG_MOE_FP8_INPUT_SLICES", 1)
        if self.fp8_input_slices not in (1,2,4,8,16):
            raise ValueError("FP8 input slices must be1,2,4,8or16")
        if self.fp8_input_slices!=1 and not self.fp8_bf16_input_pipeline:
            raise ValueError("sliced input requires the BF16 quantized producer")
        if self.fp8_input_elastic_workers and not self.fp8_input_co_resident:
            raise ValueError("elastic input workers require co-resident input")
        self.fp8_return_pipeline = fp8_flag(None, "DG_MOE_FP8_RETURN_PIPELINE")
        self.fp8_return_ctas = helper_value(None, "DG_MOE_FP8_RETURN_CTAS", 4)
        self.fp8_return_threads = helper_value(None, "DG_MOE_FP8_RETURN_THREADS", 512)
        self.fp8_return_store_bytes = helper_value(None, "DG_MOE_FP8_RETURN_STORE_BYTES", 4)
        if self.fp8_return_store_bytes not in (4,16):
            raise ValueError("FP8 return store width must be4or16bytes")
        if self.fp8_return_store_bytes != 4 and (not self.fp8_peer_compute or not fuse_return or self.fp8_return_pipeline):
            raise ValueError("wide FP8 return stores require direct fused FP8 return")
        if self.fp8_return_threads not in (256,512,1024):
            raise ValueError("FP8 return copy threads must be256,512or1024")
        if self.fp8_return_pipeline and (not self.fp8_peer_compute or not fuse_return or ring_rows is None):
            raise ValueError("FP8 return pipeline requires fused return and bounded FP8 routing windows")
        self.fp8_input_serialize = fp8_flag(fp8_input_serialize, "DG_MOE_FP8_INPUT_SERIALIZE")
        self.fp8_serial_full_grid = fp8_flag(None, "DG_MOE_FP8_INPUT_SERIAL_FULL_GRID")
        if self.fp8_input_co_resident and (not self.fp8_bf16_input_pipeline or self.fp8_serial_full_grid):
            raise ValueError("co-resident input requires the BF16 input producer without serial-full-grid mode")
        self.fp8_input_cache_scales = fp8_flag(fp8_input_cache_scales, "DG_MOE_FP8_INPUT_CACHE_SCALES")
        stage_quant = os.environ.get('DG_MOE_FP8_FUSED_STAGE_QUANT','0')
        if stage_quant not in ('0','1','2','3'):
            raise ValueError('DG_MOE_FP8_FUSED_STAGE_QUANT must be0,1,2or3')
        self.fp8_fused_stage_quant = int(stage_quant)
        if self.fp8_bf16_input_pipeline and (not self.fp8_peer_compute or self.fp8_input_pipeline
                or not (self.fp8_fused_stage_quant & 1) or ring_rows is None):
            raise ValueError("BF16 input pipeline requires bounded staged FP8 with fused input quantization")
        tiled_mode=os.environ.get('DG_MOE_FP8_TILED_GRAD_QUANT','0')
        if tiled_mode not in ('0','1','2'):
            raise ValueError('DG_MOE_FP8_TILED_GRAD_QUANT must be0,1or2')
        self.fp8_tiled_gradient_quant=tiled_mode!='0'
        self.fp8_tiled_gradient_cache=tiled_mode=='2'
        if self.fp8_tiled_gradient_quant and not (self.fp8_fused_stage_quant&2):
            raise ValueError('tiled gradient quantization requires fused DY quantization')
        # Source-quantized input owns X quantization; fused DY quantization
        # uses the same scratch only after FC1 and its producer have joined.
        if self.fp8_fused_stage_quant and (not self.fp8_peer_compute
                or (self.fp8_input_pipeline and (self.fp8_fused_stage_quant & 1))
                or not expert_local_grad or kwargs.get('block_m',128)!=128):
            raise ValueError('fused stage quantization requires staged FP8 M128 expert-local gradients')
        self.reuse_forward_context = fp8_flag(reuse_forward_context, "DG_MOE_REUSE_FORWARD_CONTEXT")
        if self.reuse_forward_context and (ring_rows is None or not expert_local_grad
                or ffn_fusion or dag_fusion or dag_backward or fuse_replay_gradients
                or (not self.fp8_peer_compute and not fuse_dhidden_scores)):
            raise ValueError("forward context reuse requires routed expert-local FP8 or BF16 dHidden-score training")
        self._forward_context_available = self._using_forward_context = False
        self._forward_context_fingerprint = None
        if self.fp8_input_cache_scales and not self.fp8_input_pipeline:
            raise ValueError("FP8 input scale caching requires its input pipeline")
        self.fp8_input_ctas = helper_value(fp8_input_ctas, "DG_MOE_FP8_INPUT_CTAS", 8)
        self.fp8_input_threads = helper_value(fp8_input_threads, "DG_MOE_FP8_INPUT_THREADS", 128)
        if self.fp8_input_threads not in (128, 256, 512):
            raise ValueError("FP8 producer threads must be128,256or512")
        if self.fp8_input_pipeline and not self.fp8_peer_compute:
            raise ValueError("FP8 input pipeline requires FP8 peer compute")
        if self.fp8_serial_full_grid and not ((self.fp8_input_pipeline or self.fp8_bf16_input_pipeline) and self.fp8_input_serialize):
            raise ValueError("full-grid serial input requires the serialized FP8 input pipeline")
        if self.fp8_input_serialize and not (self.fp8_input_pipeline or self.fp8_bf16_input_pipeline):
            raise ValueError("FP8 input serialization requires the FP8 input pipeline")
        self._fp8_backward = self._fp8_bf_input_pending = False
        if self.fp8_peer_compute and (self.transport != "peer" or fuse_dispatch or not active_rows
                or not vectorized_copy or fuse_activation_backward or dispatch_cache_n or dispatch_cache_aiu
                or staged_return or dispatch_producer or producer_local_aiu or producer_input_aiu
                or dispatch_pipeline or return_pipeline or inline_dispatch or cache_return_addresses
                or ffn_fusion or dag_fusion or dag_backward or dag_shared_pool or dispatch_assist_ctas
                or dispatch_external or fuse_fc1_activation or dgrad_compute_cus or fuse_replay_gradients
                or fuse_dhidden_scores or transpose_free_dgrad or input_helper_threads or input_copy_ilp != 1
                or return_vector_bytes != 4 or kwargs.get("block_m", 128) != 128
                or kwargs.get("gate_up_layout", "split") != "split"
                or kwargs.get("wgrad_overlap", "none") != "none" or kwargs.get("overlap_expert_dy", False)
                or not kwargs.get("fuse_activation_quant", True)):
            raise ValueError("FP8 peer compute requires staged input, active M128/split quantization and separate backward kernels")
        if fp8_weight_quantization is None:
            raw = os.environ.get("DG_MOE_FP8_WEIGHT_QUANTIZER", "2") if self.fp8_peer_compute else "0"
            if raw not in ("0", "1", "2"):
                raise ValueError("DG_MOE_FP8_WEIGHT_QUANTIZER must be0,1or2")
            fp8_weight_quantization = int(raw)
        if type(fp8_weight_quantization) is not int or fp8_weight_quantization not in (0, 1, 2):
            raise ValueError("fp8_weight_quantization must be0,1or2")
        if fp8_weight_quantization and not self.fp8_peer_compute:
            raise ValueError("FP8 weight quantization requires FP8 peer compute")
        self.fp8_weight_quantization = fp8_weight_quantization
        self.fp8_active_quantization = self.fp8_mn_major_scales = self.fp8_peer_compute
        self.communication_dtype = "bf16"
        self.fp8_input_mode = "staged_bf16_then_row_quantization" if self.fp8_peer_compute else None
        if self.fp8_input_pipeline:
            self.fp8_input_mode = "source_fp8_pull_pipeline"
            self.communication_dtype = "fp8_input_bf16_dw_and_return"
        if self.fp8_fused_stage_quant & 1:
            self.fp8_input_mode = "staged_bf16_with_fused_row_quantization"
        if self.fp8_bf16_input_pipeline:
            self.fp8_input_mode = "bf16_pull_quantized_tile_pipeline"
        self._fp8_prequantized_input = None
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
        self.reuse_packed_dy = reuse_packed_dy and (fuse_dispatch or self.fp8_peer_compute)
        super().__init__(*args, packed_rows=ring_rows, **kwargs)
        if self.h % (128*self.fp8_input_slices):
            raise ValueError("hidden width must be divisible by128times FP8 input slices")
        if self.fp8_return_pipeline:
            if not 0 < self.fp8_return_ctas < dg.get_num_sms():
                raise ValueError("FP8 return copy CTAs must leave GEMM resources")
            self._fp8_return_stream = torch.cuda.Stream(device=self.device)
        if self.fp8_input_pipeline or self.fp8_bf16_input_pipeline:
            if self.fp8_bf16_input_pipeline and self.h > 8192:
                raise ValueError("BF16 quantized input pipeline supports hidden<=8192")
            if self.fp8_input_cache_scales and self.h > 8192:
                raise ValueError("cached FP8 input scales support hidden<=8192")
            producer_limit = (4*dg.get_num_sms() if self.fp8_input_elastic_workers else dg.get_num_sms() if self.fp8_input_co_resident else
                              8 * dg.get_num_sms() if self.fp8_bf16_input_pipeline and self.fp8_serial_full_grid
                              else dg.get_num_sms() - 1)
            if not 0 < self.fp8_input_ctas <= producer_limit:
                raise ValueError("FP8 producer CTA budget exceeds its serialized or concurrent limit")
            if self.fp8_input_co_resident and (self.fp8_input_ctas%dg.get_num_sms() or
                    (not self.fp8_input_elastic_workers and self.fp8_input_ctas!=dg.get_num_sms())):
                raise ValueError("co-resident input requires one producer CTA per CU")
            self._fp8_input_stream = torch.cuda.Stream(device=self.device)
            self._fp8_bf_input_stream = torch.cuda.Stream(device=self.device)
        if self.transpose_free_dgrad:
            # dGrad reads the original expert parameters via native AIU Trans.
            # Release the now-unused transposed weight copies.
            for name in ('w1_t', 'w2_t'):
                self.buffers.pop(name)
                setattr(self, name, None)
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
                      self.inline_dispatch, self.dispatch_assist_ctas, self.dispatch_external, self.dispatch_external_serial,
                      self.fuse_dhidden_scores, self.transpose_free_dgrad,
                      self.input_helper_threads, self.input_helper_block_m, self.input_helper_block_n,
                      self.fp8_weight_quantization, self.fp8_input_pipeline,
                      self.fp8_input_ctas, self.fp8_input_serialize, self.fp8_input_threads,
                      self.fp8_input_cache_scales, self.reuse_forward_context, self.deferred_dw_compute_cus, self.fp8_serial_full_grid, self.fp8_fused_stage_quant, self.fp8_tiled_gradient_quant, self.fp8_tiled_gradient_cache, self.fp8_bf16_input_pipeline, self.fp8_return_pipeline, self.fp8_return_ctas, self.fp8_return_threads, self.fp8_return_store_bytes, self.fp8_fuse_dhidden_activation, self.fp8_input_co_resident,self.fp8_input_elastic_workers,self.fp8_input_slices)
            recipes = [None] * self.size
            dist.all_gather_object(recipes, recipe, group=self.group)
            if any(other != recipe for other in recipes):
                raise ValueError("EP ranks must agree on the peer gradient and return protocol")
        if gemm_configs is None and self.block_m == 256:
            gemm_configs = {role: (256, 64, 64, 64, 2) for role in ("fc1", "fc2", "dhidden", "dx")}
        self.gemm_configs = {} if gemm_configs is None else {name: tuple(value) for name, value in gemm_configs.items()}
        if self.fp8_input_co_resident:
            self.gemm_configs.setdefault('fc1',(128,128,32,64,3))
        if self.fp8_return_pipeline:
            for role in ('fc2', 'dx'):
                config = self.gemm_configs.setdefault(role, (256, 128, 32, 64, 3))
                if config[0] != 256:
                    raise ValueError("FP8 return publication requires N256 GEMM tiles")
        if self.input_helper_threads:
            self.gemm_configs["fc1"] = (self.input_helper_block_n, 64, 64, 64, 3)
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
        if self.fp8_input_pipeline:
            descriptions += (("local_qx", (c, h), torch.float8_e4m3fn),
                             ("local_qx_scales", (c, h // 128), torch.float32))
        layout, size = {}, 0
        for name, shape, dtype in descriptions:
            size = (size + 15) // 16 * 16
            count = math.prod(shape)
            element = torch.empty((), dtype=dtype).element_size()
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
        for name in ("gathered_x", "gathered_ids", "route_values") + (() if self.fp8_peer_compute else ("group_ids",)):
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
        if self.fp8_tiled_gradient_quant:
            self.fp8_dscore_partials=torch.empty((self.rows,self.h//128),device=self.device,dtype=torch.float32)
            self.buffers['fp8_dscore_partials']=self.fp8_dscore_partials
        if self.fuse_replay_gradients or self.fuse_dhidden_scores:
            bn = self.gemm_configs['dhidden' if self.fuse_dhidden_scores else 'fc2'][0]
            width = self.i if self.fuse_dhidden_scores else self.h
            self.score_partial_columns = (width + bn - 1) // bn
            self.score_partials = torch.empty((self.rows, self.score_partial_columns),
                                             device=self.device, dtype=torch.float32)
            self.buffers['score_partials'] = self.score_partials
            if self.fuse_dhidden_scores:
                self.packed_row_scores = torch.empty(self.rows, device=self.device, dtype=torch.float32)
                self.buffers['packed_row_scores'] = self.packed_row_scores
        self.gate_up.zero_()
        self.hidden.zero_()
        self.quantized_scale_storage = {}
        if self.fp8_peer_compute:
            for width in (self.h, self.i, 2 * self.i):
                if width in self.quantized_scale_storage:
                    continue
                storage = torch.empty((width // 128, self.rows), device=self.device, dtype=torch.float32)
                self.buffers["activation_scale_mn_" + str(width)] = storage
                self.quantized[width] = (self.quantized[width][0], storage.t())
                self.buffers.pop("activation_scale_" + str(width))
                self.quantized_scale_storage[width] = storage
            self.group_ids.zero_()
        if self.fp8_input_pipeline or self.fp8_bf16_input_pipeline:
            self.fp8_input_ready = torch.zeros(self.rows // 128, device=self.device, dtype=torch.int32)
            self.buffers["fp8_input_ready"] = self.fp8_input_ready
            if self.fp8_input_elastic_workers:
                self.fp8_input_work_counter=torch.zeros(1,device=self.device,dtype=torch.int32)
                self.buffers['fp8_input_work_counter']=self.fp8_input_work_counter
        if self.fp8_return_pipeline:
            self.fp8_return_ready = torch.zeros((self.rows // 128) * (self.h // 256), device=self.device, dtype=torch.int32)
            self.buffers['fp8_return_ready'] = self.fp8_return_ready
        self.forward_context_slots = []
        if self.reuse_forward_context:
            # Each source chunk retains its last nonempty ring window. Earlier
            # windows can still be replayed without unbounded skew allocations.
            names = ['gate_up', 'hidden', 'global_offsets', 'global_lengths', 'global_permutation']
            names += ['packed_output'] if self.fp8_peer_compute else ['packed_x']
            if self.fp8_peer_compute and not self.fp8_input_pipeline:
                names.append('packed_x')
            original = {name: getattr(self, name) for name in names}
            if self.fp8_peer_compute:
                original.update(quantized_hidden=self.quantized[self.i][0],
                                quantized_hidden_scales=self.quantized_scale_storage[self.i])
            for slot in range(max(1, math.ceil(self.max_tokens / self.chunk_tokens))):
                tensors = original if slot == 0 else {name: torch.empty_like(value) for name, value in original.items()}
                if slot:
                    self.buffers.update({f'forward_context_{slot}_{name}': value for name, value in tensors.items()})
                self.forward_context_slots.append(tensors)
            for name, shape, dtype in (('replay_offsets', (self.e + 1,), torch.int32),
                                      ('replay_lengths', (self.e,), torch.int32),
                                      ('replay_blocks', (1,), torch.int32),
                                      ('cache_only_blocks', (1,), torch.int32),
                                      ('forward_context_reused_windows', (1,), torch.int32),
                                      ('forward_context_reused_rows', (1,), torch.int64)):
                value = torch.zeros(shape, device=self.device, dtype=dtype)
                setattr(self, name, value)
                self.buffers[name] = value
        self._prepared = False
        torch.cuda.synchronize(self.device)
        if self.group is not None:
            dist.barrier(group=self.group)

    def _prepare_weights(self, w1, w2, backward=False):
        if self.fp8_peer_compute:
            self._fp8_prequantized_input = None
            self._fp8_backward = backward
            if self._shared_fp8_weights_ready:
                return
        if self.transpose_free_dgrad:
            self.w1_t, self.w2_t = w1, w2
            return
        return super()._prepare_weights(w1, w2, backward=backward)

    @staticmethod
    def _context_fingerprint(*tensors):
        return tuple((value.data_ptr(), value._version, tuple(value.shape)) for value in tensors)

    def _activate_forward_context(self, start):
        if not self.reuse_forward_context:
            return
        for name, value in self.forward_context_slots[start // self.chunk_tokens].items():
            if name == 'quantized_hidden':
                self.quantized[self.i] = (value, self.quantized[self.i][1])
            elif name == 'quantized_hidden_scales':
                self.quantized_scale_storage[self.i] = value
                self.quantized[self.i] = (self.quantized[self.i][0], value.t())
            else:
                setattr(self, name, value)

    def _prepare_context_replay(self, start, compile_only=False):
        _peer(20, [self.offsets, self.lengths, self.num_blocks, self.global_offsets,
                   self.replay_offsets, self.replay_lengths, self.replay_blocks, self.cache_only_blocks,
                   self.status, self.forward_context_reused_windows, self.forward_context_reused_rows],
              [self.e, self.rows, start], compile_only)

    def _quantize_weight(self, name, weight):
        data, scales = self.quantized[name]
        _launch((10, 29, 30)[self.fp8_weight_quantization], [weight, data, scales], list(weight.shape))

    def _prepare_pointwise(self):
        super()._prepare_pointwise()
        if self.fp8_peer_compute:
            dg._C._moe_training_precompile(self.device.index, [9, 34 if self.fp8_vector_row_quant else 26, 27, 28, (10, 29, 30)[self.fp8_weight_quantization]])

    def _barrier(self, tag, compile_only=False):
        _peer(5, [self.pointer_tables["barrier_state"], self.barrier_state, self.status],
              [self.size, self.rank, 180 * 2_000_000_000, tag], compile_only)

    def _peer_gemm(self, a, weight, output, source_rows, *, read=False, write=False,
                   source="local_x", destination="local_values", compile_only=False, pack_input=False, role=None,
                   first_n_tile=0, num_n_tiles=0, local_input_aiu=False, producer_input_aiu=False,
                   dispatch_pipeline=False, keep_gate_up=True, replay_gradients=False,
                   fc1_helper_threads=0):
        source_stride = self.chunk_tokens if source == "local_x" else self.chunk_tokens * self.k
        if role is None:
            role = ("fc1" if source == "local_x" else "dhidden") if read else ("dx" if destination == "local_dx" else "fc2")
        if self.fp8_peer_compute:
            if read or dispatch_pipeline or first_n_tile or num_n_tiles or replay_gradients:
                raise ValueError("FP8 peer GEMM requires staged local input and a complete N range")
            name = {"fc1": "w1", "fc2": "w2", "dhidden": "w2_t", "dx": "w1_t"}[role]
            width = a.shape[1]
            quantized_a = self.quantized[width]
            if (self.fp8_input_pipeline or self.fp8_bf16_input_pipeline) and role == "fc1":
                if self.fp8_serial_full_grid:
                    # _fc1 has joined the producer stream. No consumers wait
                    # on READY, so every CU may now execute the native GEMM.
                    config = self.gemm_configs.get(role)
                    options = {} if config is None else {"configs": (dg.get_num_sms(), self.block_m, *config, (0, 0, 0))}
                    dg.m_grouped_fp8_gemm_nt_nopad(quantized_a, self.quantized[name], output, self.group_ids,
                                                  m_rows=self.lengths, disable_ue8m0_cast=True, **options)
                    return None
                return dg._C._moe_training_peer_fp8_gemm(
                    quantized_a, self.quantized[name], output, self.lengths, self.permutation,
                    self.pointer_tables[destination], self.chunk_tokens * self.k,
                    keep_local=True, compile_only=compile_only,
                    config=self.gemm_configs.get(role, ()),
                    input_control=[self.fp8_input_ready, self.status]+([self.fp8_input_work_counter] if self.fp8_input_elastic_workers else []), reserved_input_ctas=self.fp8_input_ctas,
                    co_resident_input=self.fp8_input_co_resident,input_ready_count=self.fp8_input_slices)
            if self._fp8_prequantized_input != a.data_ptr():
                _launch(34 if self.fp8_vector_row_quant else 26, [a, quantized_a[0], self.quantized_scale_storage[width], self.current_rows, self.status],
                        [self.rows, width, 1])
            self._fp8_prequantized_input = None
            config = self.gemm_configs.get(role)
            if role == 'dhidden' and self.fp8_fuse_dhidden_activation:
                return dg._C._moe_training_peer_fp8_gemm(
                    quantized_a,self.quantized[name],output,self.lengths,self.permutation,
                    self.pointer_tables[destination],self.chunk_tokens*self.k,keep_local=False,
                    compile_only=compile_only,config=() if config is None else config,
                    activation_gate_up=self.gate_up)
            if write:
                if self.fp8_return_pipeline:
                    return self._fp8_return_tiles(quantized_a, self.quantized[name], output, destination,
                                                  config, compile_only)
                return dg._C._moe_training_peer_fp8_gemm(
                    quantized_a, self.quantized[name], output, self.lengths, self.permutation,
                    self.pointer_tables[destination], self.chunk_tokens * self.k,
                    keep_local=self.reuse_forward_context and role == 'fc2' and not self._fp8_backward,
                    compile_only=compile_only, config=() if config is None else config,
                    return_store_bytes=self.fp8_return_store_bytes)
            options = {} if config is None else {"configs": (dg.get_num_sms(), self.block_m, *config, (0, 0, 0))}
            # Zero lengths make this a safe local warmup during preparation.
            dg.m_grouped_fp8_gemm_nt_nopad(quantized_a, self.quantized[name], output, self.group_ids,
                                          m_rows=self.lengths, disable_ue8m0_cast=True, **options)
            return None
        return_pipeline = write and self.return_pipeline
        fc1_activation = role == "fc1" and self.fuse_fc1_activation
        interleaved = self.g8_interleaved and (fc1_activation or (role == "dhidden" and self.fuse_activation_backward))
        if return_pipeline and not compile_only:
            self.return_pipeline_ready.zero_()
        extra_arguments = []
        if replay_gradients:
            extra_arguments.append([self.pointer_tables['raw_dy'], self.pointer_tables['raw_scores'],
                                    self.packed_dy, self.score_partials, self.status])
        if self.fuse_dhidden_scores and role == 'dhidden':
            if not extra_arguments:
                extra_arguments.append([])
            extra_arguments.append([self.hidden, self.packed_row_scores, self.score_partials])
        if self.transpose_free_dgrad and role in ('dhidden', 'dx'):
            while len(extra_arguments) < 2:
                extra_arguments.append([])
            extra_arguments.append(True)
        extra_options = {}
        if fc1_helper_threads:
            extra_options = dict(fc1_helper_threads=fc1_helper_threads,
                                 fc1_compute_block_m=self.input_helper_block_m)
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
            self.input_copy_ilp if producer_input_aiu or fc1_helper_threads or (dispatch_pipeline and self.inline_dispatch) else 1,
            dispatch_pipeline and self.dispatch_external,
            self.dgrad_compute_cus if role in ("dhidden", "dx") else 0,
            *extra_arguments, **extra_options,
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
        if self.fp8_bf16_input_pipeline:
            main = torch.cuda.current_stream(self.device)
            if compile_only:
                self._fp8_stage_bf16_input(compile_only=True)
            else:
                self.fp8_input_ready.zero_()
                if self.fp8_input_elastic_workers:self.fp8_input_work_counter.zero_()
                self._fp8_input_stream.wait_stream(main)
                with torch.cuda.stream(self._fp8_input_stream):
                    for tensor in (self.permutation, self.num_blocks, self.fp8_input_ready, self.status,
                                   *self.quantized[self.h], self.quantized_scale_storage[self.h],
                                   self.packed_x, self.pointer_tables['local_x']):
                        tensor.record_stream(self._fp8_input_stream)
                    self._fp8_stage_bf16_input()
                    if self.fp8_input_elastic_workers and not self.fp8_input_serialize:
                        self._fp8_elastic_input_worker()
                if self.fp8_input_serialize:
                    main.wait_stream(self._fp8_input_stream)
                    if self.fp8_input_elastic_workers:
                        self._fp8_input_stream.wait_stream(main)
                        with torch.cuda.stream(self._fp8_input_stream):self._fp8_elastic_input_worker()
            self._peer_gemm(self.packed_x, w1, self.gate_up, self.identity_rows, role='fc1',
                            compile_only=compile_only)
            if not compile_only and self.fp8_input_elastic_workers:
                main.wait_stream(self._fp8_input_stream)
            if not compile_only:
                # Preserve the original BF16 X for DW and context reuse, and
                # join every producer before reusing quantization scratch.
                main.wait_stream(self._fp8_input_stream)
        elif self.fp8_input_pipeline:
            main = torch.cuda.current_stream(self.device)
            if compile_only:
                self._fp8_stage_quantized_input(compile_only=True)
            else:
                self.fp8_input_ready.zero_()
                self._fp8_input_stream.wait_stream(main)
                with torch.cuda.stream(self._fp8_input_stream):
                    for tensor in (self.source_rows, self.num_blocks, self.fp8_input_ready, self.status,
                                   *self.quantized[self.h], self.quantized_scale_storage[self.h],
                                   self.local_qx, self.local_qx_scales, self.packed_x,
                                   self.pointer_tables['local_qx'], self.pointer_tables['local_qx_scales'],
                                   self.pointer_tables['local_x']):
                        tensor.record_stream(self._fp8_input_stream)
                    self._fp8_stage_quantized_input()
                if self.fp8_input_serialize:
                    main.wait_stream(self._fp8_input_stream)
            self._peer_gemm(self.packed_x, w1, self.gate_up, self.identity_rows, role='fc1',
                            compile_only=compile_only)
            if not compile_only:
                if self._fp8_backward:
                    # Original BF16 X is needed only by DW1. Stage it after
                    # the FP8 producer, while FC1/FC2/DW2/dHidden can progress.
                    self._fp8_bf_input_stream.wait_stream(self._fp8_input_stream)
                    with torch.cuda.stream(self._fp8_bf_input_stream):
                        for tensor in (self.permutation, self.num_blocks, self.packed_x, self.status,
                                       self.pointer_tables['local_x']):
                            tensor.record_stream(self._fp8_bf_input_stream)
                        self._stage_input(force_bf16=True)
                    self._fp8_bf_input_pending = True
                main.wait_stream(self._fp8_input_stream)
        elif self.input_helper_threads:
            self._peer_gemm(self.packed_x, w1, self.gate_up, self.source_rows,
                            read=True, pack_input=True, role="fc1", compile_only=compile_only,
                            keep_gate_up=keep_gate_up, fc1_helper_threads=self.input_helper_threads)
        elif self.dispatch_pipeline:
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
                if self.dispatch_external_serial:
                    # Identical mover and consumer kernels. The only change
                    # is joining the complete dispatch before GEMM starts.
                    torch.cuda.current_stream(self.device).wait_stream(self._input_stream)
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
        if self.fp8_peer_compute:
            self.lengths.zero_()
            self.offsets.zero_()
            self.num_blocks.zero_()
            self._fp8_prequantized_input = None
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
        if self.reuse_forward_context:
            self._prepare_context_replay(0, compile_only=True)
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
        if self.reuse_forward_context and self.fp8_peer_compute:
            previous_backward = self._fp8_backward
            self._fp8_backward = False
            self._peer_gemm(self.hidden, w2, self.packed_output, self.identity_rows,
                            write=self.fuse_return, compile_only=True)
            self._fp8_backward = previous_backward
        if self.fuse_replay_gradients:
            self._peer_gemm(self.hidden, w2, self.packed_output, self.identity_rows,
                            compile_only=True, replay_gradients=True)
        if self.fuse_dhidden_scores:
            self._expert_gradients(compile_only=True)
            self._reduce_score_partials(compile_only=True)
        read_dy = self.fuse_dispatch and not self.reuse_packed_dy
        self._peer_gemm(self.packed_output if self.fuse_dhidden_scores else self.packed_dy, self.w2_t, self.hidden,
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
        self._activate_forward_context(start)
        _launch(13, [x, ids, self.local_x, self.local_ids, valid_tokens, self.status],
                [x.shape[0], self.chunk_tokens, self.h, self.k, start])
        if self.fp8_input_pipeline:
            _launch(9, [self.local_x, self.local_qx, self.local_qx_scales], [self.chunk_tokens, self.h])
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
                              prefetch_dy=False, _context_masked=False):
        if self._using_forward_context and not return_output and not _context_masked:
            self._prepare_context_replay(self._context_window_start)
            original = self.offsets, self.lengths, self.num_blocks, self.current_rows
            self.offsets, self.lengths, self.num_blocks = self.replay_offsets, self.replay_lengths, self.replay_blocks
            self.current_rows = self.replay_offsets[self.e:self.e + 1]
            try:
                self._chunk_forward_single(w1, w2, pack_input, return_output, synchronize_output,
                                           prefetch_dy, _context_masked=True)
            finally:
                self.offsets, self.lengths, self.num_blocks, self.current_rows = original
            if self.fp8_input_pipeline:
                # The cached forward did not stage BF16 X. Only its reused
                # window needs this additional copy; other counts are zero.
                self._fp8_bf_input_stream.wait_stream(self._fp8_input_stream)
                with torch.cuda.stream(self._fp8_bf_input_stream):
                    for tensor in (self.permutation, self.cache_only_blocks, self.packed_x,
                                   self.pointer_tables['local_x'], self.status):
                        tensor.record_stream(self._fp8_bf_input_stream)
                    self._stage_input(force_bf16=True, blocks_override=self.cache_only_blocks)
                self._fp8_bf_input_pending = True
            return
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
        # A reused BF16 forward must retain expert-packed X for DW1 even
        # when FC1 directly reads peers without a separate producer.
        self._fc1(w1, pack_input=pack_input or (return_output and self.reuse_forward_context),
                  keep_gate_up=not return_output or self.reuse_forward_context,
                  prefetch_dy=prefetch_dy)
        if not self.fuse_fc1_activation:
            self._activation()
        if self.fuse_dhidden_scores and not return_output:
            return
        self._peer_gemm(self.hidden, w2, self.packed_output, self.identity_rows,
                        write=self.fuse_return and return_output,
                        replay_gradients=self.fuse_replay_gradients and not return_output)
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

    def _fp8_return_tiles(self, activation, weight, output, destination, config, compile_only):
        main = torch.cuda.current_stream(self.device)
        def launch(copy_only):
            return dg._C._moe_training_peer_fp8_gemm(
                activation, weight, output, self.lengths, self.permutation,
                self.pointer_tables[destination], self.chunk_tokens * self.k,
                keep_local=True, compile_only=compile_only, config=config,
                return_control=[self.fp8_return_ready,self.status],reserved_return_ctas=self.fp8_return_ctas,
                copy_only=copy_only,copy_threads=self.fp8_return_threads)
        if compile_only:
            launch(True)
        else:
            self.fp8_return_ready.zero_()
            self._fp8_return_stream.wait_stream(main)
            with torch.cuda.stream(self._fp8_return_stream):
                for tensor in (output,self.lengths,self.permutation,self.pointer_tables[destination],
                               self.fp8_return_ready,self.status,*activation,*weight):
                    tensor.record_stream(self._fp8_return_stream)
                launch(True)
        grid=launch(False)
        if not compile_only:
            main.wait_stream(self._fp8_return_stream)
        return grid

    def _fp8_stage_bf16_input(self, compile_only=False):
        _peer(25, [self.pointer_tables['local_x'], self.permutation, self.packed_x,
                   self.status, self.num_blocks, self.quantized[self.h][0],
                   self.quantized_scale_storage[self.h], self.fp8_input_ready],
              [self.rows, self.h, self.chunk_tokens, self.k, self.size, 128, self.fp8_input_ctas,
               2 if self.fp8_input_elastic_workers else int(self.fp8_input_co_resident),self.fp8_input_slices], compile_only)

    def _fp8_elastic_input_worker(self):
        stream=torch.cuda.current_stream(self.device)
        for tensor in (self.gate_up,self.lengths,self.permutation,self.pointer_tables['local_values'],
                       *self.quantized[self.h],*self.quantized['w1'],self.fp8_input_ready,
                       self.status,self.fp8_input_work_counter):tensor.record_stream(stream)
        return dg._C._moe_training_peer_fp8_gemm(
            self.quantized[self.h],self.quantized['w1'],self.gate_up,self.lengths,self.permutation,
            self.pointer_tables['local_values'],self.chunk_tokens*self.k,keep_local=True,
            config=self.gemm_configs['fc1'],input_control=[self.fp8_input_ready,self.status,self.fp8_input_work_counter],
            reserved_input_ctas=self.fp8_input_ctas,co_resident_input=True,input_ready_count=self.fp8_input_slices)

    def _fp8_stage_quantized_input(self, compile_only=False):
        _peer(19, [self.pointer_tables['local_qx'], self.pointer_tables['local_qx_scales'],
                   self.pointer_tables['local_x'], self.source_rows, self.quantized[self.h][0],
                   self.quantized_scale_storage[self.h], self.packed_x, self.num_blocks,
                   self.fp8_input_ready, self.status],
              [self.rows, self.h, self.chunk_tokens, self.size, self.fp8_input_ctas, 0,
               self.fp8_input_threads, int(self.fp8_input_cache_scales)], compile_only)

    def _join_fp8_bf_input(self):
        if self._fp8_bf_input_pending:
            torch.cuda.current_stream(self.device).wait_stream(self._fp8_bf_input_stream)
            self._fp8_bf_input_pending = False

    def _stage_input(self, compile_only=False, force_bf16=False, blocks_override=None):
        if self.fp8_bf16_input_pipeline and not force_bf16:
            return
        if self.fp8_input_pipeline and not force_bf16:
            if compile_only:
                self._fp8_stage_quantized_input(compile_only=True)
                self._stage_input(compile_only=True, force_bf16=True)
            return
        if self.fp8_fused_stage_quant & 1 and not force_bf16:
            _peer(21,[self.pointer_tables['local_x'],self.permutation,self.packed_x,self.status,
                      self.num_blocks if blocks_override is None else blocks_override,
                      self.quantized[self.h][0],self.quantized_scale_storage[self.h]],
                  [self.rows,self.h,self.chunk_tokens,self.k,self.size,self.block_m],compile_only)
            if not compile_only:self._fp8_prequantized_input=self.packed_x.data_ptr()
            return
        _peer(11 if self.vectorized_copy else 7,
              [self.pointer_tables["local_x"], self.permutation, self.packed_x, self.status,
               self.num_blocks if blocks_override is None else blocks_override],
              [self.rows, self.h, self.chunk_tokens, self.k, self.size, self.block_m], compile_only)

    def _activation(self, backward=False, compile_only=False):
        if self.fp8_peer_compute:
            if not compile_only:
                width = 2 * self.i if backward else self.i
                if backward and self.fp8_fuse_dhidden_activation:
                    _launch(34 if self.fp8_vector_row_quant else 26,
                            [self.gate_up,self.quantized[width][0],self.quantized_scale_storage[width],
                             self.current_rows,self.status],[self.rows,width,1])
                    self._fp8_prequantized_input=self.gate_up.data_ptr()
                    return
                tensors = [self.gate_up, self.hidden] + ([self.gate_up] if backward else [])
                tensors.extend((self.quantized[width][0], self.quantized_scale_storage[width],
                                self.current_rows, self.status))
                _launch(28 if backward else 27, tensors, [self.rows, self.i, 1])
                self._fp8_prequantized_input = self.gate_up.data_ptr() if backward else self.hidden.data_ptr()
            return
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

    def _reduce_score_partials(self, compile_only=False):
        _peer(17, [self.score_partials, self.permutation, self.num_blocks,
                   self.pointer_tables['local_dscores'], self.status],
              [self.rows, self.score_partial_columns, self.chunk_tokens,
               self.k, self.size, self.block_m], compile_only)

    def _expert_gradients(self, compile_only=False):
        if self.fuse_dhidden_scores:
            _peer(18, [self.pointer_tables['raw_dy'], self.pointer_tables['raw_scores'],
                       self.permutation, self.packed_output, self.packed_dy,
                       self.packed_row_scores, self.num_blocks, self.status],
                  [self.rows, self.h, self.chunk_tokens, self.k, self.size, self.block_m], compile_only)
            return
        if self.fuse_replay_gradients:
            _peer(17, [self.score_partials, self.permutation, self.num_blocks,
                       self.pointer_tables['local_dscores'], self.status],
                  [self.rows, self.score_partial_columns, self.chunk_tokens,
                   self.k, self.size, self.block_m], compile_only)
            return
        tensors = [self.pointer_tables["raw_dy"], self.pointer_tables["raw_scores"],
                   self.pointer_tables["local_dscores"], self.permutation,
                   self.packed_output, self.packed_dy, self.num_blocks, self.status]
        if self.overlap_expert_dy:
            if not compile_only:
                self._join_gradient_prefetch()
            tensors.append(self.packed_raw_dy)
        if self.fp8_fused_stage_quant & 2:
            tensors.extend((self.quantized[self.h][0],self.quantized_scale_storage[self.h]))
            if self.fp8_tiled_gradient_quant:
                tensors.append(self.fp8_dscore_partials)
                _peer(24 if self.fp8_tiled_gradient_cache else 23,tensors,[self.rows,self.h,self.chunk_tokens,self.k,self.size,self.block_m],compile_only)
                _peer(17,[self.fp8_dscore_partials,self.permutation,self.num_blocks,
                          self.pointer_tables['local_dscores'],self.status],
                      [self.rows,self.h//128,self.chunk_tokens,self.k,self.size,self.block_m],compile_only)
            else:
                _peer(22,tensors,[self.rows,self.h,self.chunk_tokens,self.k,self.size,self.block_m],compile_only)
            if not compile_only:self._fp8_prequantized_input=self.packed_dy.data_ptr()
        else:
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
        windows = range(0, self.full_row_capacity, self.rows)
        for window in reversed(windows) if self._using_forward_context else windows:
            self._set_window(window)
            self._context_window_start = window
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
                self._peer_gemm(self.packed_output if self.fuse_dhidden_scores else self.packed_dy,
                                self.w2_t, self.hidden, self.identity_rows, role="dhidden")
                if self.fuse_dhidden_scores:
                    self._reduce_score_partials()
                if not self.fuse_activation_backward:
                    self._activation(backward=True)
                self._join_fp8_bf_input()
                if (self._defer_last_dw1_enabled and self._using_forward_context
                        and start+self.chunk_tokens>=self.max_tokens and window==0):
                    # dX only reads these operands. The final window/chunk has
                    # no following metadata overwrite before the caller joins.
                    self._pending_last_dw1=(self.gate_up,self.packed_x,dw1,self.lengths)
                else:
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
    def _run_deferred_dw1(self):
        with self._execution():
            pending=self._pending_last_dw1
            if pending is None:
                self._defer_last_dw1_enabled=False
                return False
            a,b,output,lengths=pending
            if self.lengths.data_ptr()!=lengths.data_ptr():
                raise RuntimeError('deferred DW1 metadata changed before completion')
            if self.deferred_dw_compute_cus:
                dg.k_grouped_bf16_gemm_tn_device(a,b,output,lengths,self.rows,compute_cus=self.deferred_dw_compute_cus)
            else:
                self._wgrad(a,b,output,asynchronous=False)
            self._pending_last_dw1=None
            self._defer_last_dw1_enabled=False
            return True

    @torch.no_grad()
    def forward(self, x, ids, scores, w1, w2, valid_tokens, *, _chunk_output_callback=None):
        if self._pending_last_dw1 is not None:
            raise RuntimeError('complete deferred DW1 before reusing this workspace')
        with self._execution():
            self._dag_slot = 0
            self._validate(x, ids, scores, w1, w2, valid_tokens)
            self._begin_forward_context()
            self.prepare(w1, w2)
            if self.reuse_forward_context:
                self.forward_context_reused_windows.zero_()
                self.forward_context_reused_rows.zero_()
            output = super().forward(x, ids, scores, w1, w2, valid_tokens, _chunk_output_callback=_chunk_output_callback)
            if self.reuse_forward_context:
                self._record_forward_context_generation()
                self._forward_context_fingerprint = self._context_fingerprint(x, ids, scores, w1, w2, valid_tokens)
                self._forward_context_available = True
            return output

    @torch.no_grad()
    def backward(self, dy, x, ids, scores, w1, w2, valid_tokens, accumulators=None, *, _defer_last_dw1=False, _chunk_output_callback=None):
        if self._pending_last_dw1 is not None:
            raise RuntimeError('complete deferred DW1 before reusing this workspace')
        if type(_defer_last_dw1) is not bool:raise TypeError('_defer_last_dw1 must be bool')
        if _chunk_output_callback is not None:
            if not callable(_chunk_output_callback):raise TypeError('chunk callback must be callable')
            if self.ring_rows is None or not self.expert_local_grad or self.wgrad_overlap!="none":
                raise ValueError('chunk callbacks require bounded serial source gradients')
        if _defer_last_dw1 and (self.compute!='bf16' or self.ring_rows is None
                or not self.expert_local_grad or not self.reuse_forward_context or self.wgrad_overlap!='none'):
            raise ValueError('deferred DW1 requires serial routed BF16 live-context gradients')
        self._defer_last_dw1_enabled=_defer_last_dw1
        self._validate(x, ids, scores, w1, w2, valid_tokens)
        _validate_accumulators(accumulators, w1, w2)
        if dy.shape != x.shape or dy.device != x.device or dy.dtype != x.dtype or not dy.is_contiguous():
            raise ValueError("dy must be contiguous and match x")
        with self._execution():
            self._using_forward_context = self._matching_forward_context((x, ids, scores, w1, w2, valid_tokens))
            self._consume_forward_context_generation()
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
                if self._using_forward_context:
                    self._activate_forward_context(start)
                    _launch(13, [x, ids, self.local_x, self.local_ids, valid_tokens, self.status],
                            [x.shape[0], self.chunk_tokens, self.h, self.k, start])
                    if self.fp8_input_pipeline:
                        _launch(9, [self.local_x, self.local_qx, self.local_qx_scales], [self.chunk_tokens, self.h])
                    self.local_dx.zero_()
                    self._barrier(start * 4)
                else:
                    self._route_chunk(x, ids, valid_tokens, start, zero_values=not self.expert_local_grad)
                if self.ring_rows is not None:
                    self._backward_windows(dy, scores, w1, w2, dw1, dw2, dx, dscores, start)
                    if _chunk_output_callback is not None:
                        _chunk_output_callback(dx,dscores,start,max(0,min(self.chunk_tokens,x.shape[0]-start)))
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
                self._join_fp8_bf_input()
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
            self._forward_context_available = self._using_forward_context = False
            return dx, dw1, dw2, dscores
