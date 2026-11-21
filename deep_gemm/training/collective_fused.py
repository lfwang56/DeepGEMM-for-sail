"""PCCL/RDMA transport with GEMM epilogues writing the return ring directly."""

import torch

import deep_gemm as dg
from .native import NativeMoEPlan


class CollectiveFusedMoEPlan(NativeMoEPlan):
    """BF16 local GEMM fusion without a direct-access-domain requirement.

    Cross-domain transport still uses fixed-capacity PCCL collectives. This
    fuses the local GEMM-to-return-ring scatter, not RDMA transactions themselves.
    """

    def __init__(self, *args, gemm_configs=None, fuse_return=True,
                 fuse_activation_backward=False, **kwargs):
        if kwargs.get("compute", "bf16") != "bf16" or kwargs.get("m_schedule", "nopad") != "nopad":
            raise ValueError("collective-fused training requires BF16 nopad compute")
        if not isinstance(fuse_return, bool) or not isinstance(fuse_activation_backward, bool):
            raise TypeError("collective fusion switches must be bool")
        super().__init__(*args, **kwargs)
        self.fuse_return, self.fuse_activation_backward = fuse_return, fuse_activation_backward
        self.gemm_configs = {} if gemm_configs is None else {name: tuple(value) for name, value in gemm_configs.items()}
        if set(self.gemm_configs) - {"fc1", "fc2", "dhidden", "dx"}:
            raise ValueError("unknown collective-fused GEMM role")
        self.identity_rows = torch.arange(self.rows, dtype=torch.int32, device=self.device)
        self.work_info = torch.empty((self.rows // 128, 4), dtype=torch.int32, device=self.device)
        self.num_blocks = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.local_tables = {}
        for name in ("gathered_x", "route_values"):
            address = getattr(self, name).data_ptr()
            address = address if address < 2**63 else address - 2**64
            self.local_tables[name] = torch.tensor([address], dtype=torch.int64, device=self.device)
        self.buffers.update(identity_rows=self.identity_rows, work_info=self.work_info,
                            num_blocks=self.num_blocks,
                            gathered_x_pointer=self.local_tables["gathered_x"],
                            route_values_pointer=self.local_tables["route_values"])

    def _prepare_local(self, w1, w2):
        # Compile every new kernel before all ranks agree to enter payload
        # collectives. The zero work count makes GEMM preparation data-free.
        self._build_work(compile_only=True)
        super()._prepare_local(w1, w2)

    def _build_work(self, compile_only=False):
        dg._C._moe_training_peer_launch(3,
            [self.offsets, self.lengths, self.work_info, self.num_blocks],
            [self.e, self.rows, 128], compile_only)

    def _route_chunk(self, x, ids, valid_tokens, start):
        super()._route_chunk(x, ids, valid_tokens, start)
        self._build_work()

    def _gemm(self, a, weight, output, name, prequantized=False):
        role = {"w1": "fc1", "w2": "fc2", "w2_t": "dhidden", "w1_t": "dx"}[name]
        returned = self.fuse_return and role in ("fc2", "dx")
        if returned:
            self.route_values.zero_()
        dg._C._moe_training_peer_gemm(
            a, weight, output, self.local_tables["gathered_x"], self.local_tables["route_values"],
            self.identity_rows, self.permutation, self.work_info, self.num_blocks,
            self.size * self.chunk_tokens, self.routes, False, returned, False, False,
            self.gemm_configs.get(role, ()),
            self.gate_up if self.fuse_activation_backward and role == "dhidden" else None,
        )

    def _activation(self, backward=False):
        if backward and self.fuse_activation_backward:
            return False
        return super()._activation(backward)

    def _scatter_output(self):
        if self.fuse_return:
            self._return_routes()
        else:
            super()._scatter_output()
