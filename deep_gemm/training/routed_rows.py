"""Device-only windows over the dropless, globally sorted local expert rows."""
import torch
import deep_gemm as dg


def allocate_route_windows(plan):
    def allocate(name, shape):
        tensor = torch.empty(shape, device=plan.device, dtype=torch.int32)
        setattr(plan, name, tensor)
        plan.buffers[name] = tensor
    allocate('global_offsets', (plan.e + 1,))
    allocate('global_lengths', (plan.e,))
    allocate('global_permutation', (plan.full_row_capacity,))
    allocate('global_source_rows', (plan.full_row_capacity,))
    if not hasattr(plan, 'source_rows'):
        allocate('source_rows', (plan.rows,))
    if not hasattr(plan, 'group_ids'):
        allocate('group_ids', (plan.rows,))
    plan.route_window_count = (plan.full_row_capacity + plan.rows - 1) // plan.rows


def set_route_window(plan, start, compile_only=False):
    dg._C._moe_training_peer_launch(14,
        [plan.global_offsets, plan.global_permutation, plan.offsets, plan.lengths,
         plan.permutation, plan.source_rows, plan.status, plan.group_ids],
        [plan.e, plan.full_row_capacity, plan.rows, start, plan.k,
         plan.size * plan.chunk_tokens], compile_only)
