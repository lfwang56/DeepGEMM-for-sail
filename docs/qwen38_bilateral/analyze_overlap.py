"""Summarize exported Asys kernel intervals without calling fused kernels pure compute."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def merge(intervals):
    result = []
    for begin, end in sorted(intervals):
        if result and begin <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([begin, end])
    return result


def duration(intervals):
    return sum(end - begin for begin, end in merge(intervals))


def intersection(left, right):
    left, right = merge(left), merge(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def exclusive_activity(intervals):
    """Partition kernel union into disjoint sets of simultaneously active categories."""
    events = defaultdict(lambda: defaultdict(int))
    for label, spans in intervals.items():
        if label == 'harness_collective':
            continue
        for begin, end in spans:
            events[begin][label] += 1
            events[end][label] -= 1
    active, result = defaultdict(int), defaultdict(int)
    previous = None
    for time in sorted(events):
        labels = sorted(label for label, count in active.items() if count)
        if previous is not None and labels:
            result[' & '.join(labels)] += time - previous
        for label, change in events[time].items():
            active[label] += change
        previous = time
    return result


def category(name):
    if name.startswith(('bf16_peer_ffn_', 'bf16_peer_dag_')):
        return 'fused_compute_communication'
    if name in ('moe_training_peer_15', 'moe_training_peer_16'):
        return 'explicit_peer_copy'
    if name in ('moe_training_peer_13', 'moe_training_peer_17'):
        return 'expert_gradient_communication_and_pointwise'
    if 'pcclKernel' in name:
        return 'harness_collective'
    if 'deep_ep::' in name:
        return 'communication_layout' if 'get_dispatch_layout' in name else 'communication_and_notify'
    if name.startswith('bf16_peer_'):
        if '_external_dispatch' in name:
            return 'local_compute_with_ready_wait'
        if '_rtrue_' in name or '_wtrue_' in name or '_pipeline' in name or '_replay_grad' in name:
            return 'fused_compute_communication'
        return 'local_compute'
    if name.startswith(('bf16_grouped_', 'bf16_k_grouped_')):
        return 'local_compute'
    if name == 'moe_training_peer_5':
        return 'rank_barrier_wait'
    if name in ('moe_training_peer_4', 'moe_training_peer_6',
                'moe_training_peer_7', 'moe_training_peer_11', 'moe_training_peer_12'):
        return 'explicit_peer_copy'
    if name == 'moe_training_22':
        return 'weight_transpose'
    return 'routing_pointwise_local_copy_zero'


def analyze(path, steps, chunks=6):
    connection = sqlite3.connect(path)
    rows = list(connection.execute('''
        SELECT k.start,k.end,s.value,k.streamId,k.gridX,k.blockX
        FROM HGPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start
    '''))
    intervals, kernels = defaultdict(list), defaultdict(list)
    phase_categories, roles = defaultdict(int), defaultdict(int)
    read_index = gemm_index = wgrad_index = 0
    phase = 'setup'
    stream_ids = defaultdict(set)
    for start, end, name, stream, *_ in rows:
        if name == 'moe_training_13':
            phase = 'forward' if (read_index % (2 * chunks)) < chunks else 'backward'
            read_index += 1
            gemm_index = wgrad_index = 0
        if name == 'moe_training_22':
            phase = 'backward'
        label = category(name)
        intervals[label].append((start, end))
        kernels[name].append((start, end))
        stream_ids[label].add(stream)
        phase_categories[(phase, label)] += end - start
        if name == 'bf16_grouped_deep_gemm_NoPad':
            names = ('fc1', 'fc2') if phase == 'forward' else ('replay_fc1', 'replay_fc2', 'dhidden', 'dx')
            roles[names[gemm_index % len(names)]] += end-start
            gemm_index += 1
        elif name == 'bf16_k_grouped_gemm':
            roles[('dw2', 'dw1')[wgrad_index % 2]] += end-start
            wgrad_index += 1
    compute = intervals['local_compute'] + intervals['local_compute_with_ready_wait']
    communication = intervals['communication_and_notify'] + intervals['explicit_peer_copy']
    mixed = intervals['fused_compute_communication']
    wgrad = [span for name, spans in kernels.items() if name.startswith('bf16_k_grouped_') for span in spans]
    other_compute = [span for name, spans in kernels.items()
                     if category(name) in ('local_compute', 'local_compute_with_ready_wait') and not name.startswith('bf16_k_grouped_')
                     for span in spans]
    relevant = [(a, b) for a, b, n, *_ in rows if category(n) != 'harness_collective']
    return {
        'source': str(path), 'profiled_steps': steps,
        'scope': 'rank0 instrumented full FWD+recomputed BWD; routed experts only',
        'categories': {
            name: {'sum_ms_per_step': sum(b-a for a,b in values) / steps / 1e6,
                   'union_ms_per_step': duration(values) / steps / 1e6,
                   'streams': sorted(stream_ids[name])}
            for name, values in intervals.items() if values
        },
        'explicit_communication_compute_intersection_ms_per_step':
            intersection(compute, communication) / steps / 1e6,
        'explicit_communication_mixed_kernel_intersection_ms_per_step':
            intersection(communication, mixed) / steps / 1e6,
        'local_compute_mixed_kernel_intersection_ms_per_step':
            intersection(compute, mixed) / steps / 1e6,
        'wgrad_mixed_kernel_intersection_ms_per_step':
            intersection(wgrad, mixed) / steps / 1e6,
        'wgrad_other_compute_intersection_ms_per_step':
            intersection(wgrad, other_compute) / steps / 1e6,
        'wgrad_explicit_communication_intersection_ms_per_step':
            intersection(wgrad, communication) / steps / 1e6,
        'exclusive_activity_ms_per_step': {
            name: value / steps / 1e6 for name, value in exclusive_activity(intervals).items()},
        'kernel_union_ms_per_step': duration(relevant) / steps / 1e6,
        'local_compute_roles_ms_per_step': {name: value / steps / 1e6 for name, value in roles.items()},
        'phase_categories_ms_per_step': {phase: {
            label: value / steps / 1e6 for (p, label), value in phase_categories.items() if p == phase
        } for phase in ('setup', 'forward', 'backward')},
        'kernels': sorted([
            {'name': name, 'category': category(name), 'calls': len(values),
             'sum_ms_per_step': sum(b-a for a,b in values) / steps / 1e6}
            for name, values in kernels.items()
        ], key=lambda row: -row['sum_ms_per_step']),
        'limits': [
            'Fused GEMM kernels include peer traffic; their duration is not pure compute.',
            'External-input GEMMs have local operands but include tile-ready waits; their span is not active MMA time.',
            'Intersection with a mixed kernel measures concurrent kernel spans, not physical wire or MMA overlap.',
            'Category sums can exceed step union when kernels run concurrently; exclusive activity partitions do not double count.',
            'Zero explicit-kernel intersection does not exclude instruction-level overlap within a kernel.',
            'Per-rank instrumented kernel sums are not uninstrumented max-rank step latency.',
            'Communication includes payload, control notifications and synchronization inside DeepEP kernels.'
            ,f'Phase labels infer {chunks} source chunks per forward/backward; setup gradient-zero kernels near step boundaries can be attributed to the previous phase.'
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--chunks', type=int, default=6)
    parser.add_argument('--output', required=True)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    if args.chunks < 1 or args.steps < 1:
        parser.error('positive steps and source chunks required')
    results = [analyze(path, args.steps, args.chunks) for path in args.paths]
    Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k != 'kernels'}, indent=2))
