"""Complete public-text LM with expert shards and replicated dense parameters.

This small validation harness explicitly reduces replicated gradients. The
operator's DDP/FSDP integration is tested separately. A gathered final model is
an evaluation checkpoint; rank files retain their local AdamW states.
"""

import argparse
from dataclasses import asdict
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch import nn

from deep_gemm.training.hierarchical import HierarchicalPeerMoEPlan, make_hierarchical_groups
from deep_gemm.training import NativeMoEPlan, PeerMoEPlan, PersistentMoEPlan
from train_moe_lm import Configuration, LanguageModel, batch, read_data
from moe_distributed_utils import destroy_training_process_group


def expert_parameter(name):
    return name.endswith((".experts.w1", ".experts.w2"))


class GradientReduction:
    def __init__(self, model, group):
        self.group = group
        self.size = dist.get_world_size(group)
        self.parameters = list(model.parameters())
        self.replicated = [p for name, p in model.named_parameters() if not expert_parameter(name)]
        self.experts = [p for name, p in model.named_parameters() if expert_parameter(name)]
        self.buffer = torch.empty(sum(p.numel() for p in self.replicated), device=self.parameters[0].device)
        self.views = []
        offset = 0
        for parameter in self.replicated:
            self.views.append(self.buffer[offset:offset + parameter.numel()].view_as(parameter))
            offset += parameter.numel()

    def reduce(self):
        for parameter, view in zip(self.replicated, self.views):
            view.copy_(parameter.grad)
        dist.all_reduce(self.buffer, group=self.group)
        for parameter, view in zip(self.replicated, self.views):
            parameter.grad.copy_(view)

    def clip(self):
        # Expert gradients are disjoint; replicated gradients are identical
        # after reduction and contribute only once to the global norm.
        expert_square = torch.stack([p.grad.square().sum() for p in self.experts]).sum()
        replicated_square = torch.stack([p.grad.square().sum() for p in self.replicated]).sum()
        square = expert_square + replicated_square / self.size
        dist.all_reduce(square, group=self.group)
        norm = square.sqrt()
        scale = (1.0 / (norm + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_([p.grad for p in self.parameters], scale)
        return norm


@torch.no_grad()
def evaluate(model, validation, config, rank, size, device, batches):
    generator = torch.Generator().manual_seed(902103)
    local_batch = config.batch // size
    losses = []
    model.eval()
    for _ in range(batches):
        x, y = batch(validation, config, generator)
        section = slice(rank * local_batch, (rank + 1) * local_batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            loss, _ = model(x[section].to(device), y[section].to(device))
        losses.append(loss)
    result = torch.stack(losses).mean()
    dist.all_reduce(result)
    value = float(result / size)
    model.train()
    return value


def reference_check(model, global_state, config, vocabulary, x, y, reduction, rank, size, device, compute,
                    fixed_routes=False):
    """Compare every complete-model gradient with an unsharded dense MoE."""
    model.zero_grad(set_to_none=True)
    for block in model.blocks:
        block.experts.capture_reference_ids = True
    local_batch = config.batch // size
    section = slice(rank * local_batch, (rank + 1) * local_batch)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        ce, auxiliary = model(x[section].to(device), y[section].to(device))
        loss = (ce + config.auxiliary_coefficient * auxiliary) / size
    loss.backward()
    reduction.reduce()
    actual = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()}
    if model.blocks[0].experts.plan.gate_up_layout == "g8":
        from deep_gemm.training import from_g8_weight
        actual = {name: from_g8_weight(value) if name.endswith(".experts.w1") else value
                  for name, value in actual.items()}
    selected_ids = []
    for block in model.blocks:
        local = block.experts.observed_reference_ids
        gathered = [torch.empty_like(local) for _ in range(size)]
        dist.all_gather(gathered, local)
        selected_ids.append(torch.cat(gathered))
        block.experts.capture_reference_ids = False
        del block.experts.observed_reference_ids
    reference_backend = "torch_fp8_reference" if compute == "fp8" else "torch"
    dense = LanguageModel(config, vocabulary, "torch", None).to(device)
    dense.load_state_dict(global_state)
    bf16_route_differences = None
    if compute == "fp8" or fixed_routes:
        for block in dense.blocks:
            block.experts.capture_reference_ids = True
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            dense(x.to(device), y.to(device))
        bf16_route_differences = [int((block.experts.observed_reference_ids.sort(-1).values != ids.sort(-1).values)
                                      .any(-1).sum()) for block, ids in zip(dense.blocks, selected_ids)]
    for block, ids in zip(dense.blocks, selected_ids):
        block.experts.backend = reference_backend
        block.experts.capture_reference_ids = True
        if compute == "fp8" or fixed_routes:
            # Compare the continuous gradient on the branch selected by the
            # actual forward. Real training always recomputes its own routes.
            block.experts.reference_ids = ids
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        reference_ce, reference_auxiliary = dense(x.to(device), y.to(device))
        reference_loss = reference_ce + config.auxiliary_coefficient * reference_auxiliary
    reference_loss.backward()
    route_differences = [int((block.experts.observed_reference_ids.sort(-1).values != ids.sort(-1).values)
                             .any(-1).sum()) for block, ids in zip(dense.blocks, selected_ids)]
    errors, difference, magnitude = {}, 0.0, 0.0
    local_experts = config.experts // size
    for name, parameter in dense.named_parameters():
        expected = parameter.grad.detach().cpu().double()
        if expert_parameter(name):
            expected = expected[rank * local_experts:(rank + 1) * local_experts]
        delta = (actual[name].double() - expected).square().sum().item()
        target = expected.square().sum().item()
        errors[name] = math.sqrt(delta / max(target, 1e-30))
        difference += delta
        magnitude += target
    limit = 0.13 if compute == "fp8" else 0.04
    result = {"case": "distributed_lm_reference", "rank": rank, "ep_size": size, "compute": compute,
              "reference_backend": reference_backend, "fixed_selected_ids": compute == "fp8" or fixed_routes,
              "selected_set_differences_per_layer": route_differences,
              "unfrozen_bf16_selected_set_differences_per_layer": bf16_route_differences,
              "relative_l2_all_gradients": math.sqrt(difference / max(magnitude, 1e-30)),
              "relative_l2_by_parameter": errors, "limit": limit}
    print(json.dumps(result), flush=True)
    if not all(math.isfinite(error) and error < limit for error in errors.values()):
        raise AssertionError(result)
    model.zero_grad(set_to_none=True)
    del dense, actual
    gc.collect()


def save_checkpoints(model, optimizer, config, manifest, options, generator, rank, size, validation, initial_loss):
    path = Path(options.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    common = dict(configuration=asdict(config), manifest=manifest, compute=options.compute,
                  backend="native" if options.backend == "collective" else options.backend,
                  evaluation_backend="peer" if options.backend == "hybrid" else "native" if options.backend == "collective" else options.backend,
                  hybrid_domain_size=options.hybrid_domain_size if options.backend == "hybrid" else None,
                  step=options.steps, seed=options.seed, planned_steps=options.steps,
                  plan_options=dict(block_m=options.block_m, gate_up_layout=options.gate_up_layout,
                                    fuse_fc1_activation=options.fuse_fc1_activation,
                                    fuse_dhidden_scores=options.fuse_dhidden_scores,
                                    transpose_free_dgrad=options.transpose_free_dgrad,
                                    dag_fusion=options.dag_fusion, dag_backward=options.dag_backward,
                                    dag_direct_return=options.dag_direct_return,
                                    dag_shared_pool=options.dag_fusion or options.dag_backward,
                                    input_copy_ilp=options.input_copy_ilp,
                                    dispatch_cache_n=options.dispatch_cache_n,
                                    dispatch_producer=options.dispatch_producer,
                                    producer_local_aiu=options.producer_local_aiu,
                                    producer_input_aiu=options.producer_input_aiu,
                                    fuse_activation_backward=options.fuse_activation_backward,
                                    expert_local_grad=options.expert_local_grad,
                                    ring_rows=options.ring_rows,
                                    dispatch_producer_tiles=options.dispatch_producer_tiles,
                                    dispatch_cache_ctas=options.dispatch_cache_ctas,
                                    dispatch_cache_aiu=options.dispatch_cache_aiu,
                                    staged_return=options.staged_return,
                                    reuse_packed_dy=options.reuse_packed_dy,
                                    gemm_config=options.gemm_config, peer_fc1_config=options.peer_fc1_config),
                  initial_validation_loss=initial_loss, validation_loss=validation, ep_size=size)
    local_path = path.with_name(path.stem + f".rank{rank:03d}" + path.suffix)
    temporary = local_path.with_suffix(local_path.suffix + ".tmp")
    torch.save(dict(common, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    generator_state=generator.get_state(), checkpoint_scope="rank-local training state"), temporary)
    temporary.replace(local_path)
    global_state = {}
    for name, value in model.state_dict().items():
        if expert_parameter(name):
            gathered = [torch.empty_like(value) for _ in range(size)]
            dist.all_gather(gathered, value)
            if rank == 0:
                global_state[name] = torch.cat(gathered).cpu()
        elif rank == 0:
            global_state[name] = value.cpu()
    if rank == 0:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(common, model=global_state, checkpoint_scope="gathered evaluation model; no global optimizer"), temporary)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("collective", "peer", "persistent", "hybrid"), default="collective")
    parser.add_argument("--hybrid-domain-size", type=int, default=1)
    parser.add_argument("--compute", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--reference-check", action="store_true")
    parser.add_argument("--reference-fixed-routes", action="store_true",
                        help="Compare continuous gradients on the actual branch; training keeps dynamic routing")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4827)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--data", default="build/campaign/data/tinyshakespeare.txt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--global-batch", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--block-m", type=int, choices=(128, 256), default=128)
    parser.add_argument("--gate-up-layout", choices=("split", "g8"), default="split")
    parser.add_argument("--fuse-fc1-activation", action="store_true")
    parser.add_argument("--fuse-dhidden-scores", action="store_true")
    parser.add_argument("--transpose-free-dgrad", action="store_true")
    parser.add_argument("--dag-fusion", action="store_true")
    parser.add_argument("--dag-backward", action="store_true")
    parser.add_argument("--dag-direct-return", action="store_true")
    parser.add_argument("--input-copy-ilp", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--dispatch-cache-n", type=int, choices=(0, 2, 4, 8, 16), default=0)
    parser.add_argument("--dispatch-cache-aiu", action="store_true")
    parser.add_argument("--staged-return", action="store_true")
    parser.add_argument("--dispatch-cache-ctas", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--dispatch-producer", action="store_true")
    parser.add_argument("--producer-local-aiu", action="store_true")
    parser.add_argument("--producer-input-aiu", action="store_true")
    parser.add_argument("--fuse-activation-backward", action="store_true")
    parser.add_argument("--expert-local-grad", action="store_true")
    parser.add_argument("--ring-rows", type=int)
    parser.add_argument("--dispatch-producer-tiles", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--peer-fc1-config", type=int, nargs=5)
    parser.add_argument("--reuse-packed-dy", action="store_true")
    parser.add_argument("--gemm-config", type=int, nargs=5)
    options = parser.parse_args()
    if options.input_copy_ilp != 1 and options.backend not in ("peer", "hybrid"):
        parser.error("input copy ILP requires --backend peer")
    if (options.dag_fusion or options.dag_backward or options.dag_direct_return) and options.backend not in ("peer", "hybrid"):
        parser.error("DAG options require --backend peer")
    if (options.dispatch_cache_n or options.dispatch_cache_aiu or options.staged_return or options.reuse_packed_dy or options.gemm_config or options.peer_fc1_config or options.dispatch_cache_ctas != 2 or options.dispatch_producer or options.producer_input_aiu or options.producer_local_aiu or options.fuse_activation_backward or options.fuse_fc1_activation or options.fuse_dhidden_scores or options.transpose_free_dgrad or options.expert_local_grad or options.ring_rows is not None) and options.backend not in ("peer", "hybrid"):
        parser.error("peer kernel options require --backend peer")
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if size < 2 or options.experts % size or options.global_batch % size:
        raise ValueError("experts and global batch must be divisible by EP size >=2")
    if options.steps < 1 or min(options.eval_interval, options.eval_batches) < 1:
        raise ValueError("step/evaluation counts must be positive")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=300))
    train, validation, vocabulary, manifest = read_data(options.data)
    config = Configuration(hidden=options.hidden, intermediate=2 * options.hidden,
                           layers=options.layers, experts=options.experts, batch=options.global_batch,
                           sequence=options.sequence, chunk_tokens=options.chunk_tokens)
    torch.manual_seed(options.seed)
    # Construct the identical global model before slicing expert parameters.
    model = LanguageModel(config, vocabulary, "native", None)
    global_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    initial_hash = hashlib.sha256(b"".join(value.numpy().tobytes() for value in global_state.values())).hexdigest()
    local_experts, local_batch = config.experts // size, config.batch // size
    classes = {"collective": NativeMoEPlan, "peer": PeerMoEPlan, "persistent": PersistentMoEPlan}
    plan_options = dict(block_m=options.block_m, gate_up_layout=options.gate_up_layout)
    if options.backend in ("peer", "hybrid"):
        plan_options.update(dispatch_cache_n=options.dispatch_cache_n,
                            dispatch_producer=options.dispatch_producer,
                            producer_local_aiu=options.producer_local_aiu,
                            producer_input_aiu=options.producer_input_aiu,
                            fuse_activation_backward=options.fuse_activation_backward,
                            fuse_fc1_activation=options.fuse_fc1_activation,
                                    fuse_dhidden_scores=options.fuse_dhidden_scores,
                                    transpose_free_dgrad=options.transpose_free_dgrad,
                            dag_fusion=options.dag_fusion, dag_backward=options.dag_backward,
                            dag_direct_return=options.dag_direct_return,
                            dag_shared_pool=options.dag_fusion or options.dag_backward,
                            input_copy_ilp=options.input_copy_ilp,
                            expert_local_grad=options.expert_local_grad,
                            ring_rows=options.ring_rows,
                            dispatch_producer_tiles=options.dispatch_producer_tiles,
                            dispatch_cache_ctas=options.dispatch_cache_ctas,
                            dispatch_cache_aiu=options.dispatch_cache_aiu,
                            staged_return=options.staged_return,
                            reuse_packed_dy=options.reuse_packed_dy)
        if options.gemm_config:
            plan_options["gemm_configs"] = {role: options.gemm_config for role in ("fc1", "fc2", "dhidden", "dx")}
        if options.peer_fc1_config:
            plan_options.setdefault("gemm_configs", {})["fc1"] = options.peer_fc1_config
    hybrid_groups = None
    if options.backend == "hybrid":
        metadata = dist.new_group(backend="gloo", timeout=timedelta(seconds=300))
        hybrid_groups = make_hierarchical_groups(options.hybrid_domain_size, metadata)
        classes["hybrid"] = HierarchicalPeerMoEPlan
    plan = classes[options.backend](config.hidden, config.intermediate, local_experts, config.topk,
                                    local_batch * config.sequence, chunk_tokens=config.chunk_tokens,
                                    compute=options.compute,
                                    **({"group":dist.group.WORLD} if hybrid_groups is None else {"groups":hybrid_groups}), device=device,
                                    check_errors=False, **plan_options)
    for block in model.blocks:
        experts = block.experts
        experts.backend, experts.plan = "native", plan
        section = slice(rank * local_experts, (rank + 1) * local_experts)
        experts.w1 = nn.Parameter(experts.w1[section].clone())
        experts.w2 = nn.Parameter(experts.w2[section].clone())
        if options.gate_up_layout == "g8":
            from deep_gemm.training import to_g8_weight
            experts.w1 = nn.Parameter(to_g8_weight(experts.w1.detach()))
    model.to(device)
    reduction = GradientReduction(model, dist.group.WORLD)
    learning_rate = torch.tensor(3e-4, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95),
                                 weight_decay=0.1, capturable=True, foreach=False)
    generator = torch.Generator().manual_seed(options.seed + 100)
    warm_x, warm_y = batch(train, config, torch.Generator().manual_seed(10911))
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream())
    if options.reference_check:
        with torch.cuda.stream(stream):
            reference_check(model, global_state, config, vocabulary, warm_x, warm_y,
                            reduction, rank, size, device, options.compute, options.reference_fixed_routes)
    del global_state
    before_warmup = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    static_x = torch.empty((local_batch, config.sequence), device=device, dtype=torch.long)
    static_y = torch.empty_like(static_x)
    local_section = slice(rank * local_batch, (rank + 1) * local_batch)
    static_x.copy_(warm_x[local_section])
    static_y.copy_(warm_y[local_section])
    stream.wait_stream(torch.cuda.current_stream())
    graph = None
    with torch.cuda.stream(stream):
        initial_loss = evaluate(model, validation, config, rank, size, device, options.eval_batches)
        print(json.dumps({"case": "distributed_lm_start", "rank": rank, "ep_size": size,
                          "backend": options.backend, "compute": options.compute, "graph": options.graph,
                          "configuration": asdict(config), "manifest": manifest,
                          "plan_options": plan_options,
                          "initial_state_sha256": initial_hash, "initial_validation_loss": initial_loss,
                          "seed": options.seed}), flush=True)

        def step():
            plan.reset_errors()
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                ce, auxiliary = model(static_x, static_y)
                loss = (ce + config.auxiliary_coefficient * auxiliary) / size
            loss.backward()
            reduction.reduce()
            norm = reduction.clip()
            optimizer.step()
            return ce, auxiliary, norm

        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            step()
        model.load_state_dict(before_warmup)
        for state in optimizer.state.values():
            for name in ("step", "exp_avg", "exp_avg_sq"):
                state[name].zero_()
        optimizer.zero_grad(set_to_none=True)
        if options.graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                ce, auxiliary, norm = step()
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        last_validation = initial_loss
        for iteration in range(options.steps):
            x, y = batch(train, config, generator)
            static_x.copy_(x[local_section])
            static_y.copy_(y[local_section])
            warmup = min(100, max(1, options.steps // 10))
            fraction = (iteration - warmup) / max(1, options.steps - warmup)
            lr = (3e-4 * (iteration + 1) / warmup if iteration < warmup else
                  3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * fraction)))
            learning_rate.fill_(lr)
            if graph is None:
                optimizer.zero_grad(set_to_none=True)
                ce, auxiliary, norm = step()
            else:
                graph.replay()
            if (iteration + 1) % options.eval_interval == 0 or iteration + 1 == options.steps:
                plan.check()
                last_validation = evaluate(model, validation, config, rank, size, device, options.eval_batches)
                metrics = torch.stack((ce.detach(), auxiliary.detach(), norm.detach()))
                dist.all_reduce(metrics)
                values = metrics.cpu().tolist()
                if not all(math.isfinite(value) for value in (*values, last_validation)):
                    raise AssertionError("nonfinite distributed training metrics")
                if rank == 0:
                    print(json.dumps({"case": "distributed_lm_progress", "step": iteration + 1,
                                      "backend": options.backend, "compute": options.compute, "ep_size": size,
                                      "train_cross_entropy": values[0] / size, "auxiliary_loss": values[1] / size,
                                      "gradient_norm": values[2] / size, "validation_loss": last_validation,
                                      "elapsed_seconds": time.perf_counter() - begin}), flush=True)
        save_checkpoints(model, optimizer, config, manifest, options, generator, rank, size, last_validation, initial_loss)
        torch.cuda.synchronize(device)
        print(json.dumps({"case": "distributed_lm_finished", "rank": rank, "ep_size": size,
                          "backend": options.backend, "compute": options.compute, "seed": options.seed,
                          "graph": options.graph, "steps": options.steps,
                          "initial_validation_loss": initial_loss, "final_validation_loss": last_validation,
                          "elapsed_seconds": time.perf_counter() - begin, "checkpoint": options.checkpoint}), flush=True)
        del graph
        optimizer.zero_grad(set_to_none=True)
        del ce, auxiliary, norm
        gc.collect()
    del model, optimizer, reduction, plan, static_x, static_y, stream
    gc.collect()
    if hybrid_groups is not None:
        hybrid_groups.close()
    destroy_training_process_group()


if __name__ == "__main__":
    main()
