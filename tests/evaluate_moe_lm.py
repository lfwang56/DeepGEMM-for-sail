"""Evaluate every successor character in the pinned held-out text once.

Context resets at nonoverlapping sequence boundaries. The last batch is padded
on the right and its extra targets are ignored. This is separate from the fixed
sampled validation batches used to monitor training.
"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import torch

from deep_gemm.training import NativeMoEPlan, PeerMoEPlan, PersistentMoEPlan, CollectiveFusedMoEPlan
from train_moe_lm import Configuration, LanguageModel, read_data


@torch.no_grad()
def evaluate_checkpoint(path, data_path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = Configuration(**saved["configuration"])
    _, validation, vocabulary, manifest = read_data(data_path)
    if manifest != saved["manifest"]:
        raise ValueError("checkpoint and validation text differ")
    device = torch.device("cuda", 0)
    backend, compute = saved["backend"], saved["compute"]
    classes = {"native": NativeMoEPlan, "peer": PeerMoEPlan, "persistent": PersistentMoEPlan,
               "collective_fused": CollectiveFusedMoEPlan}
    backend_options = dict(saved.get("backend_options", {}))
    if backend == "peer" and "backend_options" not in saved:
        training_options = saved.get("plan_options", {})
        for key in ("dispatch_cache_n", "dispatch_cache_ctas", "dispatch_cache_aiu", "reuse_packed_dy", "staged_return",
                    "dispatch_producer", "dispatch_producer_tiles"):
            if key in training_options:
                backend_options[key] = training_options[key]
        if training_options.get("gemm_config"):
            backend_options["gemm_configs"] = {
                role: training_options["gemm_config"] for role in ("fc1", "fc2", "dhidden", "dx")}
        if training_options.get("peer_fc1_config"):
            backend_options.setdefault("gemm_configs", {})["fc1"] = training_options["peer_fc1_config"]
    plan = None if backend == "torch" else classes[backend](
        config.hidden, config.intermediate, config.experts, config.topk,
        config.batch * config.sequence, chunk_tokens=config.chunk_tokens,
        compute=compute, device=device, check_errors=False, **backend_options,
    )
    model = LanguageModel(config, vocabulary, backend, plan).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    targets_count = len(validation) - 1
    width = config.batch * config.sequence
    summed_loss = torch.zeros((), dtype=torch.float64, device=device)
    evaluated = 0
    for start in range(0, targets_count, width):
        indices = start + torch.arange(width)
        valid = indices < targets_count
        safe = indices.clamp_max(targets_count - 1)
        x = validation[safe].reshape(config.batch, config.sequence)
        y = validation[safe + 1].masked_fill(~valid, -100).reshape(config.batch, config.sequence)
        count = min(width, targets_count - start)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            loss, _ = model(x.to(device), y.to(device))
        summed_loss.add_(loss.double() * count)
        evaluated += count
    if evaluated != targets_count:
        raise AssertionError((evaluated, targets_count))
    if plan is not None:
        plan.check()
    loss = float(summed_loss / evaluated)
    if not math.isfinite(loss):
        raise AssertionError(loss)
    with Path(path).open("rb") as checkpoint_file:
        checkpoint_hash = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    print(json.dumps({
        "case": "lm_full_validation", "checkpoint": str(path), "checkpoint_sha256": checkpoint_hash,
        "backend": backend, "compute": compute, "seed": saved.get("seed"), "step": saved["step"],
        "backend_options": backend_options,
        "seed_recorded_in_checkpoint": "seed" in saved,
        "configuration": saved["configuration"], "manifest": manifest,
        "targets_evaluated": evaluated, "validation_cross_entropy_nats": loss,
        "perplexity": math.exp(loss), "bits_per_character": loss / math.log(2),
        "context": "nonoverlapping causal windows, reset every sequence tokens; no train prefix",
        "padding_targets_ignored": True,
    }), flush=True)
    del model, plan, saved
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--data", default="build/campaign/data/tinyshakespeare.txt")
    options = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    for path in options.checkpoint:
        evaluate_checkpoint(path, options.data)


if __name__ == "__main__":
    main()
