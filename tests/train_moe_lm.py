"""Complete small MoE language-model convergence on pinned public text.

The dense Torch MoE is a numerical/convergence reference: it computes every
expert and is deliberately not a fair performance denominator for sparse MoE.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

import torch
from torch import nn
from torch.nn import functional as F

from deep_gemm.training import NativeMoEPlan, PeerMoEPlan, native_moe


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_BLOB = "7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9"


@dataclass
class Configuration:
    layers: int = 4
    hidden: int = 512
    heads: int = 8
    intermediate: int = 1024
    experts: int = 8
    topk: int = 2
    sequence: int = 128
    batch: int = 8
    chunk_tokens: int = 256
    auxiliary_coefficient: float = 0.01


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.hidden, 3 * config.hidden, bias=False)
        self.output = nn.Linear(config.hidden, config.hidden, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(config.sequence, config.sequence, dtype=torch.bool), diagonal=1))

    def forward(self, x):
        b, s, h = x.shape
        q, k, v = self.qkv(x).reshape(b, s, 3, self.heads, h // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-1, -2)).float() * ((h // self.heads) ** -0.5)
        probabilities = scores.masked_fill(self.mask[:s, :s], -float("inf")).softmax(-1).to(v.dtype)
        values = (probabilities @ v).transpose(1, 2).reshape(b, s, h)
        return self.output(values)


class Experts(nn.Module):
    def __init__(self, config, backend, plan):
        super().__init__()
        self.config, self.backend, self.plan = config, backend, plan
        h, i, e = config.hidden, config.intermediate, config.experts
        self.w1 = nn.Parameter(torch.empty(e, 2 * i, h))
        self.w2 = nn.Parameter(torch.empty(e, h, i))
        self.router = nn.Linear(h, e, bias=False)
        self.shared_up = nn.Linear(h, i, bias=False)
        self.shared_down = nn.Linear(i // 2, h, bias=False)
        self.shared_gate = nn.Linear(h, 1, bias=False)
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.config.hidden).bfloat16()
        logits = self.router(flat).float()
        probabilities = logits.softmax(-1)
        selected, ids = probabilities.topk(self.config.topk, dim=-1)
        scores = (selected / selected.sum(-1, keepdim=True)).contiguous()
        if self.backend in ("native", "peer"):
            routed = native_moe(flat, ids, scores, self.w1, self.w2, plan=self.plan)
        else:
            # One dense FC1 over all experts; then an independent batched FC2.
            gu = F.linear(flat, self.w1.bfloat16().flatten(0, 1)).reshape(flat.shape[0], self.config.experts, -1)
            gate, up = gu.float().chunk(2, -1)
            hidden = (F.silu(gate) * up).bfloat16()
            all_outputs = torch.bmm(hidden.transpose(0, 1).contiguous(), self.w2.bfloat16().transpose(1, 2)).transpose(0, 1)
            chosen = all_outputs.gather(1, ids[:, :, None].expand(-1, -1, self.config.hidden))
            routed = (chosen.float() * scores[:, :, None]).sum(1).bfloat16()
        shared_gu = self.shared_up(flat).float()
        shared_gate, shared_up = shared_gu.chunk(2, -1)
        shared = self.shared_down((F.silu(shared_gate) * shared_up).bfloat16())
        combined = routed.float() + self.shared_gate(flat).float().sigmoid() * shared.float()
        one_hot = (ids[:, :, None] == torch.arange(self.config.experts, device=ids.device)).float()
        load = one_hot.mean((0, 1)).detach()
        auxiliary = self.config.experts * (probabilities.mean(0) * load).sum()
        return combined.reshape(shape), auxiliary


class Block(nn.Module):
    def __init__(self, config, backend, plan):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden)
        self.attention = Attention(config)
        self.moe_norm = nn.LayerNorm(config.hidden)
        self.experts = Experts(config, backend, plan)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x)).float()
        moe, auxiliary = self.experts(self.moe_norm(x))
        return x + moe, auxiliary


class LanguageModel(nn.Module):
    def __init__(self, config, vocabulary, backend, plan):
        super().__init__()
        self.token = nn.Embedding(vocabulary, config.hidden)
        self.position = nn.Embedding(config.sequence, config.hidden)
        self.blocks = nn.ModuleList(Block(config, backend, plan) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.hidden)
        self.head = nn.Linear(config.hidden, vocabulary, bias=False)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token(tokens) + self.position(positions)
        auxiliary = x.new_zeros(())
        for block in self.blocks:
            x, aux = block(x)
            auxiliary = auxiliary + aux
        logits = self.head(self.norm(x)).float()
        cross_entropy = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return cross_entropy, auxiliary / len(self.blocks)


def read_data(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
    data = path.read_bytes()
    git_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if git_hash != DATA_BLOB:
        raise RuntimeError("public text differs from the pinned Git blob")
    text = data.decode("utf-8")
    characters = sorted(set(text))
    indices = {character: index for index, character in enumerate(characters)}
    tokens = torch.tensor([indices[character] for character in text], dtype=torch.long)
    split = int(0.9 * len(tokens))
    return tokens[:split], tokens[split:], len(characters), {
        "url": DATA_URL, "git_blob": git_hash, "sha256": hashlib.sha256(data).hexdigest(),
        "characters": len(text), "vocabulary": len(characters), "train_characters": split,
        "validation_characters": len(tokens) - split,
    }


def batch(data, config, generator):
    starts = torch.randint(len(data) - config.sequence - 1, (config.batch,), generator=generator)
    windows = data[starts[:, None] + torch.arange(config.sequence + 1)[None, :]]
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


@torch.no_grad()
def evaluate(model, data, config, device, batches):
    generator = torch.Generator().manual_seed(902103)
    losses = []
    model.eval()
    for _ in range(batches):
        x, y = batch(data, config, generator)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            loss, _ = model(x.to(device), y.to(device))
        losses.append(loss)
    result = float(torch.stack(losses).mean())
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "peer", "torch"), default="native")
    parser.add_argument("--compute", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--stop-after", type=int, help="clean checkpoint boundary without changing the planned learning-rate schedule")
    parser.add_argument("--seed", type=int, default=4827)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--data", default="build/campaign/data/tinyshakespeare.txt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    options = parser.parse_args()
    if options.backend == "torch" and options.compute != "bf16":
        raise ValueError("the independent Torch reference uses BF16")
    if options.backend == "peer" and options.compute != "bf16":
        raise ValueError("the peer-fused backend currently uses BF16")
    end_step = options.steps if options.stop_after is None else options.stop_after
    if not 0 < end_step <= options.steps:
        raise ValueError("stop-after must be within the planned training steps")
    config = Configuration(hidden=options.hidden, intermediate=2 * options.hidden,
                           layers=options.layers, experts=options.experts, batch=options.batch,
                           sequence=options.sequence, chunk_tokens=options.chunk_tokens)
    train, validation, vocabulary, manifest = read_data(options.data)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(options.seed)
    plan_class = PeerMoEPlan if options.backend == "peer" else NativeMoEPlan
    plan = plan_class(config.hidden, config.intermediate, config.experts, config.topk,
                        config.batch * config.sequence, chunk_tokens=config.chunk_tokens,
                        compute=options.compute, device=device, check_errors=False) if options.backend != "torch" else None
    model = LanguageModel(config, vocabulary, options.backend, plan).to(device)
    learning_rate = torch.tensor(3e-4, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95),
                                 weight_decay=0.1, capturable=True, foreach=False)
    generator = torch.Generator().manual_seed(options.seed + 100)
    checkpoint = Path(options.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    start_step, original_initial_loss = 0, None
    restored = None
    if options.resume and checkpoint.exists():
        restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (restored["configuration"] != asdict(config) or restored["manifest"] != manifest
                or restored["compute"] != options.compute or restored["backend"] != options.backend
                or restored.get("seed") != options.seed or restored.get("planned_steps") != options.steps):
            raise ValueError("checkpoint configuration/data/recipe mismatch")
        model.load_state_dict(restored["model"])
        start_step = restored["step"]
        generator.set_state(restored["generator_state"])
        original_initial_loss = restored["initial_validation_loss"]
    if end_step <= start_step:
        raise ValueError("checkpoint has already reached the requested stop boundary")
    # Keep a CPU copy for restoring parameters after library/optimizer warm-up.
    state_before_warmup = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    static_x = torch.zeros((config.batch, config.sequence), device=device, dtype=torch.long)
    static_y = torch.zeros_like(static_x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = None
    with torch.cuda.stream(stream):
        initial_loss = evaluate(model, validation, config, device, options.eval_batches)
        if original_initial_loss is None: original_initial_loss = initial_loss
        initial_hash = hashlib.sha256(b"".join(value.numpy().tobytes() for value in state_before_warmup.values())).hexdigest()
        print(json.dumps({"case": "lm_start", "configuration": asdict(config), "manifest": manifest,
                          "backend": options.backend, "compute": options.compute, "graph": options.graph,
                          "seed": options.seed, "parameters": sum(p.numel() for p in model.parameters()),
                          "initial_state_sha256": initial_hash, "initial_validation_loss": initial_loss,
                          "resume_step": start_step, "workspace_bytes": plan.workspace_bytes if plan else None}), flush=True)

        def step():
            if plan is not None:
                plan.reset_errors()
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                ce, auxiliary = model(static_x, static_y)
                loss = ce + config.auxiliary_coefficient * auxiliary
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            optimizer.step()
            return ce, auxiliary, norm

        warm_generator = torch.Generator().manual_seed(10911)
        warm_x, warm_y = batch(train, config, warm_generator)
        static_x.copy_(warm_x)
        static_y.copy_(warm_y)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            step()
        model.load_state_dict(state_before_warmup)
        if restored is None:
            for state in optimizer.state.values():
                state["step"].zero_()
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
        else:
            optimizer.load_state_dict(restored["optimizer"])
            for group in optimizer.param_groups: group["lr"] = learning_rate
            for name, value in model.state_dict().items():
                if not torch.equal(value.detach().cpu(), state_before_warmup[name]):
                    raise AssertionError(("checkpoint model restoration", name))
            current_optimizer = optimizer.state_dict()["state"]
            for index, saved_state in restored["optimizer"]["state"].items():
                for name, value in saved_state.items():
                    actual = current_optimizer[index][name]
                    if torch.is_tensor(value) and not torch.equal(actual.detach().cpu(), value):
                        raise AssertionError(("checkpoint optimizer restoration", index, name))
            if not torch.equal(generator.get_state(), restored["generator_state"]):
                raise AssertionError("checkpoint batch generator restoration")
            if abs(initial_loss - restored["validation_loss"]) > 1e-6:
                raise AssertionError("checkpoint validation loss changed before resumed training")
            print(json.dumps({"case": "lm_checkpoint_loaded", "step": start_step,
                              "model_bitwise_equal": True, "optimizer_bitwise_equal": True,
                              "batch_generator_bitwise_equal": True,
                              "validation_loss_difference": abs(initial_loss - restored["validation_loss"])}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        if options.graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                ce, auxiliary, norm = step()
        begin = time.perf_counter()
        last_validation = initial_loss
        for iteration in range(start_step, end_step):
            x, y = batch(train, config, generator)
            static_x.copy_(x)
            static_y.copy_(y)
            warmup = min(100, max(1, options.steps // 10))
            if iteration < warmup:
                lr = 3e-4 * (iteration + 1) / warmup
            else:
                fraction = (iteration - warmup) / max(1, options.steps - warmup)
                lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * fraction))
            learning_rate.fill_(lr)
            if graph is None:
                optimizer.zero_grad(set_to_none=True)
                ce, auxiliary, norm = step()
            else:
                graph.replay()
            if plan is not None and (iteration + 1) % options.eval_interval == 0:
                plan.check()
            if (iteration + 1) % options.eval_interval == 0 or iteration + 1 == end_step:
                last_validation = evaluate(model, validation, config, device, options.eval_batches)
                values = dict(train_cross_entropy=float(ce.detach()), auxiliary_loss=float(auxiliary.detach()),
                              gradient_norm=float(norm), validation_loss=last_validation)
                if not all(math.isfinite(value) for value in values.values()):
                    raise RuntimeError(("non-finite LM training", iteration + 1, values))
                print(json.dumps({"case": "lm_progress", "step": iteration + 1,
                                  "backend": options.backend, "compute": options.compute,
                                  "learning_rate": lr, "elapsed_seconds": time.perf_counter() - begin,
                                  **values}), flush=True)
            if (iteration + 1) % options.checkpoint_interval == 0 or iteration + 1 == end_step:
                payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                               configuration=asdict(config), manifest=manifest, compute=options.compute,
                               backend=options.backend, step=iteration + 1, generator_state=generator.get_state(),
                               seed=options.seed, planned_steps=options.steps,
                               initial_validation_loss=original_initial_loss, validation_loss=last_validation)
                temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
                torch.save(payload, temporary)
                temporary.replace(checkpoint)
        torch.cuda.synchronize()
        print(json.dumps({"case": "lm_finished", "backend": options.backend, "compute": options.compute,
                          "graph": options.graph, "seed": options.seed, "steps": end_step,
                          "planned_steps": options.steps,
                          "initial_validation_loss": original_initial_loss,
                          "final_validation_loss": last_validation,
                          "elapsed_seconds": time.perf_counter() - begin,
                          "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
                          "checkpoint": str(checkpoint)}), flush=True)
        del graph


if __name__ == "__main__":
    main()
