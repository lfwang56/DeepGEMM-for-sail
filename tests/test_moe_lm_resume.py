"""Compare uninterrupted and checkpoint-resumed complete-model training."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


def norm_error(actual, expected):
    difference, reference = 0.0, 0.0
    for a, b in zip(actual, expected):
        difference += float((a.double() - b.double()).square().sum())
        reference += float(b.double().square().sum())
    return (difference / max(reference, 1e-30)) ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "peer"), default="native")
    parser.add_argument("--compute", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--directory", required=True)
    options = parser.parse_args()
    directory = Path(options.directory)
    directory.mkdir(parents=True, exist_ok=True)
    uninterrupted, repeated, resumed = directory / "continuous.pt", directory / "repeated.pt", directory / "resumed.pt"
    common = [sys.executable, str(Path(__file__).with_name("train_moe_lm.py")),
              "--backend", options.backend, "--compute", options.compute, "--graph", "--steps", "400",
              "--seed", "7301", "--hidden", "256", "--layers", "2", "--experts", "4",
              "--batch", "2", "--sequence", "64", "--chunk-tokens", "64", "--eval-interval", "100"]
    for checkpoint, extra in ((uninterrupted, []), (repeated, []),
                              (resumed, ["--stop-after", "200"]), (resumed, ["--resume"])):
        subprocess.run(common + ["--checkpoint", str(checkpoint)] + extra, check=True)
    a = torch.load(uninterrupted, map_location="cpu", weights_only=False)
    b = torch.load(resumed, map_location="cpu", weights_only=False)
    c = torch.load(repeated, map_location="cpu", weights_only=False)
    assert a["step"] == b["step"] == 400
    assert torch.equal(a["generator_state"], b["generator_state"])
    assert a["manifest"] == b["manifest"] and a["configuration"] == b["configuration"]
    assert a["model"].keys() == b["model"].keys()
    parameter_error = norm_error(a["model"].values(), b["model"].values())
    repeat_parameter_error = norm_error(a["model"].values(), c["model"].values())
    moment_errors = {}
    for key in ("exp_avg", "exp_avg_sq"):
        moment_errors[key] = norm_error([state[key] for state in a["optimizer"]["state"].values()],
                                       [state[key] for state in b["optimizer"]["state"].values()])
    optimizer_steps = [float(state["step"]) for state in b["optimizer"]["state"].values()]
    loss_difference = abs(a["validation_loss"] - b["validation_loss"])
    result = {"case": "lm_checkpoint_resume", "backend": options.backend, "compute": options.compute,
              "continuous_validation_loss": a["validation_loss"], "resumed_validation_loss": b["validation_loss"],
              "validation_loss_difference": loss_difference, "parameter_relative_l2": parameter_error,
              "independent_repeat_parameter_relative_l2": repeat_parameter_error,
              "independent_repeat_validation_loss_difference": abs(a["validation_loss"] - c["validation_loss"]),
              "moment_relative_l2": moment_errors, "all_optimizer_steps_400": all(step == 400 for step in optimizer_steps),
              "batch_generator_identical": True}
    print(json.dumps(result), flush=True)
    # Hard routing can amplify nondeterministic floating-point reductions over
    # hundreds of steps. Compare drift against an independent uninterrupted
    # repeat; require exact state restoration at the checkpoint in the trainer.
    if not (parameter_error < max(0.002, 2 * repeat_parameter_error) and loss_difference < 0.02
            and all(step == 400 for step in optimizer_steps)):
        raise AssertionError(result)


if __name__ == "__main__":
    main()
