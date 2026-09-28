#!/usr/bin/env python3
"""Recompute held-out action metrics and inspect saturation, without game input."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_multitrack_vision_curriculum import (
    extract_causal_sequence_dataset, file_sha256, write_metrics,
)
from tmrl.custom.torch.car_brain import MultiModalCarBrain


def collect(model, data, device, amp=False):
    predictions, logits = [], []
    current_logits = []
    def capture(_module, inputs):
        current_logits.append(inputs[0].detach().float())
    hook = model.policy_head[-1].register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for start in range(0, len(data[0]), 8):
                imgs, states, previous, _targets, valid = [value[start:start + 8].to(device) for value in data]
                with torch.amp.autocast("cuda") if amp else nullcontext():
                    pred = model(imgs.float() / 255, states, prev_actions=previous)
                predictions.append(pred.float()[valid].cpu().numpy())
                logits.append(current_logits.pop()[valid].cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(predictions), np.concatenate(logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh diagnostic output path")
    data = extract_causal_sequence_dataset(args.dataset, window_len=16, stride=16,
                                          split="validation", return_mask=True)
    model = MultiModalCarBrain(d_model=256, n_layers=6, n_heads=8, max_context_len=64)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(args.device).eval()
    pred, logits = collect(model, data, args.device)
    target = data[3][data[4]].numpy()
    speed = data[1][..., 0][data[4]].numpy() * 100
    difference = np.abs(pred.astype(np.float64) - target)
    huber = np.where(difference < 0.1, difference**2 / 0.2, difference - 0.05)
    report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
              "dataset_sha256": file_sha256(args.dataset), "valid_window_tokens": len(target),
              "independent_numpy_objective": float(huber.mean() + 2.5 * huber[:, 2].mean()),
              "independent_numpy_action_mae": difference.mean(axis=0).tolist(),
              "quantile_levels": [0, 0.1, 0.5, 0.9, 1], "channels": {},
              "limitations": ["Teacher-forced observations from one held-out lap; not a closed-loop test",
                              "Metrics count window tokens, including terminal-window overlap"]}
    amp_pred, amp_logits = collect(model, data, args.device, amp=True) if torch.device(args.device).type == "cuda" else (None, None)
    for i, channel in enumerate(("gas", "brake", "steer")):
        row = {"prediction_quantiles": np.quantile(pred[:, i], report["quantile_levels"]).tolist(),
               "target_quantiles": np.quantile(target[:, i], report["quantile_levels"]).tolist(),
               "logit_quantiles": np.quantile(logits[:, i], report["quantile_levels"]).tolist(),
               "mean_tanh_derivative_fp32": float(np.mean(1 - pred[:, i]**2))}
        if amp_pred is not None:
            row.update(amp_exact_endpoint_fraction=float(np.mean(np.abs(amp_pred[:, i]) == 1)),
                       amp_wrong_endpoint_fraction=float(np.mean((np.abs(amp_pred[:, i]) == 1) & (np.abs(amp_pred[:, i] - target[:, i]) > 0.5))),
                       mean_tanh_derivative_amp=float(np.mean(1 - amp_pred[:, i]**2)))
        report["channels"][channel] = row
    fast = speed >= 20
    report["at_least_20_mps"] = {"window_tokens": int(fast.sum()),
                                 "predicted_physical_throttle_mean": float(((pred[fast, 0] + 1) / 2).mean()),
                                 "teacher_physical_throttle_mean": float(((target[fast, 0] + 1) / 2).mean())} if fast.any() else None
    write_metrics(args.output, report)
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
