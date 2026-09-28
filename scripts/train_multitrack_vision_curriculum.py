#!/usr/bin/env python3
"""
Offline Visual Robustness Fine-Tuner.

Trains the MultiModalCarBrain on labeled demonstration trajectories while applying
appearance perturbations (dirt-like contrast, low light, blur, and noise).  These
augmentations do not create new map geometry, racing lines, or driving labels.

Features:
- Recorded-frame appearance augmentation
- Complete, checked optimizer groups with a conservative backbone learning rate
- AMP with successful-update accounting and measured peak VRAM
- Episode-held-out validation, best/latest snapshots, and image ablations
- Dual format export (.pt and .safetensors)
- Candidate-only export; live deployment is a separate evaluation-gated step
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tmrl.custom.torch.car_brain import MultiModalCarBrain

try:
    from safetensors.torch import save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# =====================================================================
# 1. Multi-Track Photorealistic Vision Domain Synthesizer
# =====================================================================

class MultiTrackVisionSynthesizer:
    """
    Applies appearance perturbations to recorded camera observations.

    This is robustness augmentation, not a substitute for demonstrations or
    successful replays collected on additional tracks.
    """
    DOMAINS = ["tarmac", "dirt", "ice", "grass", "night", "fullspeed"]

    def __init__(self, device: torch.device):
        self.device = device
        # Pre-generate headlight mask for night/tunnel domain (96x96)
        y, x = np.ogrid[:96, :96]
        # Car headlights cone centered in lower middle
        center_x, center_y = 48, 60
        dist_sq = ((x - center_x) ** 2) / (32 ** 2) + ((y - center_y) ** 2) / (45 ** 2)
        cone = np.clip(1.3 - dist_sq, 0.25, 1.2).astype(np.float32)
        self.headlight_cone = torch.from_numpy(cone).to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, 96, 96)

    def augment_batch(self, imgs: torch.Tensor, domain: Optional[str] = None) -> torch.Tensor:
        """
        imgs: (B, T, 4, 96, 96) in [0.0, 1.0]
        domain: specific domain or None for random mix per sample
        """
        B, T, C, H, W = imgs.shape
        x = imgs.clone()

        if domain is None:
            active_domains = [random.choice(self.DOMAINS) for _ in range(B)]
        else:
            active_domains = [domain] * B

        for b, dom in enumerate(active_domains):
            sample = x[b]  # (T, 4, 96, 96)

            if dom == "dirt":
                # Dirt/Mud: High-frequency soil texture noise + warm earthy tone contrast
                noise = torch.randn((T, 4, H, W), device=self.device) * 0.06
                # Contrast boost on track ruts
                sample = torch.clamp((sample - 0.45) * 1.15 + 0.45 + noise, 0.0, 1.0)
                # Slight gamma compression (clamp to 1e-4 to prevent inf gradient at 0.0)
                sample = torch.pow(torch.clamp(sample, 1e-4, 1.0), 1.1)

            elif dom == "ice":
                # Ice/Snow: High brightness, specular reflective sheen, lower edge contrast
                # Brighten (clamp to 1e-4 to prevent inf gradient at 0.0)
                sample = torch.pow(torch.clamp(sample, 1e-4, 1.0), 0.78)
                # Specular glare patches
                glare = (torch.rand((T, 1, H, W), device=self.device) > 0.94).float() * 0.22
                sample = torch.clamp(sample + glare, 0.0, 1.0)

            elif dom == "grass":
                # Grass/Offroad: Texture roughness & micro-bump vibration
                shift_h = random.choice([-1, 0, 1])
                shift_w = random.choice([-1, 0, 1])
                if shift_h != 0 or shift_w != 0:
                    sample = torch.roll(sample, shifts=(shift_h, shift_w), dims=(-2, -1))
                noise = torch.randn((T, 4, H, W), device=self.device) * 0.04
                sample = torch.clamp(sample + noise, 0.0, 1.0)

            elif dom == "night":
                # Night / Tunnel: Low ambient light with headlight cone illumination
                dimmed = sample * 0.35
                illuminated = sample * self.headlight_cone
                sample = torch.clamp(torch.max(dimmed, illuminated), 0.0, 1.0)

            elif dom == "fullspeed":
                # FullSpeed / High-Speed Motion Blur: Blend consecutive frames slightly
                # Simulates 400+ km/h motion blur on 20 Hz cameras
                for c in range(1, C):
                    sample[:, c] = 0.7 * sample[:, c] + 0.3 * sample[:, c - 1]
                # High contrast asphalt
                sample = torch.clamp((sample - 0.5) * 1.25 + 0.5, 0.0, 1.0)

            else:  # tarmac / standard
                # Standard clean baseline with minor natural lighting variation
                gain = 0.95 + random.random() * 0.10
                bias = (random.random() - 0.5) * 0.05
                sample = torch.clamp(sample * gain + bias, 0.0, 1.0)

            x[b] = sample

        return x


# =====================================================================
# 2. Causal Sequence Dataset Extraction
# =====================================================================

def extract_causal_sequence_dataset(
    demo_file: str,
    window_len: int = 32,
    stride: int = 2,
    split: str = "train",
    return_mask: bool = False,
) -> tuple:
    """
    Extracts rolling causal windows of images, states, previous actions, and target actions.
    Returns:
      all_imgs: (N, T, 4, 96, 96) uint8 tensor (compact in CPU RAM)
      all_states: (N, T, 15) float32 tensor
      all_prev_acts: (N, T, 3) float32 tensor
      all_target_acts: (N, T, 3) float32 tensor
      valid_mask: optional (N, T) bool tensor, required for short padded episodes
    """
    if window_len < 1 or stride < 1:
        raise ValueError("window_len and stride must be positive")
    print(f"[+] Loading multi-modal demonstration dataset: {demo_file}...")
    demo_data = torch.load(demo_file, weights_only=False)
    episodes = select_episode_split(demo_data["episodes"], split)
    metadata = demo_data.get("metadata", {})
    track_id = metadata.get("track_id")
    if track_id:
        print(f"    Source track: {track_id}")
    else:
        print(
            "    [!] Legacy dataset has no track identity. Treating it as one "
            "unknown source track, regardless of the number of downloaded maps."
        )
    print(
        "    [!] Appearance augmentation adds zero new track trajectories; "
        "only labeled episodes in this file count as driving data."
    )

    list_imgs = []
    list_states = []
    list_prev_acts = []
    list_target_acts = []
    list_masks = []
    start_indices = []

    for ep_idx, ep in enumerate(episodes):
        imgs = ep["imgs"]      # (T_total, 4, 96, 96) uint8
        states = ep["states"]  # (T_total, 15) float32
        actions = ep["actions"]# (T_total, 3) float32

        T = len(states)
        if T < 1 or len(imgs) != T or len(actions) != T:
            raise ValueError(f"Episode {ep_idx} is empty or has inconsistent lengths")
        if T < window_len and not return_mask:
            raise ValueError("Short episodes require return_mask=True for padded loss masking")

        starts = list(range(0, max(0, T - window_len) + 1, stride))
        if T >= window_len and starts[-1] != T - window_len:
            starts.append(T - window_len)  # retain the final driving/finish window
        for start in starts:
            end = start + window_len
            w_img = imgs[start:end]
            w_state = states[start:end]
            w_target = actions[start:end]

            w_prev = np.zeros_like(w_target)
            w_prev[1:] = w_target[:-1]
            if start > 0:
                w_prev[0] = actions[start - 1]

            valid_length = len(w_state)
            mask = np.arange(window_len) < valid_length
            if valid_length < window_len:
                def pad(value):
                    return np.pad(value, [(0, window_len - valid_length)] + [(0, 0)] * (value.ndim - 1))
                w_img, w_state, w_prev, w_target = map(pad, (w_img, w_state, w_prev, w_target))
            # Right padding is future-only under the model's causal attention;
            # valid tokens cannot attend to it. Loss/metrics exclude padded tokens.
            list_imgs.append(w_img)
            list_states.append(w_state)
            list_prev_acts.append(w_prev)
            list_target_acts.append(w_target)
            list_masks.append(mask)
            if start == 0:
                start_indices.append(len(list_imgs) - 1)

    if not list_imgs:
        raise ValueError(f"No usable {split} episodes")
    if split == "train" and start_indices:
        # At least 25% of training windows start strictly at t=0. Do not
        # oversample validation windows or count duplicates as additional data.
        extra = max(0, math.ceil((0.25 * len(list_imgs) - len(start_indices)) / 0.75))
        for offset in range(extra):
            index = start_indices[offset % len(start_indices)]
            for values in (list_imgs, list_states, list_prev_acts, list_target_acts, list_masks):
                values.append(values[index])

    raw_arr = np.array(list_imgs)
    if np.issubdtype(raw_arr.dtype, np.floating) and raw_arr.max() <= 1.0:
        all_imgs = torch.from_numpy((raw_arr * 255.0).clip(0, 255).astype(np.uint8))
    else:
        all_imgs = torch.from_numpy(raw_arr.astype(np.uint8))
    all_states = torch.tensor(np.array(list_states), dtype=torch.float32)
    # Mask out ALL telemetry cheat shortcuts (steer, gas, GPS):
    # Only keep speed (0), gear (3), rpm (4), and default tires (9:13).
    # This removes current-label/GPS shortcuts, but does not prove image reliance.
    all_states[:, :, 1] = 0.0
    all_states[:, :, 2] = 0.0
    all_states[:, :, 5:] = 0.0
    all_states[:, :, 9:13] = 1.0
    print("    [+] Current steer/gas and GPS masked; speed/gear/RPM and previous actions retained.")
    all_prev_acts = torch.tensor(np.array(list_prev_acts), dtype=torch.float32)
    all_target_acts = torch.tensor(np.array(list_target_acts), dtype=torch.float32)

    result = (all_imgs, all_states, all_prev_acts, all_target_acts)
    return (*result, torch.from_numpy(np.array(list_masks))) if return_mask else result


def select_episode_split(episodes, split):
    """Split before extracting windows; windows from a lap cannot cross splits."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    if any("split" in ep for ep in episodes):
        if not all(ep.get("split") in {"train", "validation"} for ep in episodes):
            raise ValueError("Every episode must have a valid explicit split")
        selected = [ep for ep in episodes if ep["split"] == split]
    else:
        if len(episodes) < 2:
            raise ValueError("Need at least two episodes for an honest held-out validation run")
        selected = episodes[:-1] if split == "train" else episodes[-1:]
    if not selected:
        raise ValueError(f"Dataset has no episodes in {split}")
    return selected


def save_candidate(model, output_path):
    """Never hot-reload untested training weights into the active driver."""
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Refusing to save non-finite candidate weights")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, path)
    if HAS_SAFETENSORS:
        clean = {key: value.contiguous() for key, value in model.state_dict().items()}
        safe_path = path.with_suffix(".safetensors")
        temporary = safe_path.with_name(safe_path.name + ".tmp")
        save_safetensors(clean, str(temporary))
        os.replace(temporary, safe_path)


def optimizer_parameter_groups(model, lr_vision, lr_backbone):
    """Fail closed if a new trainable parameter has no deliberate LR assignment."""
    prefixes = {
        "vision": ("conv1.", "conv2.", "conv3.", "conv4.", "visual_fc.", "visual_ln."),
        "fusion_head": ("fusion_proj.", "policy_head."),
        "backbone": ("blocks.", "kinematics_proj.", "pos_emb", "ln_f."),
    }
    groups = {name: {"name": name, "params": [], "param_names": [],
                     "lr": lr_backbone if name == "backbone" else lr_vision * (0.35 if name == "fusion_head" else 1),
                     "weight_decay": 1e-4} for name in prefixes}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [group for group, roots in prefixes.items() if name.startswith(roots)]
        if len(matches) != 1:
            raise ValueError(f"Parameter {name} must belong to exactly one optimizer group: {matches}")
        groups[matches[0]]["params"].append(parameter)
        groups[matches[0]]["param_names"].append(name)
    ids = [id(p) for group in groups.values() for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(ids) != len(set(ids)) or set(ids) != expected or any(not g["params"] for g in groups.values()):
        raise ValueError("Optimizer parameter coverage is incomplete, duplicated, or empty")
    return list(groups.values())


def action_loss(predictions, targets, mask=None):
    """Per-valid-token mean Huber over actions plus 2.5 times steering Huber."""
    if mask is not None:
        predictions, targets = predictions[mask], targets[mask]
    if predictions.numel() == 0:
        raise ValueError("Cannot compute loss without valid target tokens")
    huber = F.smooth_l1_loss(predictions.float(), targets.float(), beta=0.1, reduction="none")
    return huber.mean() + 2.5 * huber[..., 2].mean()


def trigger_logit_loss(logits, targets, mask=None):
    """Signed trigger pressure is (tanh(z)+1)/2 == sigmoid(2*z).

    BCEWithLogits retains a corrective gradient when a wrong throttle/brake
    prediction saturates tanh, including exact FP16 endpoint rounding. Soft
    pressure targets are allowed; steering keeps its continuous Huber loss.
    """
    if mask is not None:
        logits, targets = logits[mask], targets[mask]
    if logits.numel() == 0:
        raise ValueError("Cannot compute trigger loss without valid tokens")
    return F.binary_cross_entropy_with_logits(2 * logits[..., :2].float(),
                                              (targets[..., :2].float() + 1) / 2)


def evaluate_windows(model, data, device, batch_size=8, image_mode="clean", seed=42, zero_previous_actions=False):
    """All held-out windows, teacher-forced actions, no random augmentation.

    Terminal windows can overlap their predecessor. Metrics count valid window
    tokens, not unique lap frames. Shuffling swaps whole image sequences while
    retaining the original states/labels/actions; it never changes training RNG.
    """
    if image_mode not in {"clean", "blank", "shuffled"} or batch_size < 1:
        raise ValueError("Invalid validation mode or batch size")
    imgs, states, previous, targets = data[:4]
    mask = data[4] if len(data) == 5 else torch.ones(targets.shape[:2], dtype=torch.bool)
    n = len(imgs)
    if n < 1 or not mask.any():
        raise ValueError("Validation data is empty")
    indices = torch.arange(n)
    offset = None
    if image_mode == "shuffled":
        if n < 2:
            raise ValueError("Image shuffling requires at least two windows")
        generator = torch.Generator().manual_seed(seed)
        offset = int(torch.randint(1, n, (1,), generator=generator))
        indices = indices.roll(offset)  # no window retains its own images
    was_training = model.training
    absolute_sum = torch.zeros(3, dtype=torch.float64)
    objective_sum, count = 0.0, 0
    steer_pred, steer_target = [], []
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_imgs = imgs[indices[start:end]].to(device).float() / 255.0
                if image_mode == "blank":
                    batch_imgs.zero_()
                batch_prev = previous[start:end].to(device)
                if zero_previous_actions:
                    batch_prev = torch.zeros_like(batch_prev)
                pred = model(batch_imgs, states[start:end].to(device), prev_actions=batch_prev)
                valid = mask[start:end].to(device)
                pred = pred[valid].float()
                target = targets[start:end].to(device)[valid]
                if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
                    raise ValueError("Non-finite held-out prediction or target")
                token_count = len(pred)
                objective_sum += float(action_loss(pred, target)) * token_count
                absolute_sum += (pred - target).abs().double().sum(dim=0).cpu()
                count += token_count
                steer_pred.append(pred[:, 2].cpu())
                steer_target.append(target[:, 2].cpu())
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Validation contains no valid tokens")
    p, t = torch.cat(steer_pred).numpy(), torch.cat(steer_target).numpy()
    correlation = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-8 and np.std(t) > 1e-8 else None
    maes = (absolute_sum / count).tolist()
    return {"objective": objective_sum / count, "action_mae": maes,
            "steer_mae": maes[2], "steer_correlation": correlation,
            "windows": n, "valid_window_tokens": count,
            "image_mode": image_mode, "shuffle_offset": offset,
            "previous_actions": "zeroed" if zero_previous_actions else "teacher_forced"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metrics(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class CandidateCheckpoints:
    """Clean held-out objective selects best; the last update is saved separately."""
    def __init__(self, path):
        self.latest = Path(path)
        self.best = self.latest.with_name(self.latest.stem + ".best.pt")
        self.best_step = None
        self.best_score = math.inf

    def consider(self, model, step, metrics):
        if metrics.get("image_mode") != "clean" or metrics.get("previous_actions") != "teacher_forced":
            raise ValueError("Only clean teacher-forced validation may select a checkpoint")
        score = metrics["objective"]
        if not math.isfinite(score):
            raise ValueError("Checkpoint selection requires a finite objective")
        if score < self.best_score:
            save_candidate(model, self.best)
            self.best_score, self.best_step = score, step
            return True
        return False

    def save_latest(self, model, step, snapshot=False):
        save_candidate(model, self.latest)
        if snapshot:
            save_candidate(model, self.latest.with_name(f"{self.latest.stem}.step_{step:07d}.pt"))


def gradient_group_norms(groups):
    result = {}
    for group in groups:
        grads = [p.grad for p in group["params"] if p.grad is not None]
        if len(grads) != len(group["params"]):
            raise ValueError(f"Some {group['name']} parameters have no gradient")
        result[group["name"]] = float(torch.stack([g.float().norm() for g in grads]).norm())
    return result


# =====================================================================
# 3. Main Multi-Track Vision Curriculum Training Loop
# =====================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Visual Demonstration Robustness Fine-Tuner")
    parser.add_argument("--demo_file", type=str, default="data/target_track_demos.pt")
    parser.add_argument("--base_model", type=str, default="weights/car_brain_1m_curriculum/car_brain_latest.pt")
    parser.add_argument("--existing_multimodal", type=str, default="weights/car_brain_1m_curriculum/car_brain_multimodal.pt")
    parser.add_argument("--reset_weights", action="store_true", help="Start fresh from base foundation physics instead of GPS-contaminated weights")
    parser.add_argument("--output_path", type=str, default="weights/car_brain_1m_curriculum/car_brain_vision_1k_diverse.pt")
    parser.add_argument("--deploy_path", type=str, default=None,
                        help="Deprecated: automatic deployment is blocked; save a candidate and live-test it")
    parser.add_argument("--steps", type=int, default=1000, help="Successful gradient updates (AMP skips do not count)")
    parser.add_argument("--batch_size", type=int, default=8, help="Training micro-batch size")
    parser.add_argument("--seq_len", type=int, default=16, help="Causal sequence length")
    parser.add_argument("--lr_vision", type=float, default=1e-4, help="Learning rate for CNN & Visual Tokenizer")
    parser.add_argument("--lr_backbone", type=float, default=1e-6, help="Learning rate for Transformer and final LayerNorm")
    parser.add_argument("--trigger_loss_weight", type=float, default=0.1,
                        help="Auxiliary pre-tanh throttle/brake BCE weight; prevents saturated wrong outputs")
    parser.add_argument("--save_every", type=int, default=250, help="Numbered checkpoint interval")
    parser.add_argument("--eval_every", type=int, default=250, help="Clean held-out validation interval")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.deploy_path is not None:
        parser.error("Automatic deployment is blocked. Use --output_path for an isolated candidate, then live-test it.")
    if any(value < 1 for value in (args.steps, args.batch_size, args.seq_len, args.save_every,
                                   args.eval_every, args.eval_batch_size)) or args.seq_len > 64:
        parser.error("Counts/intervals must be positive and seq_len must be <= 64")
    if not all(math.isfinite(value) and value > 0 for value in (args.lr_vision, args.lr_backbone)):
        parser.error("Learning rates must be finite and positive")
    if not math.isfinite(args.trigger_loss_weight) or args.trigger_loss_weight < 0:
        parser.error("trigger_loss_weight must be finite and nonnegative")
    if args.lr_backbone > args.lr_vision * 0.35 * 0.1:
        parser.error("Backbone LR must be <= 0.1 times the fusion/head LR (0.35 * lr_vision)")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    output = Path(args.output_path).resolve()
    protected = {Path(args.existing_multimodal).resolve(), Path(args.base_model).resolve()}
    best = output.with_name(output.stem + ".best.pt")
    if {output, best} & protected:
        parser.error("--output_path must not overwrite an input/deployed checkpoint")
    if output.suffix != ".pt":
        parser.error("--output_path must end in .pt")
    # No implicit resume/overwrite: use a fresh experiment stem/directory.
    if any(output.parent.glob(output.stem + ".*")):
        parser.error("Candidate artifacts already exist; choose a fresh --output_path")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)

    print("\n" + "=" * 65)
    print("          OFFLINE VISUAL ROBUSTNESS FINE-TUNER                  ")
    print("   Labeled trajectories + appearance augmentation (not new maps) ")
    print("=" * 65)
    print(f"Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")

    if not os.path.exists(args.demo_file):
        raise FileNotFoundError(args.demo_file)

    # 1. Build Causal Sequence Dataset
    dataset_imgs, dataset_states, dataset_prev_acts, dataset_target_acts, dataset_mask = extract_causal_sequence_dataset(
        demo_file=args.demo_file,
        window_len=args.seq_len,
        stride=2,
        return_mask=True,
    )
    validation_data = extract_causal_sequence_dataset(
        demo_file=args.demo_file, window_len=args.seq_len, stride=args.seq_len,
        split="validation", return_mask=True,
    )
    N_samples = dataset_imgs.shape[0]
    print(f"[+] Staged {N_samples:,} causal training sequences ({args.seq_len} steps each).")
    print(f"    RAM Footprint: {dataset_imgs.element_size() * dataset_imgs.nelement() / (1024**2):.1f} MB CPU RAM.")

    # 2. Instantiate Model
    print(f"[+] Instantiating MultiModalCarBrain...")
    source_path = None
    if not args.reset_weights and os.path.exists(args.existing_multimodal):
        print(f"[+] Loading existing multimodal foundation weights: {args.existing_multimodal}")
        model = MultiModalCarBrain(d_model=256, n_layers=6, n_heads=8, max_context_len=64)
        source_path = args.existing_multimodal
        state_dict = torch.load(source_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    elif os.path.exists(args.base_model):
        print(f"[+] Initializing from 1M foundation physics backbone: {args.base_model}")
        model = MultiModalCarBrain.from_foundation(args.base_model, device=str(device))
        source_path = args.base_model
    else:
        print("[+] Initializing standard MultiModalCarBrain architecture...")
        model = MultiModalCarBrain(d_model=256, n_layers=6, n_heads=8, max_context_len=64)

    model.to(device)
    model.train()

    groups = optimizer_parameter_groups(model, args.lr_vision, args.lr_backbone)
    optimizer = torch.optim.AdamW(groups)

    # Scale every group by the same factor. A common absolute eta_min based on
    # the CNN rate could increase the smaller backbone LR while "decaying" it.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 0.1 + 0.45 * (1 + math.cos(math.pi * step / args.steps)))
    synthesizer = MultiTrackVisionSynthesizer(device=device)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    checkpoints = CandidateCheckpoints(args.output_path)
    metrics_path = checkpoints.latest.with_suffix(".metrics.json")
    # Preserve the exact local training/model implementation for this run,
    # alongside hashes, rather than relying on the future working tree.
    checkpoints.latest.parent.mkdir(parents=True, exist_ok=True)
    checkpoints.latest.with_suffix(".trainer.py").write_bytes(Path(__file__).read_bytes())
    model_source = REPO_ROOT / "tmrl/custom/torch/car_brain.py"
    checkpoints.latest.with_suffix(".model.py").write_bytes(model_source.read_bytes())
    metrics = {
        "schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args), "pytorch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "dataset_path": str(Path(args.demo_file).resolve()), "dataset_sha256": file_sha256(args.demo_file),
        "source_checkpoint": str(Path(source_path).resolve()) if source_path else None,
        "source_sha256": file_sha256(source_path) if source_path else None,
        "trainer_sha256": file_sha256(__file__),
        "model_source_sha256": file_sha256(model_source),
        "training_objective": "selection objective + trigger_loss_weight * BCEWithLogits(2 * trigger_logits, signed_targets / 2 + 0.5)",
        "selection_metric": "clean teacher-forced mean action Huber(beta=0.1) + 2.5 * steering Huber",
        "limitations": ["One training lap and one validation lap of the same map in this experiment",
                        "Offline teacher-forced windows are not closed-loop driving or one-shot adaptation",
                        "Final window may overlap previous window; denominator is valid window tokens",
                        "Appearance augmentation adds no new geometry, trajectories, or recovery labels",
                        "Seeded RNG does not guarantee bitwise reproducibility across GPU/software versions"],
        "optimizer_groups": [{"name": group["name"], "lr": group["lr"],
                              "parameters": sum(p.numel() for p in group["params"]),
                              "param_names": group["param_names"]} for group in groups],
        "training_windows_including_start_oversampling": N_samples,
        "validation_windows": len(validation_data[0]), "validation_history": [],
        "successful_updates": 0, "amp_skipped_attempts": 0,
    }
    write_metrics(metrics_path, metrics)
    print("[+] Complete optimizer coverage:", {g["name"]: sum(p.numel() for p in g["params"]) for g in groups})
    print("[!] Offline same-map validation only; no deployment or one-shot claim.")

    def validate(step):
        score = evaluate_windows(model, validation_data, device, args.eval_batch_size)
        improved = checkpoints.consider(model, step, score)
        metrics["validation_history"].append({"step": step, **score})
        metrics.update(best_step=checkpoints.best_step, best_objective=checkpoints.best_score,
                       best_checkpoint=str(checkpoints.best.resolve()))
        write_metrics(metrics_path, metrics)
        print(f"  [Validation {step}] objective={score['objective']:.6f} steer_MAE={score['steer_mae']:.6f} best={improved}", flush=True)
        return score

    def ablations():
        return {mode: evaluate_windows(model, validation_data, device, args.eval_batch_size,
                                       image_mode=mode, seed=args.seed)
                for mode in ("blank", "shuffled")}

    t_start = time.perf_counter()
    recent_losses = []
    step = 0
    processed_tokens = 0
    try:
        metrics["baseline"] = validate(0)
        metrics["baseline_image_ablations"] = ablations()
        write_metrics(metrics_path, metrics)
        print("[+] Baseline scored; starting successful-update counter.", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        while step < args.steps:
            if step < int(args.steps * 0.33):
                active_domain = "tarmac" if random.random() < 0.70 else None
            else:
                active_domain = None
            batch_idx = torch.randint(0, N_samples, (args.batch_size,))
            raw_imgs = dataset_imgs[batch_idx].to(device).float() / 255.0
            b_states = dataset_states[batch_idx].to(device)
            b_prev_acts = dataset_prev_acts[batch_idx].to(device)
            b_targets = dataset_target_acts[batch_idx].to(device)
            b_mask = dataset_mask[batch_idx].to(device)

            # Preserve valid right-padded prefixes. Full-length windows may have
            # their contexts shortened; this augmentation is not a rollout test.
            if raw_imgs.shape[1] > 1 and bool(b_mask.all()) and random.random() < 0.40:
                sub_t = 1 if random.random() < 0.50 else random.randint(2, raw_imgs.shape[1])
                raw_imgs = raw_imgs[:, :sub_t].contiguous()
                b_states = b_states[:, :sub_t]
                b_prev_acts = b_prev_acts[:, :sub_t]
                b_targets = b_targets[:, :sub_t]
                b_mask = b_mask[:, :sub_t]
            # Discourage an action-copying shortcut; image ablations test whether
            # images actually help instead of assuming dropout guarantees it.
            if random.random() < 0.50:
                b_prev_acts = torch.zeros_like(b_prev_acts)
            aug_imgs = synthesizer.augment_batch(raw_imgs, domain=active_domain)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                pred_acts, logits = model(aug_imgs, b_states, prev_actions=b_prev_acts, return_logits=True)
                total_loss = action_loss(pred_acts, b_targets, b_mask)
                total_loss = total_loss + args.trigger_loss_weight * trigger_logit_loss(logits, b_targets, b_mask)
            if not torch.isfinite(total_loss):
                raise ValueError(f"Non-finite training loss after {step} updates")
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            norms = gradient_group_norms(groups)
            finite_gradients = all(math.isfinite(value) for value in norms.values())
            if not finite_gradients and not scaler.is_enabled():
                raise ValueError("Non-finite full-precision gradients")
            if finite_gradients:
                if step == 0:
                    if any(value <= 0 for value in norms.values()):
                        raise ValueError("Expected nonzero vision, fusion/head, and backbone gradients")
                    metrics["first_update_gradient_norms"] = norms
                    metrics["first_update_ln_f_gradient_norm"] = float(model.ln_f.weight.grad.norm())
                    print("[+] First finite pre-clipping gradient norms:", norms, flush=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                metrics["amp_skipped_attempts"] += 1
                print(f"  [AMP] Skipped overflow attempt; completed updates still {step}", flush=True)
                if metrics["amp_skipped_attempts"] > 50:
                    raise ValueError("More than 50 AMP overflows; investigate training stability")
                continue
            if not finite_gradients:
                raise ValueError("Non-finite gradients were not skipped by AMP")
            step += 1
            lr_scheduler.step()
            processed_tokens += int(b_mask.sum())
            metrics["successful_updates"] = step
            recent_losses.append(total_loss.detach().item())
            if step % 50 == 0 or step == args.steps:
                elapsed = time.perf_counter() - t_start
                peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0
                metrics.update(elapsed_seconds=elapsed, peak_vram_mb=peak_mb,
                               processed_training_tokens=processed_tokens,
                               last_gradient_norms=norms,
                               final_learning_rates=[g["lr"] for g in optimizer.param_groups])
                print(f"  Update [{step}/{args.steps}] loss={np.mean(recent_losses[-50:]):.6f} "
                      f"tokens/s={processed_tokens / max(elapsed, 0.1):.0f} peak_VRAM={peak_mb:.0f} MB", flush=True)
                write_metrics(metrics_path, metrics)
            if step % args.eval_every == 0 or step == args.steps:
                validate(step)
            if step % args.save_every == 0 or step == args.steps:
                checkpoints.save_latest(model, step, snapshot=True)
        # Latest remains the last update; diagnostics use the selected best.
        model.load_state_dict(torch.load(checkpoints.best, map_location=device, weights_only=True), strict=True)
        metrics["selected_clean_validation"] = evaluate_windows(model, validation_data, device, args.eval_batch_size)
        metrics["selected_image_ablations"] = ablations()
        metrics["selected_zero_previous_actions"] = evaluate_windows(
            model, validation_data, device, args.eval_batch_size, zero_previous_actions=True)
        metrics.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.perf_counter() - t_start,
                       latest_checkpoint=str(checkpoints.latest.resolve()),
                       latest_sha256=file_sha256(checkpoints.latest), best_sha256=file_sha256(checkpoints.best),
                       source_unchanged=file_sha256(source_path) == metrics["source_sha256"] if source_path else None)
        if metrics["source_unchanged"] is False:
            raise ValueError("Input checkpoint changed during candidate training")
    except KeyboardInterrupt:
        metrics.update(status="interrupted", successful_updates=step)
        # Once selection has loaded the best for diagnostics, do not relabel it
        # as the final training update on an interrupted ablation.
        if step < args.steps:
            checkpoints.save_latest(model, step)
        raise
    except Exception as exc:
        metrics.update(status="failed", error=f"{type(exc).__name__}: {exc}", successful_updates=step)
        raise
    finally:
        write_metrics(metrics_path, metrics)
    print(f"[+] Completed {step} successful updates. Best step: {checkpoints.best_step}.", flush=True)
    print(f"[+] Isolated best: {checkpoints.best}; metrics: {metrics_path}; deployed weights unchanged.", flush=True)


if __name__ == "__main__":
    main()
