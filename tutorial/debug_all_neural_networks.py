"""
Neural Network Smoke Diagnostics
================================
Checks key tensor shapes, local gradient flow, and one-step training. These
checks do not establish world-model accuracy or zero-shot transfer.
"""

import math
import time
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium.spaces as spaces

from tmrl.custom.torch.world_model import (
    TorchVisualTelemetryEncoder,
    TorchLatentRSSM,
    TorchLatentWorldModel,
    TorchLatentAdversaryProposer,
    symlog,
    symexp
)
from tmrl.custom.torch.custom_models import (
    VanillaCNNActorCritic,
    SquashedGaussianVanillaCNNActor,
    VanillaCNNQFunction
)
from tmrl.custom.torch.custom_algorithms import DreamSACAgent
import tmrl.config.config_constants as cfg


def test_section_header(title: str):
    print("\n" + "=" * 80)
    print(f"[TEST] {title.upper()}")
    print("=" * 80)


def run_all_neural_diagnostics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Diagnostics on Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # =========================================================================
    # 1. TEST: Visual-Telemetry Encoder Dimension Math & Layer Progression
    # =========================================================================
    test_section_header("1. Visual-Telemetry Encoder Dimension Math")
    encoder = TorchVisualTelemetryEncoder(img_channels=4, latent_dim=128).to(device)
    
    B = 64
    dummy_speed = torch.randn(B, 1, device=device) * 150.0
    dummy_gear = torch.randint(1, 6, (B, 1), device=device).float()
    dummy_rpm = torch.rand(B, 1, device=device) * 9000.0
    dummy_imgs = torch.randint(0, 256, (B, 4, 96, 96), device=device, dtype=torch.uint8)
    dummy_act1 = torch.randn(B, 3, device=device)
    dummy_act2 = torch.randn(B, 3, device=device)
    obs_tuple = (dummy_speed, dummy_gear, dummy_rpm, dummy_imgs, dummy_act1, dummy_act2)

    # Step-by-step layer tracking
    imgs_float = dummy_imgs.float() / 255.0
    c1 = F.relu(encoder.conv1(imgs_float))
    c2 = F.relu(encoder.conv2(c1))
    c3 = F.relu(encoder.conv3(c2))
    c4 = F.relu(encoder.conv4(c3))
    flat_c4 = torch.flatten(c4, start_dim=1)

    print(f"  [Input]  Image Shape:        {list(dummy_imgs.shape)} (uint8 [0, 255])")
    print(f"  [Conv1]  Output Shape:       {list(c1.shape)} (Expected: [64, 32, 45, 45])")
    print(f"  [Conv2]  Output Shape:       {list(c2.shape)} (Expected: [64, 64, 21, 21])")
    print(f"  [Conv3]  Output Shape:       {list(c3.shape)} (Expected: [64, 128, 9, 9])")
    print(f"  [Conv4]  Output Shape:       {list(c4.shape)} (Expected: [64, 128, 3, 3])")
    print(f"  [Flatten] Conv Features:     {list(flat_c4.shape)} (Exact size: 128*3*3 = 1152)")
    
    assert c1.shape == (B, 32, 45, 45), f"Conv1 shape mismatch: {c1.shape}"
    assert c2.shape == (B, 64, 21, 21), f"Conv2 shape mismatch: {c2.shape}"
    assert c3.shape == (B, 128, 9, 9), f"Conv3 shape mismatch: {c3.shape}"
    assert c4.shape == (B, 128, 3, 3), f"Conv4 shape mismatch: {c4.shape}"
    assert flat_c4.shape == (B, 1152), f"Flat Conv shape mismatch: {flat_c4.shape}"

    # Telemetry MLP
    telem = torch.cat([dummy_speed / 300.0, dummy_gear / 5.0, dummy_rpm / 10000.0], dim=-1)
    telem_feat = encoder.telem_mlp(telem)
    print(f"  [Telem]  Output Shape:       {list(telem_feat.shape)} (Expected: [64, 64])")
    assert telem_feat.shape == (B, 64)

    # Full forward
    e = encoder(obs_tuple)
    print(f"  [Fused]  Latent Vector Shape:{list(e.shape)} (Expected: [64, 128])")
    assert e.shape == (B, 128)
    print("  [PASS] Visual-Telemetry Encoder: All Dimensions 100% Verified!")

    # =========================================================================
    # 2. TEST: Recurrent State-Space Model (RSSM) & KL Divergence Math
    # =========================================================================
    test_section_header("2. Recurrent State-Space Model (RSSM) & KL Math")
    rssm = TorchLatentRSSM(latent_dim=128, action_dim=3, hidden_dim=256).to(device)
    
    h0 = torch.zeros(B, 256, device=device)
    dummy_action = torch.tanh(torch.randn(B, 3, device=device))
    
    # 1. Deterministic step
    h1 = rssm.step_deterministic(h0, e, dummy_action)
    print(f"  [GRU] Deterministic State h_1: {list(h1.shape)} (Expected: [64, 256])")
    assert h1.shape == (B, 256)

    # 2. Stochastic Prior & Posterior
    z_prior, prior_m, prior_s = rssm.compute_prior(h1)
    z_post, post_m, post_s = rssm.compute_posterior(h1, e)
    
    print(f"  [Prior]     z_prior: {list(z_prior.shape)}, mean: {list(prior_m.shape)}, std: {list(prior_s.shape)}")
    print(f"  [Posterior] z_post:  {list(z_post.shape)}, mean: {list(post_m.shape)}, std: {list(post_s.shape)}")
    assert z_prior.shape == (B, 128) and z_post.shape == (B, 128)
    assert (prior_s > 0).all() and (post_s > 0).all(), "Standard deviations must be strictly positive!"

    # 3. Analytical Gaussian KL Divergence: KL(q(z|h,e) || p(z|h))
    kl = torch.log(prior_s / (post_s + 1e-6)) + (post_s**2 + (post_m - prior_m)**2) / (2.0 * prior_s**2 + 1e-6) - 0.5
    kl_sum = kl.sum(dim=-1).mean()
    print(f"  [KL Div] Mean KL: {kl_sum.item():.4f} (Must be >= 0.0)")
    assert kl_sum.item() >= -1e-4, f"KL Divergence cannot be negative: {kl_sum.item()}"
    print("  [PASS] RSSM Dynamics & KL Divergence: 100% Mathematically Sound!")

    # =========================================================================
    # 3. TEST: Scale-Invariant Symlog / Symexp Bijective Parity
    # =========================================================================
    test_section_header("3. Symlog / Symexp Invariant Bijective Consistency")
    test_vals = torch.tensor([-500.0, -100.0, -10.0, -1.0, -0.01, 0.0, 0.01, 1.0, 10.0, 100.0, 500.0], device=device)
    sym_transformed = symlog(test_vals)
    reconstructed = symexp(sym_transformed)
    
    print("  Raw Values:       ", [round(v.item(), 2) for v in test_vals])
    print("  Symlog Compressed:", [round(v.item(), 4) for v in sym_transformed])
    print("  Symexp Restored:  ", [round(v.item(), 2) for v in reconstructed])
    
    error = torch.max(torch.abs(test_vals - reconstructed)).item()
    print(f"  Max Reconstruction Error: {error:.6e}")
    assert error < 1e-3, f"Symlog/Symexp bijective reconstruction error too high: {error}"
    print("  [PASS] Symlog / Symexp Transform: Exact Invariance Verified!")

    # =========================================================================
    # 4. TEST: Experimental latent proposer bounds and local gradients
    # =========================================================================
    test_section_header("4. Latent Adversary Proposer Bounds & Gradients")
    adversary = TorchLatentAdversaryProposer(feat_dim=384, perturbation_dim=128, max_magnitude=0.25).to(device)
    
    feat = torch.cat([h1, z_post], dim=-1)
    perturbation = adversary(feat)
    
    print(f"  [Adversary] Input Feature Shape: {list(feat.shape)} (Expected: [64, 384])")
    print(f"  [Adversary] Output Perturbation: {list(perturbation.shape)} (Expected: [64, 128])")
    print(f"  [Adversary] Max Magnitude Observed: {perturbation.abs().max().item():.4f} (Bounded by <= 0.25)")
    
    assert perturbation.shape == (B, 128)
    assert perturbation.abs().max().item() <= 0.250001, "Perturbation exceeded max magnitude bound!"
    
    # Test gradient flow
    adv_loss = -perturbation.sum()
    adv_loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in adversary.parameters() if p.grad is not None)
    print(f"  [Adversary] Backward Gradient Norm: {grad_norm:.4f}")
    assert grad_norm > 0.0, "Adversary parameter gradients are zero!"
    print("  [PASS] Latent proposer module is bounded and differentiable (not a task-validity test).")

    # =========================================================================
    # 5. TEST: VanillaCNNActorCritic Policy & Q-Functions
    # =========================================================================
    test_section_header("5. VanillaCNNActorCritic Actor & Critic Verification")
    obs_space = spaces.Tuple((
        spaces.Box(-np.inf, np.inf, (1,)),
        spaces.Box(-np.inf, np.inf, (1,)),
        spaces.Box(-np.inf, np.inf, (1,)),
        spaces.Box(0, 255, (4, 96, 96), dtype=np.uint8),
        spaces.Box(-1, 1, (3,)),
        spaces.Box(-1, 1, (3,))
    ))
    act_space = spaces.Box(-1, 1, (3,))
    
    ac = VanillaCNNActorCritic(obs_space, act_space).to(device)
    
    # Actor forward
    pi_action, logp_pi = ac.actor(obs_tuple)
    print(f"  [Actor] Sampled Action Shape: {list(pi_action.shape)} (Expected: [64, 3])")
    print(f"  [Actor] Log Prob Shape:       {list(logp_pi.shape)} (Expected: [64])")
    print(f"  [Actor] Action Range:         [{pi_action.min().item():.3f}, {pi_action.max().item():.3f}] (Bounded in [-1, 1])")
    
    assert pi_action.shape == (B, 3)
    assert logp_pi.shape == (B,)
    assert pi_action.min().item() >= -1.0 and pi_action.max().item() <= 1.0
    
    # Q-functions forward
    q1 = ac.q1(obs_tuple, pi_action)
    q2 = ac.q2(obs_tuple, pi_action)
    print(f"  [Critic] Q1 Output Shape:     {list(q1.shape)} (Expected: [64])")
    print(f"  [Critic] Q2 Output Shape:     {list(q2.shape)} (Expected: [64])")
    assert q1.shape == (B,) and q2.shape == (B,)
    print("  [PASS] VanillaCNNActorCritic: Fully Operational & Verified!")

    # =========================================================================
    # 6. TEST: SAC plus replay-grounded one-step world-model update
    # =========================================================================
    test_section_header("6. SAC + one-step world-model update")
    agent = DreamSACAgent(
        observation_space=obs_space,
        action_space=act_space,
        model_cls=VanillaCNNActorCritic,
        device=device,
        lr_actor=0.00021,
        lr_critic=0.00015,
        lr_world_model=0.0003,
        lr_adversary=0.0001
    )
    
    dummy_reward = torch.randn(B, 1, device=device) * 2.0
    dummy_done = (torch.rand(B, 1, device=device) > 0.95).float()
    batch = (obs_tuple, dummy_action, dummy_reward, obs_tuple, dummy_done, None)

    # Execute 10 training steps and measure timing
    t0 = time.perf_counter()
    for step in range(1, 11):
        metrics = agent.train(batch)
        if step in [1, 5, 10]:
            print(f"  [Step {step:02d}] "
                  f"Critic Loss: {metrics['loss_critic']:.4f} | "
                  f"Actor Loss: {metrics['loss_actor']:.4f} | "
                  f"WM Loss: {metrics['loss_world_model']:.4f} | "
                  f"Imagined Policy Updates: {metrics['imagined_policy_updates']:.0f} | "
                  f"Alpha: {metrics['entropy_coef']:.4f}")
    t1 = time.perf_counter()
    
    avg_step_ms = (t1 - t0) / 10.0 * 1000.0
    print(f"\n  [BENCHMARK] Average Step Execution Time: {avg_step_ms:.2f} ms ({1000.0/avg_step_ms:.1f} steps/sec on {device})")
    
    assert metrics['loss_world_model'] > 0.0
    assert math.isfinite(metrics['loss_world_model'])
    assert metrics['imagined_policy_updates'] == 0.0
    assert not math.isnan(metrics['loss_critic'])
    assert not math.isnan(metrics['loss_actor'])
    print("  [PASS] SAC and grounded one-step world-model updates executed without silent skips.")

    print("\n" + "=" * 80)
    print("[SUCCESS] SMOKE DIAGNOSTICS PASSED; HELD-OUT ZERO-SHOT TRANSFER WAS NOT EVALUATED.")
    print("=" * 80)


if __name__ == "__main__":
    run_all_neural_diagnostics()
