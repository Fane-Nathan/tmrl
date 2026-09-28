"""Candidate-training regression checks independent of a running game/GPU."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from scripts.train_multitrack_vision_curriculum import (
    CandidateCheckpoints, action_loss, evaluate_windows,
    extract_causal_sequence_dataset, gradient_group_norms,
    optimizer_parameter_groups, parse_args, trigger_logit_loss,
)
from tmrl.custom.torch.car_brain import MultiModalCarBrain


class ImagePolicy(nn.Module):
    def forward(self, imgs, states, prev_actions=None):
        return imgs.mean(dim=(-3, -2, -1)).unsqueeze(-1).expand(-1, -1, 3)


class OptimizerTests(unittest.TestCase):
    def test_all_parameters_covered_once_and_final_layernorm_updates(self):
        torch.manual_seed(17)
        model = MultiModalCarBrain(d_model=32, n_layers=1, n_heads=4, d_ff=64)
        groups = optimizer_parameter_groups(model, 1e-4, 1e-6)
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        backbone = next(group for group in groups if group["name"] == "backbone")
        self.assertIn("ln_f.weight", backbone["param_names"])
        self.assertIn("ln_f.bias", backbone["param_names"])
        optimizer = torch.optim.AdamW(groups)
        before = model.ln_f.weight.detach().clone()
        pred = model(torch.rand(2, 3, 4, 96, 96), torch.rand(2, 3, 15))
        action_loss(pred, torch.rand(2, 3, 3)).backward()
        norms = gradient_group_norms(groups)
        self.assertTrue(all(np.isfinite(value) and value > 0 for value in norms.values()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.ln_f.weight))
        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_unknown_parameter_is_not_silently_omitted(self):
        model = MultiModalCarBrain(d_model=32, n_layers=1, n_heads=4, d_ff=64)
        model.extra = nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(ValueError, "extra"):
            optimizer_parameter_groups(model, 1e-4, 1e-6)

    def test_saturated_wrong_trigger_has_corrective_logit_gradient(self):
        logits = torch.tensor([[[8.0, -8.0, 0.0]]], requires_grad=True)
        target = torch.tensor([[[-1.0, -1.0, 0.0]]])
        rounded = logits.half().tanh().float()
        self.assertEqual(rounded[0, 0, 0].detach().item(), 1.0)
        action_loss(rounded, target).backward(retain_graph=True)
        self.assertEqual(float(logits.grad[0, 0, 0]), 0.0)
        logits.grad = None
        trigger_logit_loss(logits, target).backward()
        self.assertGreater(float(logits.grad[0, 0, 0]), 0.9)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_optional_logits_preserve_inference_and_checkpoint_contract(self):
        model = MultiModalCarBrain(d_model=32, n_layers=1, n_heads=4, d_ff=64).eval()
        keys = set(model.state_dict())
        imgs, states = torch.rand(1, 2, 4, 96, 96), torch.rand(1, 2, 15)
        with torch.inference_mode():
            normal = model(imgs, states)
            actions, logits = model(imgs, states, return_logits=True)
        torch.testing.assert_close(normal, actions)
        torch.testing.assert_close(actions, logits.tanh())
        self.assertEqual(set(model.state_dict()), keys)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.imgs = torch.tensor([0, 51, 102, 153, 204], dtype=torch.uint8)[:, None, None, None, None].expand(5, 3, 4, 8, 8)
        self.states = torch.zeros(5, 3, 15)
        self.previous = torch.zeros(5, 3, 3)
        self.targets = self.imgs.float().mean(dim=(-3, -2, -1)).unsqueeze(-1).expand(5, 3, 3) / 255
        self.data = (self.imgs, self.states, self.previous, self.targets)
        self.model = ImagePolicy().train()

    def test_clean_and_ablated_scores_and_rng_restoration(self):
        rng = torch.random.get_rng_state().clone()
        clean = evaluate_windows(self.model, self.data, "cpu", batch_size=2)
        blank = evaluate_windows(self.model, self.data, "cpu", batch_size=2, image_mode="blank")
        shuffled = evaluate_windows(self.model, self.data, "cpu", batch_size=2, image_mode="shuffled")
        self.assertAlmostEqual(clean["steer_mae"], 0, places=6)
        self.assertGreater(blank["steer_mae"], 0.1)
        self.assertGreater(shuffled["steer_mae"], 0.1)
        self.assertEqual(clean["valid_window_tokens"], 15)
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_weighted_aggregation_matches_direct_calculation(self):
        targets = self.targets.clone()
        targets[-1] = -0.8  # last partial batch carries most of the error
        mask = torch.ones(5, 3, dtype=torch.bool)
        mask[-1, 1:] = False
        data = (self.imgs, self.states, self.previous, targets, mask)
        score = evaluate_windows(self.model, data, "cpu", batch_size=2)
        pred = self.model(self.imgs.float() / 255, self.states)
        self.assertAlmostEqual(score["objective"], float(action_loss(pred, targets, mask)), places=6)
        self.assertAlmostEqual(score["steer_mae"], float((pred - targets).abs()[mask][:, 2].mean()), places=6)
        self.assertEqual(score["valid_window_tokens"], 13)

    def test_eval_mode_restored_after_nonfinite_prediction(self):
        class BrokenPolicy(nn.Module):
            def forward(self, imgs, states, prev_actions=None):
                return torch.full((*states.shape[:2], 3), float("nan"))
        model = BrokenPolicy().train()
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            evaluate_windows(model, self.data, "cpu")
        self.assertTrue(model.training)

    def test_best_is_not_overwritten_by_worse_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = CandidateCheckpoints(Path(directory) / "candidate.pt")
            model = nn.Linear(1, 1)
            score = {"objective": 0.1, "image_mode": "clean", "previous_actions": "teacher_forced"}
            self.assertTrue(tracker.consider(model, 250, score))
            original = model.weight.detach().clone()
            with torch.no_grad():
                model.weight.add_(1)
            self.assertFalse(tracker.consider(model, 500, {**score, "objective": 0.2}))
            tracker.save_latest(model, 500, snapshot=True)
            self.assertEqual(tracker.best_step, 250)
            self.assertTrue(torch.equal(torch.load(tracker.best, weights_only=True)["weight"], original))
            self.assertFalse(torch.equal(torch.load(tracker.latest, weights_only=True)["weight"], original))
            with self.assertRaisesRegex(ValueError, "Only clean"):
                tracker.consider(model, 500, {**score, "image_mode": "blank"})


class DatasetPaddingTests(unittest.TestCase):
    def test_short_episode_is_right_padded_masked_and_causally_equivalent(self):
        ep = {"split": "train", "imgs": np.ones((2, 4, 96, 96), dtype=np.uint8),
              "states": np.ones((2, 15), dtype=np.float32),
              "actions": np.full((2, 3), 0.2, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / "demo.pt"
            torch.save({"episodes": [ep]}, path)
            data = extract_causal_sequence_dataset(path, window_len=4, return_mask=True)
            with self.assertRaisesRegex(ValueError, "return_mask"):
                extract_causal_sequence_dataset(path, window_len=4)
        self.assertEqual(data[4].tolist(), [[True, True, False, False]])
        self.assertTrue(torch.all(data[0][:, 2:] == 0))
        self.assertTrue(torch.all(data[2][:, 0] == 0))
        model = MultiModalCarBrain(d_model=32, n_layers=1, n_heads=4, d_ff=64).eval()
        with torch.inference_mode():
            padded = model(data[0].float() / 255, data[1], prev_actions=data[2])
            short = model(data[0][:, :2].float() / 255, data[1][:, :2], prev_actions=data[2][:, :2])
        torch.testing.assert_close(padded[:, :2], short, atol=1e-6, rtol=1e-5)

    def test_quarter_of_training_windows_start_at_zero(self):
        ep = {"split": "train", "imgs": np.zeros((101, 4, 96, 96), dtype=np.uint8),
              "states": np.zeros((101, 15), dtype=np.float32),
              "actions": np.ones((101, 3), dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / "demo.pt"
            torch.save({"episodes": [ep]}, path)
            data = extract_causal_sequence_dataset(path, window_len=4, return_mask=True)
        initial = (data[2][:, 0] == 0).all(dim=-1).float().mean()
        self.assertGreaterEqual(float(initial), 0.25)


class ArgumentTests(unittest.TestCase):
    def test_invalid_counts_learning_rates_and_seed_rejected(self):
        for args in (["--steps", "0"], ["--seq_len", "65"], ["--eval_every", "0"],
                     ["--lr_vision", "nan"], ["--lr_backbone", "1"], ["--seed", "-1"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)

    def test_existing_and_input_checkpoint_overwrite_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.pt"
            output.touch()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--output_path", str(output)])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--output_path", str(output), "--existing_multimodal", str(output)])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
