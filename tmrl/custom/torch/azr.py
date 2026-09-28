"""Replay-grounded AZR-style curriculum utilities for Torch imagination.

The objects in this module deliberately call their contents *latent tasks*.
They are short-lived challenges anchored to posterior states inferred from real
replay observations; they are not executable TrackMania maps.  A task only
enters the buffer after support and continuation checks, and its priority is
computed from repeated binary solver attempts on that exact stored task.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


def compute_azr_learnability(
    solver_successes: torch.Tensor,
    rollout_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-task AZR learnability rewards and pass rates.

    ``solver_successes`` must contain binary outcomes from repeated attempts at
    the same tasks.  Following AZR, impossible tasks receive zero reward and a
    partially solved task receives ``1 - pass_rate``.  Trivial tasks are also
    assigned zero here because they no longer provide curriculum signal.
    """
    if solver_successes.ndim < 2:
        raise ValueError("solver_successes must include rollout and task axes")
    if not torch.all((solver_successes == 0) | (solver_successes == 1)):
        raise ValueError("solver_successes must be binary")

    pass_rate = solver_successes.float().mean(dim=rollout_axis)
    learnable = (pass_rate > 0.0) & (pass_rate < 1.0)
    reward = torch.where(learnable, 1.0 - pass_rate, torch.zeros_like(pass_rate))
    return reward, pass_rate


@dataclass
class ReplayLatentTask:
    """A persistent, replay-anchored challenge in a local model snapshot."""

    task_id: int
    h: torch.Tensor
    z: torch.Tensor
    target_return: float
    priority: float
    pass_rate: float
    mean_survival: float
    created_step: int
    last_evaluated_step: int


class ReplayLatentTaskBuffer:
    """Small prioritized buffer for screened posterior-anchored tasks."""

    def __init__(self, capacity: int = 1024, max_age: int = 2000, seed: int = 0):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if max_age < 1:
            raise ValueError("max_age must be positive")
        self.capacity = int(capacity)
        self.max_age = int(max_age)
        self.rng = np.random.default_rng(seed)
        self.tasks: list[ReplayLatentTask] = []
        self._next_task_id = 0

    def __len__(self) -> int:
        return len(self.tasks)

    def prune(self, current_step: int) -> None:
        """Drop model-space tasks after the model has drifted too far."""
        self.tasks = [
            task
            for task in self.tasks
            if current_step - task.created_step <= self.max_age
        ]

    def add_batch(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        target_returns: torch.Tensor,
        priorities: torch.Tensor,
        pass_rates: torch.Tensor,
        mean_survival: torch.Tensor,
        valid: torch.Tensor,
        current_step: int,
    ) -> int:
        """Insert valid and currently learnable tasks, returning the count."""
        tensors: Sequence[torch.Tensor] = (
            h,
            z,
            target_returns,
            priorities,
            pass_rates,
            mean_survival,
            valid,
        )
        batch_size = h.shape[0]
        if any(t.shape[0] != batch_size for t in tensors):
            raise ValueError("all task fields must share the same batch dimension")

        self.prune(current_step)
        added = 0
        for i in range(batch_size):
            priority = float(priorities[i].detach().cpu())
            if not bool(valid[i].detach().cpu()) or priority <= 0.0:
                continue
            task = ReplayLatentTask(
                task_id=self._next_task_id,
                h=h[i].detach().to(device="cpu", dtype=torch.float32).clone(),
                z=z[i].detach().to(device="cpu", dtype=torch.float32).clone(),
                target_return=float(target_returns[i].detach().cpu()),
                priority=priority,
                pass_rate=float(pass_rates[i].detach().cpu()),
                mean_survival=float(mean_survival[i].detach().cpu()),
                created_step=int(current_step),
                last_evaluated_step=int(current_step),
            )
            self._next_task_id += 1
            if len(self.tasks) < self.capacity:
                self.tasks.append(task)
            else:
                # Prefer replacing a stale/easy task.  Age breaks priority ties.
                replace_idx = min(
                    range(len(self.tasks)),
                    key=lambda j: (
                        self.tasks[j].priority,
                        self.tasks[j].created_step,
                    ),
                )
                self.tasks[replace_idx] = task
            added += 1
        return added

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        current_step: int,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample tasks proportionally to learnability priority."""
        self.prune(current_step)
        if not self.tasks:
            raise RuntimeError("cannot sample an empty latent task buffer")

        priorities = np.asarray(
            [max(task.priority, 0.0) for task in self.tasks], dtype=np.float64
        )
        positive_indices = np.flatnonzero(priorities > 0.0)
        if positive_indices.size > 0:
            # Sampling without replacement cannot request more entries than
            # have non-zero probability. Zero-priority tasks remain stored so
            # they can expire naturally, but they do not displace currently
            # learnable tasks from a minibatch.
            candidate_indices = positive_indices
            candidate_priorities = priorities[candidate_indices]
            probabilities = candidate_priorities / candidate_priorities.sum()
        else:
            # Re-evaluate a uniform subset when every stored task has become
            # trivial or impossible; model drift may make one learnable again.
            candidate_indices = np.arange(len(self.tasks))
            probabilities = None
        count = min(int(batch_size), len(candidate_indices))
        candidate_positions = self.rng.choice(
            len(candidate_indices), size=count, replace=False, p=probabilities
        )
        indices = candidate_indices[np.atleast_1d(candidate_positions)]
        sampled = [self.tasks[int(i)] for i in np.atleast_1d(indices)]

        task_ids = [task.task_id for task in sampled]
        h = torch.stack([task.h for task in sampled]).to(device=device)
        z = torch.stack([task.z for task in sampled]).to(device=device)
        target = torch.tensor(
            [task.target_return for task in sampled], device=device, dtype=torch.float32
        )
        priority = torch.tensor(
            [task.priority for task in sampled], device=device, dtype=torch.float32
        )
        return task_ids, h, z, target, priority

    def update(
        self,
        task_ids: Iterable[int],
        priorities: torch.Tensor,
        pass_rates: torch.Tensor,
        mean_survival: torch.Tensor,
        current_step: int,
    ) -> None:
        """Refresh priorities using new attempts at the same persistent tasks."""
        by_id = {task.task_id: task for task in self.tasks}
        for i, task_id in enumerate(task_ids):
            task = by_id.get(int(task_id))
            if task is None:
                continue
            task.priority = float(priorities[i].detach().cpu())
            task.pass_rate = float(pass_rates[i].detach().cpu())
            task.mean_survival = float(mean_survival[i].detach().cpu())
            task.last_evaluated_step = int(current_step)
