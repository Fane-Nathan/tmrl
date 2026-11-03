# standard library imports
import itertools
from copy import deepcopy
from dataclasses import dataclass

# third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD

# local imports
import tmrl.custom.torch.custom_models as models
from tmrl.custom.torch.utils.nn import copy_shared, no_grad
from tmrl.core.util import cached_property
from tmrl.core.torch.util import concat_collated
from tmrl.core.training import TrainingAgent
import tmrl.config.config_constants as cfg

import logging


# Soft Actor-Critic ====================================================================================================


@dataclass(eq=0)
class SpinupSACAgent(TrainingAgent):  # Adapted from Spinup
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = models.MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning (SAC v2)
    learn_entropy_coef: bool = True  # if True, SAC v2 is used, else, SAC v1 is used
    target_entropy: float = None  # if None, the target entropy for SAC v2 is set automatically
    optimizer_actor: str = "adam"  # one of ["adam", "adamw", "sgd"]
    optimizer_critic: str = "adam"  # one of ["adam", "adamw", "sgd"]
    betas_actor: tuple = None  # for Adam and AdamW
    betas_critic: tuple = None  # for Adam and AdamW
    l2_actor: float = None  # weight decay
    l2_critic: float = None  # weight decay

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up optimizers for policy and q-function:

        self.optimizer_actor = self.optimizer_actor.lower()
        self.optimizer_critic = self.optimizer_critic.lower()
        if self.optimizer_actor not in ["adam", "adamw", "sgd"]:
            logging.warning(f"actor optimizer {self.optimizer_actor} is not valid, defaulting to sgd")
        if self.optimizer_critic not in ["adam", "adamw", "sgd"]:
            logging.warning(f"critic optimizer {self.optimizer_critic} is not valid, defaulting to sgd")
        if self.optimizer_actor == "adam":
            pi_optimizer_cls = Adam
        elif self.optimizer_actor == "adamw":
            pi_optimizer_cls = AdamW
        else:
            pi_optimizer_cls = SGD
        pi_optimizer_kwargs = {"lr": self.lr_actor}
        if self.optimizer_actor in ["adam, adamw"] and self.betas_actor is not None:
            pi_optimizer_kwargs["betas"] = tuple(self.betas_actor)
        if self.l2_actor is not None:
            pi_optimizer_kwargs["weight_decay"] = self.l2_actor

        if self.optimizer_critic == "adam":
            q_optimizer_cls = Adam
        elif self.optimizer_critic == "adamw":
            q_optimizer_cls = AdamW
        else:
            q_optimizer_cls = SGD
        q_optimizer_kwargs = {"lr": self.lr_critic}
        if self.optimizer_critic in ["adam, adamw"] and self.betas_critic is not None:
            q_optimizer_kwargs["betas"] = tuple(self.betas_critic)
        if self.l2_critic is not None:
            q_optimizer_kwargs["weight_decay"] = self.l2_critic

        self.pi_optimizer = pi_optimizer_cls(self.model.actor.parameters(), **pi_optimizer_kwargs)
        self.q_optimizer = q_optimizer_cls(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()), **q_optimizer_kwargs)

        # entropy coefficient:

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape)  # .astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        o, a, r, o2, d, _ = batch

        pi, logp_pi = self.model.actor(o)
        # FIXME? log_prob = log_prob.reshape(-1, 1)

        # loss_alpha:

        loss_alpha = None
        if self.learn_entropy_coef:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        # Run one gradient descent step for Q1 and Q2

        # loss_q:

        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.model.actor(o2)

            # Target Q-values
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = (loss_q1 + loss_q2) / 2  # averaged for homogeneity with REDQ

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)

        # Next run one gradient descent step for actor.

        # loss_pi:

        # pi, logp_pi = self.model.actor(o)
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha_t * logp_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # FIXME: remove debug info
        with torch.no_grad():

            if not cfg.DEBUG_MODE:
                ret_dict = dict(
                    loss_actor=loss_pi.detach().item(),
                    loss_critic=loss_q.detach().item(),
                )
            else:
                q1_o2_a2 = self.model.q1(o2, a2)
                q2_o2_a2 = self.model.q2(o2, a2)
                q1_targ_pi = self.model_target.q1(o, pi)
                q2_targ_pi = self.model_target.q2(o, pi)
                q1_targ_a = self.model_target.q1(o, a)
                q2_targ_a = self.model_target.q2(o, a)

                diff_q1pt_qpt = (q1_pi_targ - q_pi_targ).detach()
                diff_q2pt_qpt = (q2_pi_targ - q_pi_targ).detach()
                diff_q1_q1t_a2 = (q1_o2_a2 - q1_pi_targ).detach()
                diff_q2_q2t_a2 = (q2_o2_a2 - q2_pi_targ).detach()
                diff_q1_q1t_pi = (q1_pi - q1_targ_pi).detach()
                diff_q2_q2t_pi = (q2_pi - q2_targ_pi).detach()
                diff_q1_q1t_a = (q1 - q1_targ_a).detach()
                diff_q2_q2t_a = (q2 - q2_targ_a).detach()
                diff_q1_backup = (q1 - backup).detach()
                diff_q2_backup = (q2 - backup).detach()
                diff_q1_backup_r = (q1 - backup + r).detach()
                diff_q2_backup_r = (q2 - backup + r).detach()

                ret_dict = dict(
                    loss_actor=loss_pi.detach().item(),
                    loss_critic=loss_q.detach().item(),
                    # debug:
                    debug_log_pi=logp_pi.detach().mean().item(),
                    debug_log_pi_std=logp_pi.detach().std().item(),
                    debug_logp_a2=logp_a2.detach().mean().item(),
                    debug_logp_a2_std=logp_a2.detach().std().item(),
                    debug_q_a1=q_pi.detach().mean().item(),
                    debug_q_a1_std=q_pi.detach().std().item(),
                    debug_q_a1_targ=q_pi_targ.detach().mean().item(),
                    debug_q_a1_targ_std=q_pi_targ.detach().std().item(),
                    debug_backup=backup.detach().mean().item(),
                    debug_backup_std=backup.detach().std().item(),
                    debug_q1=q1.detach().mean().item(),
                    debug_q1_std=q1.detach().std().item(),
                    debug_q2=q2.detach().mean().item(),
                    debug_q2_std=q2.detach().std().item(),
                    debug_diff_q1=diff_q1_backup.mean().item(),
                    debug_diff_q1_std=diff_q1_backup.std().item(),
                    debug_diff_q2=diff_q2_backup.mean().item(),
                    debug_diff_q2_std=diff_q2_backup.std().item(),
                    debug_diff_r_q1=diff_q1_backup_r.mean().item(),
                    debug_diff_r_q1_std=diff_q1_backup_r.std().item(),
                    debug_diff_r_q2=diff_q2_backup_r.mean().item(),
                    debug_diff_r_q2_std=diff_q2_backup_r.std().item(),
                    debug_diff_q1pt_qpt=diff_q1pt_qpt.mean().item(),
                    debug_diff_q2pt_qpt=diff_q2pt_qpt.mean().item(),
                    debug_diff_q1_q1t_a2=diff_q1_q1t_a2.mean().item(),
                    debug_diff_q2_q2t_a2=diff_q2_q2t_a2.mean().item(),
                    debug_diff_q1_q1t_pi=diff_q1_q1t_pi.mean().item(),
                    debug_diff_q2_q2t_pi=diff_q2_q2t_pi.mean().item(),
                    debug_diff_q1_q1t_a=diff_q1_q1t_a.mean().item(),
                    debug_diff_q2_q2t_a=diff_q2_q2t_a.mean().item(),
                    debug_diff_q1pt_qpt_std=diff_q1pt_qpt.std().item(),
                    debug_diff_q2pt_qpt_std=diff_q2pt_qpt.std().item(),
                    debug_diff_q1_q1t_a2_std=diff_q1_q1t_a2.std().item(),
                    debug_diff_q2_q2t_a2_std=diff_q2_q2t_a2.std().item(),
                    debug_diff_q1_q1t_pi_std=diff_q1_q1t_pi.std().item(),
                    debug_diff_q2_q2t_pi_std=diff_q2_q2t_pi.std().item(),
                    debug_diff_q1_q1t_a_std=diff_q1_q1t_a.std().item(),
                    debug_diff_q2_q2t_a_std=diff_q2_q2t_a.std().item(),
                    debug_r=r.detach().mean().item(),
                    debug_r_std=r.detach().std().item(),
                    debug_d=d.detach().mean().item(),
                    debug_d_std=d.detach().std().item(),
                    debug_a_0=a[:, 0].detach().mean().item(),
                    debug_a_0_std=a[:, 0].detach().std().item(),
                    debug_a_1=a[:, 1].detach().mean().item(),
                    debug_a_1_std=a[:, 1].detach().std().item(),
                    debug_a_2=a[:, 2].detach().mean().item(),
                    debug_a_2_std=a[:, 2].detach().std().item(),
                    debug_a1_0=pi[:, 0].detach().mean().item(),
                    debug_a1_0_std=pi[:, 0].detach().std().item(),
                    debug_a1_1=pi[:, 1].detach().mean().item(),
                    debug_a1_1_std=pi[:, 1].detach().std().item(),
                    debug_a1_2=pi[:, 2].detach().mean().item(),
                    debug_a1_2_std=pi[:, 2].detach().std().item(),
                    debug_a2_0=a2[:, 0].detach().mean().item(),
                    debug_a2_0_std=a2[:, 0].detach().std().item(),
                    debug_a2_1=a2[:, 1].detach().mean().item(),
                    debug_a2_1_std=a2[:, 1].detach().std().item(),
                    debug_a2_2=a2[:, 2].detach().mean().item(),
                    debug_a2_2_std=a2[:, 2].detach().std().item(),
                )

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# REDQ-SAC =============================================================================================================

@dataclass(eq=0)
class REDQSACAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = models.REDQMLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning (SAC v2)
    learn_entropy_coef: bool = True  # if True, SAC v2 is used, else, SAC v1 is used
    target_entropy: float = None  # if None, the target entropy for SAC v2 is set automatically
    optimizer_actor: str = "adam"  # one of ["adam", "adamw", "sgd"]
    optimizer_critic: str = "adam"  # one of ["adam", "adamw", "sgd"]
    betas_actor: tuple = None  # for Adam and AdamW
    betas_critic: tuple = None  # for Adam and AdamW
    l2_actor: float = None  # weight decay
    l2_critic: float = None  # weight decay
    m: int = 2  # number of randomly sampled target networks
    q_updates_per_policy_update: int = 1  # in REDQ, this is the "UTD ratio" (20), this interplays with lr_actor

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.n = len(self.model.qs)

        # Set up optimizers for policy and q-function:

        self.optimizer_actor = self.optimizer_actor.lower()
        self.optimizer_critic = self.optimizer_critic.lower()
        if self.optimizer_actor not in ["adam", "adamw", "sgd"]:
            logging.warning(f"actor optimizer {self.optimizer_actor} is not valid, defaulting to sgd")
        if self.optimizer_critic not in ["adam", "adamw", "sgd"]:
            logging.warning(f"critic optimizer {self.optimizer_critic} is not valid, defaulting to sgd")
        if self.optimizer_actor == "adam":
            pi_optimizer_cls = Adam
        elif self.optimizer_actor == "adamw":
            pi_optimizer_cls = AdamW
        else:
            pi_optimizer_cls = SGD
        pi_optimizer_kwargs = {"lr": self.lr_actor}
        if self.optimizer_actor in ["adam, adamw"] and self.betas_actor is not None:
            pi_optimizer_kwargs["betas"] = tuple(self.betas_actor)
        if self.l2_actor is not None:
            pi_optimizer_kwargs["weight_decay"] = self.l2_actor

        if self.optimizer_critic == "adam":
            q_optimizer_cls = Adam
        elif self.optimizer_critic == "adamw":
            q_optimizer_cls = AdamW
        else:
            q_optimizer_cls = SGD
        q_optimizer_kwargs = {"lr": self.lr_critic}
        if self.optimizer_critic in ["adam, adamw"] and self.betas_critic is not None:
            q_optimizer_kwargs["betas"] = tuple(self.betas_critic)
        if self.l2_critic is not None:
            q_optimizer_kwargs["weight_decay"] = self.l2_critic

        self.pi_optimizer = pi_optimizer_cls(self.model.actor.parameters(), **pi_optimizer_kwargs)
        # self.q_optimizer = q_optimizer_cls(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()), **q_optimizer_kwargs)
        self.q_optimizer = q_optimizer_cls(self.model.qs.parameters(), **q_optimizer_kwargs)

        self.i_update = 0  # for UTD ratio
        self.o_minibatches = []  # we stack minibatches for the actor update

        # entropy coefficient:

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape)  # .astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)

        self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)
        self.loss_pi = 0
        self.loss_alpha = 0

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):  # TODO

        o, a, r, o2, d, _ = batch

        self.o_minibatches.append(o)

        self.i_update += 1
        update_actor = (self.i_update % self.q_updates_per_policy_update == 0)

        if update_actor:
            o_batch = concat_collated(self.o_minibatches)
            self.o_minibatches = []
            pi, logp_pi = self.model.actor(o_batch)

        alpha_t = self.alpha_t

        # loss_q:

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.model.actor(o2)

            # Target Q-values
            if self.m < self.n:
                sample_idxs = np.random.choice(self.n, self.m, replace=False)
                qs_pi_targ_list = [self.model_target.qs[i](o2, a2) for i in sample_idxs]
            else:
                qs_pi_targ_list = [q(o2, a2) for q in self.model_target.qs]
            qs_pi_targ = torch.stack(qs_pi_targ_list, -1)  # shape: (batch, m)
            min_q, _ = torch.min(qs_pi_targ, dim=1, keepdim=False)
            backup = r + self.gamma * (1 - d) * (min_q - alpha_t * logp_a2)

        # MSE loss against Bellman backup
        q_pred_list = [q(o, a) for q in self.model.qs]
        q_pred = torch.stack(q_pred_list, -1)

        # backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup

        loss_q = ((q_pred - backup.unsqueeze(-1))**2).mean()
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # loss_pi:

        if update_actor:
            self.model.qs.requires_grad_(False)
            qs_pi_list = [q(o_batch, pi) for q in self.model.qs]
            qs_pi = torch.stack(qs_pi_list, -1)
            avg_q_pi = torch.mean(qs_pi, dim=1, keepdim=False)  # NB: SAC uses min here, not mean
            loss_pi = (alpha_t * logp_pi - avg_q_pi).mean()
            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()
            self.model.qs.requires_grad_(True)
            self.loss_pi = loss_pi.detach().item()

        # loss_alpha:

        loss_alpha = None
        if self.learn_entropy_coef and update_actor:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
            self.alpha_t = alpha_t
            self.loss_alpha = loss_alpha.detach().item()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        with torch.no_grad():
            ret_dict = dict(
                loss_actor=self.loss_pi,
                loss_critic=loss_q.detach().item(),
            )

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = self.loss_alpha
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# Experimental SAC auxiliary world model =================================================

from tmrl.custom.torch.world_model import TorchLatentWorldModel, TorchLatentAdversaryProposer, symlog
from tmrl.custom.torch.azr import ReplayLatentTaskBuffer, compute_azr_learnability


@dataclass(eq=0)
class DreamSACAgent(SpinupSACAgent):
    """
    SAC plus a gated replay-grounded AZR/imagination curriculum.

    Real replay remains the source of truth for SAC and for fitting the world
    model.  Candidate tasks are bounded perturbations of posterior states from
    those real observations.  They are kept only when the model predicts
    adequate continuation support, evaluated by repeated stochastic attempts,
    and prioritized with the AZR learnability rule.  Imagination updates the
    same action head returned by :meth:`get_actor`, so successful updates are
    broadcast to rollout workers rather than remaining in a side policy.

    These are model-space curriculum tasks, not generated TrackMania maps.  The
    conservative warm-up, short horizon, support screening, age limit, and
    small loss scale are intentional safeguards against model exploitation.
    """
    horizon: int = 3
    lr_world_model: float = 3e-4
    lr_adversary: float = 1e-4
    free_nats: float = 1.0
    enable_azr_imagination: bool = False
    enable_latent_adversary: bool = False
    world_model_warmup_steps: int = 1000
    solver_attempts: int = 8
    tasks_per_proposal: int = 8
    task_batch_size: int = 8
    task_buffer_capacity: int = 1024
    task_max_age: int = 2000
    task_proposal_interval: int = 4
    imagination_interval: int = 2
    min_task_continuation: float = 0.5
    max_latent_perturbation: float = 0.25
    target_quantile: float = 0.5
    imagination_actor_scale: float = 0.05
    imagination_reward_clip: float = 10.0
    imagination_grad_clip: float = 10.0
    azr_seed: int = 0

    def __post_init__(self):
        super().__post_init__()
        actor = self.model.actor
        if not hasattr(actor, "encode_observation") or not hasattr(actor, "forward_from_features"):
            raise TypeError(
                "DreamSACAgent requires an image actor exposing encode_observation() "
                "and forward_from_features()."
            )
        if self.enable_latent_adversary and not self.enable_azr_imagination:
            raise ValueError(
                "enable_latent_adversary is no longer a standalone mode; enable the "
                "validated AZR_IMAGINATION pipeline instead."
            )
        for name in (
            "horizon",
            "solver_attempts",
            "tasks_per_proposal",
            "task_batch_size",
            "task_buffer_capacity",
            "task_max_age",
            "task_proposal_interval",
            "imagination_interval",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.target_quantile <= 1.0:
            raise ValueError("target_quantile must be in [0, 1]")

        self.world_model = TorchLatentWorldModel(
            img_channels=cfg.IMG_HIST_LEN,
            img_height=cfg.IMG_HEIGHT,
            img_width=cfg.IMG_WIDTH,
            latent_dim=128,
            action_dim=3,
            hidden_dim=256,
            policy_feature_dim=actor.feature_dim,
        ).to(self.device)

        self.adversary = TorchLatentAdversaryProposer(
            feat_dim=384,
            perturbation_dim=128,
            max_magnitude=self.max_latent_perturbation,
        ).to(self.device)

        self.wm_optimizer = Adam(self.world_model.parameters(), lr=self.lr_world_model)
        self.adversary_optimizer = Adam(self.adversary.parameters(), lr=self.lr_adversary)
        self.task_buffer = ReplayLatentTaskBuffer(
            capacity=self.task_buffer_capacity,
            max_age=self.task_max_age,
            seed=self.azr_seed,
        )
        self.world_model_updates = 0
        self.azr_train_steps = 0
        self._azr_activation_logged = False

    def _empty_azr_metrics(self):
        return {
            "azr_ready": 0.0,
            "azr_warmup_updates": float(self.world_model_updates),
            "azr_warmup_remaining": float(
                max(self.world_model_warmup_steps - self.world_model_updates, 0)
            ),
            "azr_warmup_fraction": float(
                min(
                    self.world_model_updates / max(self.world_model_warmup_steps, 1),
                    1.0,
                )
            ),
            "azr_tasks_proposed": 0.0,
            "azr_tasks_accepted": 0.0,
            "azr_task_buffer_size": float(len(self.task_buffer)),
            "azr_mean_pass_rate": 0.0,
            "azr_mean_learnability": 0.0,
            "azr_mean_survival": 0.0,
            "loss_adversary": 0.0,
            "loss_imagination_actor": 0.0,
            "mean_imagined_return": 0.0,
            "imagined_policy_updates": 0.0,
        }

    def _alpha_for_actor(self):
        if self.learn_entropy_coef:
            return torch.exp(self.log_alpha.detach())
        return self.alpha_t.detach()

    def _set_world_model_grad(self, enabled):
        for parameter in self.world_model.parameters():
            parameter.requires_grad_(enabled)

    def _posterior_start(self, obs):
        embed = self.world_model.encoder(obs)
        batch_size = embed.shape[0]
        h = torch.zeros(
            batch_size,
            self.world_model.hidden_dim,
            device=embed.device,
            dtype=embed.dtype,
        )
        _, mean, std = self.world_model.rssm.compute_posterior(h, embed)
        return h, mean, std, self.world_model.get_feature(h, mean)

    def _rollout_tasks(self, h, z, attempts=1):
        """Roll out exact task states; gradients are controlled by the caller."""
        task_count = h.shape[0]
        h = h.unsqueeze(0).expand(attempts, -1, -1).reshape(
            attempts * task_count, h.shape[-1]
        )
        z = z.unsqueeze(0).expand(attempts, -1, -1).reshape(
            attempts * task_count, z.shape[-1]
        )
        cumulative_return = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        cumulative_log_prob = torch.zeros_like(cumulative_return)
        survival = torch.ones_like(cumulative_return)
        discount = 1.0

        for _ in range(self.horizon):
            feature = self.world_model.get_feature(h, z)
            policy_feature = self.world_model.predict_policy_feature(feature)
            action, log_prob = self.model.actor.forward_from_features(policy_feature)
            h, z, reward, continuation, _ = self.world_model.imagine_step(h, z, action)
            reward = reward.squeeze(-1).clamp(
                -self.imagination_reward_clip, self.imagination_reward_clip
            )
            continuation = continuation.squeeze(-1).clamp(0.0, 1.0)
            cumulative_return = cumulative_return + discount * survival * reward
            cumulative_log_prob = cumulative_log_prob + discount * survival * log_prob
            survival = survival * continuation
            discount *= self.gamma

        shape = (attempts, task_count)
        return (
            cumulative_return.reshape(shape),
            survival.reshape(shape),
            cumulative_log_prob.reshape(shape),
        )

    @staticmethod
    def _gaussian_kl(p_mean, p_std, q_mean, q_std):
        """KL(N(p)||N(q)), reduced over the latent dimension."""
        p_var = p_std.square()
        q_var = q_std.square()
        elem = (
            torch.log(q_std / (p_std + 1e-6))
            + (p_var + (p_mean - q_mean).square()) / (2.0 * q_var + 1e-6)
            - 0.5
        )
        return elem.sum(dim=-1)

    def _train_world_model(self, o, a, r, o2, d, truncated):
        if not isinstance(o, (tuple, list)) or len(o) < 4:
            raise ValueError("DreamSACAgent world-model training requires image observations")

        embed = self.world_model.encoder(o)
        next_embed = self.world_model.encoder(o2)
        batch_size = embed.shape[0]

        h0 = torch.zeros(batch_size, 256, device=embed.device, dtype=embed.dtype)
        z0, _, _ = self.world_model.rssm.compute_posterior(h0, embed)
        h1 = self.world_model.rssm.step_deterministic(h0, z0, a)
        _, prior_mean, prior_std = self.world_model.rssm.compute_prior(h1)
        z_post, post_mean, post_std = self.world_model.rssm.compute_posterior(h1, next_embed)

        feat0 = self.world_model.get_feature(h0, z0)
        feat1 = self.world_model.get_feature(h1, z_post)
        pred_reward = self.world_model.predict_reward(feat1)
        pred_continue = self.world_model.predict_continuation(feat1)

        pred_embedding0 = self.world_model.predict_embedding(feat0)
        pred_embedding1 = self.world_model.predict_embedding(feat1)
        pred_policy_feature0 = self.world_model.predict_policy_feature_symlog(feat0)
        pred_policy_feature1 = self.world_model.predict_policy_feature_symlog(feat1)
        with torch.no_grad():
            policy_feature0 = self.model.actor.encode_observation(o)
            policy_feature1 = self.model.actor.encode_observation(o2)

        reward_target = r.reshape(batch_size, 1).to(dtype=pred_reward.dtype)
        done = d.reshape(batch_size, 1).to(dtype=pred_continue.dtype)
        if truncated is not None:
            trunc = truncated.reshape(batch_size, 1).to(dtype=pred_continue.dtype)
            done = torch.maximum(done, trunc)
        continue_target = 1.0 - done

        loss_reward = F.mse_loss(pred_reward, symlog(reward_target))
        loss_continue = F.binary_cross_entropy(pred_continue, continue_target)
        loss_embedding = 0.5 * (
            F.mse_loss(pred_embedding0, embed.detach())
            + F.mse_loss(pred_embedding1, next_embed.detach())
        )
        loss_policy_feature = 0.5 * (
            F.smooth_l1_loss(pred_policy_feature0, symlog(policy_feature0.detach()))
            + F.smooth_l1_loss(pred_policy_feature1, symlog(policy_feature1.detach()))
        )

        kl_dyn = self._gaussian_kl(
            post_mean.detach(), post_std.detach(), prior_mean, prior_std
        )
        kl_rep = self._gaussian_kl(
            post_mean, post_std, prior_mean.detach(), prior_std.detach()
        )
        free_nats = torch.as_tensor(self.free_nats, device=embed.device, dtype=embed.dtype)
        loss_kl_dyn = torch.maximum(kl_dyn, free_nats).mean()
        loss_kl_rep = torch.maximum(kl_rep, free_nats).mean()

        loss_model = (
            loss_reward
            + loss_continue
            + loss_embedding
            + loss_policy_feature
            + loss_kl_dyn
            + 0.1 * loss_kl_rep
        )

        self.wm_optimizer.zero_grad(set_to_none=True)
        loss_model.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=100.0)
        self.wm_optimizer.step()
        self.world_model_updates += 1

        return {
            "loss_world_model": loss_model.detach().item(),
            "loss_wm_reward": loss_reward.detach().item(),
            "loss_wm_continue": loss_continue.detach().item(),
            "loss_wm_embedding": loss_embedding.detach().item(),
            "loss_wm_policy_feature": loss_policy_feature.detach().item(),
            "loss_wm_kl_dyn": loss_kl_dyn.detach().item(),
            "loss_wm_kl_rep": loss_kl_rep.detach().item(),
        }

    def _propose_and_validate_tasks(self, obs):
        metrics = self._empty_azr_metrics()
        with torch.no_grad():
            h, z_mean, z_std, feature = self._posterior_start(obs)
        task_count = min(self.tasks_per_proposal, h.shape[0])
        indices = torch.randperm(h.shape[0], device=h.device)[:task_count]
        h = h[indices]
        z_mean = z_mean[indices]
        z_std = z_std[indices]
        feature = feature[indices].detach()

        perturbation, proposal_log_prob = self.adversary.sample(feature)
        # The proposal is measured in posterior standard deviations, keeping it
        # inside a small local support region around a real replay observation.
        task_z = z_mean + perturbation * z_std

        with torch.no_grad():
            returns, survival, _ = self._rollout_tasks(
                h.detach(), task_z.detach(), attempts=self.solver_attempts
            )
            target_return = torch.quantile(
                returns, self.target_quantile, dim=0
            )
            successes = (
                (returns > target_return.unsqueeze(0))
                & (survival >= self.min_task_continuation)
            ).float()
            learnability, pass_rate = compute_azr_learnability(successes)
            mean_survival = survival.mean(dim=0)
            finite = (
                torch.isfinite(returns).all(dim=0)
                & torch.isfinite(survival).all(dim=0)
                & torch.isfinite(target_return)
            )
            in_support = perturbation.detach().abs().amax(dim=-1) <= (
                self.max_latent_perturbation + 1e-6
            )
            valid = finite & in_support & (mean_survival >= self.min_task_continuation)

        signal = learnability.detach() * valid.float()
        loss_adversary = torch.zeros((), device=h.device)
        if signal.sum() > 0.0:
            loss_adversary = -(signal * proposal_log_prob).sum() / valid.float().sum().clamp_min(1.0)
            self.adversary_optimizer.zero_grad(set_to_none=True)
            loss_adversary.backward()
            torch.nn.utils.clip_grad_norm_(self.adversary.parameters(), max_norm=10.0)
            self.adversary_optimizer.step()

        accepted = self.task_buffer.add_batch(
            h=h,
            z=task_z,
            target_returns=target_return,
            priorities=learnability,
            pass_rates=pass_rate,
            mean_survival=mean_survival,
            valid=valid,
            current_step=self.azr_train_steps,
        )
        metrics.update(
            azr_tasks_proposed=float(task_count),
            azr_tasks_accepted=float(accepted),
            azr_task_buffer_size=float(len(self.task_buffer)),
            azr_mean_pass_rate=pass_rate.mean().item(),
            azr_mean_learnability=learnability.mean().item(),
            azr_mean_survival=mean_survival.mean().item(),
            loss_adversary=loss_adversary.detach().item(),
        )
        return metrics

    def _refresh_tasks_and_train_actor(self):
        metrics = self._empty_azr_metrics()
        if len(self.task_buffer) == 0:
            return metrics

        task_ids, h, z, target, _ = self.task_buffer.sample(
            batch_size=self.task_batch_size,
            device=self.device,
            current_step=self.azr_train_steps,
        )
        with torch.no_grad():
            returns, survival, _ = self._rollout_tasks(
                h, z, attempts=self.solver_attempts
            )
            successes = (
                (returns > target.unsqueeze(0))
                & (survival >= self.min_task_continuation)
            ).float()
            learnability, pass_rate = compute_azr_learnability(successes)
            mean_survival = survival.mean(dim=0)
            valid = (
                torch.isfinite(returns).all(dim=0)
                & torch.isfinite(survival).all(dim=0)
                & (mean_survival >= self.min_task_continuation)
            )
            priorities = torch.where(valid, learnability, torch.zeros_like(learnability))
        self.task_buffer.update(
            task_ids=task_ids,
            priorities=priorities,
            pass_rates=pass_rate,
            mean_survival=mean_survival,
            current_step=self.azr_train_steps,
        )

        active = priorities > 0.0
        loss_actor = torch.zeros((), device=h.device)
        imagined_return = torch.zeros((), device=h.device)
        updated = 0.0
        if active.any():
            h_active = h[active].detach()
            z_active = z[active].detach()
            weights = priorities[active].detach()
            weights = weights / weights.mean().clamp_min(1e-6)

            # Freeze model parameters while retaining derivatives from actions
            # through the dynamics.  Only the deployed actor is optimized.
            self._set_world_model_grad(False)
            try:
                imagined_returns, _, log_probs = self._rollout_tasks(
                    h_active, z_active, attempts=1
                )
                imagined_return = imagined_returns[0].mean()
                alpha = self._alpha_for_actor()
                loss_actor = self.imagination_actor_scale * (
                    weights * (alpha * log_probs[0] - imagined_returns[0])
                ).mean()
                self.pi_optimizer.zero_grad(set_to_none=True)
                loss_actor.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.actor.parameters(), max_norm=self.imagination_grad_clip
                )
                self.pi_optimizer.step()
                updated = 1.0
            finally:
                self._set_world_model_grad(True)

        metrics.update(
            azr_task_buffer_size=float(len(self.task_buffer)),
            azr_mean_pass_rate=pass_rate.mean().item(),
            azr_mean_learnability=learnability.mean().item(),
            azr_mean_survival=mean_survival.mean().item(),
            loss_imagination_actor=loss_actor.detach().item(),
            mean_imagined_return=imagined_return.detach().item(),
            imagined_policy_updates=updated,
        )
        return metrics

    def train(self, batch):
        o, a, r, o2, d, truncated = batch
        world_model_metrics = self._train_world_model(o, a, r, o2, d, truncated)
        metrics = super().train(batch)
        metrics.update(world_model_metrics)
        azr_metrics = self._empty_azr_metrics()
        self.azr_train_steps += 1

        ready = (
            self.enable_azr_imagination
            and self.world_model_updates >= self.world_model_warmup_steps
        )
        if ready and not self._azr_activation_logged:
            logging.info(
                "AZR imagination activated after %s world-model updates; "
                "task proposal and screened actor updates are now enabled.",
                self.world_model_updates,
            )
            self._azr_activation_logged = True
        azr_metrics["azr_ready"] = float(ready)
        if ready and self.azr_train_steps % self.task_proposal_interval == 0:
            azr_metrics.update(self._propose_and_validate_tasks(o))
            azr_metrics["azr_ready"] = 1.0
        if (
            ready
            and len(self.task_buffer) > 0
            and self.azr_train_steps % self.imagination_interval == 0
        ):
            proposal_metrics = azr_metrics.copy()
            azr_metrics.update(self._refresh_tasks_and_train_actor())
            # Preserve counts/loss from a proposal performed on the same step.
            for key in ("azr_tasks_proposed", "azr_tasks_accepted", "loss_adversary"):
                azr_metrics[key] = proposal_metrics[key]
            azr_metrics["azr_ready"] = 1.0

        azr_metrics["azr_task_buffer_size"] = float(len(self.task_buffer))
        metrics.update(azr_metrics)
        return metrics
