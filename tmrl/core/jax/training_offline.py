import logging
import math
import time
from typing import Callable

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

import tmrl.config.config_constants as cst
from tmrl.core.jax import ensure_spawn_start_method
from tmrl.core.training_offline import TrainingOffline


__docformat__ = "google"


@nnx.jit
def _train_jit(batch, agent):
    return agent.train(batch)


@nnx.jit(static_argnums=(2,))
def _sample_and_train_jit(memory, agent, n: int):
    """Fully jitted loop sampling from the memory and training the agent ``n`` times."""
    memory_graphdef, memory_state = nnx.split(memory)
    agent_graphdef, agent_state = nnx.split(agent)

    def step(carry, _):
        memory_state, agent_state = carry
        memory = nnx.merge(memory_graphdef, memory_state)
        agent = nnx.merge(agent_graphdef, agent_state)

        batch = memory.sample()
        ret = agent.train(batch)

        _, memory_state = nnx.split(memory)
        _, agent_state = nnx.split(agent)
        return (memory_state, agent_state), ret

    (final_memory_state, final_agent_state), training_results = jax.lax.scan(
        f=step,
        init=(memory_state, agent_state),
        xs=None,
        length=n,
    )

    nnx.update(memory, final_memory_state)
    nnx.update(agent, final_agent_state)

    return training_results


class NNXTrainingOffline(TrainingOffline):
    """
    TrainingOffline for trainers based on Flax NNX, with jit-able train() method.
    Optionally, NNXTrainingOffline can be used with memories that have a jit-able sample() method.
    This enables jitting the sampling and training loop over jit_substeps iterations within a single JIT block.
    """
    def __init__(self,
                 env_cls: type = None,
                 memory_cls: type = None,
                 training_agent_cls: type = None,
                 epochs: int = 10,
                 rounds: int = 50,
                 steps: int = 2000,
                 update_model_interval: int = 100,
                 update_buffer_interval: int = 100,
                 max_training_steps_per_env_step: float = 1.0,
                 sleep_between_buffer_retrieval_attempts: float = 1.0,
                 profiling: bool = False,
                 agent_scheduler: Callable = None,
                 start_training: int = 0,
                 device: str = None,
                 jit_sampling: bool = False,
                 jit_substeps: int = 1,
                 jit_training: bool = True):
        """
        Args:
            env_cls (type): class of a dummy environment, used only to retrieve observation and action spaces if needed.
                Alternatively, this can be a tuple of the form ``(observation_space, action_space)``.
            memory_cls (type): class of the replay memory
            training_agent_cls (type): class of the training agent
            epochs (int): total number of epochs, we save the agent every epoch
            rounds (int): number of rounds per epoch, we generate statistics every round
            steps (int): number of training steps per round
            update_model_interval (int): number of training steps between model broadcasts
            update_buffer_interval (int): number of training steps between retrieving buffered samples
            max_training_steps_per_env_step (float): training will pause when above this ratio
            sleep_between_buffer_retrieval_attempts (float): algorithm will sleep for this amount of time when waiting for needed incoming samples
            profiling (bool): if True, `train()` and `memory.sample()` will be traced using the JAX profiler
            agent_scheduler (callable): if not None, must be of the form ``f(agent, epoch)``, called at the beginning of each epoch
            start_training (int): minimum number of samples in the replay buffer before starting training
            device (str): device to use (None for automatic)
            jit_sampling (bool): whether to jit the sampling method from memory_cls with nnx.jit
            jit_substeps (int): when jit_sampling is true, sampling + training steps are clubbed into jitted blocks of jit_substeps iterations to alleviate python-XLA transfer bottlenecks
            jit_training (bool): whether to wrap the complete agent.train call in nnx.jit. Set False for agents that dispatch between separately jitted phases on the host.
        """
        ensure_spawn_start_method()

        if jit_sampling and not jit_training:
            raise ValueError("jit_sampling requires jit_training=True")
        if jit_sampling:
            if jit_substeps < 1:
                raise ValueError("jit_substeps must be >= 1 when jit_sampling is True")
            # When jit_sampling is true, every "training step" contains jit_substeps actual training steps.
            # Adapt step intervals and the per-env-step update ratio accordingly:
            update_model_interval = math.ceil(update_model_interval / jit_substeps)
            update_buffer_interval = math.ceil(update_buffer_interval / jit_substeps)
            max_training_steps_per_env_step = max_training_steps_per_env_step / jit_substeps

        super().__init__(env_cls,
                         memory_cls,
                         training_agent_cls,
                         epochs,
                         rounds,
                         steps,
                         update_model_interval,
                         update_buffer_interval,
                         max_training_steps_per_env_step,
                         sleep_between_buffer_retrieval_attempts,
                         profiling,
                         agent_scheduler,
                         start_training,
                         device)
        self.jit_sampling = jit_sampling
        self.jit_substeps = jit_substeps
        self.jit_training = jit_training

    def _train_batch(self, batch):
        if self.jit_training:
            return _train_jit(batch, self.agent)
        return self.agent.train(batch)

    def run_epoch(self, interface):
        from pandas import DataFrame
        from tmrl.core.util import pandas_dict

        stats = []

        # === JAX profiling tool ===

        if self.profiling:
            nb_profiling_steps = 10
            output_trace_path = cst.TMRL_FOLDER / "jax" / "trace"

            logging.info("PROFILING...")
            self.check_ratio(interface)

            if not self.jit_sampling:
                batch = self.memory.sample()
                t_compile_start = time.perf_counter()
                out = self._train_batch(batch)
                t_compile_end = time.perf_counter()
            else:
                t_compile_start = time.perf_counter()
                out = _sample_and_train_jit(self.memory, self.agent, self.jit_substeps)
                t_compile_end = time.perf_counter()

            jax.tree_util.tree_map(lambda v: v.block_until_ready() if hasattr(v, "block_until_ready") else v, out)
            logging.info(f"JIT compilation time: {t_compile_end - t_compile_start}")

            logging.info(f"RUNNING JAX PROFILER FOR {nb_profiling_steps} STEPS...")
            with jax.profiler.trace(str(output_trace_path)):
                for _ in range(nb_profiling_steps):
                    if not self.jit_sampling:
                        batch = self.memory.sample()
                        with jax.profiler.TraceAnnotation("train"):
                            out = self._train_batch(batch)
                    else:
                        with jax.profiler.TraceAnnotation("sample_and_train"):
                            out = _sample_and_train_jit(self.memory, self.agent, self.jit_substeps)
                jax.tree_util.tree_map(lambda v: v.block_until_ready() if hasattr(v, "block_until_ready") else v, out)

            logging.info(f"FINISHED PROFILING. OUTPUT IN {output_trace_path}")

        # === Epoch ===

        benchmarks_names = self.memory.get_benchmarks_names()

        if self.agent_scheduler is not None:
            self.agent_scheduler(self.agent, self.epoch)

        for rnd in range(self.rounds):
            logging.info(f"=== epoch {self.epoch + 1}/{self.epochs} ".ljust(20, '=') + f" round {rnd + 1}/{self.rounds} ".ljust(50, '='))
            logging.debug(f"(Training): current memory size: {len(self.memory)}")

            update_buffer_duration = 0.0
            sampling_duration = 0.0
            training_step_duration = 0.0
            sampling_and_training_duration = 0.0
            model_broadcast_duration = 0.0
            idle_duration = 0.0
            stats_training = []

            t0 = time.perf_counter()
            self.check_ratio(interface)
            t1 = time.perf_counter()
            idle_duration += t1 - t0

            for _ in range(self.steps):
                t_round_start = time.perf_counter()

                if self.total_updates % self.update_buffer_interval == 0:
                    self.update_buffer(interface)

                t_update_buffer = time.perf_counter()

                if self.total_updates == 0:
                    logging.info("Starting training")

                if not self.jit_sampling:
                    batch = self.memory.sample()
                    t_sample = time.perf_counter()
                    stats_training_dict = self._train_batch(batch)
                    t_train = time.perf_counter()
                    sampling_duration += t_sample - t_update_buffer
                    training_step_duration += t_train - t_sample
                else:
                    stats_training_dict = _sample_and_train_jit(self.memory, self.agent, self.jit_substeps)
                    t_train = time.perf_counter()
                    sampling_and_training_duration += t_train - t_update_buffer

                if self.jit_sampling:
                    stats_training_dict = {k: float(jnp.mean(v)) for k, v in stats_training_dict.items()}
                else:
                    stats_training_dict = {
                        k: float(np.asarray(v).mean()) if np.ndim(v) > 0 else float(v)
                        for k, v in stats_training_dict.items()
                    }

                self.total_updates += 1
                if self.total_updates % self.update_model_interval == 0:
                    interface.broadcast_model(self.agent.get_actor())

                t_broadcast = time.perf_counter()
                self.check_ratio(interface)
                t_round_end = time.perf_counter()

                stats_training_dict["return_test"] = self.memory.stat_test_return
                stats_training_dict["return_train"] = self.memory.stat_train_return
                stats_training_dict["episode_length_test"] = self.memory.stat_test_steps
                stats_training_dict["episode_length_train"] = self.memory.stat_train_steps

                memory_benchmarks = self.memory.get_benchmarks()
                if memory_benchmarks is not None:
                    for i, benchmark in enumerate(memory_benchmarks):
                        if benchmarks_names is not None and i < len(benchmarks_names):
                            stats_training_dict[benchmarks_names[i]] = benchmark
                        else:
                            stats_training_dict[f"memory_benchmark_{i}"] = benchmark

                update_buffer_duration += t_update_buffer - t_round_start
                model_broadcast_duration += t_broadcast - t_train
                idle_duration += t_round_end - t_broadcast

                stats_training.append(stats_training_dict)

            t3 = time.perf_counter()
            round_duration = t3 - t0
            stats.append(pandas_dict(
                memory_len=len(self.memory),
                round_duration=round_duration,
                time_spent_waiting=idle_duration,
                time_spent_updating_buffer=update_buffer_duration,
                **(
                    dict(time_spent_sampling=sampling_duration, time_spent_training=training_step_duration)
                    if not self.jit_sampling else
                    dict(time_spent_sampling_and_training=sampling_and_training_duration)
                ),
                time_spent_broadcasting_model=model_broadcast_duration,
                **DataFrame(stats_training).mean(skipna=True),
            ))

            logging.info("Round statistics:\n" + stats[-1].add_prefix("  ").to_string() + '\n')

        self.epoch += 1
        return stats
