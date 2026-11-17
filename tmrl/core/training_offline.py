# standard library imports
from dataclasses import dataclass
import logging
import time
from typing import Callable


__docformat__ = "google"


@dataclass(eq=False)
class TrainingOffline:
    """
    Training wrapper for off-policy algorithms.

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
        profiling (bool): if True, each round in ``run_epoch`` is profiled and logged at the end of the round
        agent_scheduler (callable): if not None, must be of the form ``f(agent, epoch)``, called at the beginning of each epoch
        start_training (int): minimum number of samples in the replay buffer before starting training
        device (str): device on which the memory will collate training samples
    """
    env_cls: type = None
    memory_cls: type = None
    training_agent_cls: type = None
    epochs: int = 10
    rounds: int = 50
    steps: int = 2000
    update_model_interval: int = 100
    update_buffer_interval: int = 100
    max_training_steps_per_env_step: float = 1.0
    sleep_between_buffer_retrieval_attempts: float = 1.0
    profiling: bool = False
    agent_scheduler: Callable = None
    start_training: int = 0
    device: str = None

    def __post_init__(self):
        self.total_updates = 0
        device = self.device
        self.epoch = 0
        self.memory = self.memory_cls(device=device)
        if isinstance(self.env_cls, tuple):
            observation_space, action_space = self.env_cls
        else:
            with self.env_cls() as env:
                observation_space, action_space = env.observation_space, env.action_space
        self.agent = self.training_agent_cls(observation_space=observation_space,
                                             action_space=action_space,
                                             device=device)
        self.total_samples = len(self.memory)
        logging.info(f"Initial total_samples: {self.total_samples}")

    def update_buffer(self, interface):
        buffer = interface.retrieve_buffer()
        self.memory.append(buffer)
        self.total_samples += len(buffer)

    def check_ratio(self, interface):
        ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 and self.total_samples >= self.start_training else -1.0
        if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
            logging.info("Waiting for new samples")
            while ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                self.update_buffer(interface)
                ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 and self.total_samples >= self.start_training else -1.0
                if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                    time.sleep(self.sleep_between_buffer_retrieval_attempts)
            logging.info("Resuming training")

    def run_epoch(self, interface):
        from pandas import DataFrame
        from tmrl.core.util import pandas_dict

        stats = []
        benchmarks_names = self.memory.get_benchmarks_names()

        if self.agent_scheduler is not None:
            self.agent_scheduler(self.agent, self.epoch)

        for rnd in range(self.rounds):
            logging.info(f"=== epoch {self.epoch + 1}/{self.epochs} ".ljust(20, '=') + f" round {rnd + 1}/{self.rounds} ".ljust(50, '='))
            logging.debug(f"(Training): current memory size: {len(self.memory)}")

            update_buffer_duration = 0.0
            sampling_duration = 0.0
            training_step_duration = 0.0
            model_broadcast_duration = 0.0
            idle_duration = 0.0
            stats_training = []

            t0 = time.perf_counter()
            self.check_ratio(interface)
            t1 = time.perf_counter()
            idle_duration += t1 - t0

            pro = None
            if self.profiling:
                from pyinstrument import Profiler
                pro = Profiler()
                pro.start()

            try:
                for _ in range(self.steps):
                    t_round_start = time.perf_counter()

                    if self.total_updates % self.update_buffer_interval == 0:
                        self.update_buffer(interface)

                    t_update_buffer = time.perf_counter()
                    batch = self.memory.sample()
                    t_sample = time.perf_counter()

                    if self.total_updates == 0:
                        logging.info("Starting training")

                    stats_training_dict = self.agent.train(batch)
                    t_train = time.perf_counter()

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
                    sampling_duration += t_sample - t_update_buffer
                    training_step_duration += t_train - t_sample
                    model_broadcast_duration += t_broadcast - t_train
                    idle_duration += t_round_end - t_broadcast

                    stats_training.append(stats_training_dict)

                t3 = time.perf_counter()
                round_duration = t3 - t0
                stats.append(pandas_dict(
                    memory_len=len(self.memory),
                    round_duration=round_duration,
                    time_spent_waiting=idle_duration,
                    time_spent_sampling=sampling_duration,
                    time_spent_updating_buffer=update_buffer_duration,
                    time_spent_training=training_step_duration,
                    time_spent_broadcasting_model=model_broadcast_duration,
                    **DataFrame(stats_training).mean(skipna=True),
                ))

                logging.info("Round statistics:\n" + stats[-1].add_prefix("  ").to_string() + '\n')
            finally:
                if pro is not None:
                    pro.stop()
                    logging.info(pro.output_text(unicode=True, color=False, show_all=True))

        self.epoch += 1
        return stats
