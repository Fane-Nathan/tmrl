"""Distributed networking primitives (`Server`, `Trainer`, `RolloutWorker`, `Buffer`)."""

# standard library imports
import atexit
from collections.abc import Sequence
import datetime
import functools
import itertools
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time
from typing import Callable

# third-party imports
import numpy as np
import requests
from tlspyo import Endpoint, Relay

# local imports
import tmrl.config.config_constants as cfg
from tmrl.core.actor import ActorModule
from tmrl.core.util import dump, load, partial_to_dict


__docformat__ = "google"

__all__ = [
    "Buffer",
    "RolloutWorker",
    "Server",
    "Trainer",
    "TrainerInterface",
    "dump_run_instance",
    "iterate_epochs",
    "load_run_instance",
    "log_environment_variables",
    "print_ip",
    "print_with_timestamp",
    "run",
    "run_with_wandb",
]


# === Logging & Network Helpers =======================================================================================


def print_with_timestamp(s):
    """Logs ``s`` at INFO level prefixed with the current local timestamp."""
    x = datetime.datetime.now()
    sx = x.strftime("%x %X ")
    logging.info(sx + str(s))


def print_ip():
    """Logs the machine's public and local IP addresses."""
    local_ip = socket.gethostbyname(socket.gethostname())
    try:
        resp = requests.get("https://api.ipify.org", timeout=5.0)
        resp.raise_for_status()
        public_ip = resp.text.strip()
    except requests.RequestException as exc:
        public_ip = f"unavailable ({exc})"
    print_with_timestamp(f"public IP: {public_ip}, local IP: {local_ip}")


# === Experience Buffer ===============================================================================================


class Buffer:
    """
    Buffer of training samples.

    `Server`, `RolloutWorker` and `Trainer` all have their own `Buffer` to store and send training samples.

    Samples are tuples of the form (`act`, `new_obs`, `rew`, `terminated`, `truncated`, `info`)
    """
    def __init__(self, maxlen=cfg.BUFFERS_MAXLEN):
        """
        Args:
            maxlen (int): buffer length
        """
        self.memory = []
        self.stat_train_return = 0.0  # stores the train return
        self.stat_test_return = 0.0  # stores the test return
        self.stat_train_steps = 0  # stores the number of steps per training episode
        self.stat_test_steps = 0  # stores the number of steps per test episode
        self.maxlen = maxlen

    def clip_to_maxlen(self):
        lenmem = len(self.memory)
        if lenmem > self.maxlen:
            print_with_timestamp("buffer overflow. Discarding old samples.")
            self.memory = self.memory[(lenmem - self.maxlen):]

    def append_sample(self, sample):
        """
        Appends `sample` to the buffer.

        Args:
            sample (Tuple): a training sample of the form (`act`, `new_obs`, `rew`, `terminated`, `truncated`, `info`)
        """
        self.memory.append(sample)
        self.clip_to_maxlen()

    def clear(self):
        """
        Clears the buffer but keeps train and test returns.
        """
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        self.memory += other.memory
        self.clip_to_maxlen()
        self.stat_train_return = other.stat_train_return
        self.stat_test_return = other.stat_test_return
        self.stat_train_steps = other.stat_train_steps
        self.stat_test_steps = other.stat_test_steps
        return self


# === Server ==========================================================================================================


class Server:
    """
    Central server.

    The `Server` lets 1 `Trainer` and n `RolloutWorkers` connect.
    It buffers experiences sent by workers and periodically sends these to the trainer.
    It also receives the weights from the trainer and broadcasts these to the connected workers.
    """
    def __init__(self,
                 port=cfg.PORT,
                 password=cfg.PASSWORD,
                 local_port=cfg.LOCAL_PORT_SERVER,
                 header_size=cfg.HEADER_SIZE,
                 security=cfg.SECURITY,
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,
                 accepted_groups: None | dict = None):
        """
        Args:
            port (int): tlspyo public port
            password (str): tlspyo password
            local_port (int): tlspyo local communication port
            header_size (int): tlspyo header size (bytes)
            security (Union[str, None]): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory
            accepted_groups (Union[None, dict]): `tlspyo` accepted groups (None to accept all groups)
        """
        self.__relay = Relay(port=port,
                             password=password,
                             accepted_groups=accepted_groups,
                             local_com_port=local_port,
                             header_size=header_size,
                             security=security,
                             keys_dir=keys_dir)

    def stop(self):
        """
        Stops the tlspyo Relay.
        """
        self.__relay.stop()


# === Trainer =========================================================================================================


class TrainerInterface:
    """
    This is the trainer's network interface
    This connects to the server
    This receives samples batches and sends new weights
    """
    def __init__(self,
                 server_ip=None,
                 server_port=cfg.PORT,
                 password=cfg.PASSWORD,
                 local_com_port=cfg.LOCAL_PORT_TRAINER,
                 header_size=cfg.HEADER_SIZE,
                 max_buf_len=cfg.BUFFER_SIZE,
                 security=cfg.SECURITY,
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,
                 hostname=cfg.HOSTNAME,
                 model_path=cfg.MODEL_PATH_TRAINER,
                 groups="trainers",
                 target_groups="workers"):

        self.groups = (groups,) if isinstance(groups, str) else groups
        self.target_groups = target_groups
        self.model_path = model_path
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.__endpoint = Endpoint(ip_server=self.server_ip,
                                   port=server_port,
                                   password=password,
                                   groups=groups,
                                   local_com_port=local_com_port,
                                   header_size=header_size,
                                   max_buf_len=max_buf_len,
                                   security=security,
                                   keys_dir=keys_dir,
                                   hostname=hostname)

        print_with_timestamp(f"server IP: {self.server_ip}")

        self._notify()

    def _notify(self):
        groups = {g: -1 for g in self.groups}  # retrieve everything for all member groups
        self.__endpoint.notify(groups=groups)

    def stop(self):
        """
        Stops the tlspyo Endpoint.
        """
        self.__endpoint.stop()

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule
        broadcasts the model's weights to all connected RolloutWorkers
        """
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(self.model_path)
        with open(self.model_path, 'rb') as f:
            weights = f.read()
        if isinstance(self.target_groups, str):
            self.__endpoint.broadcast(weights, self.target_groups)
        else:
            self.__endpoint.send_object(weights, {g: -1 for g in self.target_groups})

    def retrieve_buffer(self):
        """
        returns the TrainerInterface's buffer of training samples
        """
        buffers = self.__endpoint.receive_all()
        res = Buffer()
        for buf in buffers:
            res += buf
        self._notify()
        return res


def log_environment_variables():
    """
    add certain relevant environment variables to our config
    usage: `LOG_VARIABLES='HOME JOBID' python ...`
    """
    return {k: os.environ.get(k, '') for k in os.environ.get('LOG_VARIABLES', '').strip().split()}


def load_run_instance(checkpoint_path):
    """
    Default function used to load trainers from checkpoint path
    Args:
        checkpoint_path: the path where instances of run_cls are checkpointed
    Returns:
        An instance of run_cls loaded from checkpoint_path
    """
    return load(checkpoint_path)


def dump_run_instance(run_instance, checkpoint_path):
    """
    Default function used to dump trainers to checkpoint path
    Args:
        run_instance: the instance of run_cls to checkpoint
        checkpoint_path: the path where instances of run_cls are checkpointed
    """
    dump(run_instance, checkpoint_path)


def iterate_epochs(run_cls,
                   interface: TrainerInterface,
                   checkpoint_path: str,
                   dump_run_instance_fn=dump_run_instance,
                   load_run_instance_fn=load_run_instance,
                   epochs_between_checkpoints=1,
                   updater_fn=None):
    """
    Main training loop (remote).

    When ``checkpoint_path`` is provided, ``run_cls`` is saved every
    ``epochs_between_checkpoints`` epochs. Model weights are broadcast according
    to the trainer's ``update_model_interval`` schedule.

    Yields:
        list[pd.Series]: Episode/round statistics produced by ``run_instance.run_epoch``.
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance

    if checkpoint_path and epochs_between_checkpoints <= 0:
        raise ValueError("epochs_between_checkpoints must be >= 1.")

    logging.debug(f"checkpoint_path: {checkpoint_path}")
    if not checkpoint_path:
        logging.info("=== specification ".ljust(70, "="))
        run_instance = run_cls()
        logging.info("")
    elif not Path(checkpoint_path).exists():
        logging.info("=== specification ".ljust(70, "="))
        run_instance = run_cls()
        dump_run_instance_fn(run_instance, checkpoint_path)
        logging.info("")
    else:
        logging.info("Loading checkpoint...")
        t1 = time.time()
        run_instance = load_run_instance_fn(checkpoint_path)
        logging.info(f"Loaded checkpoint in {time.time() - t1} seconds.")
        if updater_fn is not None:
            logging.info("Updating checkpoint...")
            t1 = time.time()
            run_instance = updater_fn(run_instance, run_cls)
            logging.info(f"Checkpoint updated in {time.time() - t1} seconds.")

    while run_instance.epoch < run_instance.epochs:
        yield run_instance.run_epoch(interface=interface)
        if checkpoint_path and run_instance.epoch % epochs_between_checkpoints == 0:
            logging.info("Saving checkpoint...")
            t1 = time.time()
            dump_run_instance_fn(run_instance, checkpoint_path)
            logging.info(f"Saved checkpoint in {time.time() - t1} seconds.")


def _extract_wandb_config(run_cls):
    try:
        if isinstance(run_cls, functools.partial):
            config = partial_to_dict(run_cls)
        elif callable(run_cls):
            config = partial_to_dict(functools.partial(run_cls))
        else:
            config = {"run_cls": str(run_cls)}
    except Exception:
        config = {"run_cls": getattr(run_cls, "__qualname__", str(run_cls))}
    config["environ"] = log_environment_variables()
    return config


def run_with_wandb(entity, project, run_id, interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
    """
    Main training loop (remote).

    Saves config and stats to https://wandb.com
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    wandb_dir = tempfile.mkdtemp()  # prevent wandb from polluting the home directory
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
    import wandb

    logging.debug(f"run_cls: {run_cls}")
    config = _extract_wandb_config(run_cls)
    resume = bool(checkpoint_path and Path(checkpoint_path).exists())
    wandb_initialized = False
    err_cpt = 0
    while not wandb_initialized:
        try:
            wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)
            wandb_initialized = True
        except Exception as e:
            err_cpt += 1
            logging.warning(f"wandb error {err_cpt}: {e}")
            if err_cpt > 10:
                logging.warning("Could not connect to wandb, aborting.")
                raise RuntimeError(f"Could not connect to wandb after {err_cpt} attempts.") from e
            time.sleep(10.0)
    try:
        for stats in iterate_epochs(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):
            if stats:
                for s in stats:
                    wandb.log(json.loads(s.to_json()))
    finally:
        wandb.finish()


def run(interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
    """
    Main training loop (remote).
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    for _ in iterate_epochs(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):
        pass


class Trainer:
    """
    Training entity.

    The `Trainer` object is where RL training happens.
    Typically, it can be located on a HPC cluster.
    """
    def __init__(self,
                 training_cls,
                 server_ip=cfg.SERVER_IP_FOR_TRAINER,
                 server_port=cfg.PORT,
                 password=cfg.PASSWORD,
                 local_com_port=cfg.LOCAL_PORT_TRAINER,
                 header_size=cfg.HEADER_SIZE,
                 max_buf_len=cfg.BUFFER_SIZE,
                 security=cfg.SECURITY,
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,
                 hostname=cfg.HOSTNAME,
                 model_path=cfg.MODEL_PATH_TRAINER,
                 checkpoint_path=cfg.CHECKPOINT_PATH,
                 dump_run_instance_fn: Callable = None,
                 load_run_instance_fn: Callable = None,
                 updater_fn: Callable = None,
                 groups: str | tuple[str] = "trainers",
                 target_groups: str | tuple[str] = "workers"):
        """
        Args:
            training_cls (type): training class (subclass of tmrl.core.training_offline.TrainingOffline)
            server_ip (str): ip of the central `Server`
            server_port (int): public port of the central `Server`
            password (str): password of the central `Server`
            local_com_port (int): port used by `tlspyo` for local communication
            header_size (int): number of bytes used for `tlspyo` headers
            max_buf_len (int): maximum number of messages queued by `tlspyo`
            security (str): `tlspyo` security type (None or "TLS")
            keys_dir (str): custom credentials directory for `tlspyo` TLS security
            hostname (str): custom TLS hostname
            model_path (str): path where a local copy of the model will be saved
            checkpoint_path: path where the `Trainer` will be checkpointed (`None` = no checkpointing)
            dump_run_instance_fn (callable): custom serializer (`None` = pickle.dump)
            load_run_instance_fn (callable): custom deserializer (`None` = pickle.load)
            updater_fn (callable): custom updater (`None` = no updater). If provided, this must be a function \
                that takes a checkpoint and training_cls as argument and returns an updated checkpoint. \
                The updater is called after a checkpoint is loaded, e.g., to update your checkpoint with new arguments.
            groups (Union[str, Tuple[str]]): `tlspyo` groups this `Trainer` endpoint belongs to; \
                `RolloutWorker` instances must produce experience buffers targeting these groups.
            target_groups (Union[str, Tuple[str]]): `tlspyo` groups to which model weights are broadcast \
                (i.e., groups `RolloutWorker` instances belong to)
        """
        self.checkpoint_path = checkpoint_path
        self.dump_run_instance_fn = dump_run_instance_fn
        self.load_run_instance_fn = load_run_instance_fn
        self.updater_fn = updater_fn
        self.training_cls = training_cls
        self.interface = TrainerInterface(server_ip=server_ip,
                                          server_port=server_port,
                                          password=password,
                                          local_com_port=local_com_port,
                                          header_size=header_size,
                                          max_buf_len=max_buf_len,
                                          security=security,
                                          keys_dir=keys_dir,
                                          hostname=hostname,
                                          model_path=model_path,
                                          groups=groups,
                                          target_groups=target_groups)

    def stop(self):
        """
        Stops the tlspyo Endpoint.
        """
        self.interface.stop()

    def run(self):
        """
        Runs training.
        """
        run(interface=self.interface,
            run_cls=self.training_cls,
            checkpoint_path=self.checkpoint_path,
            dump_run_instance_fn=self.dump_run_instance_fn,
            load_run_instance_fn=self.load_run_instance_fn,
            updater_fn=self.updater_fn)

    def run_with_wandb(self,
                       entity=cfg.WANDB_ENTITY,
                       project=cfg.WANDB_PROJECT,
                       run_id=cfg.WANDB_RUN_ID,
                       key=None):
        """
        Runs training while logging metrics to wandb_.

        .. _wandb: https://wandb.ai

        Args:
            entity (str): wandb entity
            project (str): wandb project
            run_id (str): name of the run
            key (str): wandb API key
        """
        if key:
            os.environ['WANDB_API_KEY'] = key
        run_with_wandb(entity=entity,
                       project=project,
                       run_id=run_id,
                       interface=self.interface,
                       run_cls=self.training_cls,
                       checkpoint_path=self.checkpoint_path,
                       dump_run_instance_fn=self.dump_run_instance_fn,
                       load_run_instance_fn=self.load_run_instance_fn,
                       updater_fn=self.updater_fn)


# === Rollout Worker ==================================================================================================


class RolloutWorker:
    """Actor.

    A `RolloutWorker` deploys the current policy in the environment.
    A `RolloutWorker` may connect to a `Server` to which it sends buffered experience.
    Alternatively, it may exist in standalone mode for deployment.
    """
    def __init__(
            self,
            env_cls,
            actor_module_cls,
            sample_compressor: Callable = None,
            device="cpu",
            max_samples_per_episode=np.inf,
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor: Callable = None,
            crc_debug=False,
            model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
            model_history=cfg.MODEL_HISTORY,
            standalone=False,
            server_ip=None,
            server_port=cfg.PORT,
            password=cfg.PASSWORD,
            local_port=cfg.LOCAL_PORT_WORKER,
            header_size=cfg.HEADER_SIZE,
            max_buf_len=cfg.BUFFER_SIZE,
            security=cfg.SECURITY,
            keys_dir=cfg.CREDENTIALS_DIRECTORY,
            hostname=cfg.HOSTNAME,
            groups: str | tuple[str] = "workers",
            target_groups: str | tuple[str] = "trainers"
    ):
        """
        Args:
            env_cls (type): class of the Gymnasium environment (subclass of tmrl.core.envs.GenericGymEnv)
            actor_module_cls (type): class of the module containing the policy (subclass of tmrl.core.actor.ActorModule)
            sample_compressor (callable): compressor for sending samples over the Internet; \
                when not `None`, `sample_compressor` must be a function that takes the following arguments: \
                (prev_act, obs, rew, terminated, truncated, info), and that returns them (modified) in the same order: \
                when not `None`, a `sample_compressor` works with a corresponding decompression scheme in the `Memory` class
            device (str): device on which the policy is running
            max_samples_per_episode (int): if an episode gets longer than this, it is reset
            model_path (str): path where a local copy of the policy will be stored
            obs_preprocessor (callable): utility for modifying observations retrieved from the environment; \
                when not `None`, `obs_preprocessor` must be a function that takes an observation as input and outputs the \
                modified observation
            crc_debug (bool): useful for debugging custom pipelines; leave to False otherwise
            model_path_history (str): (include the filename but omit .tmod) path to the saved history of policies; \
                we recommend you leave this to the default
            model_history (int): policies are saved every `model_history` new policies (<= 0: not saved)
            standalone (bool): if True, the worker will not try to connect to a server
            server_ip (str): ip of the central server
            server_port (int): public port of the central server
            password (str): tlspyo password
            local_port (int): tlspyo local communication port; usually, leave this to the default
            header_size (int): tlspyo header size (bytes)
            max_buf_len (int): tlspyo max number of messages in buffer
            security (str): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory; usually, leave this to the default
            hostname (str): tlspyo hostname; usually, leave this to the default
            groups (Union[str, Tuple[str]]): `tlspyo` group(s) this `RolloutWorker` endpoint belongs to; \
                the `Trainer` must notify the relay targeting these groups to receive experience buffers
            target_groups (Union[str, Tuple[str]]): `tlspyo` group(s) to which experience buffers are produced; \
                typically the group that `TrainerInterface` instances belong to
        """
        self.target_groups = target_groups
        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = sample_compressor
        self.env = env_cls()
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.model_path_history = model_path_history
        self.device = device
        self.actor = actor_module_cls(observation_space=obs_space, action_space=act_space).to_device(self.device)
        self.standalone = standalone
        if os.path.isfile(self.model_path):
            logging.debug(f"Loading model from {self.model_path}")
            self.actor = self.actor.load(self.model_path, device=self.device)
        else:
            logging.debug(f"No model found at {self.model_path}")
        self.buffer = Buffer()
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug
        self.model_history = model_history
        self._cur_hist_cpt = 0

        self.debug_ts_cpt = 0
        self.debug_ts_res_cpt = 0

        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'

        print_with_timestamp(f"server IP: {self.server_ip}")

        if not self.standalone:
            self.__endpoint = Endpoint(ip_server=self.server_ip,
                                       port=server_port,
                                       password=password,
                                       groups=groups,
                                       local_com_port=local_port,
                                       header_size=header_size,
                                       max_buf_len=max_buf_len,
                                       security=security,
                                       keys_dir=keys_dir,
                                       hostname=hostname,
                                       deserializer_mode="synchronous")
        else:
            self.__endpoint = None

    def _require_endpoint(self, method_name: str):
        if self.__endpoint is None:
            raise RuntimeError(f"{method_name} cannot be called in standalone mode.")

    def stop(self):
        """
        Stops the tlspyo Endpoint.
        """
        if self.__endpoint is not None:
            self.__endpoint.stop()

    def act(self, obs, test=False):
        """
        Select an action based on observation `obs`

        Args:
            obs (nested structure): observation
            test (bool): directly passed to the `act()` method of the `ActorModule`

        Returns:
            numpy.array: action computed by the `ActorModule`
        """
        return self.actor.act_(obs, test=test)

    def _buffer_sample(self, prev_obs, act, new_obs, rew, terminated, truncated, info, is_reset=False):
        if self.crc_debug:
            self.debug_ts_cpt += 1
            self.debug_ts_res_cpt = 0 if is_reset else self.debug_ts_res_cpt + 1
            info['crc_sample'] = (prev_obs, act, new_obs, rew, terminated, truncated)
            info['crc_sample_ts'] = (self.debug_ts_cpt, self.debug_ts_res_cpt)
        if self.get_local_buffer_sample:
            sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
        else:
            sample = act, new_obs, rew, terminated, truncated, info
        self.buffer.append_sample(sample)

    def reset(self, collect_samples):
        """
        Starts a new episode.

        Args:
            collect_samples (bool): if True, samples are buffered and sent to the `Server`

        Returns:
            Tuple:
            (nested structure: observation retrieved from the environment,
            dict: information retrieved from the environment)
        """
        try:
            # Faster than hasattr() in real-time environments
            act = self.env.unwrapped.default_action
        except AttributeError:
            # In non-real-time environments, we sample a dummy action for the buffer at reset
            act = self.env.action_space.sample()
        new_obs, info = self.env.reset()
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        if collect_samples:
            self._buffer_sample(None, act, new_obs, 0.0, False, False, info, is_reset=True)
        return new_obs, info

    def step(self, obs, test, collect_samples, last_step=False):
        """
        Performs a full RL transition.

        A full RL transition is `obs` -> `act` -> `new_obs`, `rew`, `terminated`, `truncated`, `info`.
        Note that, in the Real-Time RL setting, `act` is appended to a buffer which is part of `new_obs`.
        This is because it does not directly affect the new observation, due to real-time delays.

        Args:
            obs (nested structure): previous observation
            test (bool): passed to the `act()` method of the `ActorModule`
            collect_samples (bool): if True, samples are buffered and sent to the `Server`
            last_step (bool): if True and `terminated` is False, `truncated` will be set to True

        Returns:
            Tuple:
            (nested structure: new observation,
            float: new reward,
            bool: episode termination signal,
            bool: episode truncation signal,
            dict: information dictionary)
        """
        act = self.act(obs, test=test)
        new_obs, rew, terminated, truncated, info = self.env.step(act)
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        if collect_samples:
            if last_step and not terminated:
                truncated = True
            # CAUTION: in the buffer, act is for the PREVIOUS transition (act, obs(act))
            self._buffer_sample(obs, act, new_obs, rew, terminated, truncated, info, is_reset=False)
        return new_obs, rew, terminated, truncated, info

    def collect_train_episode(self, max_samples=None):
        """
        Collects a maximum of `max_samples` training transitions (from reset to terminated or truncated)

        This method stores the episode and the training return in the local `Buffer` of the worker
        for sending to the `Server`.

        Args:
            max_samples (int): if the environment is not `terminated` after `max_samples` time steps,
                it is forcefully reset and `truncated` is set to True.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=True)
        for i in iterator:
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=False, collect_samples=True, last_step=i == max_samples - 1)
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps

    def run_episodes(self, max_samples_per_episode=None, nb_episodes=np.inf, train=False):
        """
        Runs `nb_episodes` episodes.

        Args:
            max_samples_per_episode (int): same as run_episode
            nb_episodes (int): total number of episodes to collect
            train (bool): same as run_episode
        """
        if max_samples_per_episode is None:
            max_samples_per_episode = self.max_samples_per_episode

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        for _ in iterator:
            self.run_episode(max_samples_per_episode, train=train)

    def run_episode(self, max_samples=None, train=False):
        """
        Collects a maximum of `max_samples` test transitions (from reset to terminated or truncated).

        Args:
            max_samples (int): At most `max_samples` samples are collected per episode.
                If the episode is longer, it is forcefully reset and `truncated` is set to True.
            train (bool): whether the episode is a training or a test episode.
                `step` is called with `test=not train`.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=False)
        for _ in iterator:
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=not train, collect_samples=False)
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps

    def run(self, test_episode_interval=0, nb_episodes=np.inf, verbose=True, expert=False):
        """
        Runs the worker for `nb_episodes` episodes.

        This method sends episodes continuously to the Server, and checks for new weights between episodes.
        For synchronous or more fine-grained sampling, use synchronous or lower-level APIs.
        For deployment, use `run_episodes` rather than `run`.

        Note:
            This method does not work in standalone mode.

        Args:
            test_episode_interval (int): a test episode is collected for every `test_episode_interval` train episodes;
                set to 0 to not collect test episodes.
            nb_episodes (int): maximum number of train episodes to collect.
            verbose (bool): whether to log INFO messages.
            expert (bool): experts send training samples without updating their model nor running test episodes.
        """
        if self.standalone:
            raise RuntimeError("In standalone mode, use the run_episodes() method.")

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        for episode in iterator:
            if not expert and test_episode_interval and episode % test_episode_interval == 0 and not self.crc_debug:
                if verbose:
                    print_with_timestamp("running test episode")
                self.run_episode(self.max_samples_per_episode, train=False)
            if verbose:
                print_with_timestamp("collecting expert episode" if expert else "collecting train episode")
            self.collect_train_episode(self.max_samples_per_episode)
            if verbose:
                print_with_timestamp("copying buffer for sending")
            self.send_and_clear_buffer()
            if expert:
                self.ignore_actor_weights()
            else:
                if verbose:
                    print_with_timestamp("checking for new weights")
                self.update_actor_weights(verbose=verbose)

    def _wait_for_model_updates(self, iteration, i_model, max_steps_per_update, verbose):
        ratio = (iteration + 1) / i_model
        while ratio > max_steps_per_update:
            if verbose:
                logging.info(f"Ratio {ratio} > {max_steps_per_update}, checking updates")
            if len(self.buffer) > 0:
                self.send_and_clear_buffer()
            i_model += self.update_actor_weights(verbose=verbose, blocking=True)
            ratio = (iteration + 1) / i_model
        return i_model

    def run_synchronous(self,
                        test_episode_interval=0,
                        nb_steps=np.inf,
                        initial_steps=1,
                        max_steps_per_update=np.inf,
                        end_episodes=True,
                        verbose=False):
        """
        Collects `nb_steps` steps while synchronizing with the Trainer.

        This method is useful for traditional (non-real-time) environments that can be stepped fast.
        It also works for rtgym environments with `wait_on_done` enabled, just set `end_episodes` to `True`.

        Note:
            This method does not collect test episodes when `end_episodes` is `False`.
            Periodically use `run_episode(train=False)` if you wish to.
            This method does not work in standalone mode.

        Args:
            test_episode_interval (int): a test episode is collected for every `test_episode_interval` train episodes;
                set to 0 to not collect test episodes. NB: `end_episodes` must be `True` to collect test episodes.
            nb_steps (int): total number of steps to collect (after `initial_steps`).
            initial_steps (int): initial number of steps to collect before waiting for the first model update.
            max_steps_per_update (float): maximum number of steps to collect per model received from the Server
                (this can be a non-integer ratio).
            end_episodes (bool): when True, waits for episodes to end before sending samples and waiting for updates.
                When False, pauses whenever the `max_steps_per_update` ratio is exceeded.
            verbose (bool): whether to log INFO messages.
        """
        if self.standalone:
            raise RuntimeError("In standalone mode, use the run_episodes() method.")
        if initial_steps < 1:
            raise RuntimeError("initial_steps must be >= 1.")
        if max_steps_per_update <= 0:
            raise RuntimeError("max_steps_per_update must be > 0.")

        # Collect initial samples before synchronizing:
        if verbose:
            logging.info(f"Collecting {initial_steps} initial steps")

        iteration = 0
        done = False
        steps = 0
        ret = 0.0
        while iteration < initial_steps:
            steps = 0
            ret = 0.0
            obs, info = self.reset(collect_samples=True)
            done = False
            iteration += 1
            while not done and (end_episodes or iteration < initial_steps):
                obs, rew, terminated, truncated, info = self.step(
                    obs=obs,
                    test=False,
                    collect_samples=True,
                    last_step=steps == self.max_samples_per_episode - 1,
                )
                iteration += 1
                steps += 1
                ret += rew
                done = terminated or truncated
            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if verbose:
                logging.info("Sending buffer (initial steps)")
            self.send_and_clear_buffer()

        i_model = self._wait_for_model_updates(iteration, 1, max_steps_per_update, verbose)

        # Collect further samples while synchronizing with the Trainer:
        iteration = 0
        episode = 0
        if done:
            steps = 0
            ret = 0.0

        while iteration < nb_steps:
            if done:
                if test_episode_interval > 0 and episode % test_episode_interval == 0 and end_episodes:
                    if verbose:
                        print_with_timestamp("running test episode")
                    self.run_episode(self.max_samples_per_episode, train=False)
                obs, info = self.reset(collect_samples=True)
                done = False
                iteration += 1
                steps = 0
                ret = 0.0
                episode += 1

            ratio = (iteration + 1) / i_model
            while not done and (end_episodes or ratio <= max_steps_per_update):
                obs, rew, terminated, truncated, info = self.step(
                    obs=obs,
                    test=False,
                    collect_samples=True,
                    last_step=steps == self.max_samples_per_episode - 1,
                )
                iteration += 1
                steps += 1
                ret += rew
                done = terminated or truncated

                if not end_episodes:
                    ratio = (iteration + 1) / i_model
                    if ratio > max_steps_per_update:
                        if done:
                            self.buffer.stat_train_return = ret
                            self.buffer.stat_train_steps = steps
                        i_model = self._wait_for_model_updates(iteration, i_model, max_steps_per_update, verbose)
                        ratio = (iteration + 1) / i_model

            if end_episodes:
                self.buffer.stat_train_return = ret
                self.buffer.stat_train_steps = steps
                i_model = self._wait_for_model_updates(iteration, i_model, max_steps_per_update, verbose)
            elif done:
                self.buffer.stat_train_return = ret
                self.buffer.stat_train_steps = steps

            if len(self.buffer) > 0:
                if verbose:
                    logging.info(
                        f"Sending buffer - DEBUG ratio {(iteration + 1) / i_model} iteration {iteration} i_model {i_model}"
                    )
                self.send_and_clear_buffer()

    def run_env_benchmark(self, nb_steps, test=False, verbose=True):
        """
        Benchmarks the environment.

        This method is only compatible with rtgym_ environments.
        Furthermore, the `"benchmark"` option of the rtgym configuration dictionary must be set to `True`.

        .. _rtgym: https://github.com/yannbouteiller/rtgym

        Args:
            nb_steps (int): number of steps to perform to compute the benchmark
            test (int): whether the actor is called in test or train mode
            verbose (bool): whether to log INFO messages
        """
        if nb_steps == np.inf or nb_steps < 0:
            raise RuntimeError(f"Invalid number of steps: {nb_steps}")

        obs, info = self.reset(collect_samples=False)
        for _ in range(nb_steps):
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=test, collect_samples=False)
            if terminated or truncated:
                break
        res = self.env.unwrapped.benchmarks()
        if verbose:
            print_with_timestamp(f"Benchmark results:\n{res}")
        return res

    def send_and_clear_buffer(self):
        """
        Sends the buffered samples to the `Server`.
        """
        self._require_endpoint("send_and_clear_buffer")
        if isinstance(self.target_groups, str):
            self.__endpoint.produce(self.buffer, self.target_groups)
        else:
            self.__endpoint.send_object(self.buffer, {g: 1 for g in self.target_groups})
        self.buffer.clear()

    def update_actor_weights(self, verbose=True, blocking=False):
        """
        Updates the actor with new weights received from the `Server` when available.

        Args:
            verbose (bool): whether to log INFO messages.
            blocking (bool): if True, blocks until a model is received; otherwise, can be a no-op.

        Returns:
            int: number of new actor models received from the Server (the latest is used).
        """
        self._require_endpoint("update_actor_weights")
        weights_list = self.__endpoint.receive_all(blocking=blocking)
        nb_received = len(weights_list)
        if nb_received > 0:
            weights = weights_list[-1]
            model_path = Path(self.model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, 'wb') as f:
                f.write(weights)
            if self.model_history and self.model_history > 0:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    hist_path = Path(f"{self.model_path_history}{x.strftime('%d_%m_%Y_%H_%M_%S')}.tmod")
                    hist_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(hist_path, 'wb') as f:
                        f.write(weights)
                    self._cur_hist_cpt = 0
                    if verbose:
                        print_with_timestamp("model weights saved in history")
            self.actor = self.actor.load(self.model_path, device=self.device)
            if verbose:
                print_with_timestamp("model weights have been updated")
        return nb_received

    def ignore_actor_weights(self):
        """
        Clears the buffer of weights received from the `Server`.

        This is useful for expert RolloutWorkers, because all RolloutWorkers receive weights.

        Returns:
            int: number of new (ignored) actor models received from the Server.
        """
        self._require_endpoint("ignore_actor_weights")
        weights_list = self.__endpoint.receive_all(blocking=False)
        nb_received = len(weights_list)
        return nb_received
