# TrackMania RL Core Framework (`tmrl/`)

> Networking, actor-critic architectures, Gym environments, memory buffers, and TrackMania interface wrappers.

**Total Modules Documented:** 78

---

## Table of Contents

- [`tmrl\__init__.py`](#tmrlinitpy)
- [`tmrl\__main__.py`](#tmrlmainpy)
- [`tmrl\config\__init__.py`](#tmrlconfiginitpy)
- [`tmrl\config\config_constants.py`](#tmrlconfigconfigconstantspy)
- [`tmrl\config\config_objects.py`](#tmrlconfigconfigobjectspy)
- [`tmrl\core\__init__.py`](#tmrlcoreinitpy)
- [`tmrl\core\actor.py`](#tmrlcoreactorpy)
- [`tmrl\core\envs.py`](#tmrlcoreenvspy)
- [`tmrl\core\jax\__init__.py`](#tmrlcorejaxinitpy)
- [`tmrl\core\jax\actor.py`](#tmrlcorejaxactorpy)
- [`tmrl\core\jax\training_offline.py`](#tmrlcorejaxtrainingofflinepy)
- [`tmrl\core\jax\util.py`](#tmrlcorejaxutilpy)
- [`tmrl\core\memory.py`](#tmrlcorememorypy)
- [`tmrl\core\networking.py`](#tmrlcorenetworkingpy)
- [`tmrl\core\torch\__init__.py`](#tmrlcoretorchinitpy)
- [`tmrl\core\torch\actor.py`](#tmrlcoretorchactorpy)
- [`tmrl\core\torch\memory.py`](#tmrlcoretorchmemorypy)
- [`tmrl\core\torch\training_offline.py`](#tmrlcoretorchtrainingofflinepy)
- [`tmrl\core\torch\util.py`](#tmrlcoretorchutilpy)
- [`tmrl\core\training.py`](#tmrlcoretrainingpy)
- [`tmrl\core\training_offline.py`](#tmrlcoretrainingofflinepy)
- [`tmrl\core\util.py`](#tmrlcoreutilpy)
- [`tmrl\core\wrappers.py`](#tmrlcorewrapperspy)
- [`tmrl\custom\__init__.py`](#tmrlcustominitpy)
- [`tmrl\custom\jax\__init__.py`](#tmrlcustomjaxinitpy)
- [`tmrl\custom\jax\continual_checkpoints.py`](#tmrlcustomjaxcontinualcheckpointspy)
- [`tmrl\custom\jax\continual_dreamer.py`](#tmrlcustomjaxcontinualdreamerpy)
- [`tmrl\custom\jax\continual_replay.py`](#tmrlcustomjaxcontinualreplaypy)
- [`tmrl\custom\jax\custom_algorithms.py`](#tmrlcustomjaxcustomalgorithmspy)
- [`tmrl\custom\jax\custom_memories.py`](#tmrlcustomjaxcustommemoriespy)
- [`tmrl\custom\jax\custom_models.py`](#tmrlcustomjaxcustommodelspy)
- [`tmrl\custom\jax\imagination_trainer.py`](#tmrlcustomjaximaginationtrainerpy)
- [`tmrl\custom\jax\latent_adversary.py`](#tmrlcustomjaxlatentadversarypy)
- [`tmrl\custom\jax\world_model.py`](#tmrlcustomjaxworldmodelpy)
- [`tmrl\custom\tm\__init__.py`](#tmrlcustomtminitpy)
- [`tmrl\custom\tm\tm_gym_interfaces.py`](#tmrlcustomtmtmgyminterfacespy)
- [`tmrl\custom\tm\tm_preprocessors.py`](#tmrlcustomtmtmpreprocessorspy)
- [`tmrl\custom\tm\utils\__init__.py`](#tmrlcustomtmutilsinitpy)
- [`tmrl\custom\tm\utils\camera_preprocessing.py`](#tmrlcustomtmutilscamerapreprocessingpy)
- [`tmrl\custom\tm\utils\compute_reward.py`](#tmrlcustomtmutilscomputerewardpy)
- [`tmrl\custom\tm\utils\control_gamepad.py`](#tmrlcustomtmutilscontrolgamepadpy)
- [`tmrl\custom\tm\utils\control_keyboard.py`](#tmrlcustomtmutilscontrolkeyboardpy)
- [`tmrl\custom\tm\utils\control_mouse.py`](#tmrlcustomtmutilscontrolmousepy)
- [`tmrl\custom\tm\utils\self_play_obs.py`](#tmrlcustomtmutilsselfplayobspy)
- [`tmrl\custom\tm\utils\self_play_reward.py`](#tmrlcustomtmutilsselfplayrewardpy)
- [`tmrl\custom\tm\utils\tm_vision.py`](#tmrlcustomtmutilstmvisionpy)
- [`tmrl\custom\tm\utils\tools.py`](#tmrlcustomtmutilstoolspy)
- [`tmrl\custom\tm\utils\trajectory_assist.py`](#tmrlcustomtmutilstrajectoryassistpy)
- [`tmrl\custom\tm\utils\window.py`](#tmrlcustomtmutilswindowpy)
- [`tmrl\custom\torch\__init__.py`](#tmrlcustomtorchinitpy)
- [`tmrl\custom\torch\azr.py`](#tmrlcustomtorchazrpy)
- [`tmrl\custom\torch\car_brain.py`](#tmrlcustomtorchcarbrainpy)
- [`tmrl\custom\torch\car_brain_actor.py`](#tmrlcustomtorchcarbrainactorpy)
- [`tmrl\custom\torch\custom_algorithms.py`](#tmrlcustomtorchcustomalgorithmspy)
- [`tmrl\custom\torch\custom_checkpoints.py`](#tmrlcustomtorchcustomcheckpointspy)
- [`tmrl\custom\torch\custom_memories.py`](#tmrlcustomtorchcustommemoriespy)
- [`tmrl\custom\torch\custom_models.py`](#tmrlcustomtorchcustommodelspy)
- [`tmrl\custom\torch\dreamer.py`](#tmrlcustomtorchdreamerpy)
- [`tmrl\custom\torch\foundation_actor.py`](#tmrlcustomtorchfoundationactorpy)
- [`tmrl\custom\torch\foundation_encoder.py`](#tmrlcustomtorchfoundationencoderpy)
- [`tmrl\custom\torch\jax_dreamer_actor.py`](#tmrlcustomtorchjaxdreameractorpy)
- [`tmrl\custom\torch\poincare_memory.py`](#tmrlcustomtorchpoincarememorypy)
- [`tmrl\custom\torch\utils\__init__.py`](#tmrlcustomtorchutilsinitpy)
- [`tmrl\custom\torch\utils\nn.py`](#tmrlcustomtorchutilsnnpy)
- [`tmrl\custom\torch\world_model.py`](#tmrlcustomtorchworldmodelpy)
- [`tmrl\tools\__init__.py`](#tmrltoolsinitpy)
- [`tmrl\tools\benchmark_continual_actor.py`](#tmrltoolsbenchmarkcontinualactorpy)
- [`tmrl\tools\check_environment.py`](#tmrltoolscheckenvironmentpy)
- [`tmrl\tools\continual_preflight.py`](#tmrltoolscontinualpreflightpy)
- [`tmrl\tools\freeze_continual_baseline.py`](#tmrltoolsfreezecontinualbaselinepy)
- [`tmrl\tools\init_package\__init__.py`](#tmrltoolsinitpackageinitpy)
- [`tmrl\tools\init_package\init_pywin32.py`](#tmrltoolsinitpackageinitpywin32py)
- [`tmrl\tools\init_package\init_tmrl.py`](#tmrltoolsinitpackageinittmrlpy)
- [`tmrl\tools\prepare_continual_jax_config.py`](#tmrltoolspreparecontinualjaxconfigpy)
- [`tmrl\tools\record.py`](#tmrltoolsrecordpy)
- [`tmrl\tools\save_replays.py`](#tmrltoolssavereplayspy)
- [`tmrl\tools\validate_continual_map_manifest.py`](#tmrltoolsvalidatecontinualmapmanifestpy)
- [`tmrl\tools\validate_continual_trainer_stack.py`](#tmrltoolsvalidatecontinualtrainerstackpy)

---
## Module Outlines

### [`tmrl\__init__.py`](file:///D:/Project/tmrl/tmrl/__init__.py)

**Role:** Source module in `tmrl`.

#### Functions

* **[`get_environment()`](file:///D:/Project/tmrl/tmrl/__init__.py#L48)**: Default TMRL Gymnasium environment for TrackMania 2020.


---

### [`tmrl\__main__.py`](file:///D:/Project/tmrl/tmrl/__main__.py)

**Role:** Source module in `tmrl`.

#### Functions

* **[`main(args)`](file:///D:/Project/tmrl/tmrl/__main__.py#L35)**


---

### [`tmrl\config\__init__.py`](file:///D:/Project/tmrl/tmrl/config/__init__.py)

**Role:** Source module in `tmrl\config`.


---

### [`tmrl\config\config_constants.py`](file:///D:/Project/tmrl/tmrl/config/config_constants.py)

**Role:** Source module in `tmrl\config`.


---

### [`tmrl\config\config_objects.py`](file:///D:/Project/tmrl/tmrl/config/config_objects.py)

**Purpose:** This file sets up the example TMRL pipeline according to the content of config.json.

#### Functions

* **[`sac_v2_entropy_scheduler(agent, epoch)`](file:///D:/Project/tmrl/tmrl/config/config_objects.py#L437)**


---

### [`tmrl\core\__init__.py`](file:///D:/Project/tmrl/tmrl/core/__init__.py)

**Role:** Source module in `tmrl\core`.


---

### [`tmrl\core\actor.py`](file:///D:/Project/tmrl/tmrl/core/actor.py)

**Role:** Source module in `tmrl\core`.

#### Classes

* **[`ActorModule(ABC)`](file:///D:/Project/tmrl/tmrl/core/actor.py#L8)**: Implement this interface for the RolloutWorker(s) to interact with your policy.
  * [`__init__(self, observation_space, action_space)`](file:///D:/Project/tmrl/tmrl/core/actor.py#L17) — *Args:*
  * [`save(self, path)`](file:///D:/Project/tmrl/tmrl/core/actor.py#L27) — *Save your `ActorModule` on the hard drive.*
  * [`load(self, path, device)`](file:///D:/Project/tmrl/tmrl/core/actor.py#L45) — *Load and return an instance of your `ActorModule` from the hard drive.*
  * [`to_device(self, device)`](file:///D:/Project/tmrl/tmrl/core/actor.py#L70) — *Set the `ActorModule`'s relevant attributes to the designated device.*
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/core/actor.py#L85) — *Must compute an action from an observation.*
  * [`act_(self, obs, test)`](file:///D:/Project/tmrl/tmrl/core/actor.py#L99)


---

### [`tmrl\core\envs.py`](file:///D:/Project/tmrl/tmrl/core/envs.py)

**Role:** Source module in `tmrl\core`.

#### Classes

* **[`GenericGymEnv(gymnasium.Wrapper)`](file:///D:/Project/tmrl/tmrl/core/envs.py#L13)**
  * [`__init__(self, id, gym_kwargs, obs_scale, wrappers)`](file:///D:/Project/tmrl/tmrl/core/envs.py#L14) — *Use this wrapper when using the framework with arbitrary environments.*


---

### [`tmrl\core\jax\__init__.py`](file:///D:/Project/tmrl/tmrl/core/jax/__init__.py)

**Role:** Source module in `tmrl\core\jax`.


---

### [`tmrl\core\jax\actor.py`](file:///D:/Project/tmrl/tmrl/core/jax/actor.py)

**Role:** Source module in `tmrl\core\jax`.

#### Classes

* **[`NNXActorModule(ActorModule, nnx.Module, ABC)`](file:///D:/Project/tmrl/tmrl/core/jax/actor.py#L15)**: Partial implementation of `ActorModule` as a `nnx.Module`.
  * [`__init__(self, observation_space, action_space, device)`](file:///D:/Project/tmrl/tmrl/core/jax/actor.py#L29) — *Args:*
  * [`save(self, path)`](file:///D:/Project/tmrl/tmrl/core/jax/actor.py#L41)
  * [`load(self, path, device)`](file:///D:/Project/tmrl/tmrl/core/jax/actor.py#L46)
  * [`act_(self, obs, test)`](file:///D:/Project/tmrl/tmrl/core/jax/actor.py#L55) — *Transforms obs into a tree of jax arrays*
  * [`to_device(self, device)`](file:///D:/Project/tmrl/tmrl/core/jax/actor.py#L66)


---

### [`tmrl\core\jax\training_offline.py`](file:///D:/Project/tmrl/tmrl/core/jax/training_offline.py)

**Role:** Source module in `tmrl\core\jax`.

#### Classes

* **[`NNXTrainingOffline(TrainingOffline)`](file:///D:/Project/tmrl/tmrl/core/jax/training_offline.py#L54)**: TrainingOffline for trainers based on Flax NNX, with jit-able train() method.
  * [`__init__(self, env_cls, memory_cls, training_agent_cls, epochs, rounds, steps, update_model_interval, update_buffer_interval, max_training_steps_per_env_step, sleep_between_buffer_retrieval_attempts, profiling, agent_scheduler, start_training, device, jit_sampling, jit_substeps, jit_training)`](file:///D:/Project/tmrl/tmrl/core/jax/training_offline.py#L60) — *Args:*
  * [`_train_batch(self, batch)`](file:///D:/Project/tmrl/tmrl/core/jax/training_offline.py#L125)
  * [`run_epoch(self, interface)`](file:///D:/Project/tmrl/tmrl/core/jax/training_offline.py#L130)

#### Functions

* **[`_train_jit(batch, agent)`](file:///D:/Project/tmrl/tmrl/core/jax/training_offline.py#L16)**
* **[`_sample_and_train_jit(memory, agent, n)`](file:///D:/Project/tmrl/tmrl/core/jax/training_offline.py#L21)**: Fully jitted loop sampling from the memory and training the agent n times


---

### [`tmrl\core\jax\util.py`](file:///D:/Project/tmrl/tmrl/core/jax/util.py)

**Role:** Source module in `tmrl\core\jax`.

#### Functions

* **[`collate_jax(batch, device)`](file:///D:/Project/tmrl/tmrl/core/jax/util.py#L13)**: Turns a batch of nested structures with numpy arrays as leaves into a single element of the same nested structure with batched jax arrays as leaves
* **[`get_rngs(params_seed, dropout_seed, noise_seed)`](file:///D:/Project/tmrl/tmrl/core/jax/util.py#L24)**: Return an RNG with default seeds for the following streams:


---

### [`tmrl\core\memory.py`](file:///D:/Project/tmrl/tmrl/core/memory.py)

**Role:** Source module in `tmrl\core`.

#### Classes

* **[`BaseMemory(ABC)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L46)**: Directly implement this instead of Memory if you want to optimize sampling (e.g., no collating...).
  * [`__init__(self, device, memory_size, batch_size, sample_preprocessor, dataset_path, crc_debug)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L52) — *Args:*
  * [`append_buffer(self, buffer)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L85) — *Must append a Buffer object to the memory.*
  * [`sample(self)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L95) — *Must sample a minibatch collated to self.device.*
  * [`get_benchmarks(self)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L105) — *Can be overridden to log a tuple of floats for benchmarking the performance of your Memory class.*
  * [`get_benchmarks_names(self)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L113) — *Use this method to name the elements of the tuple returned by get_benchmarks.*
  * [`append(self, buffer)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L121)
* **[`Memory(BaseMemory, ABC)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L130)**: Interface implementing the replay buffer.
  * [`__init__(self, device, sample_preprocessor, memory_size, batch_size, dataset_path, crc_debug)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L138) — *Args:*
  * [`get_benchmarks(self)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L184)
  * [`get_benchmarks_names(self)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L195)
  * [`__len__(self)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L199) — *Must return the length of the memory.*
  * [`get_transition(self, item)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L210) — *Must return a transition.*
  * [`collate(self, batch, device)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L225) — *Must collate `batch` onto `device`.*
  * [`__getitem__(self, item)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L247)
  * [`sample(self)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L259)
  * [`sample_indices(self)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L271)

#### Functions

* **[`check_samples_crc(original_po, original_a, original_o, original_r, original_d, original_t, rebuilt_po, rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d, rebuilt_t, debug_ts, debug_ts_res, epsilon)`](file:///D:/Project/tmrl/tmrl/core/memory.py#L18)**


---

### [`tmrl\core\networking.py`](file:///D:/Project/tmrl/tmrl/core/networking.py)

**Role:** Source module in `tmrl\core`.

#### Classes

* **[`Buffer`](file:///D:/Project/tmrl/tmrl/core/networking.py#L47)**: Buffer of training samples.
  * [`__init__(self, maxlen)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L55) — *Args:*
  * [`clip_to_maxlen(self)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L67)
  * [`append_sample(self, sample)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L73) — *Appends `sample` to the buffer.*
  * [`clear(self)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L83) — *Clears the buffer but keeps train and test returns.*
  * [`__len__(self)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L89)
  * [`__iadd__(self, other)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L92)
* **[`Server`](file:///D:/Project/tmrl/tmrl/core/networking.py#L105)**: Central server.
  * [`__init__(self, port, password, local_port, header_size, security, keys_dir, accepted_groups)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L113) — *Args:*
  * [`stop(self)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L139) — *Stops the tlspyo Relay.*
* **[`TrainerInterface`](file:///D:/Project/tmrl/tmrl/core/networking.py#L149)**: This is the trainer's network interface
  * [`__init__(self, server_ip, server_port, password, local_com_port, header_size, max_buf_len, security, keys_dir, hostname, model_path, groups, target_groups)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L155)
  * [`_notify(self)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L188)
  * [`stop(self)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L192) — *Stops the tlspyo Endpoint.*
  * [`broadcast_model(self, model)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L198) — *model must be an ActorModule*
  * [`retrieve_buffer(self)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L208) — *returns the TrainerInterface's buffer of training samples*
* **[`Trainer`](file:///D:/Project/tmrl/tmrl/core/networking.py#L345)**: Training entity.
  * [`__init__(self, training_cls, server_ip, server_port, password, local_com_port, header_size, max_buf_len, security, keys_dir, hostname, model_path, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, updater_fn, groups, target_groups)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L352) — *Args:*
  * [`stop(self)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L412) — *Stops the tlspyo Endpoint.*
  * [`run(self)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L418) — *Runs training.*
  * [`run_with_wandb(self, entity, project, run_id, key)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L429) — *Runs training while logging metrics to wandb_.*
* **[`RolloutWorker`](file:///D:/Project/tmrl/tmrl/core/networking.py#L461)**: Actor.
  * [`__init__(self, env_cls, actor_module_cls, sample_compressor, device, max_samples_per_episode, model_path, obs_preprocessor, crc_debug, model_path_history, model_history, standalone, server_ip, server_port, password, local_port, header_size, max_buf_len, security, keys_dir, hostname, groups, target_groups)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L468) — *Args:*
  * [`stop(self)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L571) — *Stops the tlspyo Endpoint.*
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L578) — *Select an action based on observation `obs`*
  * [`reset(self, collect_samples)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L594) — *Starts a new episode.*
  * [`step(self, obs, test, collect_samples, last_step)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L631) — *Performs a full RL transition.*
  * [`collect_train_episode(self, max_samples)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L672) — *Collects a maximum of `max_samples` training transitions (from reset to terminated or truncated)*
  * [`run_episodes(self, max_samples_per_episode, nb_episodes, train)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L700) — *Runs `nb_episodes` episodes.*
  * [`run_episode(self, max_samples, train)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L717) — *Collects a maximum of `max_samples` test transitions (from reset to terminated or truncated).*
  * [`run(self, test_episode_interval, nb_episodes, verbose, expert)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L744) — *Runs the worker for `nb_episodes` episodes.*
  * [`run_synchronous(self, test_episode_interval, nb_steps, initial_steps, max_steps_per_update, end_episodes, verbose)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L806) — *Collects `nb_steps` steps while synchronizing with the Trainer.*
  * [`run_env_benchmark(self, nb_steps, test, verbose)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L953) — *Benchmarks the environment.*
  * [`send_and_clear_buffer(self)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L980) — *Sends the buffered samples to the `Server`.*
  * [`update_actor_weights(self, verbose, blocking)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L987) — *Updates the actor with new weights received from the `Server` when available.*
  * [`ignore_actor_weights(self)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L1018) — *Clears the buffer of weights received from the `Server`.*

#### Functions

* **[`print_with_timestamp(s)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L32)**
* **[`print_ip()`](file:///D:/Project/tmrl/tmrl/core/networking.py#L38)**
* **[`log_environment_variables()`](file:///D:/Project/tmrl/tmrl/core/networking.py#L220)**: add certain relevant environment variables to our config
* **[`load_run_instance(checkpoint_path)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L228)**: Default function used to load trainers from checkpoint path
* **[`dump_run_instance(run_instance, checkpoint_path)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L239)**: Default function used to dump trainers to checkpoint path
* **[`iterate_epochs(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, epochs_between_checkpoints, updater_fn)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L249)**: Main training loop (remote)
* **[`run_with_wandb(entity, project, run_id, interface, run_cls, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, updater_fn)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L300)**: Main training loop (remote).
* **[`run(interface, run_cls, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, updater_fn)`](file:///D:/Project/tmrl/tmrl/core/networking.py#L335)**: Main training loop (remote).


---

### [`tmrl\core\torch\__init__.py`](file:///D:/Project/tmrl/tmrl/core/torch/__init__.py)

**Role:** Source module in `tmrl\core\torch`.


---

### [`tmrl\core\torch\actor.py`](file:///D:/Project/tmrl/tmrl/core/torch/actor.py)

**Role:** Source module in `tmrl\core\torch`.

#### Classes

* **[`TorchActorModule(ActorModule, .Module, ABC)`](file:///D:/Project/tmrl/tmrl/core/torch/actor.py#L9)**: Partial implementation of `ActorModule` as a `torch.nn.Module`.
  * [`__init__(self, observation_space, action_space, device)`](file:///D:/Project/tmrl/tmrl/core/torch/actor.py#L25) — *Args:*
  * [`save(self, path)`](file:///D:/Project/tmrl/tmrl/core/torch/actor.py#L36)
  * [`load(self, path, device)`](file:///D:/Project/tmrl/tmrl/core/torch/actor.py#L39)
  * [`act_(self, obs, test)`](file:///D:/Project/tmrl/tmrl/core/torch/actor.py#L44) — *Transforms obs into a tree of torch tensors*
  * [`to(self, device)`](file:///D:/Project/tmrl/tmrl/core/torch/actor.py#L54)
  * [`to_device(self, device)`](file:///D:/Project/tmrl/tmrl/core/torch/actor.py#L58)


---

### [`tmrl\core\torch\memory.py`](file:///D:/Project/tmrl/tmrl/core/torch/memory.py)

**Role:** Source module in `tmrl\core\torch`.

#### Classes

* **[`TorchMemory(Memory, ABC)`](file:///D:/Project/tmrl/tmrl/core/torch/memory.py#L7)**: Partial implementation of the `Memory` class collating samples into batched torch tensors.
  * [`__init__(self, device, sample_preprocessor, memory_size, batch_size, dataset_path, crc_debug)`](file:///D:/Project/tmrl/tmrl/core/torch/memory.py#L15) — *Args:*
  * [`collate(self, batch, device)`](file:///D:/Project/tmrl/tmrl/core/torch/memory.py#L38)


---

### [`tmrl\core\torch\training_offline.py`](file:///D:/Project/tmrl/tmrl/core/torch/training_offline.py)

**Role:** Source module in `tmrl\core\torch`.

#### Classes

* **[`TorchTrainingOffline(TrainingOffline)`](file:///D:/Project/tmrl/tmrl/core/torch/training_offline.py#L6)**: TrainingOffline for trainers based on PyTorch.
  * [`__init__(self, env_cls, memory_cls, training_agent_cls, epochs, rounds, steps, update_model_interval, update_buffer_interval, max_training_steps_per_env_step, sleep_between_buffer_retrieval_attempts, profiling, agent_scheduler, start_training, device)`](file:///D:/Project/tmrl/tmrl/core/torch/training_offline.py#L12) — *Same arguments as `TrainingOffline`, but when `device` is `None` it is selected automatically for torch.*


---

### [`tmrl\core\torch\util.py`](file:///D:/Project/tmrl/tmrl/core/torch/util.py)

**Role:** Source module in `tmrl\core\torch`.

#### Functions

* **[`collate_torch(batch, device)`](file:///D:/Project/tmrl/tmrl/core/torch/util.py#L7)**: Turns a batch of nested structures with numpy arrays as leaves into a single element of the same nested structure with batched torch tensors as leaves
* **[`concat_collated(batches)`](file:///D:/Project/tmrl/tmrl/core/torch/util.py#L29)**: Concatenates a list of already-collated batches (with torch tensors as leaves) along the batch dimension.


---

### [`tmrl\core\training.py`](file:///D:/Project/tmrl/tmrl/core/training.py)

**Role:** Source module in `tmrl\core`.

#### Classes

* **[`TrainingAgent(ABC)`](file:///D:/Project/tmrl/tmrl/core/training.py#L7)**: Training algorithm.
  * [`__init__(self, observation_space, action_space, device)`](file:///D:/Project/tmrl/tmrl/core/training.py#L13) — *Args:*
  * [`train(self, batch)`](file:///D:/Project/tmrl/tmrl/core/training.py#L28) — *Executes a training step.*
  * [`get_actor(self)`](file:///D:/Project/tmrl/tmrl/core/training.py#L41) — *Returns the current ActorModule to be broadcast to the RolloutWorkers.*


---

### [`tmrl\core\training_offline.py`](file:///D:/Project/tmrl/tmrl/core/training_offline.py)

**Role:** Source module in `tmrl\core`.

#### Classes

* **[`TrainingOffline`](file:///D:/Project/tmrl/tmrl/core/training_offline.py#L18)**: Training wrapper for off-policy algorithms.
  * [`__post_init__(self)`](file:///D:/Project/tmrl/tmrl/core/training_offline.py#L55)
  * [`update_buffer(self, interface)`](file:///D:/Project/tmrl/tmrl/core/training_offline.py#L70)
  * [`check_ratio(self, interface)`](file:///D:/Project/tmrl/tmrl/core/training_offline.py#L75)
  * [`run_epoch(self, interface)`](file:///D:/Project/tmrl/tmrl/core/training_offline.py#L87)


---

### [`tmrl\core\util.py`](file:///D:/Project/tmrl/tmrl/core/util.py)

**Role:** Source module in `tmrl\core`.

#### Classes

* **[`cached_property`](file:///D:/Project/tmrl/tmrl/core/util.py#L40)**: Similar to `property` but after calling the getter/init function the result is cached.
  * [`__init__(self, init)`](file:///D:/Project/tmrl/tmrl/core/util.py#L44)
  * [`__get__(self, instance, owner)`](file:///D:/Project/tmrl/tmrl/core/util.py#L48)
  * [`__set__(self, instance, value)`](file:///D:/Project/tmrl/tmrl/core/util.py#L54)
* **[`DelayInterrupt`](file:///D:/Project/tmrl/tmrl/core/util.py#L212)**: Catches SIGINT and SIGTERM and re-raises them after the context manager exits.
  * [`__enter__(self)`](file:///D:/Project/tmrl/tmrl/core/util.py#L220)
  * [`on_signal(self)`](file:///D:/Project/tmrl/tmrl/core/util.py#L224)
  * [`__exit__(self)`](file:///D:/Project/tmrl/tmrl/core/util.py#L228)

#### Functions

* **[`pandas_dict()`](file:///D:/Project/tmrl/tmrl/core/util.py#L26)**
* **[`shallow_copy(obj)`](file:///D:/Project/tmrl/tmrl/core/util.py#L30)**
* **[`default()`](file:///D:/Project/tmrl/tmrl/core/util.py#L61)**
* **[`partial(func)`](file:///D:/Project/tmrl/tmrl/core/util.py#L65)**: Like `functools.partial`, except if used as a keyword argument for another `partial` and no function is supplied.
* **[`partial_to_dict(p, version)`](file:///D:/Project/tmrl/tmrl/core/util.py#L79)**: Only for wandb.
* **[`get_class_or_function(func)`](file:///D:/Project/tmrl/tmrl/core/util.py#L105)**
* **[`partial_from_args(func, kwargs)`](file:///D:/Project/tmrl/tmrl/core/util.py#L110)**
* **[`get_output()`](file:///D:/Project/tmrl/tmrl/core/util.py#L133)**
* **[`git_info(path)`](file:///D:/Project/tmrl/tmrl/core/util.py#L141)**: returns a dict with information about the git repo at path (path can be a sub-directory of the git repo)
* **[`dump(obj, path)`](file:///D:/Project/tmrl/tmrl/core/util.py#L167)**
* **[`load(path)`](file:///D:/Project/tmrl/tmrl/core/util.py#L177)**
* **[`cloudpickle_dump(obj, path)`](file:///D:/Project/tmrl/tmrl/core/util.py#L182)**
* **[`cloudpickle_load(path)`](file:///D:/Project/tmrl/tmrl/core/util.py#L192)**
* **[`save_json(d, path)`](file:///D:/Project/tmrl/tmrl/core/util.py#L197)**
* **[`load_json(path)`](file:///D:/Project/tmrl/tmrl/core/util.py#L204)**
* **[`prod(iterable)`](file:///D:/Project/tmrl/tmrl/core/util.py#L237)**


---

### [`tmrl\core\wrappers.py`](file:///D:/Project/tmrl/tmrl/core/wrappers.py)

**Role:** Source module in `tmrl\core`.

#### Classes

* **[`AffineObservationWrapper(gymnasium.ObservationWrapper)`](file:///D:/Project/tmrl/tmrl/core/wrappers.py#L9)**
  * [`__init__(self, env, shift, scale)`](file:///D:/Project/tmrl/tmrl/core/wrappers.py#L10)
  * [`observation(self, obs)`](file:///D:/Project/tmrl/tmrl/core/wrappers.py#L17)
* **[`Float64ToFloat32(gymnasium.ObservationWrapper)`](file:///D:/Project/tmrl/tmrl/core/wrappers.py#L21)**: Converts np.float64 arrays in the observations to np.float32 arrays.
  * [`observation(self, observation)`](file:///D:/Project/tmrl/tmrl/core/wrappers.py#L25)
  * [`step(self, action)`](file:///D:/Project/tmrl/tmrl/core/wrappers.py#L32)

#### Functions

* **[`deepmap(f, m)`](file:///D:/Project/tmrl/tmrl/core/wrappers.py#L40)**: Apply functions to the leaves of a dictionary or list, depending type of the leaf value.
* **[`float64_to_float32(x)`](file:///D:/Project/tmrl/tmrl/core/wrappers.py#L53)**
* **[`float_to_float32(x)`](file:///D:/Project/tmrl/tmrl/core/wrappers.py#L57)**


---

### [`tmrl\custom\__init__.py`](file:///D:/Project/tmrl/tmrl/custom/__init__.py)

**Role:** Source module in `tmrl\custom`.


---

### [`tmrl\custom\jax\__init__.py`](file:///D:/Project/tmrl/tmrl/custom/jax/__init__.py)

**Role:** Source module in `tmrl\custom\jax`.


---

### [`tmrl\custom\jax\continual_checkpoints.py`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_checkpoints.py)

**Purpose:** Checkpoint guards for the separately versioned JAX Dreamer pipeline.

#### Functions

* **[`dump_continual_jax_run_instance(run_instance, checkpoint_path)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_checkpoints.py#L7)**: Atomically serialize NNX/Optax state, including optimizer closures.
* **[`load_continual_jax_run_instance(checkpoint_path)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_checkpoints.py#L17)**
* **[`validate_continual_jax_run_instance(run_instance, training_cls)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_checkpoints.py#L21)**: Reject accidental loading of a Torch/SAC checkpoint into JAX Dreamer.


---

### [`tmrl\custom\jax\continual_dreamer.py`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py)

**Purpose:** Deployed JAX/Flax NNX Dreamer foundation for the continual pipeline.

#### Classes

* **[`JAXDreamerPolicy(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L169)**
  * [`__init__(self, feature_dim, action_scale, hidden_dim, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L170)
  * [`__call__(self, feature, rngs, test, with_logprob)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L185)
* **[`JAXDreamerValue(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L210)**
  * [`__init__(self, feature_dim, hidden_dim, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L211)
  * [`__call__(self, feature)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L218)
* **[`JAXDreamerActor(NNXActorModule)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L251)**: Recurrent NNX actor whose latent policy is optimized in imagination.
  * [`__init__(self, observation_space, action_space, latent_dim, hidden_dim, policy_hidden_dim, img_channels, img_height, img_width, encoder_channels, rngs_seed, internal_rngs_seed, device)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L254)
  * [`feature_dim(self)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L314)
  * [`initial(self, batch_size)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L317)
  * [`reset(self)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L324)
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L330)
  * [`warmup(self)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L340)
  * [`architecture_manifest(self)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L350)
  * [`save(self, path)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L368)
  * [`load(self, path, device)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L407)
  * [`to_device(self, device)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L441)
* **[`JAXDreamerAgent(nnx.Module, TrainingAgent)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L446)**: M1 JAX Dreamer agent with real-replay and latent-imagination updates.
  * [`__init__(self, observation_space, action_space, device, latent_dim, hidden_dim, policy_hidden_dim, lr_world_model, lr_actor, lr_critic, gamma, lambda_, free_nats, dyn_kl_scale, rep_kl_scale, telemetry_scale, embedding_scale, imagination_horizon, imagination_batch_size, burn_in, world_model_warmup_steps, entropy_scale, target_polyak, grad_clip, encoder_channels, seed)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L449)
  * [`get_actor(self)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L558)
  * [`_flatten_observation_sequence(obs)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L564)
  * [`_observation_at(obs, time_index)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L571)
  * [`_train_world_model(self, batch)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L575)
  * [`_train_imagination(self, h_sequence, z_sequence)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L720)
  * [`train(self, batch)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L824)

#### Functions

* **[`_scalar_sequence(value)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L38)**
* **[`gaussian_kl(p_mean, p_std, q_mean, q_std)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L44)**
* **[`lambda_returns(rewards, discounts, next_values, lambda_)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L55)**
* **[`portable_parameter_checksum(parameters)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L68)**
* **[`validate_portable_parameters(parameters, label)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L82)**: Reject checksummed-but-non-finite weights before they reach a worker.
* **[`nonfinite_nnx_state_paths(module)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L98)**: Return floating-point NNX state paths containing NaN or infinity.
* **[`assert_finite_nnx_state(module, label)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L116)**
* **[`_tree_all_finite(tree)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L126)**
* **[`_assert_finite_training_metrics(metrics, stage, finite_flags)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L137)**
* **[`_safe_optimizer(inner)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L162)**
* **[`_compiled_actor_step(actor, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_dreamer.py#L225)**


---

### [`tmrl\custom\jax\continual_replay.py`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_replay.py)

**Purpose:** JAX-facing adapters for the existing episode-safe TrackMania replay.

#### Classes

* **[`ArrayJaxMemoryTMFullSequence(ArrayTorchMemoryTMFullSequence)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_replay.py#L20)**: Reuse compatible replay storage and return JAX sequence batches.
  * [`__init__(self, device)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_replay.py#L28)
  * [`_target_device(self)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_replay.py#L34)
  * [`sample(self)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_replay.py#L44)
  * [`get_benchmarks(self)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_replay.py#L56)
  * [`get_benchmarks_names(self)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_replay.py#L66)

#### Functions

* **[`_torch_to_numpy(value)`](file:///D:/Project/tmrl/tmrl/custom/jax/continual_replay.py#L14)**


---

### [`tmrl\custom\jax\custom_algorithms.py`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_algorithms.py)

**Role:** Source module in `tmrl\custom\jax`.

#### Classes

* **[`_AlphaModule(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_algorithms.py#L16)**: Wraps the temperature alpha in an nnx.Module to make it learnable
  * [`__init__(self, initial_alpha)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_algorithms.py#L20)
* **[`NNXSACAgent(nnx.Module, TrainingAgent)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_algorithms.py#L27)**: NNX implementation of Soft Actor-Critic (v1 and v2 depending on learn_entropy_coef)
  * [`__init__(self, observation_space, action_space, device, model_cls, gamma, polyak, alpha, lr_actor, lr_critic, lr_entropy, learn_entropy_coef, target_entropy, optimizer_actor, optimizer_critic, betas_actor, betas_critic, l2_actor, l2_critic, rngs_seed, internal_rngs_seed)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_algorithms.py#L31)
  * [`get_actor(self)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_algorithms.py#L119)
  * [`train(self, batch)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_algorithms.py#L123)


---

### [`tmrl\custom\jax\custom_memories.py`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_memories.py)

**Role:** Source module in `tmrl\custom\jax`.

#### Classes

* **[`NNXVar(nnx.Variable)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_memories.py#L11)**
* **[`ArrayNNXMemory(nnx.Module, BaseMemory)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_memories.py#L14)**: Memory optimized for vector obsevations and actions.
  * [`__init__(self, observation_vector_dim, action_vector_dim, memory_size, batch_size, dataset_path, sample_preprocessor, crc_debug, device, seed)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_memories.py#L23)
  * [`__len__(self)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_memories.py#L80)
  * [`append_buffer(self, buffer)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_memories.py#L83) — *Appends a `tmrl.networking.Buffer` of samples to the memory.*
  * [`sample(self)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_memories.py#L154) — *JIT-compilable*


---

### [`tmrl\custom\jax\custom_models.py`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py)

**Role:** Source module in `tmrl\custom\jax`.

#### Classes

* **[`NNXSquashedGaussianMLPActor(NNXActorModule)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L55)**
  * [`__init__(self, observation_space, action_space, hidden_sizes, activation, layer_norm, rngs, internal_rngs_seed)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L56)
  * [`__call__(self, obs, rngs, test, with_logprob, epsilon)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L86) — *Note: this function assumes a batch dimension in obs.*
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L120)
* **[`NNXMLPQFunction(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L128)**
  * [`__init__(self, observation_space, action_space, hidden_sizes, activation, dropout, layer_norm, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L129)
  * [`__call__(self, obs, act)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L151)
* **[`NNXREDQMLPActorCritic(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L157)**: By default, this holds 2 critics for SAC.
  * [`__init__(self, observation_space, action_space, n, hidden_sizes, activation, critic_dropout, critic_layer_norm, actor_layer_norm, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L162)
* **[`VanillaCNN(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L198)**: Vanilla CNN feature extractor and MLP fusion head in Flax NNX.
  * [`__init__(self, q_net, img_height, img_width, img_hist_len, dropout, layer_norm, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L203)
  * [`__call__(self, x)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L243)
* **[`NNXSquashedGaussianVanillaCNNActor(NNXActorModule)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L267)**: Squashed Gaussian CNN Actor Policy in Flax NNX.
  * [`__init__(self, observation_space, action_space, layer_norm, rngs, internal_rngs_seed)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L271)
  * [`__call__(self, obs, rngs, test, with_logprob, epsilon)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L307)
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L331)
* **[`NNXVanillaCNNQFunction(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L339)**: Vanilla CNN Critic Q-Network in Flax NNX.
  * [`__init__(self, observation_space, action_space, dropout, layer_norm, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L343)
  * [`__call__(self, obs, act)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L373)
* **[`NNXREDQVanillaCNNActorCritic(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L379)**: REDQ / SAC Ensemble Container for CNN Models.
  * [`__init__(self, observation_space, action_space, n, critic_dropout, critic_layer_norm, actor_layer_norm, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L384)
* **[`NNXSquashedGaussianVanillaColorCNNActor(NNXSquashedGaussianVanillaCNNActor)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L422)**
  * [`__call__(self, obs, rngs, test, with_logprob, epsilon)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L423)
* **[`NNXVanillaColorCNNQFunction(NNXVanillaCNNQFunction)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L430)**
  * [`__call__(self, obs, act)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L431)
* **[`NNXREDQVanillaColorCNNActorCritic(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L438)**
  * [`__init__(self, observation_space, action_space, n, critic_dropout, critic_layer_norm, actor_layer_norm, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L439)

#### Functions

* **[`mlp(sizes, activation, output_activation, dropout, layer_norm, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L24)**
* **[`conv2d_out_dims(kernel_size, strides, h_in, w_in)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L186)**: Calculates output spatial dimensions for 2D convolution with VALID padding.
* **[`remove_colors(images)`](file:///D:/Project/tmrl/tmrl/custom/jax/custom_models.py#L414)**: Removes color channels so that grayscale-based models can be used.


---

### [`tmrl\custom\jax\imagination_trainer.py`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py)

**Role:** Source module in `tmrl\custom\jax`.

#### Classes

* **[`LatentActor(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L11)**: Policy network operating on latent world model state features (h_t, z_t).
  * [`__init__(self, state_dim, act_dim, hidden_dim, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L16)
  * [`__call__(self, state_feat, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L28)
* **[`LatentCritic(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L47)**: State-value critic network V(s) operating on latent state features.
  * [`__init__(self, state_dim, hidden_dim, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L51)
  * [`__call__(self, state_feat)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L61)
* **[`LatentImaginationTrainer`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L96)**: Experimental replay-grounded latent imagination engine.
  * [`__init__(self, horizon, gamma, lambda_, lr_model, lr_actor, lr_critic, lr_adv, entropy_scale, free_nats, max_grad_norm, enable_latent_adversary, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L106)
  * [`train_world_model_step(self, obs_tuple, actions, rewards, continues, next_obs_tuple)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L142) — *Fit one replay-grounded transition.*
  * [`_posterior_start(self, obs_tuple)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L251)
  * [`_rollout(self, actor, h0, z0, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L258)
  * [`_survival_weights(self, continues)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L291)
  * [`train_imagination_step(self, initial_obs_tuple)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L296) — *Execute an H-step posterior-anchored rollout and update actor/critic.*
  * [`propose_latent_tasks(self, initial_obs_tuple)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L366) — *Sample persistent proposals for an external repeated-solve verifier.*
  * [`train_adversary_step(self, task_features, raw_proposals, solver_successes)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L378) — *Update the proposer from repeated binary outcomes of matched tasks.*

#### Functions

* **[`compute_lambda_returns(rewards, values, continues, bootstrap_val, lambda_, gamma)`](file:///D:/Project/tmrl/tmrl/custom/jax/imagination_trainer.py#L67)**: Computes generalized lambda-returns across imagined sequence horizon.


---

### [`tmrl\custom\jax\latent_adversary.py`](file:///D:/Project/tmrl/tmrl/custom/jax/latent_adversary.py)

**Role:** Source module in `tmrl\custom\jax`.

#### Classes

* **[`LatentAdversaryProposer(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/latent_adversary.py#L28)**: Experimental bounded proposer operating in model latent space.
  * [`__init__(self, feature_dim, latent_dim, hidden_dim, perturbation_scale, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/latent_adversary.py#L35)
  * [`distribution(self, state_feat)`](file:///D:/Project/tmrl/tmrl/custom/jax/latent_adversary.py#L51) — *Return the Gaussian mean and log standard deviation in raw space.*
  * [`log_prob(self, state_feat, raw_sample)`](file:///D:/Project/tmrl/tmrl/custom/jax/latent_adversary.py#L59) — *Evaluate a stored raw proposal without resampling it.*
  * [`sample(self, state_feat, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/latent_adversary.py#L78) — *Return ``(bounded_perturbation, raw_sample, log_probability)``.*
  * [`__call__(self, state_feat, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/latent_adversary.py#L93) — *Proposes a continuous latent perturbation vector xi_t.*

#### Functions

* **[`compute_latent_learnability_reward(solver_successes, rollout_axis)`](file:///D:/Project/tmrl/tmrl/custom/jax/latent_adversary.py#L8)**: Compute the AZR proposer reward from repeated binary solver attempts.


---

### [`tmrl\custom\jax\world_model.py`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py)

**Role:** Source module in `tmrl\custom\jax`.

#### Classes

* **[`VisualTelemetryEncoder(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L29)**: Encodes visual frames and vehicle telemetry into a compact latent representation.
  * [`__init__(self, img_hist_len, img_height, img_width, latent_dim, channels, normalized_observations, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L33)
  * [`__call__(self, obs_tuple)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L84)
* **[`LatentRSSM(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L118)**: Recurrent State-Space Model (RSSM) predicting deterministic belief (h_t)
  * [`__init__(self, latent_dim, action_dim, hidden_dim, stable, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L123)
  * [`initial_state(self, batch_size)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L150)
  * [`step_deterministic(self, prev_h, prev_z, action)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L155)
  * [`distribution_parameters(self, raw_mean, raw_std)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L164)
  * [`predict_prior(self, h, rngs, latent_perturbation)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L178) — *Predicts the prior distribution and samples a stochastic latent z_t.*
  * [`compute_posterior(self, h, embed, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L192) — *Calculates the posterior distribution from real observations.*
* **[`LatentWorldModel(nnx.Module)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L204)**: Complete TrackMania Latent World Model combining:
  * [`__init__(self, img_hist_len, img_height, img_width, latent_dim, action_dim, hidden_dim, encoder_channels, normalized_observations, stable_rssm, rngs)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L211)
  * [`get_feature(self, h, z)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L255)
  * [`predict_reward(self, feat)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L258) — *Predicts step reward in symlog space.*
  * [`predict_continuation(self, feat)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L262) — *Predicts episode continuation probability (1 = continue, 0 = terminated).*
  * [`predict_continuation_logits(self, feat)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L266)
  * [`predict_telemetry(self, feat)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L269)
  * [`predict_embedding(self, feat)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L272) — *Predicts the next encoded observation as a decoder-free grounding loss.*
  * [`imagine_step(self, prev_h, prev_z, action, rngs, latent_perturbation)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L276) — *Rolls forward 1 step in pure latent imagination without real observations.*

#### Functions

* **[`_configured_image_dimension(name, fallback)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L8)**: Resolve legacy config defaults only when a caller omits dimensions.
* **[`symlog(x)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L19)**: Symmetric logarithmic transformation for scale-invariant regression.
* **[`symexp(x)`](file:///D:/Project/tmrl/tmrl/custom/jax/world_model.py#L24)**: Inverse symmetric exponential transformation with numerical clamping.


---

### [`tmrl\custom\tm\__init__.py`](file:///D:/Project/tmrl/tmrl/custom/tm/__init__.py)

**Role:** Source module in `tmrl\custom\tm`.


---

### [`tmrl\custom\tm\tm_gym_interfaces.py`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py)

**Role:** Source module in `tmrl\custom\tm`.

#### Classes

* **[`TM2020Interface(RealTimeGymInterface)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L36)**: This is the API needed for the algorithm to control TrackMania 2020
  * [`__init__(self, img_hist_len, gamepad, save_replays, grayscale, resize_to, signed_analog_triggers)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L40) — *Base rtgym interface for TrackMania 2020 (Full environment)*
  * [`initialize_common(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L84)
  * [`_sha256_file(path)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L117)
  * [`validate_deployment_track(self, data)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L124) — *Refuse control if the loaded map is not the demonstrated map.*
  * [`initialize(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L174)
  * [`send_control(self, control)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L179) — *Non-blocking function*
  * [`grab_data_and_img(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L206)
  * [`reset_race(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L223)
  * [`reset_common(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L229)
  * [`reset(self, seed, options)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L258) — *obs must be a list of numpy arrays*
  * [`close_finish_pop_up_tm20(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L286)
  * [`wait(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L292) — *Non-blocking function*
  * [`get_obs_rew_terminated_info(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L306) — *returns the observation, the reward, and a terminated signal for end of episode*
  * [`get_observation_space(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L343) — *must be a Tuple*
  * [`get_action_space(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L360) — *must return a Box*
  * [`get_default_action(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L366) — *initial action at episode start*
* **[`TM2020InterfaceLidar(TM2020Interface)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L374)**
  * [`__init__(self, img_hist_len, gamepad, save_replays)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L375)
  * [`grab_lidar_speed_and_data(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L380)
  * [`initialize(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L389)
  * [`reset(self, seed, options)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L395) — *obs must be a list of numpy arrays*
  * [`get_obs_rew_terminated_info(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L408) — *returns the observation, the reward, and a terminated signal for end of episode*
  * [`get_observation_space(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L431) — *must be a Tuple*
* **[`TM2020InterfaceLidarProgress(TM2020InterfaceLidar)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L443)**
  * [`reset(self, seed, options)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L445) — *obs must be a list of numpy arrays*
  * [`get_obs_rew_terminated_info(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L459) — *returns the observation, the reward, and a terminated signal for end of episode*
  * [`get_observation_space(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_gym_interfaces.py#L483) — *must be a Tuple*


---

### [`tmrl\custom\tm\tm_preprocessors.py`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_preprocessors.py)

**Role:** Source module in `tmrl\custom\tm`.

#### Functions

* **[`obs_preprocessor_tm_act_in_obs(obs)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_preprocessors.py#L9)**: Preprocessor for TM2020 full environment with grayscale images
* **[`obs_preprocessor_tm_lidar_act_in_obs(obs)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_preprocessors.py#L19)**: Preprocessor for the LIDAR environment, flattening LIDARs
* **[`obs_preprocessor_tm_lidar_progress_act_in_obs(obs)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_preprocessors.py#L27)**: Preprocessor for the LIDAR environment, flattening LIDARs
* **[`sample_preprocessor_tm_lidar_act_in_obs(last_obs, act, rew, new_obs, terminated, truncated)`](file:///D:/Project/tmrl/tmrl/custom/tm/tm_preprocessors.py#L41)**


---

### [`tmrl\custom\tm\utils\__init__.py`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/__init__.py)

**Role:** Source module in `tmrl\custom\tm\utils`.


---

### [`tmrl\custom\tm\utils\camera_preprocessing.py`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/camera_preprocessing.py)

**Purpose:** Shared camera conversion without copying a full-resolution BGRA slice.

#### Functions

* **[`preprocess_camera_frame(frame, resize_to, grayscale)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/camera_preprocessing.py#L6)**: Resize contiguous BGR/BGRA first; discard alpha at policy resolution.


---

### [`tmrl\custom\tm\utils\compute_reward.py`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/compute_reward.py)

**Role:** Source module in `tmrl\custom\tm\utils`.

#### Classes

* **[`RewardFunction`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/compute_reward.py#L10)**: Computes a reward from the Openplanet API for Trackmania 2020.
  * [`__init__(self, reward_data_path, nb_obs_forward, nb_obs_backward, nb_zero_rew_before_failure, min_nb_steps_before_failure, max_dist_from_traj)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/compute_reward.py#L14) — *Instantiates a reward function for TM2020.*
  * [`compute_reward(self, pos, speed, distance)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/compute_reward.py#L57) — *Computes the current reward given the position pos, speed, and distance*
  * [`reset(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/compute_reward.py#L170) — *Resets the reward function for a new episode.*


---

### [`tmrl\custom\tm\utils\control_gamepad.py`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/control_gamepad.py)

**Role:** Source module in `tmrl\custom\tm\utils`.

#### Functions

* **[`signed_trigger_controls(control)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/control_gamepad.py#L6)**: Convert BC's signed triggers to physical [0, 1] pressure, retaining steer.


---

### [`tmrl\custom\tm\utils\control_keyboard.py`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/control_keyboard.py)

**Role:** Source module in `tmrl\custom\tm\utils`.


---

### [`tmrl\custom\tm\utils\control_mouse.py`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/control_mouse.py)

**Role:** Source module in `tmrl\custom\tm\utils`.


---

### [`tmrl\custom\tm\utils\self_play_obs.py`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_obs.py)

**Role:** Source module in `tmrl\custom\tm\utils`.

#### Classes

* **[`SelfPlayGhostObserver`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_obs.py#L6)**: Computes comparative ghost observation features for Self-Play in TMRL.
  * [`__init__(self, trajectory_manager)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_obs.py#L13)
  * [`get_ghost_features(self, pos, speed, timestamp, cur_idx)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_obs.py#L16) — *Computes 5-dimensional ghost delta features:*


---

### [`tmrl\custom\tm\utils\self_play_reward.py`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py)

**Role:** Source module in `tmrl\custom\tm\utils`.

#### Classes

* **[`SelfPlayTrajectoryManager`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L9)**: Manages self-evolving reference trajectories for ghost-based self-imitation.
  * [`__init__(self, save_path, min_points_to_save, resample_step)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L17) — *Args:*
  * [`_load_trajectory(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L59)
  * [`_compute_total_distance(self, traj)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L81)
  * [`_compute_progress_distance(self, traj)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L87) — *Loop-resistant cold-start progress measured from the episode start.*
  * [`_compute_duration(traj)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L95)
  * [`_resample_trajectory(self, traj)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L101) — *Resample the ghost at stable spatial intervals for index-based progress.*
  * [`record_step(self, pos, speed, timestamp)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L118) — *Records a single transition state into the current episode trajectory.*
  * [`on_episode_end(self, finished_track)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L123) — *Evaluates current episode against reference ghost.*
  * [`_save_trajectory(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L171)
  * [`has_reference_ghost(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L190)
  * [`get_positions(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L193)
* **[`SelfPlayRewardFunction`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L199)**: Self-play reward function implementing ghost ratcheting for TMRL.
  * [`__init__(self, trajectory_manager, reward_data_path, nb_obs_forward, nb_obs_backward, nb_zero_rew_before_failure, min_nb_steps_before_failure, max_dist_from_traj, ghost_speed_weight, personal_record_bonus)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L209)
  * [`data(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L237)
  * [`compute_reward(self, pos, speed, timestamp, finished_track)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L240) — *Computes current reward and termination status.*
  * [`reset(self, finished_track)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/self_play_reward.py#L345) — *Resets reward state and triggers self-play ratcheting evaluation.*


---

### [`tmrl\custom\tm\utils\tm_vision.py`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tm_vision.py)

**Purpose:** High-Performance Real-Time Game Window Vision & Visual Tokenizer for TrackMania 2020.

#### Classes

* **[`FastWindowCapture`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tm_vision.py#L25)**: Ultra-low-latency TrackMania viewport grabber using Windows GDI BitBlt.
  * [`__init__(self, window_name, target_size)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tm_vision.py#L30)
  * [`_find_window(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tm_vision.py#L39)
  * [`move_and_resize(self, x, y, w, h)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tm_vision.py#L86) — *Snaps TrackMania window to a compact rectangle (default 512x256 at top-left).*
  * [`grab_frame(self, grayscale)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tm_vision.py#L105) — *Grabs the current game frame and returns a (H, W) or (H, W, 3) uint8 numpy array.*
  * [`grab_tensor(self, device)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tm_vision.py#L150) — *Returns normalized float tensor (1, 1, H, W) in [0.0, 1.0].*
* **[`VisualEncoder(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tm_vision.py#L157)**: Lightweight 4-layer Convolutional Tokenizer for 96x96 road images.
  * [`__init__(self, in_channels, embed_dim)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tm_vision.py#L163)
  * [`forward(self, x)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tm_vision.py#L176) — *x: (B, C, H, W) or (B, T, C, H, W)*


---

### [`tmrl\custom\tm\utils\tools.py`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tools.py)

**Role:** Source module in `tmrl\custom\tm\utils`.

#### Classes

* **[`TM2020OpenPlanetClient`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tools.py#L18)**
  * [`__init__(self, host, port, struct_str)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tools.py#L19)
  * [`__client_thread(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tools.py#L37) — *Thread of the client.*
  * [`retrieve_data(self, sleep_if_empty, timeout, max_age)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tools.py#L72) — *Retrieves the most recently received data*
  * [`close(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tools.py#L100) — *Stop the socket reader and wake any waiting consumer.*
* **[`Lidar`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tools.py#L128)**
  * [`__init__(self, im)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tools.py#L129)
  * [`_set_axis_lidar(self, im)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tools.py#L133)
  * [`lidar_20(self, img, show)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tools.py#L164)

#### Functions

* **[`save_ghost(host, port)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tools.py#L108)**: Saves the current ghost
* **[`armin(tab)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/tools.py#L120)**


---

### [`tmrl\custom\tm\utils\trajectory_assist.py`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/trajectory_assist.py)

**Purpose:** Map-locked demonstration trajectory safety controller.

#### Classes

* **[`DemonstrationTrajectoryAssist`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/trajectory_assist.py#L11)**: Replay demonstrated controls by spatial progress and correct drift.
  * [`__init__(self, config)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/trajectory_assist.py#L19)
  * [`_load(self)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/trajectory_assist.py#L72)
  * [`reset(self, log_previous)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/trajectory_assist.py#L111)
  * [`_ternary(value, deadzone)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/trajectory_assist.py#L131)
  * [`_nearest_index(self, position)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/trajectory_assist.py#L134)
  * [`_update_heading(self, position_xz, reference_index)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/trajectory_assist.py#L149)
  * [`action(self, telemetry, proposed_action)`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/trajectory_assist.py#L176)


---

### [`tmrl\custom\tm\utils\window.py`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/window.py)

**Role:** Source module in `tmrl\custom\tm\utils`.

#### Functions

* **[`profile_screenshot()`](file:///D:/Project/tmrl/tmrl/custom/tm/utils/window.py#L498)**


---

### [`tmrl\custom\torch\__init__.py`](file:///D:/Project/tmrl/tmrl/custom/torch/__init__.py)

**Role:** Source module in `tmrl\custom\torch`.


---

### [`tmrl\custom\torch\azr.py`](file:///D:/Project/tmrl/tmrl/custom/torch/azr.py)

**Purpose:** Replay-grounded AZR-style curriculum utilities for Torch imagination.

#### Classes

* **[`ReplayLatentTask`](file:///D:/Project/tmrl/tmrl/custom/torch/azr.py#L40)**: A persistent, replay-anchored challenge in a local model snapshot.
* **[`ReplayLatentTaskBuffer`](file:///D:/Project/tmrl/tmrl/custom/torch/azr.py#L54)**: Small prioritized buffer for screened posterior-anchored tasks.
  * [`__init__(self, capacity, max_age, seed)`](file:///D:/Project/tmrl/tmrl/custom/torch/azr.py#L57)
  * [`__len__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/azr.py#L68)
  * [`prune(self, current_step)`](file:///D:/Project/tmrl/tmrl/custom/torch/azr.py#L71) — *Drop model-space tasks after the model has drifted too far.*
  * [`add_batch(self, h, z, target_returns, priorities, pass_rates, mean_survival, valid, current_step)`](file:///D:/Project/tmrl/tmrl/custom/torch/azr.py#L79) — *Insert valid and currently learnable tasks, returning the count.*
  * [`sample(self, batch_size, device, current_step)`](file:///D:/Project/tmrl/tmrl/custom/torch/azr.py#L137) — *Sample tasks proportionally to learnability priority.*
  * [`update(self, task_ids, priorities, pass_rates, mean_survival, current_step)`](file:///D:/Project/tmrl/tmrl/custom/torch/azr.py#L183) — *Refresh priorities using new attempts at the same persistent tasks.*

#### Functions

* **[`compute_azr_learnability(solver_successes, rollout_axis)`](file:///D:/Project/tmrl/tmrl/custom/torch/azr.py#L17)**: Return per-task AZR learnability rewards and pass rates.


---

### [`tmrl\custom\torch\car_brain.py`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py)

**Purpose:** TrackMania Causal Transformer "Car Brain" for In-Context Learning.

#### Classes

* **[`TimestepEmbedder(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L36)**: Embeds multi-modal timestep features [state_t, action_{t-1}, reward_{t-1}, done_{t-1}]
  * [`__init__(self, state_dim, action_dim, d_model, include_prev_action, include_prev_reward, include_prev_done)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L41)
  * [`forward(self, states, prev_actions, prev_rewards, prev_dones)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L73) — *Args:*
* **[`CausalSelfAttention(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L124)**: Multi-head causal self-attention with pre-norm and dropout.
  * [`__init__(self, d_model, n_heads, dropout)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L129)
  * [`forward(self, x, mask)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L142)
* **[`TransformerBlock(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L168)**: Pre-norm Transformer block with causal attention and MLP.
  * [`__init__(self, d_model, n_heads, d_ff, dropout)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L170)
  * [`forward(self, x, mask)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L183)
* **[`CarBrainPolicyHead(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L189)**: Maps transformer output tokens to continuous action predictions:
  * [`__init__(self, d_model, action_dim, hidden_dim)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L194)
  * [`forward(self, hidden_states)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L203)
* **[`CarBrain(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L208)**: Full TrackMania Car Brain:
  * [`__init__(self, state_dim, action_dim, d_model, n_layers, n_heads, d_ff, max_context_len, dropout)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L217)
  * [`forward(self, states, prev_actions, prev_rewards, prev_dones)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L260) — *Batch sequence forward pass (used during offline pretraining).*
  * [`reset_context(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L286) — *Clears the in-memory context history (call at episode resets).*
  * [`step_in_context(self, state, prev_action, prev_reward, prev_done, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L293) — *Real-time closed-loop action selection at 20 Hz.*
  * [`save_pretrained(self, save_dir, filename_prefix)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L357) — *Saves model weights in SafeTensors and PyTorch formats along with config.json.*
  * [`from_pretrained(cls, filepath, state_dim, action_dim, d_model, n_layers, n_heads, d_ff, max_context_len, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L392) — *Loads a CarBrain instance from .safetensors or .pt, auto-reading config.json if available.*
* **[`MultiModalCarBrain(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L448)**: Multi-Modal Foundation Brain for TrackMania 2020.
  * [`__init__(self, img_channels, img_size, state_dim, action_dim, d_model, n_layers, n_heads, d_ff, max_context_len, dropout)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L454)
  * [`encode_visual(self, imgs)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L521) — *imgs: (B, C, H, W) or (B, T, C, H, W) in [0, 1]*
  * [`forward(self, imgs, states, prev_actions, prev_rewards, prev_dones, return_logits)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L545) — *imgs: (B, T, 4, 96, 96)*
  * [`reset_context(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L589) — *Clears in-context history at lap start / reset.*
  * [`step_in_context(self, img, state, prev_action, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L595) — *Real-time closed loop step at 20 Hz.*
  * [`from_foundation(cls, foundation_weights_path, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain.py#L642) — *Constructs MultiModalCarBrain and initializes its Transformer backbone and policy head*


---

### [`tmrl\custom\torch\car_brain_actor.py`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain_actor.py)

**Purpose:** TMRL ActorModule wrapper for the Causal Transformer Car Brain.

#### Classes

* **[`CarBrainActor(TorchActorModule)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain_actor.py#L20)**: TMRL ActorModule backed by a Causal Transformer Car Brain.
  * [`__init__(self, observation_space, action_space, state_dim, d_model, n_layers, n_heads, d_ff, max_context_len, pretrained_path, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain_actor.py#L26)
  * [`load_pretrained(self, filepath)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain_actor.py#L59) — *Loads weights from .safetensors or .pt.*
  * [`set_adaptive(self, adaptive)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain_actor.py#L74) — *Enables or disables in-context history (No-History ablation).*
  * [`reset_context(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain_actor.py#L80) — *Resets in-context rolling history (call on episode reset).*
  * [`_extract_state_vector(self, obs)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain_actor.py#L85) — *Extracts or formats a 1D state tensor of size (state_dim) from TMRL obs.*
  * [`forward(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain_actor.py#L130) — *Batched forward pass conforming to TMRL TorchActorModule.*
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/torch/car_brain_actor.py#L144) — *Real-time closed-loop action step at 20 Hz.*


---

### [`tmrl\custom\torch\custom_algorithms.py`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py)

**Role:** Source module in `tmrl\custom\torch`.

#### Classes

* **[`SpinupSACAgent(TrainingAgent)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L27)**
  * [`__post_init__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L49)
  * [`get_actor(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L107)
  * [`train(self, batch)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L110)
* **[`REDQSACAgent(TrainingAgent)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L301)**
  * [`__post_init__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L325)
  * [`get_actor(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L390)
  * [`train(self, batch)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L393)
* **[`DreamSACAgent(SpinupSACAgent)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L498)**: SAC plus a gated replay-grounded AZR/imagination curriculum.
  * [`__post_init__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L536)
  * [`_empty_azr_metrics(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L591)
  * [`_alpha_for_actor(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L616)
  * [`_set_world_model_grad(self, enabled)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L621)
  * [`_posterior_start(self, obs)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L625)
  * [`_rollout_tasks(self, h, z, attempts)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L637) — *Roll out exact task states; gradients are controlled by the caller.*
  * [`_gaussian_kl(p_mean, p_std, q_mean, q_std)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L673) — *KL(N(p)||N(q)), reduced over the latent dimension.*
  * [`_train_world_model(self, o, a, r, o2, d, truncated)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L684)
  * [`_propose_and_validate_tasks(self, obs)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L764)
  * [`_refresh_tasks_and_train_actor(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L833)
  * [`train(self, batch)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_algorithms.py#L910)


---

### [`tmrl\custom\torch\custom_checkpoints.py`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_checkpoints.py)

**Role:** Source module in `tmrl\custom\torch`.

#### Functions

* **[`load_run_instance_images_dataset(checkpoint_path)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_checkpoints.py#L15)**: function used to load trainers from checkpoint path
* **[`dump_run_instance_images_dataset(run_instance, checkpoint_path)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_checkpoints.py#L34)**: function used to dump trainers to checkpoint path
* **[`update_memory(run_instance)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_checkpoints.py#L54)**
* **[`dump_dreamer_run_instance(run_instance, checkpoint_path)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_checkpoints.py#L84)**: Safely dumps the TorchDreamer trainer checkpoint.
* **[`load_dreamer_run_instance(checkpoint_path)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_checkpoints.py#L122)**
* **[`update_run_instance(run_instance, training_cls)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_checkpoints.py#L126)**: Updates the checkpoint after loading with compatible values from config.json


---

### [`tmrl\custom\torch\custom_memories.py`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py)

**Role:** Source module in `tmrl\custom\torch`.

#### Classes

* **[`GenericTorchMemory(TorchMemory)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L109)**: Generic Torch memory implementation.
  * [`__init__(self, memory_size, batch_size, dataset_path, sample_preprocessor, crc_debug, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L115)
  * [`append_buffer(self, buffer)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L129)
  * [`__len__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L169)
  * [`get_transition(self, item)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L178)
* **[`ArrayTorchMemory(BaseMemory)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L199)**: Optimized generic Torch memory implementation for simple numpy arrays.
  * [`__init__(self, memory_size, batch_size, dataset_path, sample_preprocessor, crc_debug, device, replace, shuffle)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L206)
  * [`__len__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L226)
  * [`append_buffer(self, buffer)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L235)
  * [`sample(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L279)
* **[`MemoryTM(TorchMemory)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L333)**
  * [`__init__(self, memory_size, batch_size, dataset_path, imgs_obs, act_buf_len, sample_preprocessor, crc_debug, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L334)
  * [`append_buffer(self, buffer)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L355)
  * [`__len__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L358)
  * [`get_transition(self, item)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L367)
* **[`MemoryTMLidar(MemoryTM)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L371)**
  * [`get_transition(self, item)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L372) — *CAUTION: item is the first index of the 4 images in the images history of the OLD observation*
  * [`load_imgs(self, item)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L426)
  * [`load_acts(self, item)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L430)
  * [`append_buffer(self, buffer)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L434) — *buffer is a list of samples (act, obs, rew, terminated, truncated, info)*
* **[`MemoryTMLidarProgress(MemoryTM)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L488)**
  * [`get_transition(self, item)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L489) — *CAUTION: item is the first index of the 4 images in the images history of the OLD observation*
  * [`load_imgs(self, item)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L543)
  * [`load_acts(self, item)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L547)
  * [`append_buffer(self, buffer)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L551) — *buffer is a list of samples (act, obs, rew, truncated, terminated, info)*
* **[`MemoryTMFull(MemoryTM)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L609)**
  * [`get_transition(self, item)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L610) — *CAUTION: item is the first index of the 4 images in the images history of the OLD observation*
  * [`load_imgs(self, item)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L661)
  * [`load_acts(self, item)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L665)
  * [`append_buffer(self, buffer)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L669) — *buffer is a list of samples ( act, obs, rew, terminated, truncated, info)*
* **[`NumpyArrayTorchMemoryTMFull(BaseMemory)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L766)**: data is stored as concatenated numpy arrays
  * [`__init__(self, memory_size, batch_size, dataset_path, imgs_obs, act_buf_len, sample_preprocessor, crc_debug, device, replace, shuffle)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L770)
  * [`__len__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L813)
  * [`append_buffer(self, buffer)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L822) — *buffer is a list of samples ( act, obs, rew, terminated, truncated, info)*
  * [`load_batch_acts(self, indices)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L883)
  * [`load_batch_imgs(self, indices)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L889)
  * [`load_batch_eoes(self, indices)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L895)
  * [`sample_batch_indices(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L900)
  * [`sample(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L916)
  * [`get_benchmarks(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1024)
  * [`get_benchmarks_names(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1056)
* **[`ArrayTorchMemoryTMFull(BaseMemory)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1061)**: data stored as concatenated torch tensors
  * [`__init__(self, memory_size, batch_size, dataset_path, imgs_obs, act_buf_len, sample_preprocessor, crc_debug, device, replace, shuffle)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1065)
  * [`__len__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1103)
  * [`append_buffer(self, buffer)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1112) — *buffer is a list of samples ( act, obs, rew, terminated, truncated, info)*
  * [`load_batch_acts(self, indices)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1175)
  * [`load_batch_imgs(self, indices)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1181)
  * [`sample_batch_indices(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1187)
  * [`load_batch_eoes(self, indices)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1203)
  * [`sample(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1208)
  * [`get_benchmarks(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1331)
  * [`get_benchmarks_names(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1355)
* **[`ArrayTorchMemoryTMFullSequence(ArrayTorchMemoryTMFull)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1359)**: Episode-safe contiguous sequence sampler for recurrent world models.
  * [`__init__(self, sequence_length, replay_sampler, replay_candidate_count, replay_uniform_fraction, replay_embedding_dim, replay_curvature, replay_tangent_scale, replay_max_radius, replay_seed)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1371)
  * [`_normalize_replay_sampler(mode)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1407)
  * [`configure_replay_sampler(self, replay_sampler, replay_candidate_count, replay_uniform_fraction, replay_embedding_dim, replay_curvature, replay_tangent_scale, replay_max_radius, replay_seed)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1417) — *Configure the detached sequence-selection index.*
  * [`_ensure_replay_sampler_state(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1520) — *Supply defaults when loading a checkpoint created before this index.*
  * [`_sample_episode_safe_starts(self, count)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1555)
  * [`_series_summary(series)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1596)
  * [`_sequence_descriptors(self, starts)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1611) — *Build cheap, detached descriptors for candidate replay sequences.*
  * [`_project_replay_descriptors(self, descriptors)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1665)
  * [`_record_replay_metrics(self, candidate_count, distance, radius)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1695)
  * [`_sample_diverse_sequence_starts(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1701)
  * [`sample_sequence_indices(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1757)
  * [`get_benchmarks(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1767)
  * [`get_benchmarks_names(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1793)
  * [`sample(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1802)
* **[`NewArrayTorchMemoryTMFull(BaseMemory)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L2035)**
  * [`__init__(self, memory_size, batch_size, dataset_path, imgs_obs, act_buf_len, sample_preprocessor, crc_debug, device, replace, shuffle)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L2036)
  * [`__getstate__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L2083)
  * [`__setstate__(self, state)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L2089)
  * [`__len__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L2094)
  * [`append_buffer(self, buffer)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L2103) — *buffer is a list of samples ( act, obs, rew, terminated, truncated, info)*
  * [`sample_batch_indices(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L2170)
  * [`sample(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L2186)
  * [`get_benchmarks(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L2205)
  * [`get_benchmarks_names(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L2233)

#### Functions

* **[`get_local_buffer_sample_lidar(prev_act, obs, rew, terminated, truncated, info)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L25)**: Input:
* **[`get_local_buffer_sample_lidar_progress(prev_act, obs, rew, terminated, truncated, info)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L43)**: Input:
* **[`get_local_buffer_sample_tm20_imgs(prev_act, obs, rew, terminated, truncated, info)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L61)**: Sample compressor for MemoryTMFull
* **[`last_true_in_list(li)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L85)**
* **[`replace_hist_before_eoe(hist, eoe_idx_in_hist)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L92)**: Pads the history hist before the End Of Episode (EOE) index.
* **[`fix_batch_history_around_eoes_numpy(batch_history, eoes)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L731)**
* **[`fix_batch_history_around_eoes_torch(batch_history, eoes)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L748)**
* **[`load_batch_acts(indices, start_acts_offset, act_buf_len, torch_data)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1933)**
* **[`load_batch_imgs(indices, start_imgs_offset, imgs_obs, torch_data)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1939)**
* **[`get_tensors_tm20_full(indices, min_samples, start_acts_offset, act_buf_len, start_imgs_offset, imgs_obs, torch_data, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_memories.py#L1946)**


---

### [`tmrl\custom\torch\custom_models.py`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py)

**Role:** Source module in `tmrl\custom\torch`.

#### Classes

* **[`SquashedGaussianMLPActor(TorchActorModule)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L56)**
  * [`__init__(self, observation_space, action_space, hidden_sizes, activation, layer_norm)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L57)
  * [`forward(self, obs, test, with_logprob)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L72) — *Note: this function assumes a batch dimension in obs.*
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L107)
* **[`MLPQFunction(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L116)**
  * [`__init__(self, obs_space, act_space, hidden_sizes, activation, dropout, layer_norm)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L117)
  * [`forward(self, obs, act)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L130)
* **[`MLPActorCritic(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L136)**
  * [`__init__(self, observation_space, action_space, hidden_sizes, activation, critic_dropout, critic_layer_norm, actor_layer_norm)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L137)
* **[`REDQMLPActorCritic(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L164)**
  * [`__init__(self, observation_space, action_space, hidden_sizes, activation, n, critic_dropout, critic_layer_norm, actor_layer_norm)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L165)
* **[`VanillaCNN(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L228)**
  * [`__init__(self, q_net, dropout, layer_norm)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L229)
  * [`forward(self, x)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L254)
* **[`SquashedGaussianVanillaCNNActor(TorchActorModule)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L278)**
  * [`__init__(self, observation_space, action_space, layer_norm)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L279)
  * [`encode_observation(self, obs)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L289) — *Encode a real observation into the deployed policy feature space.*
  * [`forward_from_features(self, net_out, test, with_logprob)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L293) — *Apply the deployed action head to real or grounded imagined features.*
  * [`forward(self, obs, test, with_logprob)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L328)
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L332)
* **[`VanillaCNNQFunction(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L338)**
  * [`__init__(self, observation_space, action_space, dropout, layer_norm)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L339)
  * [`forward(self, obs, act)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L343)
* **[`VanillaCNNActorCritic(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L349)**
  * [`__init__(self, observation_space, action_space, critic_dropout, critic_layer_norm, actor_layer_norm)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L350)
* **[`REDQVanillaCNNActorCritic(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L369)**
  * [`__init__(self, observation_space, action_space, n, critic_dropout, critic_layer_norm, actor_layer_norm)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L370)
* **[`SquashedGaussianVanillaColorCNNActor(SquashedGaussianVanillaCNNActor)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L405)**
  * [`encode_observation(self, obs)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L406)
  * [`forward(self, obs, test, with_logprob)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L411)
* **[`VanillaColorCNNQFunction(VanillaCNNQFunction)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L416)**
  * [`forward(self, obs, act)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L417)
* **[`VanillaColorCNNActorCritic(VanillaCNNActorCritic)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L424)**
  * [`__init__(self, observation_space, action_space)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L425)
* **[`SELayer(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L451)**
  * [`__init__(self, inp, oup, reduction)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L452)
  * [`forward(self, x)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L462)
* **[`MBConv(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L485)**
  * [`__init__(self, inp, oup, stride, expand_ratio, use_se)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L486)
  * [`forward(self, x)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L518)
* **[`EffNetV2(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L525)**
  * [`__init__(self, cfgs, nb_channels_in, dim_output, width_mult)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L526)
  * [`forward(self, x)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L549)
  * [`_initialize_weights(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L557)
* **[`SquashedGaussianEffNetActor(TorchActorModule)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L639)**
  * [`__init__(self, observation_space, action_space)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L640)
  * [`forward(self, obs, test, with_logprob)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L651)
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L686)
* **[`EffNetQFunction(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L694)**
  * [`__init__(self, obs_space, act_space, hidden_sizes, activation)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L695)
  * [`forward(self, obs, act)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L701)
* **[`EffNetActorCritic(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L707)**
  * [`__init__(self, observation_space, action_space, hidden_sizes, activation)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L708)
* **[`SquashedGaussianRNNActor(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L741)**
  * [`__init__(self, obs_space, act_space, rnn_size, rnn_len, mlp_sizes, activation)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L742)
  * [`forward(self, obs_seq, test, with_logprob, save_hidden)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L756) — *obs: observation*
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L812)
* **[`RNNQFunction(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L819)**: The action is merged in the latent space after the RNN
  * [`__init__(self, obs_space, act_space, rnn_size, rnn_len, mlp_sizes, activation)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L823)
  * [`forward(self, obs_seq, act, save_hidden)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L833) — *obs: observation*
* **[`RNNActorCritic(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L875)**
  * [`__init__(self, observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L876)

#### Functions

* **[`combined_shape(length, shape)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L26)**
* **[`mlp(sizes, activation, output_activation, dropout, layer_norm)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L32)**
* **[`_make_divisible(v, divisor, min_value)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L191)**: This function is taken from the original tf repo.
* **[`num_flat_features(x)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L214)**
* **[`conv2d_out_dims(conv_layer, h_in, w_in)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L222)**
* **[`remove_colors(images)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L393)**: We remove colors so that we can simply use the same structure as the grayscale model.
* **[`conv_3x3_bn(inp, oup, stride)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L469)**
* **[`conv_1x1_bn(inp, oup)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L477)**
* **[`effnetv2_s()`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L572)**: Constructs a EfficientNetV2-S model
* **[`effnetv2_m()`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L588)**: Constructs a EfficientNetV2-M model
* **[`effnetv2_l()`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L605)**: Constructs a EfficientNetV2-L model
* **[`effnetv2_xl()`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L622)**: Constructs a EfficientNetV2-XL model
* **[`rnn(input_size, rnn_size, rnn_len)`](file:///D:/Project/tmrl/tmrl/custom/torch/custom_models.py#L729)**: sizes is ignored for now, expect first values and length


---

### [`tmrl\custom\torch\dreamer.py`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py)

**Purpose:** Torch Dreamer-style recurrent control for the production TMRL pipeline.

#### Classes

* **[`TorchDreamerActor(TorchActorModule)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L40)**: Recurrent latent actor deployed directly in the RolloutWorker.
  * [`__init__(self, observation_space, action_space, latent_dim, hidden_dim, policy_hidden_dim, img_channels, img_height, img_width, device, use_foundation_encoder, foundation_weights_path, freeze_foundation, foundation_only, foundation_discrete_actions, foundation_steer_threshold, residual_scale, reload_foundation_on_actor_load)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L43)
  * [`feature_dim(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L137)
  * [`reset(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L140) — *Forget recurrent state at a real environment episode boundary.*
  * [`initial(self, batch_size, device, dtype)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L149)
  * [`observe(self, obs, h, z, previous_action, is_first, sample)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L158) — *Filter a real observation into the recurrent posterior state.*
  * [`forward_features(self, feature, test, with_logprob)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L183)
  * [`policy_parameters(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L200)
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L207)
  * [`load(self, path, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L298) — *Load Dreamer state while protecting the canonical driving policy.*
* **[`DreamerPredictionHeads(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L363)**: World-model likelihood heads used only by the Trainer.
  * [`__init__(self, feature_dim, latent_dim, img_channels, reconstruction_size)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L366)
  * [`image_logits(self, feature)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L388)
* **[`DreamerValue(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L398)**: Symlog state-value model for imagined lambda returns.
  * [`__init__(self, feature_dim, hidden_dim)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L401)
  * [`forward(self, feature)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L412)
* **[`PercentileReturnNormalizer`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L416)**: EMA percentile range used to keep imagined actor targets well scaled.
  * [`__init__(self, rate, low, high, minimum_scale)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L419)
  * [`update(self, values)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L428)
* **[`TorchDreamerAgent(TrainingAgent)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L462)**: Sequence-trained Dreamer controller with latent imagination updates.
  * [`__post_init__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L515)
  * [`_build_world_model_optimizer(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L585)
  * [`configure_foundation_safety(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L620) — *Apply deployment safety settings to new or legacy checkpoints.*
  * [`get_actor(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L661)
  * [`_gaussian_kl(p_mean, p_std, q_mean, q_std)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L665)
  * [`_flatten_observation_sequence(obs)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L676)
  * [`_observation_at(obs, time_index)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L683)
  * [`_train_world_model(self, batch)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L686)
  * [`_set_world_model_grad(self, enabled)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L858)
  * [`_sample_imagination_starts(self, h_sequence, z_sequence)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L869)
  * [`_empty_azr_metrics(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L876)
  * [`_rollout_task_attempts(self, h, z, attempts)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L889) — *Evaluate repeated solver attempts at identical stored latent tasks.*
  * [`_propose_azr_tasks(self, h_sequence, z_sequence)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L927)
  * [`_sample_azr_starts(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L999)
  * [`_merge_imagination_starts(self, replay_h, replay_z, task_h, task_z)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L1047)
  * [`_train_imagination(self, h, z)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L1060)
  * [`train(self, batch)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L1152)

#### Functions

* **[`lambda_returns(rewards, discounts, next_values, lambda_)`](file:///D:/Project/tmrl/tmrl/custom/torch/dreamer.py#L447)**: Compute reverse-time generalized lambda returns for ``[time, batch]``.


---

### [`tmrl\custom\torch\foundation_actor.py`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_actor.py)

**Purpose:** Squashed Gaussian Foundation Actor for TMRL.

#### Classes

* **[`SquashedGaussianFoundationActor(TorchActorModule)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_actor.py#L27)**: TMRL Actor wrapping the 1,000,000-replay Foundation In-Context Learning Transformer.
  * [`__init__(self, observation_space, action_space, model_path, base_foundation_path, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_actor.py#L33)
  * [`reset_context(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_actor.py#L87) — *Flushes rolling in-context history upon track restart and hot-reloads new weights if available.*
  * [`_format_inputs(self, obs)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_actor.py#L110) — *Converts TMRL TM20IMAGES observation tuple:*
  * [`forward_features(self, imgs, state, prev_act)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_actor.py#L152) — *Passes inputs through MultiModal visual encoder and Transformer backbone.*
  * [`forward(self, obs, test, with_logprob)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_actor.py#L181) — *Computes SAC squashed Gaussian action and log probability.*
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_actor.py#L209) — *Real-time 20 Hz action generation for TMRL worker.*
* **[`FoundationCNNActorCritic(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_actor.py#L226)**: Actor-Critic architecture pairing the 1,000,000-replay Foundation ICL Actor
  * [`__init__(self, observation_space, action_space, critic_dropout, critic_layer_norm, actor_layer_norm)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_actor.py#L231)
* **[`REDQFoundationCNNActorCritic(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_actor.py#L245)**: Actor-Critic architecture pairing the 1,000,000-replay Foundation ICL Actor
  * [`__init__(self, observation_space, action_space, n, critic_dropout, critic_layer_norm, actor_layer_norm)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_actor.py#L250)


---

### [`tmrl\custom\torch\foundation_encoder.py`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_encoder.py)

**Purpose:** Foundation Visual-Telemetry Encoder for PyTorch Dreamer.

#### Classes

* **[`FoundationVisualTelemetryEncoder(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_encoder.py#L18)**: PyTorch Visual-Telemetry Encoder powered by the 1M Foundation Model.
  * [`__init__(self, img_channels, img_height, img_width, latent_dim, model_path, base_foundation_path, freeze_foundation, reload_on_change)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_encoder.py#L25)
  * [`_read_state_dict(model_file, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_encoder.py#L79)
  * [`set_foundation_trainable(self, trainable)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_encoder.py#L90) — *Freeze/unfreeze only the demonstrated driving policy backbone.*
  * [`train(self, mode)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_encoder.py#L100)
  * [`reset_context(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_encoder.py#L106) — *Clear the causal action context at an environment episode boundary.*
  * [`reload_foundation_weights(self, model_path)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_encoder.py#L111) — *Restore the canonical demonstrated policy without touching Dreamer.*
  * [`_maybe_reload_foundation_weights(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_encoder.py#L126) — *Hot-reload a newer canonical checkpoint during no-grad inference.*
  * [`forward(self, obs_tuple)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_encoder.py#L154) — *obs_tuple: (speed, gear, rpm, images, [act1, act2])*
  * [`get_foundation_action(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/foundation_encoder.py#L287) — *Returns the most recently inferred foundation action tensor [B, 3] in [-1, 1].*


---

### [`tmrl\custom\torch\jax_dreamer_actor.py`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py)

**Purpose:** Torch/CUDA rollout mirror for a JAX-trained Dreamer actor.

#### Classes

* **[`TorchPortableEncoder(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L62)**
  * [`__init__(self, img_channels, img_height, img_width, latent_dim, channels)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L63)
  * [`forward_packed(self, telemetry, images)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L82)
  * [`forward_channel_first(self, telemetry, images)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L87)
  * [`forward(self, obs)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L99)
* **[`TorchPortableRSSM(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L104)**
  * [`__init__(self, latent_dim, action_dim, hidden_dim)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L105)
  * [`step_deterministic(self, previous_h, previous_z, action)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L113)
  * [`posterior(self, h, embedding, sample)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L118)
* **[`TorchPortablePolicy(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L132)**
  * [`__init__(self, feature_dim, hidden_dim, action_dim, action_scale)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L133)
  * [`forward(self, feature, test)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L144)
* **[`TorchPortableRolloutCore(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L157)**: One graph for the recurrent filter, encoder, and deployed policy.
  * [`__init__(self, encoder, rssm, policy, deterministic)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L160)
  * [`forward(self, telemetry, images, h, z, previous_action)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L167)
* **[`TorchJAXDreamerActor(TorchActorModule)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L182)**: Real-time Torch mirror loaded only from versioned JAX actor bundles.
  * [`__init__(self, observation_space, action_space, latent_dim, hidden_dim, policy_hidden_dim, img_channels, img_height, img_width, encoder_channels, realtime_cpu_tuning, realtime_cpu_threads, realtime_cpu_affinity_count, realtime_high_priority, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L185)
  * [`_configure_realtime_cpu_runtime(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L242) — *Best-effort worker tuning; only enabled by the production config.*
  * [`architecture_manifest(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L268)
  * [`reset(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L286)
  * [`_safe_action_result(self, action)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L291)
  * [`_compile_rollout_cores(self, telemetry, images)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L303)
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L333)
  * [`act_(self, obs, test)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L362) — *Fast batch-one collation for the real-time TrackMania worker.*
  * [`warmup(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L415)
  * [`_copy_linear(module, parameters, prefix)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L430)
  * [`_copy_conv(module, parameters, prefix)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L439)
  * [`_load_portable_parameters(self, parameters)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L448)
  * [`load(self, path, device)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L487)

#### Functions

* **[`_parameter_checksum(parameters)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L34)**
* **[`_validate_portable_parameters(parameters)`](file:///D:/Project/tmrl/tmrl/custom/torch/jax_dreamer_actor.py#L48)**


---

### [`tmrl\custom\torch\poincare_memory.py`](file:///D:/Project/tmrl/tmrl/custom/torch/poincare_memory.py)

**Purpose:** Small, dependency-free helpers for hyperbolic replay experiments.

#### Functions

* **[`replay_sampler_kwargs_from_config(dreamer_config)`](file:///D:/Project/tmrl/tmrl/custom/torch/poincare_memory.py#L27)**: Translate ``DREAMER.REPLAY_MEMORY`` into memory constructor kwargs.
* **[`poincare_expmap0(vectors, curvature, max_radius)`](file:///D:/Project/tmrl/tmrl/custom/torch/poincare_memory.py#L82)**: Map tangent vectors at the origin into a Poincare ball.
* **[`pairwise_euclidean_distance(points)`](file:///D:/Project/tmrl/tmrl/custom/torch/poincare_memory.py#L108)**: Return a stable all-pairs Euclidean distance matrix.
* **[`pairwise_poincare_distance(points, curvature)`](file:///D:/Project/tmrl/tmrl/custom/torch/poincare_memory.py#L117)**: Return all-pairs geodesic distances inside a Poincare ball.
* **[`greedy_diverse_subset(distance_matrix, count, rng, initial_count)`](file:///D:/Project/tmrl/tmrl/custom/torch/poincare_memory.py#L141)**: Choose a farthest-first subset, seeded by random uniform anchors.
* **[`mean_off_diagonal(distance_matrix)`](file:///D:/Project/tmrl/tmrl/custom/torch/poincare_memory.py#L172)**: Mean pairwise distance excluding the diagonal.


---

### [`tmrl\custom\torch\utils\__init__.py`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/__init__.py)

**Role:** Source module in `tmrl\custom\torch\utils`.


---

### [`tmrl\custom\torch\utils\nn.py`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py)

**Role:** Source module in `tmrl\custom\torch\utils`.

#### Classes

* **[`PopArt(Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L47)**: PopArt http://papers.nips.cc/paper/6076-learning-values-across-many-orders-of-magnitude
  * [`__init__(self, output_layer, beta, zero_debias, start_pop)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L49)
  * [`update(self, targets)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L65)
  * [`normalize(self, x)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L88)
  * [`unnormalize(self, x)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L91)
  * [`normalize_sum(self, s)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L94) — *normalize x.sum(1) preserving relative weightings between elements*
* **[`TanhNormal(Distribution)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L100)**: Distribution of X ~ tanh(Z) where Z ~ N(mean, std)
  * [`__init__(self, normal_mean, normal_std, epsilon)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L104)
  * [`log_prob(self, x)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L111)
  * [`sample(self, sample_shape)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L119)
  * [`rsample(self, sample_shape)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L125)
* **[`Independent(.Independent)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L133)**
  * [`sample_test(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L134)
* **[`TanhNormalLayer(.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L138)**
  * [`__init__(self, n, m)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L139)
  * [`forward(self, x)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L150)
* **[`RlkitLinear(.Linear)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L160)**
  * [`__init__(self)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L161)
* **[`SacLinear(.Linear)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L172)**
  * [`__init__(self, in_features, out_features)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L173)
* **[`BasicReLU(.Linear)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L180)**
  * [`forward(self, x)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L181)
* **[`AffineReLU(BasicReLU)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L186)**
  * [`__init__(self, in_features, out_features, init_weight_bound, init_bias)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L187)
* **[`NormalizedReLU(.Sequential)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L194)**
  * [`__init__(self, in_features, out_features, prenorm_bias)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L195)
* **[`KaimingReLU(.Linear)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L199)**
  * [`__init__(self, in_features, out_features)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L200)
  * [`forward(self, x)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L206)

#### Functions

* **[`detach(x)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L16)**
* **[`no_grad(model)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L23)**
* **[`exponential_moving_average(averages, values, factor)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L29)**
* **[`copy_shared(model_a)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L35)**: Create a deepcopy of a model but with the underlying state_dict shared. E.g. useful in combination with `no_grad`.
* **[`dqn_conv(n)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L218)**
* **[`big_conv(n)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L223)**
* **[`hd_conv(n)`](file:///D:/Project/tmrl/tmrl/custom/torch/utils/nn.py#L237)**


---

### [`tmrl\custom\torch\world_model.py`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py)

**Role:** Source module in `tmrl\custom\torch`.

#### Classes

* **[`TorchVisualTelemetryEncoder(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L20)**: PyTorch Visual-Telemetry Encoder:
  * [`__init__(self, img_channels, img_height, img_width, latent_dim)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L25)
  * [`forward(self, obs_tuple)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L65) — *obs_tuple: (speed, gear, rpm, images, [act1, act2])*
* **[`TorchLatentRSSM(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L108)**: Recurrent State Space Model:
  * [`__init__(self, latent_dim, action_dim, hidden_dim)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L113)
  * [`compute_prior(self, h)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L136)
  * [`compute_posterior(self, h, e)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L144)
  * [`step_deterministic(self, h, z, a)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L153)
* **[`TorchLatentWorldModel(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L158)**: Full Latent World Model: Encoder + RSSM + Prediction Heads
  * [`__init__(self, img_channels, img_height, img_width, latent_dim, action_dim, hidden_dim, policy_feature_dim)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L162)
  * [`get_feature(self, h, z)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L203)
  * [`predict_reward(self, feat)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L206)
  * [`predict_continuation(self, feat)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L209)
  * [`predict_embedding(self, feat)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L212)
  * [`predict_policy_feature(self, feat)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L215)
  * [`predict_policy_feature_symlog(self, feat)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L218)
  * [`imagine_step(self, h, z, a)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L221) — *Takes one imagined step in latent dynamics:*
* **[`TorchLatentAdversaryProposer(nn.Module)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L237)**: Experimental bounded latent-perturbation module.
  * [`__init__(self, feat_dim, perturbation_dim, max_magnitude)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L244)
  * [`forward(self, feat)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L254)
  * [`sample(self, feat)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L258) — *Sample a bounded proposal and return its matched log probability.*

#### Functions

* **[`symlog(x)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L10)**: Symmetric logarithmic transformation for scale-invariant regression.
* **[`symexp(x)`](file:///D:/Project/tmrl/tmrl/custom/torch/world_model.py#L15)**: Inverse symmetric exponential transformation with numerical clamping.


---

### [`tmrl\tools\__init__.py`](file:///D:/Project/tmrl/tmrl/tools/__init__.py)

**Role:** Source module in `tmrl\tools`.


---

### [`tmrl\tools\benchmark_continual_actor.py`](file:///D:/Project/tmrl/tmrl/tools/benchmark_continual_actor.py)

**Purpose:** Measure warmed single-observation JAX Dreamer rollout latency.

#### Functions

* **[`_write_json_atomic(path, payload)`](file:///D:/Project/tmrl/tmrl/tools/benchmark_continual_actor.py#L20)**
* **[`benchmark_actor(image_size, image_history, latent_dim, hidden_dim, policy_hidden_dim, encoder_channels, iterations, runtime, device, stochastic, realtime_cpu_tuning)`](file:///D:/Project/tmrl/tmrl/tools/benchmark_continual_actor.py#L44)**
* **[`main(argv)`](file:///D:/Project/tmrl/tmrl/tools/benchmark_continual_actor.py#L122)**


---

### [`tmrl\tools\check_environment.py`](file:///D:/Project/tmrl/tmrl/tools/check_environment.py)

**Role:** Source module in `tmrl\tools`.

#### Functions

* **[`check_env_tm20lidar()`](file:///D:/Project/tmrl/tmrl/tools/check_environment.py#L14)**
* **[`show_imgs(imgs, scale)`](file:///D:/Project/tmrl/tmrl/tools/check_environment.py#L34)**
* **[`check_env_tm20full()`](file:///D:/Project/tmrl/tmrl/tools/check_environment.py#L52)**


---

### [`tmrl\tools\continual_preflight.py`](file:///D:/Project/tmrl/tmrl/tools/continual_preflight.py)

**Purpose:** Standalone environment preflight for the continual JAX trainer.

#### Functions

* **[`_package_versions()`](file:///D:/Project/tmrl/tmrl/tools/continual_preflight.py#L29)**
* **[`_run_command(command, cwd)`](file:///D:/Project/tmrl/tmrl/tools/continual_preflight.py#L39)**
* **[`_git_state(repo_root)`](file:///D:/Project/tmrl/tmrl/tools/continual_preflight.py#L59)**
* **[`_is_wsl2()`](file:///D:/Project/tmrl/tmrl/tools/continual_preflight.py#L70)**
* **[`_tiny_nnx_train_step()`](file:///D:/Project/tmrl/tmrl/tools/continual_preflight.py#L78)**
* **[`collect_preflight(repo_root)`](file:///D:/Project/tmrl/tmrl/tools/continual_preflight.py#L124)**
* **[`_atomic_write_json(path, payload)`](file:///D:/Project/tmrl/tmrl/tools/continual_preflight.py#L168)**
* **[`main(argv)`](file:///D:/Project/tmrl/tmrl/tools/continual_preflight.py#L191)**


---

### [`tmrl\tools\freeze_continual_baseline.py`](file:///D:/Project/tmrl/tmrl/tools/freeze_continual_baseline.py)

**Purpose:** Create an immutable, checksummed baseline snapshot for continual learning.

#### Functions

* **[`_copy_and_hash(source, destination)`](file:///D:/Project/tmrl/tmrl/tools/freeze_continual_baseline.py#L21)**
* **[`_git_commit(repo_root)`](file:///D:/Project/tmrl/tmrl/tools/freeze_continual_baseline.py#L32)**
* **[`_atomic_json(path, payload)`](file:///D:/Project/tmrl/tmrl/tools/freeze_continual_baseline.py#L47)**
* **[`freeze_sources(sources, output_dir, repo_root)`](file:///D:/Project/tmrl/tmrl/tools/freeze_continual_baseline.py#L69)**
* **[`main(argv)`](file:///D:/Project/tmrl/tmrl/tools/freeze_continual_baseline.py#L128)**


---

### [`tmrl\tools\init_package\__init__.py`](file:///D:/Project/tmrl/tmrl/tools/init_package/__init__.py)

**Role:** Source module in `tmrl\tools\init_package`.


---

### [`tmrl\tools\init_package\init_pywin32.py`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py)

**Role:** Source module in `tmrl\tools\init_package`.

#### Classes

* **[`Tee`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L26)**
  * [`__init__(self, file)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L27)
  * [`write(self, what)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L30)
  * [`flush(self)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L38)

#### Functions

* **[`CopyTo(desc, src, dest)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L147)**
* **[`LoadSystemModule(lib_dir, modname)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L182)**
* **[`SetPyKeyVal(key_name, value_name, value)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L200)**
* **[`UnsetPyKeyVal(key_name, value_name, delete_key)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L215)**
* **[`RegisterCOMObjects(register)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L238)**
* **[`RegisterHelpFile(register, lib_dir)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L257)**
* **[`RegisterPythonwin(register, lib_dir)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L275)**: Add (or remove) Pythonwin to context menu for python scripts.
* **[`get_shortcuts_folder()`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L339)**
* **[`get_system_dir()`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L362)**
* **[`fixup_dbi()`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L380)**
* **[`install(lib_dir)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L407)**
* **[`uninstall(lib_dir)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L587)**
* **[`verify_destination(location)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L685)**
* **[`fix_pywin32()`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L691)**: Use API this to fix pywin32 programatically.
* **[`main()`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_pywin32.py#L701)**


---

### [`tmrl\tools\init_package\init_tmrl.py`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_tmrl.py)

**Role:** Source module in `tmrl\tools\init_package`.

#### Functions

* **[`rmdir(directory)`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_tmrl.py#L7)**
* **[`init_tmrl_data()`](file:///D:/Project/tmrl/tmrl/tools/init_package/init_tmrl.py#L17)**: Wipes and re-generates the TmrlData folder.


---

### [`tmrl\tools\prepare_continual_jax_config.py`](file:///D:/Project/tmrl/tmrl/tools/prepare_continual_jax_config.py)

**Purpose:** Create a separate CONTINUAL_DREAMER_JAX config without touching a live run.

#### Functions

* **[`build_config(source, run_name, scrub_wandb_key, trainer_server_ip)`](file:///D:/Project/tmrl/tmrl/tools/prepare_continual_jax_config.py#L16)**
* **[`write_json_atomic(path, payload, overwrite)`](file:///D:/Project/tmrl/tmrl/tools/prepare_continual_jax_config.py#L83)**
* **[`main(argv)`](file:///D:/Project/tmrl/tmrl/tools/prepare_continual_jax_config.py#L111)**


---

### [`tmrl\tools\record.py`](file:///D:/Project/tmrl/tmrl/tools/record.py)

**Role:** Source module in `tmrl\tools`.

#### Functions

* **[`record_reward_dist(path_reward, use_keyboard)`](file:///D:/Project/tmrl/tmrl/tools/record.py#L18)**
* **[`line(pt1, pt2, dist)`](file:///D:/Project/tmrl/tmrl/tools/record.py#L80)**: Creates a point between pt1 and pt2, at distance dist from pt1.


---

### [`tmrl\tools\save_replays.py`](file:///D:/Project/tmrl/tmrl/tools/save_replays.py)

**Role:** Source module in `tmrl\tools`.

#### Functions

* **[`save_replays(nb_replays)`](file:///D:/Project/tmrl/tmrl/tools/save_replays.py#L14)**


---

### [`tmrl\tools\validate_continual_map_manifest.py`](file:///D:/Project/tmrl/tmrl/tools/validate_continual_map_manifest.py)

**Purpose:** Validate and checksum the immutable continual-learning map split.

#### Functions

* **[`canonical_sha256(payload)`](file:///D:/Project/tmrl/tmrl/tools/validate_continual_map_manifest.py#L23)**
* **[`validate_manifest(payload, require_frozen)`](file:///D:/Project/tmrl/tmrl/tools/validate_continual_map_manifest.py#L33)**
* **[`main(argv)`](file:///D:/Project/tmrl/tmrl/tools/validate_continual_map_manifest.py#L79)**


---

### [`tmrl\tools\validate_continual_trainer_stack.py`](file:///D:/Project/tmrl/tmrl/tools/validate_continual_trainer_stack.py)

**Purpose:** Instantiate and serialize the configured JAX trainer without networking.

#### Functions

* **[`_sha256(path)`](file:///D:/Project/tmrl/tmrl/tools/validate_continual_trainer_stack.py#L16)**
* **[`_write_json_atomic(path, payload)`](file:///D:/Project/tmrl/tmrl/tools/validate_continual_trainer_stack.py#L24)**
* **[`validate(output_directory)`](file:///D:/Project/tmrl/tmrl/tools/validate_continual_trainer_stack.py#L48)**
* **[`main(argv)`](file:///D:/Project/tmrl/tmrl/tools/validate_continual_trainer_stack.py#L100)**


---
