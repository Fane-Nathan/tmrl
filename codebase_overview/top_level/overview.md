# Top-Level Root Modules

> Root package configuration, entry points, and setup scripts.

**Total Modules Documented:** 27

---

## Table of Contents

- [`docs\source\conf.py`](#docssourceconfpy)
- [`setup.py`](#setuppy)
- [`test_output_brain\reliability_20260912_v3\candidate.model.py`](#testoutputbrainreliability20260912v3candidatemodelpy)
- [`test_output_brain\reliability_20260912_v3\candidate.trainer.py`](#testoutputbrainreliability20260912v3candidatetrainerpy)
- [`test_output_brain\visual_policy_1k_20260911_v2\candidate.model.py`](#testoutputbrainvisualpolicy1k20260911v2candidatemodelpy)
- [`test_output_brain\visual_policy_1k_20260911_v2\candidate.trainer.py`](#testoutputbrainvisualpolicy1k20260911v2candidatetrainerpy)
- [`tests\test_camera_preprocessing.py`](#teststestcamerapreprocessingpy)
- [`tests\test_confirmatory_fault_benchmark.py`](#teststestconfirmatoryfaultbenchmarkpy)
- [`tests\test_driving_data.py`](#teststestdrivingdatapy)
- [`tests\test_driving_safety.py`](#teststestdrivingsafetypy)
- [`tests\test_foundation_observation_contract.py`](#teststestfoundationobservationcontractpy)
- [`tests\test_gamepad_action_contract.py`](#teststestgamepadactioncontractpy)
- [`tests\test_unseen_map_evaluation.py`](#teststestunseenmapevaluationpy)
- [`tests\test_visual_training.py`](#teststestvisualtrainingpy)
- [`tmp\finalize_sensor_pack.py`](#tmpfinalizesensorpackpy)
- [`tmp\pdfs\qa_sensor_paper.py`](#tmppdfsqasensorpaperpy)
- [`tutorial\competition\competition_eval.py`](#tutorialcompetitioncompetitionevalpy)
- [`tutorial\competition\custom_actor_module.py`](#tutorialcompetitioncustomactormodulepy)
- [`tutorial\debug_all_neural_networks.py`](#tutorialdebugallneuralnetworkspy)
- [`tutorial\test_world_model_imagination.py`](#tutorialtestworldmodelimaginationpy)
- [`tutorial\train_world_model_on_replay.py`](#tutorialtrainworldmodelonreplaypy)
- [`tutorial\tuto.py`](#tutorialtutopy)
- [`tutorial\tuto_envs\dummy_rc_drone_interface.py`](#tutorialtutoenvsdummyrcdroneinterfacepy)
- [`tutorial\tuto_minimal_drone.py`](#tutorialtutominimaldronepy)
- [`tutorial\tuto_minimal_pendulum_jax.py`](#tutorialtutominimalpendulumjaxpy)
- [`tutorial\tuto_minimal_pendulum_torch.py`](#tutorialtutominimalpendulumtorchpy)
- [`tutorial\tuto_self_play_jax.py`](#tutorialtutoselfplayjaxpy)

---
## Module Outlines

### [`docs\source\conf.py`](file:///D:/Project/tmrl/docs/source/conf.py)

**Role:** Source module in `docs\source`.


---

### [`setup.py`](file:///D:/Project/tmrl/setup.py)

**Role:** Source module in `.`.

#### Functions

* **[`url_retrieve(url, outfile, overwrite)`](file:///D:/Project/tmrl/setup.py#L27)**: Adapted from https://www.scivision.dev/python-switch-urlretrieve-requests-timeout/


---

### [`test_output_brain\reliability_20260912_v3\candidate.model.py`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py)

**Purpose:** TrackMania Causal Transformer "Car Brain" for In-Context Learning.

#### Classes

* **[`TimestepEmbedder(nn.Module)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L36)**: Embeds multi-modal timestep features [state_t, action_{t-1}, reward_{t-1}, done_{t-1}]
  * [`__init__(self, state_dim, action_dim, d_model, include_prev_action, include_prev_reward, include_prev_done)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L41)
  * [`forward(self, states, prev_actions, prev_rewards, prev_dones)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L73) — *Args:*
* **[`CausalSelfAttention(nn.Module)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L124)**: Multi-head causal self-attention with pre-norm and dropout.
  * [`__init__(self, d_model, n_heads, dropout)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L129)
  * [`forward(self, x, mask)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L142)
* **[`TransformerBlock(nn.Module)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L168)**: Pre-norm Transformer block with causal attention and MLP.
  * [`__init__(self, d_model, n_heads, d_ff, dropout)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L170)
  * [`forward(self, x, mask)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L183)
* **[`CarBrainPolicyHead(nn.Module)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L189)**: Maps transformer output tokens to continuous action predictions:
  * [`__init__(self, d_model, action_dim, hidden_dim)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L194)
  * [`forward(self, hidden_states)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L203)
* **[`CarBrain(nn.Module)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L208)**: Full TrackMania Car Brain:
  * [`__init__(self, state_dim, action_dim, d_model, n_layers, n_heads, d_ff, max_context_len, dropout)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L217)
  * [`forward(self, states, prev_actions, prev_rewards, prev_dones)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L260) — *Batch sequence forward pass (used during offline pretraining).*
  * [`reset_context(self)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L286) — *Clears the in-memory context history (call at episode resets).*
  * [`step_in_context(self, state, prev_action, prev_reward, prev_done, device)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L293) — *Real-time closed-loop action selection at 20 Hz.*
  * [`save_pretrained(self, save_dir, filename_prefix)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L357) — *Saves model weights in SafeTensors and PyTorch formats along with config.json.*
  * [`from_pretrained(cls, filepath, state_dim, action_dim, d_model, n_layers, n_heads, d_ff, max_context_len, device)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L392) — *Loads a CarBrain instance from .safetensors or .pt, auto-reading config.json if available.*
* **[`MultiModalCarBrain(nn.Module)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L448)**: Multi-Modal Foundation Brain for TrackMania 2020.
  * [`__init__(self, img_channels, img_size, state_dim, action_dim, d_model, n_layers, n_heads, d_ff, max_context_len, dropout)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L454)
  * [`encode_visual(self, imgs)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L521) — *imgs: (B, C, H, W) or (B, T, C, H, W) in [0, 1]*
  * [`forward(self, imgs, states, prev_actions, prev_rewards, prev_dones, return_logits)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L545) — *imgs: (B, T, 4, 96, 96)*
  * [`reset_context(self)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L589) — *Clears in-context history at lap start / reset.*
  * [`step_in_context(self, img, state, prev_action, device)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L595) — *Real-time closed loop step at 20 Hz.*
  * [`from_foundation(cls, foundation_weights_path, device)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py#L642) — *Constructs MultiModalCarBrain and initializes its Transformer backbone and policy head*


---

### [`test_output_brain\reliability_20260912_v3\candidate.trainer.py`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py)

**Purpose:** Offline Visual Robustness Fine-Tuner.

#### Classes

* **[`MultiTrackVisionSynthesizer`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L52)**: Applies appearance perturbations to recorded camera observations.
  * [`__init__(self, device)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L61)
  * [`augment_batch(self, imgs, domain)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L71) — *imgs: (B, T, 4, 96, 96) in [0.0, 1.0]*
* **[`CandidateCheckpoints`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L419)**: Clean held-out objective selects best; the last update is saved separately.
  * [`__init__(self, path)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L421)
  * [`consider(self, model, step, metrics)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L427)
  * [`save_latest(self, model, step, snapshot)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L439)

#### Functions

* **[`extract_causal_sequence_dataset(demo_file, window_len, stride, split, return_mask)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L141)**: Extracts rolling causal windows of images, states, previous actions, and target actions.
* **[`select_episode_split(episodes, split)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L256)**: Split before extracting windows; windows from a lap cannot cross splits.
* **[`save_candidate(model, output_path)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L273)**: Never hot-reload untested training weights into the active driver.
* **[`optimizer_parameter_groups(model, lr_vision, lr_backbone)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L290)**: Fail closed if a new trainable parameter has no deliberate LR assignment.
* **[`action_loss(predictions, targets, mask)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L315)**: Per-valid-token mean Huber over actions plus 2.5 times steering Huber.
* **[`trigger_logit_loss(logits, targets, mask)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L325)**: Signed trigger pressure is (tanh(z)+1)/2 == sigmoid(2*z).
* **[`evaluate_windows(model, data, device, batch_size, image_mode, seed, zero_previous_actions)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L340)**: All held-out windows, teacher-forced actions, no random augmentation.
* **[`file_sha256(path)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L403)**
* **[`write_metrics(path, values)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L411)**
* **[`gradient_group_norms(groups)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L445)**
* **[`parse_args(argv)`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L459)**
* **[`main()`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py#L507)**


---

### [`test_output_brain\visual_policy_1k_20260911_v2\candidate.model.py`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py)

**Purpose:** TrackMania Causal Transformer "Car Brain" for In-Context Learning.

#### Classes

* **[`TimestepEmbedder(nn.Module)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L36)**: Embeds multi-modal timestep features [state_t, action_{t-1}, reward_{t-1}, done_{t-1}]
  * [`__init__(self, state_dim, action_dim, d_model, include_prev_action, include_prev_reward, include_prev_done)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L41)
  * [`forward(self, states, prev_actions, prev_rewards, prev_dones)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L73) — *Args:*
* **[`CausalSelfAttention(nn.Module)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L124)**: Multi-head causal self-attention with pre-norm and dropout.
  * [`__init__(self, d_model, n_heads, dropout)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L129)
  * [`forward(self, x, mask)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L142)
* **[`TransformerBlock(nn.Module)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L168)**: Pre-norm Transformer block with causal attention and MLP.
  * [`__init__(self, d_model, n_heads, d_ff, dropout)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L170)
  * [`forward(self, x, mask)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L183)
* **[`CarBrainPolicyHead(nn.Module)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L189)**: Maps transformer output tokens to continuous action predictions:
  * [`__init__(self, d_model, action_dim, hidden_dim)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L194)
  * [`forward(self, hidden_states)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L203)
* **[`CarBrain(nn.Module)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L208)**: Full TrackMania Car Brain:
  * [`__init__(self, state_dim, action_dim, d_model, n_layers, n_heads, d_ff, max_context_len, dropout)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L217)
  * [`forward(self, states, prev_actions, prev_rewards, prev_dones)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L260) — *Batch sequence forward pass (used during offline pretraining).*
  * [`reset_context(self)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L286) — *Clears the in-memory context history (call at episode resets).*
  * [`step_in_context(self, state, prev_action, prev_reward, prev_done, device)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L293) — *Real-time closed-loop action selection at 20 Hz.*
  * [`save_pretrained(self, save_dir, filename_prefix)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L357) — *Saves model weights in SafeTensors and PyTorch formats along with config.json.*
  * [`from_pretrained(cls, filepath, state_dim, action_dim, d_model, n_layers, n_heads, d_ff, max_context_len, device)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L392) — *Loads a CarBrain instance from .safetensors or .pt, auto-reading config.json if available.*
* **[`MultiModalCarBrain(nn.Module)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L448)**: Multi-Modal Foundation Brain for TrackMania 2020.
  * [`__init__(self, img_channels, img_size, state_dim, action_dim, d_model, n_layers, n_heads, d_ff, max_context_len, dropout)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L454)
  * [`encode_visual(self, imgs)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L521) — *imgs: (B, C, H, W) or (B, T, C, H, W) in [0, 1]*
  * [`forward(self, imgs, states, prev_actions, prev_rewards, prev_dones, return_logits)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L545) — *imgs: (B, T, 4, 96, 96)*
  * [`reset_context(self)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L589) — *Clears in-context history at lap start / reset.*
  * [`step_in_context(self, img, state, prev_action, device)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L595) — *Real-time closed loop step at 20 Hz.*
  * [`from_foundation(cls, foundation_weights_path, device)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py#L642) — *Constructs MultiModalCarBrain and initializes its Transformer backbone and policy head*


---

### [`test_output_brain\visual_policy_1k_20260911_v2\candidate.trainer.py`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py)

**Purpose:** Offline Visual Robustness Fine-Tuner.

#### Classes

* **[`MultiTrackVisionSynthesizer`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L52)**: Applies appearance perturbations to recorded camera observations.
  * [`__init__(self, device)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L61)
  * [`augment_batch(self, imgs, domain)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L71) — *imgs: (B, T, 4, 96, 96) in [0.0, 1.0]*
* **[`CandidateCheckpoints`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L419)**: Clean held-out objective selects best; the last update is saved separately.
  * [`__init__(self, path)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L421)
  * [`consider(self, model, step, metrics)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L427)
  * [`save_latest(self, model, step, snapshot)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L439)

#### Functions

* **[`extract_causal_sequence_dataset(demo_file, window_len, stride, split, return_mask)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L141)**: Extracts rolling causal windows of images, states, previous actions, and target actions.
* **[`select_episode_split(episodes, split)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L256)**: Split before extracting windows; windows from a lap cannot cross splits.
* **[`save_candidate(model, output_path)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L273)**: Never hot-reload untested training weights into the active driver.
* **[`optimizer_parameter_groups(model, lr_vision, lr_backbone)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L290)**: Fail closed if a new trainable parameter has no deliberate LR assignment.
* **[`action_loss(predictions, targets, mask)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L315)**: Per-valid-token mean Huber over actions plus 2.5 times steering Huber.
* **[`trigger_logit_loss(logits, targets, mask)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L325)**: Signed trigger pressure is (tanh(z)+1)/2 == sigmoid(2*z).
* **[`evaluate_windows(model, data, device, batch_size, image_mode, seed, zero_previous_actions)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L340)**: All held-out windows, teacher-forced actions, no random augmentation.
* **[`file_sha256(path)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L403)**
* **[`write_metrics(path, values)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L411)**
* **[`gradient_group_norms(groups)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L445)**
* **[`parse_args(argv)`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L459)**
* **[`main()`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py#L507)**


---

### [`tests\test_camera_preprocessing.py`](file:///D:/Project/tmrl/tests/test_camera_preprocessing.py)

**Purpose:** Pixel parity between legacy and copy-free camera preprocessing.

#### Classes

* **[`CameraPreprocessingTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_camera_preprocessing.py#L14)**
  * [`test_preserved_capture_never_calls_window_resizer_on_reinitialization(self)`](file:///D:/Project/tmrl/tests/test_camera_preprocessing.py#L16)
  * [`test_bgra_keeps_exact_grayscale_and_rgb_pixels(self)`](file:///D:/Project/tmrl/tests/test_camera_preprocessing.py#L36)
  * [`test_three_channel_captures_also_work(self)`](file:///D:/Project/tmrl/tests/test_camera_preprocessing.py#L46)
  * [`test_invalid_frame_is_rejected(self)`](file:///D:/Project/tmrl/tests/test_camera_preprocessing.py#L51)


---

### [`tests\test_confirmatory_fault_benchmark.py`](file:///D:/Project/tmrl/tests/test_confirmatory_fault_benchmark.py)

**Role:** Source module in `tests`.

#### Functions

* **[`_episode(length, obs_dim, act_dim)`](file:///D:/Project/tmrl/tests/test_confirmatory_fault_benchmark.py#L13)**
* **[`test_sequence_replay_can_sample_final_terminal_window(monkeypatch)`](file:///D:/Project/tmrl/tests/test_confirmatory_fault_benchmark.py#L22)**
* **[`test_short_episode_padding_is_explicitly_masked(monkeypatch)`](file:///D:/Project/tmrl/tests/test_confirmatory_fault_benchmark.py#L38)**
* **[`test_history_blocked_final_query_cannot_read_prior_tokens()`](file:///D:/Project/tmrl/tests/test_confirmatory_fault_benchmark.py#L54)**
* **[`test_inference_context_never_exceeds_trained_window()`](file:///D:/Project/tmrl/tests/test_confirmatory_fault_benchmark.py#L71)**


---

### [`tests\test_driving_data.py`](file:///D:/Project/tmrl/tests/test_driving_data.py)

**Purpose:** Causal export and episode-isolated training regressions.

#### Classes

* **[`LiveExportTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L14)**
  * [`setUp(self)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L15)
  * [`tearDown(self)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L33)
  * [`load(self)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L36)
  * [`test_history_and_previous_action_never_look_ahead(self)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L42)
  * [`test_failed_run_cannot_be_exported_as_success(self)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L52)
  * [`test_command_before_observation_is_rejected(self)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L57)
  * [`test_long_sampling_gap_is_rejected(self)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L62)
  * [`test_missing_frames_are_rejected(self)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L67)
* **[`EpisodeSplitTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L73)**
  * [`test_explicit_splits_are_preserved(self)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L74)
  * [`test_legacy_data_holds_out_last_entire_episode(self)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L79)
  * [`test_partial_split_metadata_is_rejected(self)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L84)
  * [`test_last_window_retained_without_validation_leakage(self)`](file:///D:/Project/tmrl/tests/test_driving_data.py#L88)


---

### [`tests\test_driving_safety.py`](file:///D:/Project/tmrl/tests/test_driving_safety.py)

**Purpose:** Offline regression tests; these never connect to or control Trackmania.

#### Classes

* **[`TelemetryTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L16)**
  * [`setUp(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L17)
  * [`tearDown(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L25)
  * [`packet(value)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L31)
  * [`wait_for_packet(self, packet)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L34)
  * [`test_partial_packet_and_latest_packet_in_backlog(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L40)
  * [`test_disconnect_raises_instead_of_spinning(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L49)
  * [`test_missing_data_has_bounded_timeout(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L55)
  * [`test_stale_packet_is_not_returned(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L59)
  * [`test_close_is_idempotent_and_refuses_retrieval(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L68)
* **[`DrivingSafetyTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L75)**
  * [`test_capture_failure_releases_controls(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L76)
  * [`test_telemetry_failure_releases_controls(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L87)
* **[`TrajectoryTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L101)**
  * [`setUp(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L102)
  * [`tearDown(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L110)
  * [`make_controller(self, legacy)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L113)
  * [`test_legacy_mislabeled_speed_values_are_not_rescaled(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L118)
  * [`test_straight_path_accelerates_without_steering(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L121)
  * [`test_large_deviation_stops_acceleration(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L126)
  * [`test_progress_never_rewinds(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L133)
  * [`test_nonfinite_reference_is_rejected(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L142)
  * [`test_invalid_heading_filter_is_rejected(self)`](file:///D:/Project/tmrl/tests/test_driving_safety.py#L147)


---

### [`tests\test_foundation_observation_contract.py`](file:///D:/Project/tmrl/tests/test_foundation_observation_contract.py)

**Purpose:** Regression checks for the data actually supplied to the pretrained brain.

#### Classes

* **[`SpyKinematics(nn.Module)`](file:///D:/Project/tmrl/tests/test_foundation_observation_contract.py#L13)**
  * [`forward(self, x)`](file:///D:/Project/tmrl/tests/test_foundation_observation_contract.py#L14)
* **[`StubBrain(nn.Module)`](file:///D:/Project/tmrl/tests/test_foundation_observation_contract.py#L19)**
  * [`__init__(self)`](file:///D:/Project/tmrl/tests/test_foundation_observation_contract.py#L20)
  * [`encode_visual(self, images)`](file:///D:/Project/tmrl/tests/test_foundation_observation_contract.py#L25)
* **[`FoundationObservationContractTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_foundation_observation_contract.py#L29)**
  * [`test_latest_applied_action_conditions_foundation(self)`](file:///D:/Project/tmrl/tests/test_foundation_observation_contract.py#L30)


---

### [`tests\test_gamepad_action_contract.py`](file:///D:/Project/tmrl/tests/test_gamepad_action_contract.py)

**Purpose:** No hardware is created: verify signed BC controls and unchanged legacy transport.

#### Classes

* **[`GamepadContractTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_gamepad_action_contract.py#L11)**
  * [`test_signed_trigger_endpoints_and_midpoint(self)`](file:///D:/Project/tmrl/tests/test_gamepad_action_contract.py#L12)
  * [`test_invalid_signed_controls_are_rejected(self)`](file:///D:/Project/tmrl/tests/test_gamepad_action_contract.py#L16)
  * [`test_analog_pressure_does_not_press_digital_accelerator(self)`](file:///D:/Project/tmrl/tests/test_gamepad_action_contract.py#L21)
  * [`test_legacy_defaults_are_unchanged(self)`](file:///D:/Project/tmrl/tests/test_gamepad_action_contract.py#L30)
  * [`test_signed_interface_neutral_releases_both_triggers(self)`](file:///D:/Project/tmrl/tests/test_gamepad_action_contract.py#L40)
  * [`test_capture_failure_releases_signed_interface(self)`](file:///D:/Project/tmrl/tests/test_gamepad_action_contract.py#L50)


---

### [`tests\test_unseen_map_evaluation.py`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py)

**Purpose:** Unseen-map evaluation must not replay or score against the old route.

#### Classes

* **[`UnseenMapTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L13)**
  * [`test_signed_triggers_require_explicit_continuous_foundation(self)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L14)
  * [`test_preserve_window_size_is_explicit(self)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L21)
  * [`test_unseen_map_ignores_missing_old_reference(self)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L25)
  * [`test_unseen_map_cannot_use_reference_controller(self)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L29)
  * [`test_known_map_keeps_reference_configuration_unchanged(self)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L34)
  * [`test_motion_guard_allows_launch_then_stops_a_stationary_car(self)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L41)
  * [`test_motion_guard_does_not_consult_route_direction(self)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L47)
  * [`test_motion_guard_rejects_nonfinite_positions(self)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L53)
  * [`test_repeated_frame_is_allowed_while_stationary(self)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L57)
  * [`test_frozen_camera_is_rejected_during_motion(self)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L63)
  * [`test_changing_pixels_reset_camera_age(self)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L70)
  * [`test_deadline_guard_allows_jitter_not_persistent_slow_loop(self)`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py#L76)


---

### [`tests\test_visual_training.py`](file:///D:/Project/tmrl/tests/test_visual_training.py)

**Purpose:** Candidate-training regression checks independent of a running game/GPU.

#### Classes

* **[`ImagePolicy(nn.Module)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L20)**
  * [`forward(self, imgs, states, prev_actions)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L21)
* **[`OptimizerTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L25)**
  * [`test_all_parameters_covered_once_and_final_layernorm_updates(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L26)
  * [`test_unknown_parameter_is_not_silently_omitted(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L47)
  * [`test_saturated_wrong_trigger_has_corrective_logit_gradient(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L53)
  * [`test_optional_logits_preserve_inference_and_checkpoint_contract(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L65)
* **[`ValidationTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L77)**
  * [`setUp(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L78)
  * [`test_clean_and_ablated_scores_and_rng_restoration(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L86)
  * [`test_weighted_aggregation_matches_direct_calculation(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L98)
  * [`test_eval_mode_restored_after_nonfinite_prediction(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L110)
  * [`test_best_is_not_overwritten_by_worse_latest(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L119)
* **[`DatasetPaddingTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L137)**
  * [`test_short_episode_is_right_padded_masked_and_causally_equivalent(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L138)
  * [`test_quarter_of_training_windows_start_at_zero(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L157)
* **[`ArgumentTests(unittest.TestCase)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L169)**
  * [`test_invalid_counts_learning_rates_and_seed_rejected(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L170)
  * [`test_existing_and_input_checkpoint_overwrite_blocked(self)`](file:///D:/Project/tmrl/tests/test_visual_training.py#L176)


---

### [`tmp\finalize_sensor_pack.py`](file:///D:/Project/tmrl/tmp/finalize_sensor_pack.py)

**Purpose:** Finalize private preparation notes and a portable paper/results review package.

#### Functions

* **[`add_file(path, arc)`](file:///D:/Project/tmrl/tmp/finalize_sensor_pack.py#L183)**


---

### [`tmp\pdfs\qa_sensor_paper.py`](file:///D:/Project/tmrl/tmp/pdfs/qa_sensor_paper.py)

**Purpose:** Read-only structural checks for the anonymous sensor pilot manuscript.

#### Functions

* **[`inspect_font(font)`](file:///D:/Project/tmrl/tmp/pdfs/qa_sensor_paper.py#L31)**
* **[`inspect_resources(resources)`](file:///D:/Project/tmrl/tmp/pdfs/qa_sensor_paper.py#L44)**


---

### [`tutorial\competition\competition_eval.py`](file:///D:/Project/tmrl/tutorial/competition/competition_eval.py)

**Purpose:** This script is used to evaluate your submission to the TMRL competition.


---

### [`tutorial\competition\custom_actor_module.py`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py)

**Purpose:** ==============================================================================

#### Classes

* **[`VanillaCNN(nn.Module)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L296)**
  * [`__init__(self, q_net)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L297) — *Simple CNN (Convolutional Neural Network) model for SAC (Soft Actor-Critic).*
  * [`forward(self, x)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L339) — *In Pytorch, the forward function is where our neural network computes its output from its input.*
* **[`TorchJSONEncoder(json.JSONEncoder)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L401)**: Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
  * [`default(self, obj)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L405)
* **[`TorchJSONDecoder(json.JSONDecoder)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L411)**: Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
  * [`__init__(self)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L415)
  * [`object_hook(self, dct)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L418)
* **[`MyActorModule(TorchActorModule)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L425)**: Our policy wrapped in the TMRL ActorModule class.
  * [`__init__(self, observation_space, action_space)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L434) — *When implementing __init__, we need to take the observation_space and action_space arguments.*
  * [`save(self, path)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L457) — *JSON-serialize a detached copy of the ActorModule and save it in path.*
  * [`load(self, path, device)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L474) — *Load the parameters of your trained ActorModule from a JSON file.*
  * [`forward(self, obs, test, compute_logprob)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L495) — *Computes the output action of our policy from the input observation.*
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L547) — *Computes an action from an observation.*
* **[`VanillaCNNQFunction(nn.Module)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L573)**: Critic module for SAC.
  * [`__init__(self, observation_space, action_space)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L577)
  * [`forward(self, obs, act)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L581) — *Estimates the action-value of the (obs, act) state-action pair.*
* **[`VanillaCNNActorCritic(nn.Module)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L604)**: Actor-critic module for the SAC algorithm.
  * [`__init__(self, observation_space, action_space)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L608)
* **[`SACTrainingAgent(TrainingAgent)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L646)**: Our custom training algorithm (SAC in this tutorial).
  * [`__init__(self, observation_space, action_space, device, model_cls, gamma, polyak, alpha, lr_actor, lr_critic)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L664)
  * [`get_actor(self)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L694) — *Returns a copy of the current ActorModule.*
  * [`train(self, batch)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L705) — *Executes a training iteration from batched training samples (batches of RL transitions).*

#### Functions

* **[`mlp(sizes, activation, output_activation)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L260)**: A simple MLP (MultiLayer Perceptron).
* **[`num_flat_features(x)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L280)**
* **[`conv2d_out_dims(conv_layer, h_in, w_in)`](file:///D:/Project/tmrl/tutorial/competition/custom_actor_module.py#L289)**


---

### [`tutorial\debug_all_neural_networks.py`](file:///D:/Project/tmrl/tutorial/debug_all_neural_networks.py)

**Purpose:** Neural Network Smoke Diagnostics

#### Functions

* **[`test_section_header(title)`](file:///D:/Project/tmrl/tutorial/debug_all_neural_networks.py#L40)**
* **[`run_all_neural_diagnostics()`](file:///D:/Project/tmrl/tutorial/debug_all_neural_networks.py#L46)**


---

### [`tutorial\test_world_model_imagination.py`](file:///D:/Project/tmrl/tutorial/test_world_model_imagination.py)

**Role:** Source module in `tutorial`.

#### Functions

* **[`main()`](file:///D:/Project/tmrl/tutorial/test_world_model_imagination.py#L15)**


---

### [`tutorial\train_world_model_on_replay.py`](file:///D:/Project/tmrl/tutorial/train_world_model_on_replay.py)

**Role:** Source module in `tutorial`.

#### Functions

* **[`main()`](file:///D:/Project/tmrl/tutorial/train_world_model_on_replay.py#L17)**


---

### [`tutorial\tuto.py`](file:///D:/Project/tmrl/tutorial/tuto.py)

**Role:** Source module in `tutorial`.

#### Classes

* **[`MyActorModule(TorchActorModule)`](file:///D:/Project/tmrl/tutorial/tuto.py#L80)**: Directly adapted from the Spinup implementation of SAC
  * [`__init__(self, observation_space, action_space, hidden_sizes, activation)`](file:///D:/Project/tmrl/tutorial/tuto.py#L84)
  * [`forward(self, obs, test, with_logprob)`](file:///D:/Project/tmrl/tutorial/tuto.py#L94)
  * [`act(self, obs, test)`](file:///D:/Project/tmrl/tutorial/tuto.py#L115)
* **[`MyMemory(TorchMemory)`](file:///D:/Project/tmrl/tutorial/tuto.py#L229)**
  * [`__init__(self, act_buf_len, device, nb_steps, sample_preprocessor, memory_size, batch_size, dataset_path)`](file:///D:/Project/tmrl/tutorial/tuto.py#L230)
  * [`append_buffer(self, buffer)`](file:///D:/Project/tmrl/tutorial/tuto.py#L249) — *buffer.memory is a list of compressed (act_mod, new_obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod) samples*
  * [`__len__(self)`](file:///D:/Project/tmrl/tutorial/tuto.py#L307)
  * [`get_transition(self, item)`](file:///D:/Project/tmrl/tutorial/tuto.py#L316) — *Args:*
* **[`MyCriticModule(.Module)`](file:///D:/Project/tmrl/tutorial/tuto.py#L407)**
  * [`__init__(self, observation_space, action_space, hidden_sizes, activation)`](file:///D:/Project/tmrl/tutorial/tuto.py#L408)
  * [`forward(self, obs, act)`](file:///D:/Project/tmrl/tutorial/tuto.py#L414)
* **[`MyActorCriticModule(.Module)`](file:///D:/Project/tmrl/tutorial/tuto.py#L420)**
  * [`__init__(self, observation_space, action_space, hidden_sizes, activation)`](file:///D:/Project/tmrl/tutorial/tuto.py#L421)
* **[`MyTrainingAgent(TrainingAgent)`](file:///D:/Project/tmrl/tutorial/tuto.py#L431)**
  * [`__init__(self, observation_space, action_space, device, model_cls, gamma, polyak, alpha, lr_actor, lr_critic, lr_entropy, learn_entropy_coef, target_entropy)`](file:///D:/Project/tmrl/tutorial/tuto.py#L435)
  * [`get_actor(self)`](file:///D:/Project/tmrl/tutorial/tuto.py#L475)
  * [`train(self, batch)`](file:///D:/Project/tmrl/tutorial/tuto.py#L478)

#### Functions

* **[`mlp(sizes, activation, output_activation)`](file:///D:/Project/tmrl/tutorial/tuto.py#L72)**
* **[`my_sample_compressor(prev_act, obs, rew, terminated, truncated, info)`](file:///D:/Project/tmrl/tutorial/tuto.py#L126)**: Compresses samples before sending over network.
* **[`last_true_in_list(li)`](file:///D:/Project/tmrl/tutorial/tuto.py#L219)**: Returns the index of the last True element in list li, or None.
* **[`run_worker(worker)`](file:///D:/Project/tmrl/tutorial/tuto.py#L583)**
* **[`run_trainer(trainer)`](file:///D:/Project/tmrl/tutorial/tuto.py#L587)**


---

### [`tutorial\tuto_envs\dummy_rc_drone_interface.py`](file:///D:/Project/tmrl/tutorial/tuto_envs/dummy_rc_drone_interface.py)

**Purpose:** RealTimeGym interface used for the TMRL library tutorial.

#### Classes

* **[`DummyRCDroneInterface(RealTimeGymInterface)`](file:///D:/Project/tmrl/tutorial/tuto_envs/dummy_rc_drone_interface.py#L17)**
  * [`__init__(self)`](file:///D:/Project/tmrl/tutorial/tuto_envs/dummy_rc_drone_interface.py#L19)
  * [`_rendering_thread(self)`](file:///D:/Project/tmrl/tutorial/tuto_envs/dummy_rc_drone_interface.py#L26)
  * [`get_observation_space(self)`](file:///D:/Project/tmrl/tutorial/tuto_envs/dummy_rc_drone_interface.py#L32)
  * [`get_action_space(self)`](file:///D:/Project/tmrl/tutorial/tuto_envs/dummy_rc_drone_interface.py#L39)
  * [`get_default_action(self)`](file:///D:/Project/tmrl/tutorial/tuto_envs/dummy_rc_drone_interface.py#L42)
  * [`send_control(self, control)`](file:///D:/Project/tmrl/tutorial/tuto_envs/dummy_rc_drone_interface.py#L45)
  * [`reset(self, seed, options)`](file:///D:/Project/tmrl/tutorial/tuto_envs/dummy_rc_drone_interface.py#L50)
  * [`get_obs_rew_terminated_info(self)`](file:///D:/Project/tmrl/tutorial/tuto_envs/dummy_rc_drone_interface.py#L63)
  * [`wait(self)`](file:///D:/Project/tmrl/tutorial/tuto_envs/dummy_rc_drone_interface.py#L76)
  * [`render(self)`](file:///D:/Project/tmrl/tutorial/tuto_envs/dummy_rc_drone_interface.py#L79)


---

### [`tutorial\tuto_minimal_drone.py`](file:///D:/Project/tmrl/tutorial/tuto_minimal_drone.py)

**Purpose:** Tutorial: a minimal TMRL pipeline for real-time robots.

#### Functions

* **[`run_worker(worker)`](file:///D:/Project/tmrl/tutorial/tuto_minimal_drone.py#L225)**
* **[`run_trainer(trainer)`](file:///D:/Project/tmrl/tutorial/tuto_minimal_drone.py#L229)**


---

### [`tutorial\tuto_minimal_pendulum_jax.py`](file:///D:/Project/tmrl/tutorial/tuto_minimal_pendulum_jax.py)

**Purpose:** Tutorial: a minimal TMRL pipeline for non-real-time environments.

#### Functions

* **[`run_worker(worker)`](file:///D:/Project/tmrl/tutorial/tuto_minimal_pendulum_jax.py#L225)**
* **[`run_trainer(trainer)`](file:///D:/Project/tmrl/tutorial/tuto_minimal_pendulum_jax.py#L240)**


---

### [`tutorial\tuto_minimal_pendulum_torch.py`](file:///D:/Project/tmrl/tutorial/tuto_minimal_pendulum_torch.py)

**Purpose:** Tutorial: a minimal TMRL pipeline for non-real-time environments.

#### Functions

* **[`run_worker(worker)`](file:///D:/Project/tmrl/tutorial/tuto_minimal_pendulum_torch.py#L202)**
* **[`run_trainer(trainer)`](file:///D:/Project/tmrl/tutorial/tuto_minimal_pendulum_torch.py#L217)**


---

### [`tutorial\tuto_self_play_jax.py`](file:///D:/Project/tmrl/tutorial/tuto_self_play_jax.py)

**Role:** Source module in `tutorial`.

#### Functions

* **[`run_self_play_demo()`](file:///D:/Project/tmrl/tutorial/tuto_self_play_jax.py#L20)**


---
