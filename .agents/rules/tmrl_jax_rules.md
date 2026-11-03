# TMRL JAX & Real-Time RL Development Guidelines

## 1. Flax NNX & PyTorch Architecture Parity
- **Convolution Padding**: Always explicitly specify `padding='VALID'` on `nnx.Conv` layers when matching PyTorch `nn.Conv2d` default behavior to preserve exact spatial output shapes calculated by `conv2d_out_dims`.
- **Channel Permutations**: JAX/Flax expects channels-last `(B, H, W, C)` layout. Maintain dynamic dimension transposition checks when consuming Gym observations formatted as `(B, C, H, W)`.
- **RNG State Management**: Use `rngs=rngs` for layer parameter initialization and `rngs.noise()` for stochastic actor sampling during JIT training.

## 2. Real-Time Execution Guardrails
- **Inference Latency**: Rollout worker policy forward passes (`act_()`) must execute in $< 2\text{ms}$ to respect the 20 Hz ($50\text{ms}$) RealTimeGym time-step boundary. Heavy sequence modeling, ODEs, or un-jitted operations must not block the main environment thread.
- **Cross-Platform String Safety**: Never use non-ASCII unicode symbols in logging statements to prevent Windows `cp1252` console encode crashes.
- **Safe Serialization**: Ensure `path.parent.mkdir(parents=True, exist_ok=True)` is called before writing temporary or checkpoint files.
