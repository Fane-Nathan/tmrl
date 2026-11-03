# In-Context Reinforcement Learning & Sequence Replay Invariants

When designing, implementing, or debugging sequence models (Causal Transformers, RL^2, Decision Transformers) for real-time continuous control, robotics, or autonomous racing:

### 1. The Replay Distribution Invariant
> *"Every deterministic-policy failure traces to states that the training distribution did not contain."*
Never train a sequence agent on a window distribution that starves operational edge-cases:
* **Episode-Start Oversampling**: Always allocate a dedicated fraction (≥ 20–25%) of training windows to start strictly at $t = 0$ (the launch state). Uniform window sampling over long episodes starves launch states, causing the agent to stall or crash at deployment.
* **Terminal Inclusion**: Sliding windows must extend up to and include the episode terminal transition. True terminations must zero out the TD bootstrap ($V(s_{\text{term}}) = 0$), while truncations bootstrap from the true successor.
* **Short-Episode Masking**: Do not discard episodes shorter than the context window length. Right-pad and mask short episodes with causal attention masks.

### 2. The Shared-Trunk Stability Invariant
When actor and critic share a single sequence encoder/transformer trunk:
* **Feature-Maintaining Anchor**: Do not apply imitation/BC losses solely to a detached policy head. Backpropagate the imitation gradient through the hidden representations into the transformer encoder to prevent high-UTD critic gradients from repurposing trunk features.
* **RESeL Learning Rate Split**: Decouple optimization into two parameter groups. Set context encoder learning rate $\eta_{\text{context}} \le 0.1 \times \eta_{\text{heads}}$ (e.g., $5 \times 10^{-5}$ vs. $5 \times 10^{-4}$).
* **Layer-Normalized Critic Heads**: In shared-trunk REDQ/SAC architectures, ensemble critics are correlated. Add LayerNorm after each linear layer in critic heads (DroQ) to bound gradient magnitudes.
* **Detached Critic Gradients (AMAGO Formulation)**: In simultaneous joint loss updates ($\mathcal{L}_{\text{total}} = \lambda_{\text{TD}} \mathcal{L}_{\text{TD}} + \lambda_{\text{PG}} \mathcal{L}_{\text{PG}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}}$), explicitly stop gradients from $\mathcal{L}_{\text{PG}}$ into the critic parameters.

### 3. Competence vs. Adaptation Hierarchy
* **Memory Buys Adaptation**: Causal transformer context windows enable zero-gradient, in-context adaptation to unmodeled latency, surface friction shifts, and actuator faults.
* **Demonstrations Buy Competence**: In real-time physical control, exploration cannot outpace wall-clock time. Seed training with human demonstrations and maintain an RLPD-style pinned buffer (50% of each batch) to prevent policy collapse.
* **Foundation Pretraining Scaling (GEN-1.5 Principle)**: Past a critical threshold of multi-task physical pretraining data (e.g., 1M+ diverse community replays), adaptation becomes an emergent property of the context window rather than an online optimization bottleneck.
