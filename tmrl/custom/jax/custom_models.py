from typing import Sequence
from math import floor

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from tmrl.core.util import prod

from tmrl.core.jax.util import get_rngs
from tmrl.core.jax.actor import NNXActorModule

import tmrl.config.config_constants as cfg


LOG_STD_MAX = 2
LOG_STD_MIN = -20


# TODO: set Rngs as class attributes for persistence (NNX side effects should be OK)


def mlp(
    sizes: Sequence[int],
    activation = nnx.relu,
    output_activation = nnx.identity,
    dropout: float | Sequence[float] = 0.0,
    layer_norm: bool | Sequence[bool] = False,
    rngs: nnx.Rngs | None = None,
) -> nnx.Sequential:
    
    rngs = rngs or get_rngs()

    layers = []

    if not isinstance(dropout, (list, tuple)):
        dropout = [dropout, ] * (len(sizes) - 1)
    if not isinstance(layer_norm, (list, tuple)):
        layer_norm = [layer_norm, ] * (len(sizes) - 1)
    if len(dropout) != (len(sizes) - 1) or len(layer_norm) != (len(sizes) - 1):
        raise RuntimeError(f"Invalid argument shapes. sizes:{len(sizes)}, dropout:{len(dropout)}, layer_norm:{len(layer_norm)}")
    for j in range(len(sizes) - 1):
        layers.append(nnx.Linear(sizes[j], sizes[j + 1], rngs=rngs))
        if dropout[j]:
            layers.append(nnx.Dropout(rate=dropout[j], rngs=rngs))
        if layer_norm[j]:
            layers.append(nnx.LayerNorm(sizes[j + 1], rngs=rngs))
        act = activation if j < len(sizes) - 2 else output_activation
        layers.append(act)

    return nnx.Sequential(*layers)


class NNXSquashedGaussianMLPActor(NNXActorModule):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nnx.relu,
                 layer_norm=False,
                 rngs: nnx.Rngs=None,
                 internal_rngs_seed: int=None):
        super().__init__(observation_space, action_space)

        rngs = rngs or get_rngs()
        self._rngs = get_rngs() if internal_rngs_seed is None else get_rngs(internal_rngs_seed, internal_rngs_seed+1, internal_rngs_seed+2)

        try:
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            dim_obs = prod(observation_space.shape)
            self.tuple_obs = False
        dim_act = action_space.shape[0]
        self.act_limit = action_space.high[0]
        self.net = mlp(sizes=[dim_obs] + list(hidden_sizes),
                       activation=activation,
                       output_activation=activation,
                       layer_norm=layer_norm,
                       rngs=rngs)
        self.mu_layer = nnx.Linear(hidden_sizes[-1], dim_act, rngs=rngs)
        self.log_std_layer = nnx.Linear(hidden_sizes[-1], dim_act, rngs=rngs)
    
    # jit-able:
    def __call__(self, obs, rngs: nnx.Rngs, test=False, with_logprob=True, epsilon=1e-8):
        """
        Note: this function assumes a batch dimension in obs.
        Obs can be either a simple batched tensor, or a collated tuple of batched tensors.
        """
        x = jnp.concatenate(obs, axis=-1) if self.tuple_obs else obs.reshape(obs.shape[0], -1)
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        # Pre-squash distribution and sample
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            eps = jax.random.normal(rngs.noise(), mu.shape)
            pi_action = mu + std * eps

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: explanation at https://github.com/openai/spinningup/issues/279
            # explicit log prob calculation
            logp_pi = (-0.5 * ((pi_action - mu) / (std + epsilon))**2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)).sum(axis=-1)
            logp_pi -= (2 * (jnp.log(2) - pi_action - jax.nn.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = jnp.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def act(self, obs, test=False):
        a, _ = self.__call__(obs=obs, rngs=self._rngs, test=test, with_logprob=False)
        res = np.array(a.squeeze())
        if not len(res.shape):
            res = np.expand_dims(res, 0)
        return res


class NNXMLPQFunction(nnx.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nnx.relu,
                 dropout=0.0,
                 layer_norm=False,
                 rngs: nnx.Rngs = None):
        
        rngs = rngs or get_rngs()

        try:
            obs_dim = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            obs_dim = prod(observation_space.shape)
            self.tuple_obs = False
        act_dim = action_space.shape[0]
        dropout_list = [dropout] * len(hidden_sizes) + [0.0]
        layer_norm_list = [layer_norm] * len(hidden_sizes) + [False]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation, dropout=dropout_list, layer_norm=layer_norm_list, rngs=rngs)

    def __call__(self, obs, act):
        x = jnp.concatenate((*obs, act), -1) if self.tuple_obs else jnp.concatenate((obs.reshape(obs.shape[0], -1), act), axis=-1)
        q = self.q(x)
        return q.squeeze(-1)


class NNXREDQMLPActorCritic(nnx.Module):
    """
    By default, this holds 2 critics for SAC.
    Set n to a higher value for REDQ-SAC.
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 n=2,
                 hidden_sizes=(256, 256),
                 activation=nnx.relu,
                 critic_dropout=0.0,
                 critic_layer_norm=False,
                 actor_layer_norm=False,
                 rngs: nnx.Rngs = None):
        
        rngs = rngs or get_rngs()
        self.n = n

        # build policy and value functions
        self.actor = NNXSquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation, layer_norm=actor_layer_norm, rngs=rngs)
        self.qs = nnx.List([
            NNXMLPQFunction(observation_space=observation_space, action_space=action_space, hidden_sizes=hidden_sizes, activation=activation, dropout=critic_dropout, layer_norm=critic_layer_norm, rngs=rngs)
            for _ in range(self.n)
        ])

# --- CNN helpers ---


def conv2d_out_dims(kernel_size: tuple[int, int], strides: tuple[int, int], h_in: int, w_in: int) -> tuple[int, int]:
    """
    Calculates output spatial dimensions for 2D convolution with VALID padding.
    """
    h_out = floor((h_in - kernel_size[0]) / strides[0] + 1)
    w_out = floor((w_in - kernel_size[1]) / strides[1] + 1)
    return h_out, w_out


# --- Vanilla CNN FOR GRAYSCALE IMAGES ---


class VanillaCNN(nnx.Module):
    """
    Vanilla CNN feature extractor and MLP fusion head in Flax NNX.
    Extracts features from stacked frames and concatenates vehicle telemetry.
    """
    def __init__(self,
                 q_net: bool = False,
                 img_height: int = cfg.IMG_HEIGHT,
                 img_width: int = cfg.IMG_WIDTH,
                 img_hist_len: int = cfg.IMG_HIST_LEN,
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 rngs: nnx.Rngs = None):
        super().__init__()
        rngs = rngs or get_rngs()
        self.q_net = q_net
        self.img_height = img_height
        self.img_width = img_width
        self.img_hist_len = img_hist_len
        h, w = img_height, img_width

        # NNX Conv: (in_features, out_features, kernel_size, strides) — channels-last by default
        self.conv1 = nnx.Conv(img_hist_len, 64, kernel_size=(8, 8), strides=(2, 2), padding='VALID', rngs=rngs)
        h, w = conv2d_out_dims((8, 8), (2, 2), h, w)
        self.conv2 = nnx.Conv(64, 64, kernel_size=(4, 4), strides=(2, 2), padding='VALID', rngs=rngs)
        h, w = conv2d_out_dims((4, 4), (2, 2), h, w)
        self.conv3 = nnx.Conv(64, 128, kernel_size=(4, 4), strides=(2, 2), padding='VALID', rngs=rngs)
        h, w = conv2d_out_dims((4, 4), (2, 2), h, w)
        self.conv4 = nnx.Conv(128, 128, kernel_size=(4, 4), strides=(2, 2), padding='VALID', rngs=rngs)
        h, w = conv2d_out_dims((4, 4), (2, 2), h, w)

        self.flat_features = 128 * h * w
        mlp_input = self.flat_features + 12 if q_net else self.flat_features + 9

        if q_net:
            mlp_sizes = [mlp_input, 256, 256, 1]
            drop = [dropout, dropout, 0.0]
            ln = [layer_norm, layer_norm, False]
        else:
            mlp_sizes = [mlp_input, 256, 256]
            drop = [dropout, dropout]
            ln = [layer_norm, layer_norm]

        self.mlp = mlp(mlp_sizes, nnx.relu, dropout=drop, layer_norm=ln, rngs=rngs)

    def __call__(self, x):
        if self.q_net:
            speed, gear, rpm, images, act1, act2, act = x
        else:
            speed, gear, rpm, images, act1, act2 = x

        # NNX Conv expects (batch, H, W, C); permute if images are (batch, C, H, W)
        if images.ndim == 4 and images.shape[1] == self.img_hist_len and images.shape[-1] != self.img_hist_len:
            images = jnp.transpose(images, (0, 2, 3, 1))

        x_img = nnx.relu(self.conv1(images))
        x_img = nnx.relu(self.conv2(x_img))
        x_img = nnx.relu(self.conv3(x_img))
        x_img = nnx.relu(self.conv4(x_img))
        x_img = x_img.reshape(x_img.shape[0], -1)

        if self.q_net:
            fused = jnp.concatenate([speed, gear, rpm, x_img, act1, act2, act], axis=-1)
        else:
            fused = jnp.concatenate([speed, gear, rpm, x_img, act1, act2], axis=-1)

        return self.mlp(fused)


class NNXSquashedGaussianVanillaCNNActor(NNXActorModule):
    """
    Squashed Gaussian CNN Actor Policy in Flax NNX.
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 layer_norm: bool = False,
                 rngs: nnx.Rngs = None,
                 internal_rngs_seed: int = None):
        super().__init__(observation_space, action_space)

        rngs = rngs or get_rngs()
        self._rngs = get_rngs() if internal_rngs_seed is None else get_rngs(internal_rngs_seed, internal_rngs_seed + 1, internal_rngs_seed + 2)

        dim_act = action_space.shape[0]
        self.act_limit = float(action_space.high[0])

        img_hist_len = cfg.IMG_HIST_LEN
        img_height = cfg.IMG_HEIGHT
        img_width = cfg.IMG_WIDTH
        if observation_space is not None and hasattr(observation_space, "__getitem__") and len(observation_space) > 3:
            img_shape = observation_space[3].shape
            if len(img_shape) == 3:
                if img_shape[0] in [1, 3, 4]:
                    img_hist_len, img_height, img_width = img_shape
                else:
                    img_height, img_width, img_hist_len = img_shape

        self.net = VanillaCNN(
            q_net=False,
            img_height=img_height,
            img_width=img_width,
            img_hist_len=img_hist_len,
            layer_norm=layer_norm,
            rngs=rngs,
        )
        self.mu_layer = nnx.Linear(256, dim_act, rngs=rngs)
        self.log_std_layer = nnx.Linear(256, dim_act, rngs=rngs)

    def __call__(self, obs, rngs: nnx.Rngs, test: bool = False, with_logprob: bool = True, epsilon: float = 1e-8):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        if test:
            pi_action = mu
        else:
            eps = jax.random.normal(rngs.noise(), mu.shape)
            pi_action = mu + std * eps

        if with_logprob:
            logp_pi = (-0.5 * ((pi_action - mu) / (std + epsilon))**2 - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(axis=-1)
            logp_pi -= (2 * (jnp.log(2) - pi_action - jax.nn.softplus(-2 * pi_action))).sum(axis=-1)
        else:
            logp_pi = None

        pi_action = jnp.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def act(self, obs, test: bool = False):
        a, _ = self.__call__(obs=obs, rngs=self._rngs, test=test, with_logprob=False)
        res = np.array(a.squeeze())
        if not len(res.shape):
            res = np.expand_dims(res, 0)
        return res


class NNXVanillaCNNQFunction(nnx.Module):
    """
    Vanilla CNN Critic Q-Network in Flax NNX.
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 rngs: nnx.Rngs = None):
        super().__init__()
        rngs = rngs or get_rngs()

        img_hist_len = cfg.IMG_HIST_LEN
        img_height = cfg.IMG_HEIGHT
        img_width = cfg.IMG_WIDTH
        if observation_space is not None and hasattr(observation_space, "__getitem__") and len(observation_space) > 3:
            img_shape = observation_space[3].shape
            if len(img_shape) == 3:
                if img_shape[0] in [1, 3, 4]:
                    img_hist_len, img_height, img_width = img_shape
                else:
                    img_height, img_width, img_hist_len = img_shape

        self.net = VanillaCNN(
            q_net=True,
            img_height=img_height,
            img_width=img_width,
            img_hist_len=img_hist_len,
            dropout=dropout,
            layer_norm=layer_norm,
            rngs=rngs,
        )

    def __call__(self, obs, act):
        x = (*obs, act)
        q = self.net(x)
        return q.squeeze(-1)


class NNXREDQVanillaCNNActorCritic(nnx.Module):
    """
    REDQ / SAC Ensemble Container for CNN Models.
    Holds 1 actor and n critics (default n=2 for standard SAC).
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 n: int = 2,
                 critic_dropout: float = 0.0,
                 critic_layer_norm: bool = False,
                 actor_layer_norm: bool = False,
                 rngs: nnx.Rngs = None):
        super().__init__()
        rngs = rngs or get_rngs()
        self.n = n

        self.actor = NNXSquashedGaussianVanillaCNNActor(
            observation_space, action_space, layer_norm=actor_layer_norm, rngs=rngs
        )
        self.qs = nnx.List([
            NNXVanillaCNNQFunction(
                observation_space=observation_space,
                action_space=action_space,
                dropout=critic_dropout,
                layer_norm=critic_layer_norm,
                rngs=rngs,
            )
            for _ in range(self.n)
        ])


# --- Vanilla CNN FOR COLOR IMAGES ---


def remove_colors(images):
    """
    Removes color channels so that grayscale-based models can be used.
    Assumes images have shape (..., C, 3) or (..., 3) where the last dimension is RGB.
    """
    return images[..., 0]


class NNXSquashedGaussianVanillaColorCNNActor(NNXSquashedGaussianVanillaCNNActor):
    def __call__(self, obs, rngs: nnx.Rngs, test: bool = False, with_logprob: bool = True, epsilon: float = 1e-8):
        speed, gear, rpm, images, act1, act2 = obs
        images = remove_colors(images)
        obs = (speed, gear, rpm, images, act1, act2)
        return super().__call__(obs, rngs=rngs, test=test, with_logprob=with_logprob, epsilon=epsilon)


class NNXVanillaColorCNNQFunction(NNXVanillaCNNQFunction):
    def __call__(self, obs, act):
        speed, gear, rpm, images, act1, act2 = obs
        images = remove_colors(images)
        obs = (speed, gear, rpm, images, act1, act2)
        return super().__call__(obs, act)


class NNXREDQVanillaColorCNNActorCritic(nnx.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 n: int = 2,
                 critic_dropout: float = 0.0,
                 critic_layer_norm: bool = False,
                 actor_layer_norm: bool = False,
                 rngs: nnx.Rngs = None):
        super().__init__()
        rngs = rngs or get_rngs()
        self.n = n
        self.actor = NNXSquashedGaussianVanillaColorCNNActor(
            observation_space, action_space, layer_norm=actor_layer_norm, rngs=rngs
        )
        self.qs = nnx.List([
            NNXVanillaColorCNNQFunction(
                observation_space=observation_space,
                action_space=action_space,
                dropout=critic_dropout,
                layer_norm=critic_layer_norm,
                rngs=rngs,
            )
            for _ in range(self.n)
        ])


if __name__ == "__main__":
    import gymnasium as gym

    # 1. Test MLP
    print("Testing MLP Actor-Critic on Pendulum-v1...")
    env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
    act_space = env.action_space
    obs_space = env.observation_space

    ac = NNXREDQMLPActorCritic(observation_space=obs_space, action_space=act_space)
    x = obs_space.sample()

    model = ac.actor
    model.train()
    y = model.act_(x)
    print("MLP act_ output shape:", y.shape)

    # 2. Test CNN
    print("\nTesting Vanilla CNN Actor-Critic...")
    batch_size = 4
    # Mock observation tuple matching TrackMania:
    # (speed: 1, gear: 1, rpm: 1, images: C x H x W, act1: 3, act2: 3)
    mock_obs = (
        jnp.zeros((batch_size, 1), dtype=jnp.float32),
        jnp.zeros((batch_size, 1), dtype=jnp.float32),
        jnp.zeros((batch_size, 1), dtype=jnp.float32),
        jnp.zeros((batch_size, cfg.IMG_HIST_LEN, cfg.IMG_HEIGHT, cfg.IMG_WIDTH), dtype=jnp.float32),
        jnp.zeros((batch_size, 3), dtype=jnp.float32),
        jnp.zeros((batch_size, 3), dtype=jnp.float32),
    )
    mock_act = jnp.zeros((batch_size, 3), dtype=jnp.float32)

    dummy_act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
    dummy_obs_space = gym.spaces.Tuple((
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        gym.spaces.Box(low=0.0, high=255.0, shape=(cfg.IMG_HIST_LEN, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)),
    ))

    cnn_ac = NNXREDQVanillaCNNActorCritic(
        observation_space=dummy_obs_space,
        action_space=dummy_act_space,
        n=2
    )

    rngs = get_rngs()
    action, logprob = cnn_ac.actor(mock_obs, rngs=rngs)
    print("CNN Actor action shape:", action.shape, "logprob shape:", logprob.shape)

    q1 = cnn_ac.qs[0](mock_obs, mock_act)
    print("CNN Critic Q1 shape:", q1.shape)

    # Test single-step inference act() for RolloutWorker
    single_obs = (
        np.zeros((1,), dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
        np.zeros((cfg.IMG_HIST_LEN, cfg.IMG_HEIGHT, cfg.IMG_WIDTH), dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
    )
    act_res = cnn_ac.actor.act_(single_obs)
    print("CNN Actor act_ output shape:", act_res.shape)

    print("\nAll tests passed successfully!")
