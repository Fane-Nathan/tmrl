# rtgym interfaces for Trackmania

# standard library imports
import logging
import time
from collections import deque

# third-party imports
import cv2
import gymnasium.spaces as spaces
import numpy as np

# third-party imports
from rtgym import RealTimeGymInterface

# local imports
import tmrl.config.config_constants as cfg
from tmrl.custom.tm.utils.compute_reward import RewardFunction
from tmrl.custom.tm.utils.control_gamepad import control_gamepad, gamepad_reset, gamepad_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_mouse import mouse_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_keyboard import apply_control, keyres
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.custom.tm.utils.tools import Lidar, TM2020OpenPlanetClient, save_ghost

# Globals ==============================================================================================================

CHECK_FORWARD = 500  # this allows (and rewards) 50m cuts


# Interface for Trackmania 2020 ========================================================================================

class TM2020Interface(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control TrackMania 2020
    """
    def __init__(self,
                 img_hist_len: int = 4,
                 gamepad: bool = True,
                 save_replays: bool = False,
                 grayscale: bool = True,
                 resize_to=(64, 64)):
        """
        Base rtgym interface for TrackMania 2020 (Full environment)

        Args:
            img_hist_len: int: history of images that are part of observations
            gamepad: bool: whether to use a virtual gamepad for control
            save_replays: bool: whether to save TrackMania replays on successful episodes
            grayscale: bool: whether to output grayscale images or color images
            resize_to: Tuple[int, int]: resize output images to this (width, height)
        """
        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.img = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        self.j = None
        self.window_interface = None
        self.small_window = None
        self.save_replays = save_replays
        self.grayscale = grayscale
        self.resize_to = resize_to
        self.finish_reward = cfg.REWARD_CONFIG['END_OF_TRACK']
        self.constant_penalty = cfg.REWARD_CONFIG['CONSTANT_PENALTY']

        self.initialized = False

    def initialize_common(self):
        if self.gamepad:
            import vgamepad as vg
            self.j = vg.VX360Gamepad()
            logging.debug(" virtual joystick in use")
        self.window_interface = WindowInterface("Trackmania")
        self.window_interface.move_and_resize()
        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)
        self.img = None
        self.reward_function = RewardFunction(reward_data_path=cfg.REWARD_PATH,
                                              nb_obs_forward=cfg.REWARD_CONFIG['CHECK_FORWARD'],
                                              nb_obs_backward=cfg.REWARD_CONFIG['CHECK_BACKWARD'],
                                              nb_zero_rew_before_failure=cfg.REWARD_CONFIG['FAILURE_COUNTDOWN'],
                                              min_nb_steps_before_failure=cfg.REWARD_CONFIG['MIN_STEPS'],
                                              max_dist_from_traj=cfg.REWARD_CONFIG['MAX_STRAY'])
        self.client = TM2020OpenPlanetClient()

    def initialize(self):
        self.initialize_common()
        self.small_window = True
        self.initialized = True

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        If control is None, does nothing (e.g. to record)
        Args:
            control: np.array: [forward,backward,right,left]
        """
        if self.gamepad:
            if control is not None:
                control_gamepad(self.j, control)
        else:
            if control is not None:
                actions = []
                if control[0] > 0:
                    actions.append('f')
                if control[1] > 0:
                    actions.append('b')
                if control[2] > 0.5:
                    actions.append('r')
                elif control[2] < -0.5:
                    actions.append('l')
                apply_control(actions)

    def grab_data_and_img(self):
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        if self.resize_to is not None:  # cv2.resize takes dim as (width, height)
            img = cv2.resize(img, self.resize_to)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, ::-1]  # reversed view for numpy RGB convention
        data = self.client.retrieve_data()
        self.img = img  # for render()
        return data, img

    def reset_race(self):
        if self.gamepad:
            gamepad_reset(self.j)
        else:
            keyres()

    def reset_common(self):
        if not self.initialized:
            self.initialize()
        self.send_control(self.get_default_action())
        self.reset_race()
        time_sleep = max(0, cfg.SLEEP_TIME_AT_RESET - 0.1) if self.gamepad else cfg.SLEEP_TIME_AT_RESET
        time.sleep(time_sleep)  # must be long enough for image to be refreshed

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        self.reward_function.reset()
        return obs, {}

    def close_finish_pop_up_tm20(self):
        if self.gamepad:
            gamepad_close_finish_pop_up_tm20(self.j)
        else:
            mouse_close_finish_pop_up_tm20(small_window=self.small_window)

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        self.send_control(self.get_default_action())
        if self.save_replays:
            save_ghost()
            time.sleep(1.0)
        self.reset_race()
        time.sleep(0.5)
        self.close_finish_pop_up_tm20()

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            terminated = True
            rew += self.finish_reward
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=6, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))  # cv2 grayscale images are (h, w)
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))  # cv2 images are (h, w, c)
        return spaces.Tuple((speed, gear, rpm, img))

    def get_action_space(self):
        """
        must return a Box
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3, ))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0], dtype='float32')


class TM2020InterfaceLidar(TM2020Interface):
    def __init__(self, img_hist_len=1, gamepad=False, save_replays: bool = False):
        super().__init__(img_hist_len, gamepad, save_replays)
        self.window_interface = None
        self.lidar = None

    def grab_lidar_speed_and_data(self):
        img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([
            data[0],
        ], dtype='float32')
        lidar = self.lidar.lidar_20(img=img, show=False)
        return lidar, speed, data

    def initialize(self):
        super().initialize_common()
        self.small_window = False
        self.lidar = Lidar(self.window_interface.screenshot())
        self.initialized = True

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, imgs))


class TM2020InterfaceLidarProgress(TM2020InterfaceLidar):

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        progress = np.array([0], dtype='float32')
        obs = [speed, progress, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        progress = np.array([self.reward_function.cur_idx / self.reward_function.datalen], dtype='float32')
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, progress, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, progress, imgs))


# RL² wrappers ==========================================================================================================
#
# Per-episode meta-distribution randomization:
#   action_scale     ~ U(action_scale_low, action_scale_high) per action dim
#   action_noise_std ~ U(0, action_noise_max), Gaussian noise added per step
#   sensor_noise_std ~ U(0, sensor_noise_max), Gaussian noise on scalar obs per step
#   script_prefix_action: random throttle ∈ [0.3, 1.0], steer ∈ [-0.5, 0.5], fixed across prefix
#   script_prefix_len   ~ U(script_prefix_min, script_prefix_max + 1)
#
# Observation augmentation: each wrapper packs (a_prev, r_prev, d_prev) as the last
# 3 elements of the observation tuple. The RL² actor extracts these in act().
#
# Action convention: the action stored as "a_prev" (and seen by the policy in obs[-3])
# is the POLICY action (pre-scale, pre-noise), so the agent observes its emitted action.
# The env-side scale + noise are the meta-perturbations the agent must adapt to via
# the resulting obs and reward.
#
# Reward: zeroed during the scripted prefix; the reward function itself still ticks
# normally so the failure counters advance.


def _draw_meta_params(rng: np.random.Generator, act_dim: int,
                      action_scale_low: float, action_scale_high: float,
                      action_noise_max: float, sensor_noise_max: float,
                      script_prefix_min: int, script_prefix_max: int,
                      num_tasks: int = 1) -> dict:
    task_id = int(rng.integers(0, max(1, num_tasks)))
    profile = task_id % 5
    action_scale = np.ones(act_dim, dtype=np.float32)
    action_noise_std = 0.0
    sensor_noise_std = 0.0

    if num_tasks <= 1:
        action_scale = rng.uniform(action_scale_low, action_scale_high,
                                   size=act_dim).astype(np.float32)
        action_noise_std = float(rng.uniform(0.0, action_noise_max))
        sensor_noise_std = float(rng.uniform(0.0, sensor_noise_max))
    elif profile == 1:
        action_scale[-1] = np.float32(action_scale_low)
    elif profile == 2:
        action_scale[-1] = np.float32(action_scale_high)
    elif profile == 3:
        action_noise_std = float(action_noise_max)
    elif profile == 4:
        sensor_noise_std = float(sensor_noise_max)

    return {
        "task_id": task_id,
        "task_profile": profile,
        "action_scale": action_scale,
        "action_noise_std": action_noise_std,
        "sensor_noise_std": sensor_noise_std,
        "script_action": np.array([
            float(rng.uniform(0.3, 1.0)),   # throttle
            0.0,                             # no brake
            float(rng.uniform(-0.5, 0.5)),   # steer
        ], dtype=np.float32),
        "script_len": int(rng.integers(script_prefix_min, script_prefix_max + 1)),
    }


class _TM2020RL2Mixin:
    """Shared RL² logic. Concrete wrappers below mix this into specific TM2020 bases.

    The mixin assumes the subclass provides:
      - act_dim attribute (we initialize it in __init__)
      - _inner_observation_space(): the base interface's observation_space (unaugmented)
      - _inner_reset(seed, options): the base interface's reset (returns (obs, info))
      - _inner_get_obs(): the base interface's get_obs_rew_terminated_info
      - _inner_send_control(): the base interface's send_control
    """

    def _init_rl2(self, sensor_noise_max: float, action_noise_max: float,
                  action_scale_low: float, action_scale_high: float,
                  script_prefix_min: int, script_prefix_max: int,
                  apply_sensor_noise_to_images: bool = False,
                  num_tasks: int = 1):
        self.sensor_noise_max = sensor_noise_max
        self.action_noise_max = action_noise_max
        self.action_scale_low = action_scale_low
        self.action_scale_high = action_scale_high
        self.script_prefix_min = script_prefix_min
        self.script_prefix_max = script_prefix_max
        self.apply_sensor_noise_to_images = apply_sensor_noise_to_images
        self.num_tasks = max(1, int(num_tasks))
        self._rng = np.random.default_rng()
        self._meta: dict | None = None
        self._prefix_remaining = 0
        # Per-episode tracking
        self.act_dim = 3
        self._last_policy_action = np.zeros(self.act_dim, dtype=np.float32)
        self._last_reward = 0.0
        self._last_done = 1.0  # "just reset" signal for the first obs of an episode

    def _draw_meta(self) -> dict:
        return _draw_meta_params(
            self._rng, self.act_dim,
            self.action_scale_low, self.action_scale_high,
            self.action_noise_max, self.sensor_noise_max,
            self.script_prefix_min, self.script_prefix_max,
            self.num_tasks,
        )

    def _meta_reset(self):
        self._meta = self._draw_meta()
        self._prefix_remaining = self._meta["script_len"]
        self._last_policy_action = np.zeros(self.act_dim, dtype=np.float32)
        self._last_reward = 0.0
        self._last_done = 1.0

    def _augment_obs(self, obs_inner) -> tuple:
        return tuple(obs_inner) + (
            self._last_policy_action.astype(np.float32, copy=True),
            np.array([self._last_reward], dtype=np.float32),
            np.array([self._last_done], dtype=np.float32),
        )

    def _augment_observation_space(self, inner_space):
        a_prev = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,))
        r_prev = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        d_prev = spaces.Box(low=0.0, high=1.0, shape=(1,))
        return spaces.Tuple(tuple(inner_space.spaces) + (a_prev, r_prev, d_prev))

    def _with_task_info(self, info):
        info = dict(info or {})
        info["rl2_task_id"] = int(self._meta.get("task_id", 0))
        info["rl2_task_profile"] = int(self._meta.get("task_profile", 0))
        return info

    def _perturb_action(self, control):
        if self._prefix_remaining > 0:
            return self._meta["script_action"].copy()
        scale = self._meta["action_scale"]
        std = self._meta["action_noise_std"]
        ctrl = np.asarray(control, dtype=np.float32)
        noise = (self._rng.normal(0.0, std, size=ctrl.shape).astype(np.float32)
                 if std > 0 else 0.0)
        return np.clip(ctrl * scale + noise, -1.0, 1.0).astype(np.float32)

    def _apply_sensor_noise(self, scalar_arr):
        std = self._meta["sensor_noise_std"]
        if std <= 0:
            return scalar_arr
        return (scalar_arr + self._rng.normal(0.0, std, size=scalar_arr.shape)
                .astype(np.float32))


class TM2020InterfaceLidarRL2(_TM2020RL2Mixin, TM2020InterfaceLidarProgress):
    """RL² wrapper over the lidar-progress interface. Used for initial smoke testing."""

    def __init__(self, *args,
                 sensor_noise_max: float = 0.02,
                 action_noise_max: float = 0.05,
                 action_scale_low: float = 0.85, action_scale_high: float = 1.15,
                 script_prefix_min: int = 10, script_prefix_max: int = 40,
                 num_tasks: int = 1,
                 **kwargs):
        TM2020InterfaceLidarProgress.__init__(self, *args, **kwargs)
        self._init_rl2(sensor_noise_max, action_noise_max,
                       action_scale_low, action_scale_high,
                       script_prefix_min, script_prefix_max,
                       num_tasks=num_tasks)

    def reset(self, seed=None, options=None):
        self._meta_reset()
        obs_inner, info = TM2020InterfaceLidarProgress.reset(self, seed=seed, options=options)
        return self._augment_obs(obs_inner), self._with_task_info(info)

    def send_control(self, control):
        if control is None:
            TM2020InterfaceLidarProgress.send_control(self, control)
            return
        effective = self._perturb_action(control)
        self._last_policy_action = np.asarray(control, dtype=np.float32).copy()
        TM2020InterfaceLidarProgress.send_control(self, effective)

    def get_obs_rew_terminated_info(self):
        obs_inner, rew, terminated, info = \
            TM2020InterfaceLidarProgress.get_obs_rew_terminated_info(self)
        speed, progress, imgs = obs_inner
        speed = self._apply_sensor_noise(speed)
        progress = self._apply_sensor_noise(progress)
        if self.apply_sensor_noise_to_images:
            imgs = self._apply_sensor_noise(imgs)
        obs_inner = (speed, progress, imgs)
        if self._prefix_remaining > 0:
            rew = np.float32(0.0)
            self._prefix_remaining -= 1
        self._last_reward = float(rew)
        self._last_done = 1.0 if bool(terminated) else 0.0
        return self._augment_obs(obs_inner), rew, terminated, self._with_task_info(info)

    def get_observation_space(self):
        inner = TM2020InterfaceLidarProgress.get_observation_space(self)
        return self._augment_observation_space(inner)


class TM2020InterfaceRL2(_TM2020RL2Mixin, TM2020Interface):
    """RL² wrapper over the full-image interface. Production target after lidar verifies."""

    def __init__(self, *args,
                 sensor_noise_max: float = 0.02,
                 action_noise_max: float = 0.05,
                 action_scale_low: float = 0.85, action_scale_high: float = 1.15,
                 script_prefix_min: int = 10, script_prefix_max: int = 40,
                 num_tasks: int = 1,
                 **kwargs):
        TM2020Interface.__init__(self, *args, **kwargs)
        self._init_rl2(sensor_noise_max, action_noise_max,
                       action_scale_low, action_scale_high,
                       script_prefix_min, script_prefix_max,
                       num_tasks=num_tasks)

    def reset(self, seed=None, options=None):
        self._meta_reset()
        obs_inner, info = TM2020Interface.reset(self, seed=seed, options=options)
        return self._augment_obs(obs_inner), self._with_task_info(info)

    def send_control(self, control):
        if control is None:
            TM2020Interface.send_control(self, control)
            return
        effective = self._perturb_action(control)
        self._last_policy_action = np.asarray(control, dtype=np.float32).copy()
        TM2020Interface.send_control(self, effective)

    def get_obs_rew_terminated_info(self):
        obs_inner, rew, terminated, info = TM2020Interface.get_obs_rew_terminated_info(self)
        speed, gear, rpm, imgs = obs_inner
        speed = self._apply_sensor_noise(speed)
        gear = self._apply_sensor_noise(gear)
        rpm = self._apply_sensor_noise(rpm)
        if self.apply_sensor_noise_to_images:
            imgs = self._apply_sensor_noise(imgs.astype(np.float32)).astype(imgs.dtype, copy=False)
        obs_inner = (speed, gear, rpm, imgs)
        if self._prefix_remaining > 0:
            rew = np.float32(0.0)
            self._prefix_remaining -= 1
        self._last_reward = float(rew)
        self._last_done = 1.0 if bool(terminated) else 0.0
        return self._augment_obs(obs_inner), rew, terminated, self._with_task_info(info)

    def get_observation_space(self):
        inner = TM2020Interface.get_observation_space(self)
        return self._augment_observation_space(inner)


if __name__ == "__main__":
    pass
