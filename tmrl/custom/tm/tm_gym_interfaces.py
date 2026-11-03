# rtgym interfaces for Trackmania

# standard library imports
import hashlib
import logging
import time
from collections import deque
from pathlib import Path

# third-party imports
import cv2
import gymnasium.spaces as spaces
import numpy as np

# third-party imports
from rtgym import RealTimeGymInterface

# local imports
import tmrl.config.config_constants as cfg
from tmrl.custom.tm.utils.compute_reward import RewardFunction
from tmrl.custom.tm.utils.control_gamepad import control_gamepad, gamepad_reset, gamepad_close_finish_pop_up_tm20, signed_trigger_controls
from tmrl.custom.tm.utils.control_mouse import mouse_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_keyboard import apply_control, keyres
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.custom.tm.utils.camera_preprocessing import preprocess_camera_frame
from tmrl.custom.tm.utils.tools import Lidar, TM2020OpenPlanetClient, save_ghost
from tmrl.custom.tm.utils.trajectory_assist import DemonstrationTrajectoryAssist

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
                 resize_to=(64, 64),
                 signed_analog_triggers: bool = False):
        """
        Base rtgym interface for TrackMania 2020 (Full environment)

        Args:
            img_hist_len: int: history of images that are part of observations
            gamepad: bool: whether to use a virtual gamepad for control
            save_replays: bool: whether to save TrackMania replays on successful episodes
            grayscale: bool: whether to output grayscale images or color images
            resize_to: Tuple[int, int]: resize output images to this (width, height)
            signed_analog_triggers: opt-in BC trigger decoding; legacy RL defaults are unchanged
        """
        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.img = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        if signed_analog_triggers and not gamepad:
            raise ValueError("Signed analog triggers require a gamepad")
        self.signed_analog_triggers = signed_analog_triggers
        self.j = None
        self.window_interface = None
        self.small_window = None
        self.save_replays = save_replays
        self.grayscale = grayscale
        self.resize_to = resize_to
        self.finish_reward = cfg.REWARD_CONFIG['END_OF_TRACK']
        self.constant_penalty = cfg.REWARD_CONFIG['CONSTANT_PENALTY']

        self.initialized = False
        self.latest_data = None
        self.trajectory_assist = None
        self._trajectory_assist_active = False
        self._skip_assist_controls = 0
        self.last_assist_info = {}

    def initialize_common(self):
        self.window_interface = WindowInterface("Trackmania")
        self.window_interface.move_and_resize()
        if self.gamepad:
            import vgamepad as vg
            self.j = vg.VX360Gamepad()
            logging.debug(" virtual joystick in use")
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
        self.auto_map_cycle = bool(cfg.TMRL_CONFIG.get("AUTO_MAP_CYCLE", True))
        self.episodes_per_map = int(cfg.TMRL_CONFIG.get("EPISODES_PER_MAP", 5))
        self.episodes_on_current_map = 0
        self.driving_safety = cfg.TMRL_CONFIG.get("DRIVING_SAFETY", {})
        assist_config = self.driving_safety.get("TRAJECTORY_ASSIST", {})
        if assist_config.get("ENABLED", False):
            raise RuntimeError(
                "Automatic trajectory assist is not supported in asynchronous RTGym: "
                "it can mislabel replay actions. Disable DRIVING_SAFETY.TRAJECTORY_ASSIST "
                "and use scripts/evaluate_live_driving.py --mode tracking for explicit, "
                "synchronously logged assisted driving."
            )
        self.trajectory_assist = DemonstrationTrajectoryAssist(assist_config)
        self._track_identity_logged = False

    @staticmethod
    def _sha256_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def validate_deployment_track(self, data):
        """Refuse control if the loaded map is not the demonstrated map."""
        safety = getattr(self, "driving_safety", {})
        if not safety or not bool(safety.get("ENABLED", False)):
            return

        track_id = safety.get("EXPECTED_TRACK_ID", "configured track")
        active_map_path = safety.get("ACTIVE_MAP_PATH")
        expected_map_sha256 = safety.get("EXPECTED_MAP_SHA256")
        if active_map_path and expected_map_sha256:
            active_map = Path(active_map_path)
            if not active_map.is_file():
                raise RuntimeError(
                    f"Driving safety gate: active map file is missing: {active_map}"
                )
            actual_sha256 = self._sha256_file(active_map)
            if actual_sha256.lower() != str(expected_map_sha256).lower():
                self.send_control(self.get_default_action())
                raise RuntimeError(
                    "Driving safety gate: active map file does not match "
                    f"{track_id} (expected {expected_map_sha256}, got {actual_sha256})"
                )

        expected_start = safety.get("EXPECTED_START_POSITION")
        if expected_start is not None:
            expected_start = np.asarray(expected_start, dtype=np.float32)
            if expected_start.shape != (3,):
                raise ValueError(
                    "DRIVING_SAFETY.EXPECTED_START_POSITION must contain [x, y, z]"
                )
            actual_start = np.asarray([data[2], data[3], data[4]], dtype=np.float32)
            tolerance = float(safety.get("START_TOLERANCE_METERS", 10.0))
            distance = float(np.linalg.norm(actual_start - expected_start))
            if not np.isfinite(distance) or distance > tolerance:
                self.send_control(self.get_default_action())
                raise RuntimeError(
                    "Driving safety gate: Trackmania has a different map cached. "
                    f"Expected {track_id} start {expected_start.tolist()}, got "
                    f"{actual_start.tolist()} ({distance:.1f} m away). Reload the "
                    "configured map before starting the worker."
                )
            if not self._track_identity_logged:
                logging.info(
                    "Driving safety gate accepted %s: start position is %.2f m "
                    "from the demonstrated start.",
                    track_id,
                    distance,
                )
                self._track_identity_logged = True

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
                if getattr(self, "signed_analog_triggers", False):
                    control_gamepad(self.j, signed_trigger_controls(control), digital_acceleration=False)
                else:
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
        try:
            img = self.window_interface.screenshot()  # retain contiguous BGR/BGRA until resize
            data = self.client.retrieve_data(timeout=0.5)
            if not np.isfinite(np.asarray(data, dtype=np.float64)).all():
                raise ValueError("OpenPlanet returned non-finite telemetry")
        except Exception:
            # Never leave the last throttle/steering command held while the
            # worker waits for a missing stream or an invalid capture window.
            self._trajectory_assist_active = False
            self.send_control(self.get_default_action())
            raise
        img = preprocess_camera_frame(img, self.resize_to, self.grayscale)
        self.latest_data = data
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
        self._trajectory_assist_active = False
        self._skip_assist_controls = 0
        self.last_assist_info = {}
        if self.trajectory_assist is not None:
            self.trajectory_assist.reset()
        self.send_control(self.get_default_action())
        self.episodes_on_current_map += 1

        if self.auto_map_cycle and self.episodes_on_current_map >= self.episodes_per_map:
            logging.info(
                f"[AutoMapCycler] Reached {self.episodes_on_current_map} episodes on current track. "
                f"Automatically cycling to next track..."
            )
            try:
                from scripts.cycle_map_macro import cycle_now
                cycle_now(strategy="active_file", gamepad=self.j if self.gamepad else None)
            except Exception as e:
                logging.error(f"[AutoMapCycler] Exception cycling map: {e}")
                self.reset_race()
            self.episodes_on_current_map = 0
        else:
            self.reset_race()

        time_sleep = max(0, cfg.SLEEP_TIME_AT_RESET - 0.1) if self.gamepad else cfg.SLEEP_TIME_AT_RESET
        time.sleep(time_sleep)  # must be long enough for image to be refreshed

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, img = self.grab_data_and_img()
        self.validate_deployment_track(data)
        if self.trajectory_assist is not None and self.trajectory_assist.enabled:
            self._trajectory_assist_active = True
            # RTGym launches one default-action thread immediately after reset.
            # Keep that reset action neutral, then assist policy actions.
            self._skip_assist_controls = 1
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
        self._trajectory_assist_active = False
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
        rew, terminated = self.reward_function.compute_reward(
            pos=np.array([data[2], data[3], data[4]]),
            speed=speed,
            distance=data[1],
        )
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        end_of_track = bool(data[8])
        info = {}
        if self.last_assist_info:
            info["trajectory_assist"] = dict(self.last_assist_info)
            if self.last_assist_info.get("emergency_stop", False):
                terminated = True
                info["termination_reason"] = "trajectory_assist_emergency_stop"
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
        neutral = [-1.0, -1.0, 0.0] if getattr(self, "signed_analog_triggers", False) else [0.0, 0.0, 0.0]
        return np.array(neutral, dtype='float32')


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
        rew, terminated = self.reward_function.compute_reward(
            pos=np.array([data[2], data[3], data[4]]),
            speed=speed,
            distance=data[1],
        )
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
        rew, terminated = self.reward_function.compute_reward(
            pos=np.array([data[2], data[3], data[4]]),
            speed=speed,
            distance=data[1],
        )
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


if __name__ == "__main__":
    pass
