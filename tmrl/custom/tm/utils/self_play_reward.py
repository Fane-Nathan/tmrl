import os
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


class SelfPlayTrajectoryManager:
    """
    Manages self-evolving reference trajectories for ghost-based self-imitation.
    
    Instead of relying on a pre-recorded human demonstration, the TrajectoryManager
    bootstraps from empty/zero-knowledge and autonomously updates the reference trajectory
    whenever the agent achieves a new best distance or faster lap time.
    """
    def __init__(self,
                 save_path: Optional[str] = None,
                 min_points_to_save: int = 10,
                 resample_step: float = 1.0):
        """
        Args:
            save_path: Path to save the best self-evolved trajectory (defaults to TmrlData/reward/self_play_reward.pkl)
            min_points_to_save: Minimum trajectory points before establishing a reference ghost
            resample_step: Spatial distance between resampled trajectory points
        """
        if save_path is None:
            reward_folder = Path.home() / "TmrlData" / "reward"
            reward_folder.mkdir(parents=True, exist_ok=True)
            self.save_path = str(reward_folder / "self_play_reward.pkl")
        else:
            self.save_path = save_path

        self.min_points_to_save = min_points_to_save
        self.resample_step = resample_step

        # Active best ghost trajectory: list/array of [x, y, z, speed, timestamp]
        self._loaded_best_distance = None
        self._loaded_best_time = None
        self._loaded_best_finished = None
        self.best_trajectory = self._load_trajectory()
        self.best_distance = (
            float(self._loaded_best_distance)
            if self._loaded_best_distance is not None
            else self._compute_progress_distance(self.best_trajectory)
        )
        self.best_time = (
            float(self._loaded_best_time)
            if self._loaded_best_time is not None
            else self._compute_duration(self.best_trajectory)
        )
        self.best_finished = bool(self._loaded_best_finished) if self._loaded_best_finished is not None else False

        # Current episode buffer: list of (x, y, z, speed, time)
        self.current_trajectory: List[np.ndarray] = []
        self.episode_count = 0
        self.ratchet_count = 0

    def _load_trajectory(self) -> np.ndarray:
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "trajectory" in data:
                    self._loaded_best_distance = data.get("best_distance")
                    self._loaded_best_time = data.get("best_time")
                    self._loaded_best_finished = data.get("best_finished")
                    data = data["trajectory"]
                if isinstance(data, list):
                    data = np.array(data, dtype=np.float32)
                if isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] >= 3:
                    data = data.astype(np.float32, copy=False)
                    logging.info(f"Self-Play TrajectoryManager: Loaded reference ghost ({len(data)} pts) from {self.save_path}")
                    return data
            except Exception as e:
                logging.warning(f"Self-Play TrajectoryManager: Failed to load existing trajectory: {e}")
        
        # Default empty trajectory
        return np.empty((0, 5), dtype=np.float32)

    def _compute_total_distance(self, traj: np.ndarray) -> float:
        if len(traj) < 2:
            return 0.0
        diffs = np.diff(traj[:, :3], axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def _compute_progress_distance(self, traj: np.ndarray) -> float:
        """Loop-resistant cold-start progress measured from the episode start."""
        if len(traj) < 2:
            return 0.0
        offsets = traj[:, :3] - traj[0, :3]
        return float(np.max(np.linalg.norm(offsets, axis=1)))

    @staticmethod
    def _compute_duration(traj: np.ndarray) -> float:
        if len(traj) < 2 or traj.shape[1] < 5:
            return np.inf
        duration = float(traj[-1, 4] - traj[0, 4])
        return duration if duration >= 0.0 else np.inf

    def _resample_trajectory(self, traj: np.ndarray) -> np.ndarray:
        """Resample the ghost at stable spatial intervals for index-based progress."""
        if len(traj) < 2 or self.resample_step <= 0.0:
            return traj
        segment_lengths = np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        keep = np.concatenate(([True], np.diff(cumulative) > 1e-6))
        cumulative = cumulative[keep]
        unique_traj = traj[keep]
        if len(unique_traj) < 2 or cumulative[-1] <= self.resample_step:
            return unique_traj.astype(np.float32, copy=False)
        targets = np.arange(0.0, cumulative[-1], self.resample_step, dtype=np.float32)
        if targets.size == 0 or targets[-1] < cumulative[-1]:
            targets = np.append(targets, cumulative[-1])
        columns = [np.interp(targets, cumulative, unique_traj[:, i]) for i in range(unique_traj.shape[1])]
        return np.stack(columns, axis=1).astype(np.float32)

    def record_step(self, pos: np.ndarray, speed: float = 0.0, timestamp: float = 0.0):
        """Records a single transition state into the current episode trajectory."""
        point = np.array([pos[0], pos[1], pos[2], speed, timestamp], dtype=np.float32)
        self.current_trajectory.append(point)

    def on_episode_end(self, finished_track: bool = False) -> bool:
        """
        Evaluates current episode against reference ghost.
        Ratchets the reference trajectory if a new record was set.
        
        Returns:
            bool: True if reference trajectory was updated (ratcheted), False otherwise.
        """
        self.episode_count += 1
        if len(self.current_trajectory) < self.min_points_to_save:
            self.current_trajectory = []
            return False

        curr_arr = np.array(self.current_trajectory, dtype=np.float32)
        curr_dist = self._compute_progress_distance(curr_arr)
        curr_time = self._compute_duration(curr_arr)

        is_new_best = False

        if not self.has_reference_ghost:
            # Cold start bootstrap
            is_new_best = True
        elif not finished_track and curr_dist > (self.best_distance + 5.0):
            # Explored further down the track
            is_new_best = True
        elif finished_track and (
            not self.best_finished
            or curr_time < self.best_time
            or not np.isfinite(self.best_time)
        ):
            # Finished track faster
            is_new_best = True

        if is_new_best:
            self.best_trajectory = self._resample_trajectory(curr_arr)
            self.best_distance = curr_dist
            self.best_time = curr_time
            self.best_finished = bool(finished_track)
            self.ratchet_count += 1
            self._save_trajectory()
            logging.info(
                f"* [Self-Play Ratchet #{self.ratchet_count}] New Ghost Record! "
                f"Dist: {curr_dist:.1f}m, Time: {curr_time:.2f}s (Episode #{self.episode_count})"
            )

        self.current_trajectory = []
        return is_new_best

    def _save_trajectory(self):
        try:
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, 'wb') as f:
                pickle.dump(
                    {
                        "version": 1,
                        "trajectory": self.best_trajectory,
                        "best_distance": self.best_distance,
                        "best_time": self.best_time,
                        "best_finished": self.best_finished,
                    },
                    f,
                    pickle.HIGHEST_PROTOCOL,
                )
        except Exception as e:
            logging.error(f"Self-Play TrajectoryManager: Error saving trajectory: {e}")

    @property
    def has_reference_ghost(self) -> bool:
        return len(self.best_trajectory) >= 2 and self.best_distance > 0.0

    def get_positions(self) -> np.ndarray:
        if len(self.best_trajectory) == 0:
            return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        return self.best_trajectory[:, :3]


class SelfPlayRewardFunction:
    """
    Self-play reward function implementing ghost ratcheting for TMRL.
    
    Provides:
    1. Spatial progress along self-discovered racing line.
    2. Pace differential reward relative to the active ghost (racing against oneself).
    3. Personal record exploration bonuses.
    4. Autonomous reference ghost ratcheting without external human data.
    """
    def __init__(self,
                 trajectory_manager: Optional[SelfPlayTrajectoryManager] = None,
                 reward_data_path: Optional[str] = None,
                 nb_obs_forward: int = 50,
                 nb_obs_backward: int = 10,
                 nb_zero_rew_before_failure: int = 15,
                 min_nb_steps_before_failure: int = int(3.5 * 20),
                 max_dist_from_traj: float = 80.0,
                 ghost_speed_weight: float = 0.15,
                 personal_record_bonus: float = 10.0):
        
        self.manager = trajectory_manager or SelfPlayTrajectoryManager(save_path=reward_data_path)
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_failure = nb_zero_rew_before_failure
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.max_dist_from_traj = max_dist_from_traj
        self.ghost_speed_weight = ghost_speed_weight
        self.personal_record_bonus = personal_record_bonus

        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0
        self.start_time = None
        self.highest_index_reached = 0
        self._finished_track = False

    @property
    def data(self) -> np.ndarray:
        return self.manager.get_positions()

    def compute_reward(self,
                       pos: np.ndarray,
                       speed: float = 0.0,
                       timestamp: float = 0.0,
                       finished_track: bool = False) -> Tuple[float, bool]:
        """
        Computes current reward and termination status.
        
        Args:
            pos: Current 3D position [x, y, z]
            speed: Current vehicle speed (float)
            timestamp: Current step timestamp (float)
            finished_track: Whether vehicle crossed finish line
            
        Returns:
            Tuple[float, bool]: (reward, terminated)
        """
        self.step_counter += 1
        self._finished_track = self._finished_track or bool(finished_track)
        self.manager.record_step(pos=pos, speed=speed, timestamp=timestamp)

        # 1. Cold-start exploration mode (no reference ghost established yet)
        if not self.manager.has_reference_ghost:
            # Reward forward speed and survival
            reward = float(np.clip(speed / 100.0, 0.0, 1.0)) * 0.20
            if speed < 5.0 and self.step_counter > 20:
                reward -= 0.08
            terminated = False
            if speed < 1.0 and self.step_counter > self.min_nb_steps_before_failure:
                self.failure_counter += 1
                if self.failure_counter > self.nb_zero_rew_before_failure:
                    terminated = True
            else:
                self.failure_counter = 0
            return reward, terminated

        # 2. Ghost-guided mode
        ref_data = self.data
        datalen = len(ref_data)
        min_dist = np.inf
        index = self.cur_idx
        temp = self.nb_obs_forward
        best_index = self.cur_idx

        while True:
            dist = np.linalg.norm(pos - ref_data[index])
            if dist <= min_dist:
                min_dist = dist
                best_index = index
                temp = self.nb_obs_forward
            index += 1
            temp -= 1
            if index >= datalen or temp <= 0:
                if min_dist > self.max_dist_from_traj:
                    best_index = self.cur_idx
                break

        # Spatial progress along reference ghost line
        progress = (best_index - self.cur_idx) / 100.0
        reward = progress

        # Racing velocity incentive (+0.10 at 100 km/h, +0.20 at 200 km/h)
        reward += float(np.clip(speed / 200.0, 0.0, 1.0)) * 0.20

        # Standstill penalty (ensures high speed strictly dominates sitting idle)
        if speed < 5.0 and self.step_counter > 20:
            reward -= 0.08

        # Pace differential bonus (are we moving faster than the ghost at this segment?)
        if len(self.manager.best_trajectory) > best_index and self.manager.best_trajectory.shape[1] > 3:
            ghost_speed = self.manager.best_trajectory[best_index, 3]
            speed_delta = (speed - ghost_speed) / 100.0
            reward += float(np.clip(speed_delta, -0.5, 0.5)) * self.ghost_speed_weight

        # Personal Record / Trailblazing Bonus (exploring past the previous best point)
        if best_index > self.highest_index_reached and best_index >= (datalen - 5):
            reward += self.personal_record_bonus
            self.highest_index_reached = best_index

        # Check termination
        terminated = False
        if best_index == self.cur_idx:
            min_dist = np.inf
            index = self.cur_idx
            while True:
                dist = np.linalg.norm(pos - ref_data[index])
                if dist <= min_dist:
                    min_dist = dist
                    best_index = index
                    temp = self.nb_obs_backward
                index -= 1
                temp -= 1
                if index <= 0 or temp <= 0:
                    break

            if self.step_counter > self.min_nb_steps_before_failure:
                self.failure_counter += 1
                if self.failure_counter > self.nb_zero_rew_before_failure:
                    terminated = True
        else:
            self.failure_counter = 0

        self.cur_idx = best_index
        return reward, terminated

    def reset(self, finished_track: Optional[bool] = None) -> bool:
        """
        Resets reward state and triggers self-play ratcheting evaluation.
        
        Returns:
            bool: True if a new ghost was established/ratcheted.
        """
        episode_finished = self._finished_track or bool(finished_track)
        ratcheted = self.manager.on_episode_end(finished_track=episode_finished)
        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0
        self.highest_index_reached = 0
        self._finished_track = False
        return ratcheted
