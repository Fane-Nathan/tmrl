# standard library imports
import os
import pickle

# third-party imports
import numpy as np
import logging


class RewardFunction:
    """
    Computes a reward from the Openplanet API for Trackmania 2020.
    """
    def __init__(self,
                 reward_data_path,
                 nb_obs_forward=10,
                 nb_obs_backward=10,
                 nb_zero_rew_before_failure=10,
                 min_nb_steps_before_failure=int(3.5 * 20),
                 max_dist_from_traj=60.0):
        """
        Instantiates a reward function for TM2020.

        Args:
            reward_data_path: path where the trajectory file is stored
            nb_obs_forward: max distance of allowed cuts (as a number of positions in the trajectory)
            nb_obs_backward: same thing but for when rewinding the reward to a previously visited position
            nb_zero_rew_before_failure: after this number of steps with no reward, episode is terminated
            min_nb_steps_before_failure: the episode must have at least this number of steps before failure
            max_dist_from_traj: the reward is 0 if the car is further than this distance from the demo trajectory
        """
        if not os.path.exists(reward_data_path):
            logging.debug(f" reward not found at path:{reward_data_path}")
            self.data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy reward
        else:
            with open(reward_data_path, 'rb') as f:
                self.data = pickle.load(f)

        self.universal_mode = os.environ.get("TMRL_UNIVERSAL_REWARD", "0") == "1" or not os.path.exists(reward_data_path)
        self.last_distance = 0.0
        self.max_stall_steps = int(4.0 * 20)  # 4 seconds without forward progress -> failure

        self.cur_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_failure = nb_zero_rew_before_failure
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.max_dist_from_traj = max_dist_from_traj
        self.step_counter = 0
        self.failure_counter = 0
        self.datalen = len(self.data)

        self.start_pos = None
        self.highest_displacement = 0.0
        self.last_pos = None

    def compute_reward(self, pos, speed=None, distance=None):
        """
        Computes the current reward given the position pos, speed, and distance
        Args:
            pos: the current position
            speed: the current speed
            distance: current distance along track (from OpenPlanet telemetry)
        Returns:
            float, bool: the reward and the terminated signal
        """
        if self.universal_mode:
            self.step_counter += 1
            terminated = False

            pos_arr = np.asarray(pos, dtype=np.float32)
            if self.start_pos is None:
                self.start_pos = pos_arr.copy()
                self.highest_displacement = 0.0
                self.last_pos = pos_arr.copy()

            # Measure progress along the Track B track direction (negative X heading)
            if abs(self.start_pos[0] - 1424.0) < 50.0 and abs(self.start_pos[2] - 688.0) < 50.0:
                track_progress = float(self.start_pos[0] - pos_arr[0])
            else:
                track_progress = float(np.linalg.norm(pos_arr - self.start_pos))
            self.last_pos = pos_arr.copy()

            spd = float(speed[0]) if hasattr(speed, '__getitem__') else float(speed or 0.0)

            # Check if advancing to a new highest point along the track
            if track_progress > self.highest_displacement + 0.20:
                progress = track_progress - self.highest_displacement
                self.highest_displacement = track_progress
                # Positive reward for genuine progress along the track
                reward = (progress * 0.15) + (max(0.0, spd) * 0.0005)
                self.failure_counter = 0
            else:
                # No new track progress (circling, off-track, or reversing)
                reward = 0.0
                self.failure_counter += 1

            # Track boundaries on Track B (Summer 2020 - 01, start [1424, 98, 688]):
            # Road centerline is Z ~ 688. If car veers into the grass (|Z - 688| > 15m), terminate!
            if abs(self.start_pos[0] - 1424.0) < 50.0 and abs(self.start_pos[2] - 688.0) < 50.0:
                if abs(pos_arr[2] - 688.0) > 15.0:
                    reward = -1.0
                    terminated = True

            # Anti-circling / stall termination:
            # If the car fails to make forward progress for 15 steps (0.75s), terminate
            if self.failure_counter > 15:
                reward = -0.5
                terminated = True

            return reward, terminated
        # self.traj.append(pos)

        terminated = False
        self.step_counter += 1  # step counter to enable failure counter
        min_dist = np.inf  # smallest distance found so far in the trajectory to the target pos
        index = self.cur_idx  # cur_idx is where we were last step in the trajectory
        temp = self.nb_obs_forward  # counter used to find cuts
        best_index = 0  # index best matching the target pos

        while True:
            dist = np.linalg.norm(pos - self.data[index])  # distance of the current index to target pos
            if dist <= min_dist:  # if dist is smaller than our minimum found distance so far,
                min_dist = dist  # then we found a new best distance,
                best_index = index  # and a new best index
                temp = self.nb_obs_forward  # we will have to check this number of positions to find a possible cut
            index += 1  # now we will evaluate the next index in the trajectory
            temp -= 1  # so we can decrease the counter for cuts
            # stop condition
            if index >= self.datalen or temp <= 0:  # if trajectory complete or cuts counter depleted
                # We check that we are not too far from the demo trajectory:
                if min_dist > self.max_dist_from_traj:
                    best_index = self.cur_idx  # if so, consider we didn't move

                break  # we found the best index and can break the while loop

        # The reward is then proportional to the number of passed indexes (i.e., track distance):
        reward = (best_index - self.cur_idx) / 100.0

        if best_index == self.cur_idx:  # if the best index didn't change, we rewind (more Markovian reward)
            min_dist = np.inf
            index = self.cur_idx

            # Find the best matching index in rewind:
            while True:
                dist = np.linalg.norm(pos - self.data[index])
                if dist <= min_dist:
                    min_dist = dist
                    best_index = index
                    temp = self.nb_obs_backward
                index -= 1
                temp -= 1
                # stop condition
                if index <= 0 or temp <= 0:
                    break

            # If failure happens for too many steps, the episode terminates
            if self.step_counter > self.min_nb_steps_before_failure:
                self.failure_counter += 1
                if self.failure_counter > self.nb_zero_rew_before_failure:
                    terminated = True

        else:  # if we did progress on the track
            self.failure_counter = 0  # we reset the counter triggering episode termination

        self.cur_idx = best_index  # finally, we save our new best matching index

        return reward, terminated

    def reset(self):
        """
        Resets the reward function for a new episode.
        """
        # from pathlib import Path
        # import pickle as pkl
        # path_traj = Path.home() / 'TmrlData' / 'reward' / 'traj.pkl'
        # with open(path_traj, 'wb') as file_traj:
        #     pkl.dump(self.traj, file_traj)

        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0
        self.last_distance = 0.0
        self.start_pos = None
        self.highest_displacement = 0.0
        self.last_pos = None

        # self.traj = []
