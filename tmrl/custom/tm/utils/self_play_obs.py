import numpy as np
from typing import Optional
from tmrl.custom.tm.utils.self_play_reward import SelfPlayTrajectoryManager


class SelfPlayGhostObserver:
    """
    Computes comparative ghost observation features for Self-Play in TMRL.
    
    Provides relative state features comparing the agent's current position,
    velocity, and elapsed time against the active reference ghost.
    """
    def __init__(self, trajectory_manager: SelfPlayTrajectoryManager):
        self.manager = trajectory_manager

    def get_ghost_features(self,
                           pos: np.ndarray,
                           speed: float,
                           timestamp: float,
                           cur_idx: int) -> np.ndarray:
        """
        Computes 5-dimensional ghost delta features:
        [delta_x, delta_y, delta_z, delta_speed, delta_time]
        
        Args:
            pos: Current 3D position [x, y, z]
            speed: Current vehicle speed (float)
            timestamp: Elapsed episode time (float)
            cur_idx: Index of closest reference waypoint
            
        Returns:
            np.ndarray: Shape (5,), float32
        """
        if not self.manager.has_reference_ghost or cur_idx >= len(self.manager.best_trajectory):
            # No reference ghost available yet: return zeros
            return np.zeros(5, dtype=np.float32)

        ghost_pt = self.manager.best_trajectory[cur_idx]
        ghost_pos = ghost_pt[:3]
        ghost_speed = ghost_pt[3] if len(ghost_pt) > 3 else 0.0
        ghost_time = ghost_pt[4] if len(ghost_pt) > 4 else 0.0

        pos_delta = (pos - ghost_pos) / 50.0  # Normalized spatial offset
        speed_delta = (speed - ghost_speed) / 100.0  # Normalized velocity offset
        time_delta = np.clip((timestamp - ghost_time) / 10.0, -1.0, 1.0)  # Normalized split delta

        features = np.array([
            pos_delta[0],
            pos_delta[1],
            pos_delta[2],
            speed_delta,
            time_delta
        ], dtype=np.float32)

        return features
