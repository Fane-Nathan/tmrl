# third-party imports
import gymnasium

# local imports
from tmrl.core.wrappers import AffineObservationWrapper


__docformat__ = "google"


class GenericGymEnv(gymnasium.Wrapper):
    def __init__(self, id: str = "Pendulum-v1", gym_kwargs=None, obs_scale: float = 0., wrappers=None):
        """
        Use this wrapper when using the framework with arbitrary environments.

        Args:
            id (str): gymnasium environment id
            gym_kwargs (dict): keyword arguments passed to ``gymnasium.make``
            obs_scale (float): scalar factor used to rescale observations when non-zero
            wrappers (list): list of tuples ``(wrapper_cls, args, kwargs)``
        """
        if gym_kwargs is None:
            gym_kwargs = {}
        env = gymnasium.make(id, **gym_kwargs, disable_env_checker=True)
        if obs_scale:
            env = AffineObservationWrapper(env, 0, obs_scale)

        if wrappers is not None:
            for wrapper, args, kwargs in wrappers:
                env = wrapper(env, *args, **kwargs)

        super().__init__(env)
