from typing import Any, Optional

import numpy as np
import gymnasium
from gymnasium import spaces


class BulletSafetyGymWrapper(gymnasium.Env):
    """
    Wraps a bullet_safety_gym NavTask environment to the gymnasium API for LTL training.

    The inner env must be an EnvironmentBuilder instance (not wrapped in gym's TimeLimit).
    Propositions correspond to the named obstacles in the scene; a proposition is true
    whenever the agent is within detection distance of the corresponding obstacle.
    """

    metadata = {}

    def __init__(self, env, render_mode: Optional[str] = None):
        super().__init__()
        self.env = env
        self.render_mode = render_mode

        task = env.task
        assert hasattr(task, 'get_collisions'), (
            "BulletSafetyGymWrapper only supports NavTask environments."
        )
        self._nav_task = task

        # Use dummy task setup so specific_reset() doesn't raise.
        # goals=[frozenset()] ensures goal_achieved is never True
        # (no obstacle has empty symbols), so the inner env never terminates.
        self._setup_nav_task()

        # Propositions = obstacle names, already sorted in bases.Task.__init__
        self._propositions = [obs.name for obs in self._nav_task.obstacles]

        # Convert gym.spaces -> gymnasium.spaces
        gym_obs = env.observation_space
        self.observation_space = spaces.Box(
            low=gym_obs.low.copy(),
            high=gym_obs.high.copy(),
            dtype=gym_obs.dtype,
        )
        gym_act = env.action_space
        self.action_space = spaces.Box(
            low=gym_act.low.copy(),
            high=gym_act.high.copy(),
            dtype=gym_act.dtype,
        )

    def _setup_nav_task(self):
        """Configure NavTask with dummy values so it never self-terminates."""
        self._nav_task.set_task_goal(task=set(), goals=[frozenset()])
        self._nav_task.set_penalty(False)

    def _get_active_propositions(self) -> set:
        """Return propositions currently true (agent within detection distance)."""
        return {obs.name for obs in self._nav_task.get_collisions()}

    def step(self, action) -> tuple:
        obs, _reward, _done, info = self.env.step(action)
        info['propositions'] = self._get_active_propositions()
        # Always return terminated=False so inner NavTask goal never truncates LTL episode.
        return obs, 0.0, False, False, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # Re-apply dummy setup and clear stutter-free state before each episode.
        self._setup_nav_task()
        self._nav_task._curr_region_symbols = set()
        obs = self.env.reset()
        return obs, {'propositions': set()}

    def get_propositions(self) -> list[str]:
        return self._propositions

    def get_possible_assignments(self) -> list:
        from ltl.logic import Assignment
        return Assignment.zero_or_one_propositions(set(self._propositions))

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
