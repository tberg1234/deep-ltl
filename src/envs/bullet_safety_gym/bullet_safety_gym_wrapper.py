from typing import Any, Optional

import numpy as np
import gymnasium
from gymnasium import spaces


class BulletSafetyGymWrapper(gymnasium.Env):
    """
    Wraps a bullet_safety_gym NavTask environment to the gymnasium API for LTL training.

    The inner env must be an EnvironmentBuilder instance (not wrapped in gym's TimeLimit).
    Propositions are the individual color/shape tokens (e.g. 'blue', 'box', 'sphere').
    When the agent touches an obstacle, the two tokens that make up its name become true
    (e.g. touching blue_box → {'blue', 'box'}).  This mirrors the repoman AP scheme.
    """

    metadata = {}

    def __init__(self, env, render_mode: Optional[str] = None,
                 randomize_agent: bool = True, randomize_objects: bool = True):
        super().__init__()
        self.env = env
        self.render_mode = render_mode
        self.randomize_agent = randomize_agent
        self.randomize_objects = randomize_objects
        self._fixed_obs_positions = None  # captured on first reset when randomize_objects=False

        task = env.task
        assert hasattr(task, 'get_collisions'), (
            "BulletSafetyGymWrapper only supports NavTask environments."
        )
        self._nav_task = task

        # Use dummy task setup so specific_reset() doesn't raise.
        # goals=[frozenset()] ensures goal_achieved is never True
        # (no obstacle has empty symbols), so the inner env never terminates.
        self._setup_nav_task()

        # Propositions are individual color/shape tokens.  Each obstacle's symbols
        # attribute already splits its name (e.g. blue_box → {'blue', 'box'}).
        all_tokens: set[str] = set()
        for obs in self._nav_task.obstacles:
            all_tokens |= obs.symbols
        self._propositions = sorted(all_tokens)

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
        """Return tokens of the first obstacle the agent is touching, or empty set."""
        for obs in self._nav_task.get_collisions():
            return set(obs.symbols)
        return set()

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

        needs_obs_refresh = False

        if not self.randomize_objects:
            if self._fixed_obs_positions is None:
                # Capture positions from the first reset and lock them in for all future resets.
                self._fixed_obs_positions = {ob.name: ob.get_position().copy()
                                              for ob in self._nav_task.obstacles}
                self._nav_task.set_locs(self._fixed_obs_positions)
            # Obstacles are already at the fixed positions (set_locs hooks into set_obstacles).
            # No position change here, so no obs refresh needed for objects alone.

        if self.randomize_agent:
            random_pos = self._nav_task.world.generate_random_xyz_position()
            random_pos[2] = self._nav_task.agent.init_xyz[2]
            self._nav_task.agent.set_position(random_pos)
            yaw = np.random.uniform(-np.pi, np.pi)
            quat = self._nav_task.bc.getQuaternionFromEuler([0, 0, yaw])
            self._nav_task.agent.set_orientation(quat)
            needs_obs_refresh = True

        if needs_obs_refresh:
            self._nav_task.bc.stepSimulation()
            obs = self.env.get_observation()

        return obs, {'propositions': set()}

    def get_propositions(self) -> list[str]:
        return self._propositions

    def get_possible_assignments(self) -> list:
        from ltl.logic import Assignment
        from sequence.samplers.bullet_sequence_samplers import _OBJECTS
        props = set(self._propositions)
        # Zero assignment (agent not touching anything) + one per object type
        assignments = [Assignment({p: False for p in props})]
        for shape, colour in _OBJECTS:
            assignments.append(Assignment({p: (p in {shape, colour}) for p in props}))
        return assignments

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
