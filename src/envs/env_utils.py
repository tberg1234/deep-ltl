from typing import Callable, Optional

import gymnasium
from gymnasium.wrappers import FlattenObservation, TimeLimit

from envs.remove_trunc_wrapper import RemoveTruncWrapper


def get_env_attr(env, attr: str):
    if hasattr(env, attr):
        return getattr(env, attr)
    if hasattr(env, 'env'):
        return getattr(env.unwrapped, attr)
    else:
        raise AttributeError(f'Attribute {attr} not found in env.')


def make_env(
        name: str,
        sampler: Callable[[list[str]], Callable],
        max_steps: Optional[int] = None,
        render_mode: str | None = None,
        sequence=False,
        randomize_agent: bool = True,
        randomize_objects: bool = True,
        **env_kwargs,
):
    from envs.seq_wrapper import SequenceWrapper
    from envs.ldba_wrapper import LDBAWrapper
    from envs.ltl_wrapper import LTLWrapper

    if is_safety_gym_env(name):
        env = make_safety_gym_env(name, render_mode)
        max_steps = max_steps or 1000
    elif is_bullet_safety_gym_env(name):
        env = make_bullet_safety_gym_env(name, render_mode, randomize_agent=randomize_agent,
                                         randomize_objects=randomize_objects)
        max_steps = max_steps or 1000
    elif name.startswith('Letter'):
        env = make_letter_env(name, render_mode)
        max_steps = max_steps or 75
    elif name.startswith('OfficeWorld'):
        env = make_office_env(name, render_mode)
        max_steps = max_steps or 200
    elif name.startswith('FlatWorld'):
        env = make_flatworld_env(name)
        max_steps = max_steps or 500
    elif name.startswith('RepoMan'):
        env = make_repoman_env(name, render_mode,
                               randomize_agent=randomize_agent,
                               randomize_objects=randomize_objects,
                               **env_kwargs)
        max_steps = max_steps or 200
    else:
        raise ValueError(f'Unknown environment: {name}')

    propositions = get_env_attr(env, 'get_propositions')()
    sample_task = sampler(propositions)
    if not sequence:
        # env = PartiallyOrderedWrapper(env, sample_task)
        env = LTLWrapper(env, sample_task)
        env = LDBAWrapper(env)
    else:
        env = SequenceWrapper(env, sample_task)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = RemoveTruncWrapper(env)
    return env


def is_safety_gym_env(name: str) -> bool:
    return any([name.startswith(agent_name) for agent_name in ['Point', 'Car', 'Racecar', 'Doggo', 'Ant']])


def is_bullet_safety_gym_env(name: str) -> bool:
    return name in BULLET_SAFETY_GYM_CONFIGS


def make_safety_gym_env(name: str, render_mode: str | None = None):
    # noinspection PyUnresolvedReferences
    import safety_gymnasium
    from envs.zones.safety_gym_wrapper import SafetyGymWrapper

    env = safety_gymnasium.make(name, render_mode=render_mode)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)
    return env


def make_letter_env(name: str, render_mode: str | None = None):
    import envs.letter_world

    env = gymnasium.make(name, render_mode=render_mode)
    return env


def make_office_env(name: str, render_mode: str | None = None):
    import envs.office_world

    env = gymnasium.make(name, render_mode=render_mode)
    return env


def make_flatworld_env(name: str):
    import envs.flatworld

    env = gymnasium.make(name)
    return env


def make_repoman_env(name: str, render_mode: str | None = None,
                    randomize_agent: bool = True, randomize_objects: bool = True,
                    **env_kwargs):
    import envs.gym_repoman

    env = gymnasium.make(name, render_mode=render_mode,
                         randomize_agent=randomize_agent,
                         randomize_objects=randomize_objects,
                         **env_kwargs)
    return env


# Configurations for supported bullet_safety_gym environments.
# Each entry mirrors the kwargs from bullet_safety_gym/__init__.py registration.
BULLET_SAFETY_GYM_CONFIGS: dict[str, dict] = {
    'SafetyBallNav-v0': dict(
        agent='Ball',
        task='NavTask',
        obstacles={
            'BlueBox': {'number': 1, 'fixed_base': True, 'movement': 'static'},
            'PurpleBox': {'number': 1, 'fixed_base': True, 'movement': 'static'},
            'BeigeBox': {'number': 1, 'fixed_base': True, 'movement': 'static'},
            'BlueSphere': {'number': 1, 'fixed_base': True, 'movement': 'static'},
            'PurpleSphere': {'number': 1, 'fixed_base': True, 'movement': 'static'},
            'BeigeSphere': {'number': 1, 'fixed_base': True, 'movement': 'static'},
        },
        world={'name': 'SmallRoom', 'factor': 1},
    ),
}


def make_bullet_safety_gym_env(name: str, render_mode: str | None = None,
                               randomize_agent: bool = True, randomize_objects: bool = True):
    from bullet_safety_gym.envs.builder import EnvironmentBuilder
    from bullet_safety_gym.bullet_safety_gym_wrapper import BulletSafetyGymWrapper

    if name not in BULLET_SAFETY_GYM_CONFIGS:
        raise ValueError(f'Unknown bullet_safety_gym environment: {name}')

    kwargs = BULLET_SAFETY_GYM_CONFIGS[name].copy()
    kwargs['graphics'] = (render_mode == 'human')
    env = EnvironmentBuilder(**kwargs)
    env = BulletSafetyGymWrapper(env, render_mode=render_mode,
                                  randomize_agent=randomize_agent, randomize_objects=randomize_objects)
    return env
