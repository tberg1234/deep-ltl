from gymnasium.envs.registration import register

register(
    id='RepoMan-v0',
    entry_point='envs.gym_repoman.envs:CollectEnv',
)
