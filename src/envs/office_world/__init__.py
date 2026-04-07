from gymnasium.envs.registration import register

register(
    id='OfficeWorldEnv-v0',
    entry_point='envs.office_world.office_env:OfficeWorldEnv',
    kwargs=dict(
        use_fixed_map=False,
    )
)

register(
    id='OfficeWorldEnv-v0.fixed',
    entry_point='envs.office_world.office_env:OfficeWorldEnv',
    kwargs=dict(
        use_fixed_map=True,
    )
)
