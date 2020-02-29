from gym.envs.registration import register

register(
    id='Mosaic-v0',
    entry_point='ai.gym.envs:MosaicEnv',
    kwargs={
        'size': 7,
        'done_on_illegal_move': True,
    },
)
