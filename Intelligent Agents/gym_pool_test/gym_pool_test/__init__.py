
from gym.envs.registration import register

register(
    id='pool-v0',
    entry_point='gym_pool_test.envs:Pool_test',
)
