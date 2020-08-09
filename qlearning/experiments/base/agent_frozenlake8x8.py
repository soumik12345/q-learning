from gym.envs.registration import register
from qlearning.agents.base import BaseAgent


def run_test():
    # try:
    #     register(
    #         id='FrozenLakeNoSlip-v0',
    #         entry_point='gym.envs.toy_text:FrozenLakeEnv',
    #         kwargs={'map_name': '4x4', 'is_slippery': False},
    #         max_episode_steps=100,
    #         reward_threshold=0.78,
    #     )
    # except Exception as e:
    #     pass
    agent = BaseAgent(env_name='FrozenLake8x8-v0')
    print(agent.is_env_discrete)
    agent.act(iterations=200, render=True, collect_frames=False, log_metrics=False)
    agent.close_env()
