from gym.envs.registration import register
from qlearning.agents.qagent import QAgent


def run_test():
    try:
        register(
            id='FrozenLakeNoSlip-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name': '4x4', 'is_slippery': False},
            max_episode_steps=100,
            reward_threshold=0.78,
        )
    except Exception as e:
        pass
    agent = QAgent(
        env_name='FrozenLakeNoSlip-v0',
        hyper_parameters={
            'discount': 0.97,
            'learning_rate': 0.01,
            'epsilon': 1.0,
            'gamma': 0.99,
            'initialization_scale': 1e-4
        }
    )
    agent.train(episodes=400, render=True, log_metrics=False)
    train_total_reward = agent.total_reward
    test_total_reward = agent.act(
        iterations=200, render=True,
        collect_frames=False, log_metrics=False
    )
    print('Total reward in Training: {}'.format(train_total_reward))
    print('Total reward in Testing: {}'.format(test_total_reward))
    agent.close_env()
