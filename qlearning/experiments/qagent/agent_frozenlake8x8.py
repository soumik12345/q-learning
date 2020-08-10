from qlearning.agents.qagent import QAgent


def run_test():
    agent = QAgent(
        env_name='FrozenLake8x8-v0',
        hyper_parameters={
            'discount': 0.97,
            'learning_rate': 0.01,
            'epsilon': 1.0,
            'gamma': 0.999,
            'initialization_scale': 1e-4
        }
    )
    agent.train(episodes=800, render=True, log_metrics=False)
    train_total_reward = agent.total_reward
    test_total_reward = agent.act(
        iterations=200, render=True,
        collect_frames=False, log_metrics=False
    )
    print('Total reward in Training: {}'.format(train_total_reward))
    print('Total reward in Testing: {}'.format(test_total_reward))
    agent.close_env()
