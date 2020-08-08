from qlearning.agents import NaiveCartPoleAgent


def run_test():
    agent = NaiveCartPoleAgent(env_name='CartPole-v0')
    agent.act(iterations=200, render=True)
