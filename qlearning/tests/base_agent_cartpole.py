from ..agents import BaseAgent


def run_test():
    agent = BaseAgent(env_name='CartPole-v0')
    agent.act(iterations=200, render=True)
