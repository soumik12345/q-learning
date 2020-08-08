import wandb
from qlearning.agents import BaseAgent


def run_test():
    agent = BaseAgent(env_name='CartPole-v0')
    frames = agent.act(iterations=200, render=True)
    wandb.log({"video": wandb.Video(frames, fps=10, format="mp4")})
    agent.close_env()
