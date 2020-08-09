import wandb
from qlearning.agents import NaiveCartPoleAgent


def run_test():
    agent = NaiveCartPoleAgent(env_name='CartPole-v0')
    frames = agent.act(iterations=200, render=True, log_metrics=False)
    wandb.log({"video": wandb.Video(frames, fps=10, format="mp4")})
    agent.close_env()
