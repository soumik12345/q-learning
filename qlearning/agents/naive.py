from .base import BaseAgent


class NaiveCartPoleAgent(BaseAgent):

    def __init__(self, env_name):
        super(NaiveCartPoleAgent, self).__init__(env_name)

    def discrete_action(self, state):
        pole_angle = state[2]
        return 0 if pole_angle < 0 else 1
