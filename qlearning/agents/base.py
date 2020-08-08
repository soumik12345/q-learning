import gym
import random
import numpy as np


class BaseAgent:

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.is_env_discrete = type(self.env.action_space) == gym.spaces.discrete.Discrete
        self._display_details()

    def _display_details(self):
        print(
            'Observation Space Range: ({}, {})'.format(
                self.env.observation_space.high, self.env.observation_space.high
            )
        )
        if self.is_env_discrete:
            print('Number of Actions:', self.env.action_space.n)
        else:
            print(
                'Action Space Range: ({}, {})'.format(
                    self.env.action_space.low, self.env.action_space.high
                )
            )

    def _discrete_action(self):
        return random.choice(range(self.env.action_space.n))

    def _continuos_action(self):
        return np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            self.env.action_space.shape
        )

    def get_action(self, state):
        if self.is_env_discrete:
            return self._discrete_action()
        return self._continuos_action()

    def act(self, iterations, render=True):
        state = self.env.reset()
        for _ in range(iterations):
            action = self.get_action(state)
            state, reward, done, info = self.env.step(action)
            if render:
                self.env.render()
            if done:
                break
