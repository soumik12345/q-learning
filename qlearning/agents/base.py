import os
import gym
import random
import numpy as np


class BaseAgent:

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.is_env_discrete = type(self.env.action_space) == gym.spaces.discrete.Discrete
        self.display_details()

    def display_details(self):
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

    def discrete_action(self, state):
        return random.choice(range(self.env.action_space.n))

    def continuos_action(self, state):
        return np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            self.env.action_space.shape
        )

    def get_action(self, state):
        if self.is_env_discrete:
            return self.discrete_action(state)
        return self.continuos_action(state)

    def act(self, iterations, render=True):
        state = self.env.reset()
        for _ in range(iterations):
            os.system('clear')
            if render:
                self.env.render()
            action = self.get_action(state)
            state, reward, done, info = self.env.step(action)
