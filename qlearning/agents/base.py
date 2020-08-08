import os
import gym
import random
import numpy as np


class BaseAgent:

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.is_env_discrete = type(self.env.action_space) == gym.spaces.discrete.Discrete

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

    def act(self, iterations, render=True, collect_frames=True):
        state = self.env.reset()
        frames = []
        for _ in range(iterations):
            os.system('clear')
            if render:
                if collect_frames:
                    frame = self.env.render(mode='rgb_array')
                    frames.append(frame)
                else:
                    self.env.render()
            action = self.get_action(state)
            state, reward, done, info = self.env.step(action)
        if render and collect_frames:
            frames = np.array(frames)
            frames = np.transpose(frames, (0, 3, 1, 2))
            return frames

    def close_env(self):
        self.env.close()
