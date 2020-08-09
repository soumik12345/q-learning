import os
import wandb
import numpy as np
from random import random
from .base import BaseAgent


class QAgent(BaseAgent):

    def __init__(self, env_name: str, hyper_parameters: dict):
        super(QAgent, self).__init__(env_name)
        self.gamma = hyper_parameters['gamma']
        self.epsilon = hyper_parameters['epsilon']
        self.discount = hyper_parameters['discount']
        self.learning_rate = hyper_parameters['learning_rate']
        self.initialization_scale = hyper_parameters['initialization_scale']
        self.model = self.build_model()
        self.total_reward = 0

    def build_model(self):
        return self.initialization_scale * np.random.random([self.state_size, self.action_size])

    def get_action(self, state):
        q_state = self.model[state]
        greedy_action = np.argmax(q_state)
        exploratory_action = super(QAgent, self).get_action(state)
        if random() < self.epsilon:
            return exploratory_action
        return greedy_action

    def _train_step(self, experience):
        state, action, next_state, reward, done = experience
        q_next = self.model[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.discount * np.max(q_next)
        q_update = q_target - self.model[state, action]
        self.model[state, action] += self.learning_rate * q_update
        if done:
            self.epsilon = self.epsilon * self.gamma

    def train(self, episodes: int, render=True, log_metrics=True):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                experience = (state, action, next_state, reward, done)
                self._train_step(experience)
                state = next_state
                self.total_reward += reward
                if render:
                    print(
                        'Episode: {}\nTotal Reward: {}\nEpsilon: {}'.format(
                            episode, self.total_reward, self.epsilon
                        )
                    )
                    self.env.render()
                    os.system('clear')
                if log_metrics:
                    wandb.log({
                        'train_reward': reward,
                        'train_total_reward': self.total_reward,
                        'epsilon': self.epsilon
                    })
