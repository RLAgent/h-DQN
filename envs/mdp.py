import random
from enum import IntEnum
import gym
from gym.envs.registration import register as gymRegister


class StochasticMDP(gym.Env):
    """
    There are 6 possible states {0, 1, ..., 5} and the agent always starts
    at 1. The agent moves left deterministically when it chooses left action;
    but the action right only succeeds 50% of the time, resulting in a left
    move otherwise. The terminal state is 0. The agent receives a reward of
    1 when it first visits 5 and then 0. The reward for going to 0 without
    visiting 5 is 0.01.
    """
    class Actions(IntEnum):
        left = 0
        right = 1

    def __init__(self):
        self.actions = StochasticMDP.Actions
        self.reset()

    def reset(self):
        self.visited_five = False
        self.current_state = 1
        return self.current_state

    def step(self, action):
        if self.current_state == 0:
            done = True
            if self.visited_five:
                reward = 1.
            else:
                reward = 0.01
        else:
            reward = 0.0
            done = False

            if action == self.actions.right:
                if random.random() < 0.5 and self.current_state < 5:
                    self.current_state += 1
                else:
                    self.current_state -= 1
            if action == self.actions.left:
                self.current_state -= 1
            if self.current_state == 5:
                self.visited_five = True
        return self.current_state, reward, done

gymRegister(id='StochasticMDP-v0',
            entry_point='envs.mdp:StochasticMDP')
