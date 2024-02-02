from __future__ import print_function
from __future__ import division
from time import sleep
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
import os


# from ..networks.network import Net


class BaseAgent(object):
    """Agent is the base class for implementing agents to play the game of solitaire"""

    def __init__(self, name="Random Agent", gamma=1.0, render=False):
        self.name = name
        self.gamma = gamma
        self.render = render

    def play(self, env):
        '''
        Plays a game given the environment `env` until the end, selecting moves at random.

        Parameters
        ----------
        env : Env
            The environment with which the agent will interact.
        '''
        G = 0.0
        discount = 1.0
        end = False

        if self.render:  # render the state of the board at the begining of the game
            env.init_fig()
            env.render()
            sleep(1.5)

        while not end:
            action = self.select_action(env.feasible_actions)
            if self.render:
                env.render(action=action, show_action=True)  # render a first time displaying the action selected
                sleep(0.8)
            reward, _, end = env.step(action)
            G += discount * reward
            discount = discount * self.gamma
            if self.render:
                env.render()  # render a second time the state of the board after the action is played
                sleep(0.6)

        if self.render:
            env.render()
            sleep(2)
            plt.close()

        return (G, env.n_pegs)

    def select_action(self, state, feasible_actions):
        pass
