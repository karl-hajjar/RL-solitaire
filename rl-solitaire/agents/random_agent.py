import numpy as np

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """RandomAgent is class of agents which select actions randomly"""

    def __init__(self, name="Random Agent", gamma=1.0, render=False):
        '''
        Instanciates an object of the class RandomAgent by initializing the seed, defining the name of the agent, and setting
        the render parameter.

        Parameters
        ----------
        name : string (default "Random Agent")
            The name of the agent.
        seed : int or None (default None)
            The seed to use in numpy.random. If None, the seed is set using the current time by default.
        render : bool (default False)
            Whether or not to display a visual representation of the game as the agent plays.

        Attributes
        ----------
        name : string
            The name of the agent.
        render : bool
            Whether or not to display a visual representation of the game as the agent plays.
        '''
        super().__init__(name, gamma, render)

    def select_action(self, state, feasible_actions):
        '''
        Selects an action at random from the legal actions in the current state of the env, which are given by `feasible_actions`.

        Parameters
        ----------
        feasible_actions : 2d-array of bools
            An array indicating for each position on the board, whether each action is legal (True) or not (False).

        Returns
        -------
        out : tuple of ints (pos_id, move_id)
            a tuple representing the action selected : which peg to pick up, and where to move it.
        '''
        actions = np.argwhere(feasible_actions)
        return actions[np.random.randint(0, len(actions))]
