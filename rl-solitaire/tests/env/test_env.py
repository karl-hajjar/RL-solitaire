import unittest
import numpy as np

from env.env import Env, ACTIONS, GRID, MOVES, N_ACTIONS, OUT_OF_BORDER_ACTIONS
from agents.random_agent import RandomAgent


class TestEnv(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_constant_definitions(self):
        self.assertTrue(N_ACTIONS == len(ACTIONS) == len(GRID) * len(MOVES) == OUT_OF_BORDER_ACTIONS.size)

    def test_feasible_actions(self):
        def test_feasible_actions_(env, feasible_actions):
            feasible_actions_ = np.ones(shape=(len(GRID), len(MOVES)), dtype=bool)
            for i, pos in enumerate(GRID):
                if env.pegs[pos] == 0:
                    feasible_actions_[i, :] = False
                else:
                    for move_id in range(len(MOVES)):
                        if OUT_OF_BORDER_ACTIONS[i, move_id]:
                            feasible_actions_[i, move_id] = False
                        else:
                            if not env.action_jump_feasible(i, move_id):
                                feasible_actions_[i, move_id] = False

            np.testing.assert_array_equal(feasible_actions, feasible_actions_)

        random_agent = RandomAgent()
        env = Env()
        end = False
        cmpt = 0
        while not end:
            feasible_actions = env.feasible_actions
            test_feasible_actions_(env, feasible_actions)
            action_index = random_agent.select_action(env.state, feasible_actions)
            action = env.convert_action_id_to_action(action_index)
            reward, next_state, end = env.step(action)
            cmpt += 1


if __name__ == '__main__':
    unittest.main()
