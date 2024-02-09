import unittest

from env.env import Env, GRID, MOVES
from agents.random_agent import RandomAgent


class TestRandomAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = RandomAgent()

    def test_select_action(self):
        env = Env()
        state = env.state
        feasible_actions = env.feasible_actions
        action_index = self.agent.select_action(state, feasible_actions)
        action = env.convert_action_id_to_action(action_index)
        pos_id, move_id = action
        self.assertLess(pos_id, len(GRID))
        self.assertLess(move_id, len(MOVES))

        pos = GRID[pos_id]
        move = MOVES[move_id]
        x, y = pos

        reward, next_state, end = env.step(action)

        self.assertTrue(True)

    def test_play(self):
        env = Env()
        reward, n_pegs_left = self.agent.play(env)
        self.assertTrue(type(reward) == float)
        self.assertTrue(type(n_pegs_left) == int)

    def test_collect_data(self):
        env = Env()
        data, end = self.agent.collect_data(env, T=50)

        self.assertTrue(type(data) == list)
        self.assertTrue(type(end) == bool)


if __name__ == '__main__':
    unittest.main()
