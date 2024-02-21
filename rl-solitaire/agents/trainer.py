from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
import numpy as np
import torch
import pickle

from .base_agent import BaseAgent
from env.env import Env


NEPTUNE_API_KEY = "RLSOL"
NEPTUNE_PROJECT_NAME = "karl-hajjar/RL-solitaire"

logger = logging.getLogger()


class BaseTrainer:
    """
    Implementing a trainer class for training RL agents.
    """
    def __init__(self, env: Env, agent: BaseAgent, n_iter: int, n_games_train: int, agent_results_filepath: str,
                 n_steps_update: int = None, log_every: int = None, n_games_eval: int = 10):
        self.env = env
        self.agent = agent
        self.n_iter = n_iter
        self.n_games_train = n_games_train
        self.n_steps_update = n_steps_update
        self.log_every = log_every
        self.n_games_eval = n_games_eval
        self._name = "BaseTrainer"
        self.current_iteration = 0
        self.agent_results_file = open(agent_results_filepath, "wb+")  # open log file to write in during evaluation

    @property
    def name(self):
        return self._name

    def train(self):
        with logging_redirect_tqdm():
            for i in tqdm(range(self.n_iter)):
                self.current_iteration = i
                # prepare data
                with torch.no_grad():
                    self.agent.set_evaluation_mode()
                    data = self.collect_data()
                    dataset = self.prepare_dataset(data)
                    dataloader = self.prepare_dataloader(dataset)

                # update agent with collected gameplay interactions with the environment
                self.update_agent(dataloader)
                if self.current_iteration % 300 == 0:
                    self.save_agent()

                # evaluate current agent and logs the results
                with torch.no_grad():
                    self.evaluate_agent()

            self.save_agent()
            self.agent_results_file.close()

    def collect_data(self) -> dict[str, np.ndarray]:
        # play once to get the keys from agent.collect_data
        self.env.reset()
        data0 = self.agent.collect_data(self.env, T=self.n_steps_update)
        data = {key: [] for key in data0.keys()}

        for _ in range(self.n_games_train - 1):
            self.env.reset()
            data_ = self.agent.collect_data(self.env, T=self.n_steps_update)
            for key in data.keys():
                data[key].append(data_[key])

        for key in data.keys():
            data[key] = np.concatenate(data[key], axis=0)
        return data

    def prepare_dataset(self, data: dict[np.array]):
        data = self.reformat_data(data)
        return data

    def prepare_dataloader(self, dataset):
        return dataset

    def reformat_data(self, data: dict):
        return data

    def update_agent(self, dataloader):
        pass

    def evaluate_agent(self):
        self.agent.set_evaluation_mode()
        rewards, pegs_left = self.agent.evaluate(self.env, self.n_games_eval, greedy=False)
        greedy_reward, greedy_pegs_left = self.agent.evaluate(self.env, greedy=True)  # only 1 game of greedy evaluation
        self.log_evaluation_results(rewards, pegs_left, greedy_reward[0], greedy_pegs_left[0])
        pickle.dump({"rewards": rewards, "pegs_left": pegs_left}, self.agent_results_file)

    def log_evaluation_results(self, rewards: list[float], pegs_left: list[float], greedy_reward: float,
                               greedy_pegs_left: float):
        mean_reward = np.mean(rewards)
        mean_pegs_left = np.mean(pegs_left)
        if (self.log_every is not None) and (self.current_iteration % self.log_every == 0):
            logger.info("Iteration {:,}: mean reward: {:.3f}, mean pegs left: {:.2f}".format(self.current_iteration,
                                                                                             mean_reward,
                                                                                             mean_pegs_left))

    def save_agent(self):
        pass
