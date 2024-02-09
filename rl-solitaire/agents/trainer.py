from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
import numpy as np

from .base_agent import BaseAgent
from env.env import Env


logger = logging.getLogger()


class BaseTrainer:
    """
    Implementing a trainer class for training RL agents.
    """
    def __init__(self, env: Env, agent: BaseAgent, n_iter: int, n_games_train: int, n_steps_update: int,
                 log_every: int = None, n_games_eval: int = 10):
        self.env = env
        self.agent = agent
        self.n_iter = n_iter
        self.n_games_train = n_games_train
        self.n_steps_update = n_steps_update
        self.log_every = log_every
        self.n_games_eval = n_games_eval
        self._name = "BaseTrainer"
        self.current_iteration = 0

    @property
    def name(self):
        return self._name

    def train(self):
        with logging_redirect_tqdm():
            for i in tqdm(range(self.n_iter)):
                self.current_iteration = i
                # prepare data
                data = self.collect_data(self.agent, self.env, self.n_games_train, self.n_steps_update)
                dataset = self.prepare_dataset(data)
                dataloader = self.prepare_dataloader(dataset)

                # update agent with collected gameplay interactions with the environment
                self.update(self.agent, dataloader)

                # evaluate current agent and logs the results
                self.evaluate_agent(self.agent, self.env, self.n_games_eval)

    def collect_data(self, agent, env, n_games_train, n_steps_update):
        data = []
        for _ in range(n_games_train):
            env.reset()
            data += agent.collect_data(env, T=n_steps_update)
        return data

    def prepare_dataset(self, data):
        data = self.reformat_data(data)
        return data

    def prepare_dataloader(self, dataset):
        return dataset

    def reformat_data(self, data):
        return data

    def update(self, agent, dataloader):
        pass

    def evaluate_agent(self, agent, env, n_games_eval):
        rewards, pegs_left = agent.evaluate(env, n_games_eval)
        self.log_evaluation_results(rewards, pegs_left)

    def log_evaluation_results(self, rewards, pegs_left):
        mean_reward = np.mean(rewards)
        mean_pegs_left = np.mean(pegs_left)
        if (self.log_every is not None) and (self.current_iteration % self.log_every == 0):
            logger.info("Iteration {:,}: mean reward: {:.3f}, mean pegs left: {:.2f}".format(self.current_iteration,
                                                                                             mean_reward,
                                                                                             mean_pegs_left))



