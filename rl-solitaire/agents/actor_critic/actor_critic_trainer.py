import logging
from copy import deepcopy
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader

from agents.actor_critic.actor_critic import ActorCriticAgent
from agents.trainer import BaseTrainer
from env.env import Env

logger = logging.getLogger()


class ActorCriticDataset(Dataset):
    def __init__(self, states, action_indices, action_masks, advantages, value_targets):
        super().__init__()
        assert (len(states) == len(action_indices) == len(action_masks) == len(advantages) == len(value_targets))
        self.states = states
        self.action_indices = action_indices
        self.action_maks = action_masks
        self.advantages = advantages
        self.value_targets = value_targets

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        return self.states[index], self.action_indices[index], self.action_maks[index], self.advantages[index], \
               self.value_targets[index]

    def __len__(self):
        return len(self.action_indices)


class ActorCriticTrainer(BaseTrainer):
    """
    A Trainer class for actor-critic agents.
    """
    def __init__(self, env: Env, agent: ActorCriticAgent, n_iter: int, n_games_train: int, n_steps_update: int = None,
                 log_every: int = None, n_games_eval: int = 10, batch_size: int = 64, log_dir: str = None,
                 checkpoints_dir: str = None):
        super().__init__(env, agent, n_iter, n_games_train, n_steps_update, log_every, n_games_eval)
        # TODO: create a config class for trainers
        self._name = "ActorCriticTrainer"
        self.batch_size = batch_size
        tb_logger = TensorBoardLogger(save_dir=log_dir)
        # save agent network at the end of each training loop (call to trainer.fit())
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_dir,
                                              filename='{epoch}_{step}_{val_loss:.4f}',
                                              save_on_train_epoch_end=True,
                                              verbose=True)
        if n_steps_update is None:
            n_epochs = 1
        else:
            n_epochs = None
        # PL trainer with automatic validation disabled: limit_val_batches=0.0, because "validation" is performed
        # manually in the agent evaluation
        self.trainer = Trainer(max_epochs=n_epochs,
                               max_steps=n_steps_update,
                               limit_val_batches=0.0,
                               logger=tb_logger,
                               callbacks=checkpoint_callback,
                               log_every_n_steps=10,
                               deterministic=True,
                               num_sanity_val_steps=0)

    def prepare_dataset(self, data: dict[str, np.array]) -> ActorCriticDataset:
        data = self.reformat_data(data)
        return ActorCriticDataset(states=data["states"],
                                  action_indices=data["actions"],
                                  action_masks=data["action_masks"],
                                  advantages=data["advantages"],
                                  value_targets=data["value_targets"])

    def prepare_dataloader(self, dataset: ActorCriticDataset) -> DataLoader:
        return DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

    def reformat_data(self, data: dict[str, np.array]) -> dict[str, torch.Tensor]:
        return {key: torch.from_numpy(data) for key in data.keys()}

    def update_agent(self, agent: ActorCriticAgent, dataloader: DataLoader):
        self.trainer.fit(model=self.agent.network, train_dataloaders=dataloader)

    def log_evaluation_results(self, rewards, pegs_left):
        mean_reward = np.mean(rewards)
        mean_pegs_left = np.mean(pegs_left)
        if (self.log_every is not None) and (self.current_iteration % self.log_every == 0):
            logger.info("Iteration {:,}: mean reward: {:.3f}, mean pegs left: {:.2f}".format(self.current_iteration,
                                                                                             mean_reward,
                                                                                             mean_pegs_left))
        self.agent.network.log('eval/mean_reward', mean_reward)
        self.agent.network.log('eval/mean_pegs_left', mean_pegs_left)
        # TODO: check if the appropriate way is not rather self.trainer.current_epoch/global_step.
        self.agent.network.log('eval/n_train_epochs', self.agent.network.current_epoch)
        self.agent.network.log('eval/n_train_steps', self.agent.network.global_step)
