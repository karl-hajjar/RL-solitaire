import torch

from ..base_net import BaseNet
from ..utils import get_loss


DEFAULT_ACTOR_LOSS_DICT = {"name": "cross_entropy",
                           "reduction": "mean",
                           "coef": 1.0}
DEFAULT_CRITIC_LOSS_DICT = {"name": "mse",
                            "reduction": "mean",
                            "coef": 1.0}


class BasePolicyValueNet(BaseNet):
    def __init__(self, config):
        super().__init__(config)

    def _build_model(self, architecture_config: dict):
        self.state_embeddings = torch.nn.Module()
        self.policy_head = torch.nn.Module()
        self.value_head = torch.nn.Module()

    def _set_loss(self, loss_config: dict):
        # actor loss = policy loss
        if "actor_loss" not in loss_config.keys():
            loss_config["actor_loss"] = DEFAULT_ACTOR_LOSS_DICT
        else:
            for key in DEFAULT_ACTOR_LOSS_DICT.keys():
                if key not in loss_config["actor_loss"].keys():
                    loss_config["actor_loss"][key] = DEFAULT_ACTOR_LOSS_DICT[key]

        self.actor_coef = loss_config["actor_loss"]["coef"]
        loss_config["actor_loss"].pop("coef")
        self.actor_loss = self._get_loss(loss_config["actor_loss"])

        # critic loss = value loss
        if "critic_loss" not in loss_config.keys():
            loss_config["critic_loss"] = DEFAULT_CRITIC_LOSS_DICT
        else:
            for key in DEFAULT_CRITIC_LOSS_DICT.keys():
                if key not in loss_config["critic_loss"].keys():
                    loss_config["critic_loss"][key] = DEFAULT_CRITIC_LOSS_DICT[key]

        self.critic_coef = loss_config["critic_loss"]["coef"]
        loss_config["critic_loss"].pop("coef")
        self.critic_loss = self._get_loss(loss_config["critic_loss"])

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Outputs the policies and values associated with a batch of states.
        Given a tensor x representing a state or a batch of states of shape (n_batch, state_shape), produces the
        :param x: a torch.Tensor representing a state or a batch of states of shape (n_batch, state_shape).
        :return: (policies, values) of type (torch.Tensor, torch.Tensor) where policies is of shape (n_batch, N_ACTIONS)
        and values is oh shape (n_batch, 1).
        """
        x = self.state_embeddings(x)
        return self.policy_head(x), self.value_head(x)

    def training_step(self, batch, batch_nb) -> dict:
        """
        Runs one step of training on the given batch as follows:
                # put model in train mode
                model.train()
                torch.set_grad_enabled(True)

                losses = []
                for batch in train_dataloader:
                    # calls hooks like this one
                    on_train_batch_start()

                    # train step
                    loss = training_step(batch)

                    # clear gradients
                    optimizer.zero_grad()

                    # backward
                    loss.backward()

                    # update parameters
                    optimizer.step()

                    losses.append(loss)
        :param batch: a batch of samples (states, action_indices, value_targets)
        :param batch_nb: int, the index of the current batch
        :return: dict with the training metrics.
        """
        states, action_indices, value_targets, action_masks = batch
        logits, values = self.forward(states)
        actor_loss = self.actor_loss(logits, action_indices)
        critic_loss = self.critic_loss(values, value_targets)
        weighted_actor_loss = self.actor_coef * actor_loss
        weighted_critic_loss = self.critic_coef * critic_loss
        loss = weighted_actor_loss + weighted_critic_loss
        self.log('train/actor_loss', actor_loss.detach().item())
        self.log('train/critic_loss', critic_loss.detach().item())
        self.log('train/weighted_actor_loss', weighted_actor_loss.detach().item())
        self.log('train/weighted_critic_loss', weighted_critic_loss.detach().item())
        self.log('train/loss', loss.detach().item(), prog_bar=True, logger=True, on_step=True)

        if self.regularization:
            if self.regularization_type == "entropy":
                with torch.no_grad():
                    feasible_actions_uniform_dist = action_masks / torch.sum(action_masks, dim=-1, keepdim=True)
                regularized_loss = loss + \
                                   self.regularization_coef * self.regularization_loss(logits,
                                                                                       feasible_actions_uniform_dist)
                self.log('train/regularized_loss', regularized_loss.detach().item(), prog_bar=True, logger=True,
                         on_step=True)
            else:
                # TODO : implement KLDiv Loss wrt to reference policy later
                regularized_loss = loss

        else:
            regularized_loss = loss

        # see https://stackoverflow.com/questions/37304461/tensorflow-importing-data-from-a-tensorboard-tfevent-file to
        # read from TensorBoard events file
        self.log('lr', self._get_opt_lr()[0], prog_bar=True, logger=True, on_step=True)
        # return {'loss': loss, 'regularized_loss': regularized_loss, 'log': tensorboard_logs}
        return regularized_loss

    def validation_step(self, batch, batch_nb):
        pass

