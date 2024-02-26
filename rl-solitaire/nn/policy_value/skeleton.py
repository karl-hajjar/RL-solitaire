import torch
from torch.nn.functional import softmax

from ..base_net import BaseNet
from ..utils import compute_entropies_from_logits


DEFAULT_ACTOR_LOSS_DICT = {"name": "cross_entropy",
                           "reduction": "mean",
                           "coef": 1.0}
DEFAULT_CRITIC_LOSS_DICT = {"name": "mse",
                            "reduction": "mean",
                            "coef": 1.0}

DEFAULT_VALUE_HEAD_OUPUT_DIM = 1


class BasePolicyValueNet(BaseNet):
    def __init__(self, config):
        super().__init__(config)

    def _build_model(self, architecture_config: dict):
        architecture_config["policy_head"]["input_dim"] = architecture_config["embeddings"]["hidden_dim"]
        architecture_config["value_head"]["input_dim"] = architecture_config["embeddings"]["hidden_dim"]
        if ("output_dim" not in architecture_config["value_head"].keys()) or \
           (architecture_config["value_head"]["output_dim"] is None):
            architecture_config["value_head"]["output_dim"] = DEFAULT_VALUE_HEAD_OUPUT_DIM

        self._build_state_embeddings(**architecture_config["embeddings"])
        self._build_policy_head(**architecture_config["policy_head"])
        self._build_value_head(**architecture_config["value_head"])

    def _build_state_embeddings(self, **kwargs):
        self.state_embeddings = torch.nn.Module()

    def _build_policy_head(self, **kwargs):
        self.policy_head = torch.nn.Module()

    def _build_value_head(self, **kwargs):
        self.value_head = torch.nn.Module()

    def _set_loss(self, loss_config: dict):
        # set regularization
        self._set_regularization(loss_config)  # pops the key "regularization" from loss_config

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

    def get_policy(self, x: torch.Tensor):
        return softmax(self.policy_head(self.state_embeddings(x)), dim=-1)

    def get_value(self, x: torch.Tensor):
        return self.value_head(self.state_embeddings(x))

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Outputs the policies and values associated with a batch of states.
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
        if len(batch) == 5:  # standard way of training the network
            states, action_indices, action_masks, advantages, value_targets = batch
            logits, values = self.forward(states)
            actor_loss = self.actor_loss(logits, action_indices)
            advantage_actor_loss = torch.mean(advantages * actor_loss)
        elif len(batch) == 6:  # PPO network needs action probabilities from old policy
            states, action_indices, action_probas, action_masks, advantages, value_targets = batch
            logits, values = self.forward(states)
            advantage_actor_loss = self.actor_loss(logits, action_indices, action_probas, advantages)
        else:
            raise ValueError(f"Batch length must be 5 or 6 but was {len(batch)}")

        critic_loss = self.critic_loss(values, value_targets)
        weighted_actor_loss = self.actor_coef * advantage_actor_loss
        weighted_critic_loss = self.critic_coef * critic_loss
        loss = weighted_actor_loss + weighted_critic_loss

        # compute uniform distribution over feasible actions for later logging / KL-penalty computation
        with torch.no_grad():
            feasible_actions_uniform_dist = action_masks / torch.sum(action_masks, dim=-1, keepdim=True)

        if self.regularization:
            if self.regularization_type == "entropy":
                kl_div_uniform = self.regularization_loss(torch.nn.functional.log_softmax(logits, dim=-1),
                                                          feasible_actions_uniform_dist)
                weighted_penalty = self.regularization_coef * kl_div_uniform
                regularized_loss = loss + weighted_penalty
                self.log('train/penalty', kl_div_uniform.detach().item(),
                         on_step=True, logger=True)
                self.log('train/weighted_penalty', weighted_penalty.detach().item(),
                         prog_bar=True, on_step=True, logger=True)
                self.log('train/regularized_loss', regularized_loss.detach().item(),
                         prog_bar=True, on_step=True, logger=True)
            else:
                # TODO : implement KLDiv Loss wrt to reference policy later
                regularized_loss = loss
                with torch.no_grad():
                    kl_div_uniform = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(logits),
                        feasible_actions_uniform_dist,
                        reduction="batchmean",
                        log_target=False)

        else:
            regularized_loss = loss
            with torch.no_grad():
                kl_div_uniform = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(logits),
                    feasible_actions_uniform_dist,
                    reduction="batchmean",
                    log_target=False)

        self.log('train/loss', loss.detach().item(), prog_bar=True, on_step=True)

        # entropies and policy diversity
        with torch.no_grad():
            full_entropy, masked_entropy = compute_entropies_from_logits(logits=logits, mask=action_masks)
            self.log('train/full_entropy', torch.mean(full_entropy).detach().item(), logger=True)
            self.log('train/masked_entropy', torch.mean(masked_entropy).detach().item(), logger=True)
            self.log('train/kl_div_uniform', kl_div_uniform.detach().item(), on_step=True, logger=True)
            self.log('train/feasible_actions_proba', 
                     torch.mean(torch.sum(softmax(logits) * action_masks, dim=-1)).detach().item(),
                     on_step=True, logger=True)
            self.log('train/n_feasible_actions', torch.mean(torch.sum(action_masks, dim=-1)).detach().item(),
                     on_step=True, logger=True)

        # auxiliary loss metrics
        if len(batch) == 5:
            self.log('train/actor_loss', torch.mean(actor_loss).detach().item(), logger=True)
        self.log('train/advantages', torch.mean(advantages).detach().item(), logger=True)
        self.log('train/advantage_actor_loss', advantage_actor_loss.detach().item(), logger=True)
        self.log('train/critic_loss', critic_loss.detach().item(), logger=True)
        self.log('train/weighted_actor_loss', weighted_actor_loss.detach().item(), logger=True)
        self.log('train/weighted_critic_loss', weighted_critic_loss.detach().item(), logger=True)

        # see https://stackoverflow.com/questions/37304461/tensorflow-importing-data-from-a-tensorboard-tfevent-file to
        # read from TensorBoard events file
        self.log('lr', self._get_opt_lr()[0], prog_bar=True, logger=True, on_step=True)
        # return {'loss': loss, 'regularized_loss': regularized_loss, 'log': tensorboard_logs}
        return regularized_loss

    def validation_step(self, batch, batch_nb):
        # TODO: the validation_step might need to be removed altogether (no overriding of the base method).
        pass

