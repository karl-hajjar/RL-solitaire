import torch
import importlib
import os

ACTIVATION_DICT = {'relu': torch.nn.ReLU,
                   'elu': torch.nn.ELU,
                   'gelu': torch.nn.GELU,
                   'sigmoid': torch.nn.modules.activation.Sigmoid,
                   'tanh': torch.nn.modules.activation.Tanh,
                   'leaky_relu': torch.nn.modules.activation.LeakyReLU,
                   'identity': torch.nn.Identity}
DEFAULT_ACTIVATION = "gelu"

OPTIMIZER_DICT = {'adam': torch.optim.Adam,
                  'rmsprop': torch.optim.RMSprop,
                  'sgd': torch.optim.SGD}
DEFAULT_OPT = "adam"

INIT_DICT = {'glorot_uniform': torch.nn.init.xavier_uniform_,
             'glorot_normal': torch.nn.init.xavier_normal_,
             'he_uniform': torch.nn.init.kaiming_uniform_,
             'he_normal': torch.nn.init.kaiming_normal_,
             'normal': torch.nn.init.normal_,
             'uniform': torch.nn.init.uniform_}
DEFAULT_INIT = "he_normal"

SUPPORTED_NETS_TO_CLASS_NAME = {'fc_policy_value': "policy_value.fully_connected.FCPolicyValueNet",
                                'conv_policy_value': "policy_value.conv.ConvPolicyValueNet"}


class PPOClipLoss(torch.nn.Module):
    SUPPORTED_REDUCTIONS = {"none", "mean", "sum"}

    def __init__(self, reduction: str = 'mean', epsilon_clip: float = 0.2, epsilon_proba_ratio: float = 1e-6) -> None:
        """
        A class implementing the PPO clipped loss. First, we compute the ratio of new over old action probabilities as
        well as a clipped version between 1-epsilon_clip and 1+epsilon_clip. Then we multiply both by the advantages and
        return the minimum of the two values.
        :param reduction: string indicating which type of reduction to apply to the batch losses. Must be one of
        {"none", "mean", "sum"}.
        :param epsilon_clip: float indicating the value to use when clipping around 1.
        :param epsilon_proba_ratio: float indicating the value to add to the old action probas to prevent numerical
        overflow when the probabilities are too low.
        """
        super().__init__()
        if reduction not in self.SUPPORTED_REDUCTIONS:
            raise ValueError(f"`reduction` argument must be one of {self.SUPPORTED_REDUCTIONS}, but was {reduction}")
        self.reduction = reduction
        self.epsilon_clip = epsilon_clip
        self.epsilon_proba_ratio = epsilon_proba_ratio

    def forward(self, logits: torch.Tensor, action_indices: torch.Tensor, action_probas: torch.Tensor,
                advantages: torch.Tensor) -> torch.Tensor:
        """
        Computes the PPO clipped loss.
        :param logits: Tensor of shape (N, N_ACTIONS) containing the logits of the possible actions for each state in the batch.
        :param action_indices: Tensor of shape (N, 1) containing the index of the actions taken in the batch.
        :param action_probas: Tensor of shape (N, 1) containing the old policy probabilities for the actions taken.
        :param advantages: Tensor of shape (N, 1) containing the advantage estimated for each state-action in the batch.
        :return: Tensor of shape (1) if reduction is "mean" or "sum", and of shape (N, 1) if reduction is "none".
        """
        probas = torch.nn.functional.softmax(logits, dim=-1)  # compute softmax over last dimension
        new_action_probas = probas[torch.arange(len(action_indices)), action_indices]
        proba_ratios = new_action_probas / (action_probas + self.epsilon_proba_ratio)
        clipped_proba_ratios = torch.clip(proba_ratios, min=1-self.epsilon_clip, max=1 + self.epsilon_clip)

        # maximizing advantage * action_proba <=> minimizing - loss
        batch_losses = -torch.minimum(proba_ratios * advantages, clipped_proba_ratios * advantages)
        if self.reduction == "none":
            return batch_losses
        elif self.reduction == "mean":
            return torch.mean(batch_losses)
        else:
            return torch.sum(batch_losses)


LOSS_DICT = {'cross_entropy': torch.nn.CrossEntropyLoss,
             'ppo_clip': PPOClipLoss,
             'bce': torch.nn.BCELoss,
             'kl': torch.nn.KLDivLoss,
             'mse': torch.nn.MSELoss,
             'logistic': torch.nn.BCEWithLogitsLoss}

DEFAULT_LOSS = "cross_entropy"


def get_activation(activation=None):
    if activation is None:
        return ACTIVATION_DICT[DEFAULT_ACTIVATION]
    elif isinstance(activation, str):
        if activation in ACTIVATION_DICT.keys():
            return ACTIVATION_DICT[activation]
        else:
            raise ValueError("Activation name must be one of {} but was {}".format(list(ACTIVATION_DICT.keys()),
                                                                                   activation))
    elif isinstance(activation, torch.nn.Module):
        return activation
    else:
        raise ValueError("activation argument must be of type None, str, or torch.nn.Module but was of type {}".
                         format(type(activation)))


def get_optimizer(optimizer=None):
    if optimizer is None:
        return OPTIMIZER_DICT[DEFAULT_OPT]
    elif isinstance(optimizer, str):
        if optimizer in OPTIMIZER_DICT.keys():
            return OPTIMIZER_DICT[optimizer]
        else:
            raise ValueError("Optimizer name must be one of {} but was {}".format(list(OPTIMIZER_DICT.keys()),
                                                                                  optimizer))
    elif isinstance(optimizer, torch.nn.Module):
        return optimizer
    else:
        raise ValueError("optimizer argument must be of type None, str, or torch.nn.Module but was of type {}".
                         format(type(optimizer)))


def get_loss(loss=None):
    if loss is None:
        return LOSS_DICT[DEFAULT_LOSS]
    elif isinstance(loss, str):
        if loss in LOSS_DICT.keys():
            return LOSS_DICT[loss]
        else:
            raise ValueError("Loss name must be one of {} but was {}".format(list(LOSS_DICT.keys()),
                                                                             loss))
    elif isinstance(loss, torch.nn.Module):
        return loss
    else:
        raise ValueError("loss argument must be of type None, str, or torch.nn.Module but was of type {}".
                         format(type(loss)))


def get_initializer(initializer=None):
    if initializer is None:
        return INIT_DICT[DEFAULT_INIT]
    elif isinstance(initializer, str):
        if initializer in INIT_DICT.keys():
            return INIT_DICT[initializer]
        else:
            raise ValueError("Initializer name must be one of {} but was {}".format(list(INIT_DICT.keys()),
                                                                                    initializer))
    elif isinstance(initializer, torch.nn.Module):
        return initializer
    else:
        raise ValueError("optimizer argument must be of type None, str, or torch.nn.Module but was of type {}". \
                         format(type(initializer)))


def compute_entropies_from_logits(logits: torch.Tensor, mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
    """
    Computes and returns the full entropy from the logits as well as the entropy with only the non-masked classes if
    mask is not None.
    :param logits: input tensor of shape (batch_size, n_classes) where each row contain the logits of the corresponding
    sample
    :param mask: a tensor of shape (batch_size, n_classes) with elements in {0, 1} representing the classes to mask when
    computing the entropy. If it is None then only the full entropy is computed.
    :return:
    """
    log_probas = torch.nn.functional.log_softmax(logits, dim=1)
    probas = torch.nn.functional.softmax(logits, dim=1)
    p_log_p = probas * log_probas
    if mask is None:
        return -torch.sum(p_log_p, dim=1), torch.Tensor([])
    else:
        return -torch.sum(p_log_p, dim=1), -torch.sum(mask * p_log_p, dim=1)


def get_network_class_from_name(name: str):
    if name not in SUPPORTED_NETS_TO_CLASS_NAME.keys():
        raise ValueError(f"Network {name} not in supported networks: {SUPPORTED_NETS_TO_CLASS_NAME.keys()}")
    else:
        dot_split_net_name = SUPPORTED_NETS_TO_CLASS_NAME[name].split('.')
        net_class_name = dot_split_net_name[-1]
        net_module = importlib.import_module(name='.'.join(['nn'] + dot_split_net_name[:-1]))
        net_class = getattr(net_module, net_class_name)
        return net_class


def get_network_dir_from_name(name: str):
    if name not in SUPPORTED_NETS_TO_CLASS_NAME.keys():
        raise ValueError(f"Network {name} not in supported networks: {SUPPORTED_NETS_TO_CLASS_NAME.keys()}")
    else:
        dot_split_net_name = SUPPORTED_NETS_TO_CLASS_NAME[name].split('.')
        net_dir = os.path.join('nn', *dot_split_net_name[:-2])  # last dot is class and penultimate dot is module
        return net_dir
