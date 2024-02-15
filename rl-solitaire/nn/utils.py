import torch

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

LOSS_DICT = {'cross_entropy': torch.nn.CrossEntropyLoss,
             'bce'
             'kl': torch.nn.KLDivLoss,
             'mse': torch.nn.MSELoss,
             'logistic': torch.nn.BCEWithLogitsLoss}

DEFAULT_LOSS = "cross_entropy"

INIT_DICT = {'glorot_uniform': torch.nn.init.xavier_uniform_,
             'glorot_normal': torch.nn.init.xavier_normal_,
             'he_uniform': torch.nn.init.kaiming_uniform_,
             'he_normal': torch.nn.init.kaiming_normal_,
             'normal': torch.nn.init.normal_,
             'uniform': torch.nn.init.uniform_}
DEFAULT_INIT = "he_normal"


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
    probas = torch.nn.functional.log_softmax(logits, dim=1)
    p_log_p = probas * log_probas
    if mask is None:
        return torch.sum(p_log_p, dim=1), torch.Tensor([])
    else:
        return torch.sum(p_log_p, dim=1), torch.sum(mask * p_log_p, dim=1)
