import torch
from pytorch_lightning import LightningModule

from .network_config import NetConfig
from .utils import get_activation, get_initializer, get_loss, get_optimizer


INIT_EXCLUDE_MODULES = {"norm", "bias"}
REGULARIZATION_KEYS = {"entropy", "kl"}


class BaseNet(LightningModule):
    def __init__(self, config: NetConfig):
        super().__init__()
        self._set_name(config.name)
        self._set_activation(**config.activation_config)
        self._build_model(config.architecture_config)
        self._set_initializer(config.initializer_config["name"])
        config.initializer_config.pop("name")
        self._set_loss(config.loss_config)
        self.initialize(**config.initializer_config)
        self._set_optimizer(**config.optimizer_config)

        self.save_hyperparameters(config.to_dict())  # stores the hparams for later saving in the Lightning module
        # self.save_hyperparameters()  # stores the hparams for later saving in the Lightning module

    @property
    def name(self):
        return self._name

    def _set_name(self, name: str):
        self._name = name

    def _set_activation(self, name: str = None, **kwargs):
        activation_class = get_activation(name)
        self.activation = activation_class(**kwargs)

    def _build_model(self, architecture_config: dict):
        pass

    def _set_initializer(self, name: str = None):
        self.initializer_class = get_initializer(name)

    def _set_loss(self, loss_config: dict):
        self._set_regularization(loss_config)
        self.loss = self._get_loss(loss_config)

    def _set_regularization(self, loss_config):
        if "regularization" in loss_config.keys():
            if "name" not in loss_config["regularization"].keys():
                self.regularization = False
                self.regularization_coef = 0.
                self.regularization_loss = None
                self.regularization_type = None
            else:
                if loss_config["regularization"]["name"] not in REGULARIZATION_KEYS:
                    raise ValueError(f"regularization name must be one of {REGULARIZATION_KEYS} "
                                     f"but was {loss_config['regularization']['name']}")
                else:
                    if "coef" not in loss_config["regularization"].keys():
                        self.regularization = False
                        self.regularization_coef = 0.
                        self.regularization_loss = None
                        self.regularization_type = None
                    else:
                        self.regularization_type = loss_config["regularization"]["name"]
                        self.regularization = True
                        self.regularization_coef = loss_config["regularization"]["coef"]
                        loss_config["regularization"].pop("coef")  # remove "coef" from the regularization keys
                        # Regularization loss is a KLDivergence in any case, depending on the teg type different
                        # arguments will be fed to it
                        # self.regularization_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
                        self.regularization_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

            loss_config.pop("regularization")  # remove "regularization" from the loss_config keys
        else:
            self.regularization = False
            self.regularization_coef = 0.
            self.regularization_loss = None
            self.regularization_type = None

    @staticmethod
    def _get_loss(loss_config: dict) -> torch.nn.Module:
        if "name" in loss_config.keys():
            name = loss_config["name"]
            loss_config.pop("name")
        else:
            name = None
        loss_class = get_loss(name)
        return loss_class(**loss_config)  # name has been popped, all other loss params are kwargs

    def initialize(self, **kwargs):
        with torch.no_grad():
            for name, p in self.named_parameters():
                # initialize param if it is not one of the excluded module names
                if not any([module_name in name.lower() for module_name in INIT_EXCLUDE_MODULES]):
                    self.initializer_class(p, **kwargs)

    def _set_optimizer(self, name: str = None, **kwargs):
        if len(list(self.parameters())) > 0:
            optimizer_class = get_optimizer(name)
            self.optimizer = optimizer_class(self.parameters(), **kwargs)

    def configure_optimizers(self):
        return self.optimizer

    def _get_opt_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def __str__(self):
        s = LightningModule.__str__(self)
        return self.name + " ({:,} params)".format(self.count_parameters()) + ":\n" + s

    def count_parameters(self):
        if hasattr(self, "parameters"):
            return sum(p.numel() for p in self.parameters())
        else:
            return None
