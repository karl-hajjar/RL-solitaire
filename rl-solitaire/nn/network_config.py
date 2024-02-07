from ..utils.tools import read_yaml


class NetConfig:
    def __init__(self, config_path: str = None, config_dict: dict = None, **kwargs):
        self.config_dict = None
        if config_path is not None:
            self.config_dict = read_yaml(config_path)
        else:
            if config_dict is not None:
                self.config_dict = config_dict
            else:
                self.config_dict = kwargs

        self._set_config_attributes()

    def _set_config_attributes(self):
        if self.config_dict is None:
            pass
        else:
            self._set_name()
            self._set_architecture_config()
            self._set_loss_config()
            self._set_initializer_config()
            self._set_optimizer_config()

    def _set_name(self):
        if "name" in self.config_dict:
            self.name = self.config_dict["name"]
        else:
            self.name = "Net"

    def _set_architecture_config(self):
        if "architecture" in self.config_dict:
            self.architecture_config = self.config_dict["architecture"]
        else:
            self.architecture_config = None

    def _set_loss_config(self):
        if "loss" in self.config_dict:
            self.loss_config = self.config_dict["loss"]
        else:
            self.loss_config = None

    def _set_initializer_config(self):
        if "initializer" in self.config_dict:
            self.initializer_config = self.config_dict["initializer"]
        else:
            self.initializer_config = None

    def _set_optimizer_config(self):
        if "optimizer" in self.config_dict:
            self.optimizer_config = self.config_dict["optimizer"]
        else:
            self.optimizer_config = None

    def to_dict(self):
        res = dict()
        res["name"] = self.name
        res["architecture"] = self.architecture_config
        res["loss"] = self.loss_config
        res["initializer"] = self.initializer_config
        res["optimizer"] = self.optimizer_config
