from utils.tools import read_yaml


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

        self._set_config_attributes(**kwargs)

    def _set_config_attributes(self, **kwargs):
        if self.config_dict is None:
            # TODO: set configs from **kwargs
            pass
        else:
            self._set_name()
            self.activation_config = self._get_attribute_from_config("activation")
            self.architecture_config = self._get_attribute_from_config("architecture")
            self.loss_config = self._get_attribute_from_config("loss")
            self.initializer_config = self._get_attribute_from_config("initializer")
            self.optimizer_config = self._get_attribute_from_config("optimizer")

    def _set_name(self):
        if ("name" not in self.config_dict) or (self.config_dict["name"] is None):
            self.name = "Net"
        else:
            self.name = self.config_dict["name"]

    def _get_attribute_from_config(self, attribute_name: str):
        if attribute_name in self.config_dict.keys():
            return self.config_dict[attribute_name]
        else:
            return None

    def to_dict(self) -> dict:
        res = dict()
        res["name"] = self.name
        res["activation"] = self.activation_config
        res["architecture"] = self.architecture_config
        res["loss"] = self.loss_config
        res["initializer"] = self.initializer_config
        res["optimizer"] = self.optimizer_config

        return res
