import unittest

from nn.network_config import NetConfig

BASE_NET_CONFIG_FILE_PATH = './base_net_config.yaml'


class TestNetworkConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.config_dict = {
            'name': "BasicNet",
            'architecture': {
                'n_layers': 4,
                'bias': False
            },
            'activation': {'name': "elu"},
            'loss': {
                'name': "mse",
                'reduction': "mean"
            },
            'initializer': {
                'name': "glorot_normal"
            },
            'optimizer': {
                'name': "rmsprop",
                'lr': 1.0e-5
            }
        }

    def test_read_from_dict(self):
        net_config = NetConfig(config_dict=self.config_dict)
        self.assertEqual(net_config.name, self.config_dict['name'])
        self.assertDictEqual(net_config.architecture_config, self.config_dict['architecture'])
        self.assertDictEqual(net_config.activation_config, self.config_dict['activation'])
        self.assertDictEqual(net_config.loss_config, self.config_dict['loss'])
        self.assertDictEqual(net_config.initializer_config, self.config_dict['initializer'])
        self.assertDictEqual(net_config.optimizer_config, self.config_dict['optimizer'])

    def test_read_from_yaml(self):
        net_config = NetConfig(config_path=BASE_NET_CONFIG_FILE_PATH)
        self.assertEqual(net_config.name, "BasicNetwork")
        self.assertDictEqual(net_config.architecture_config,
                             {
                                 'n_layers': 3,
                                 'bias': True
                             })
        self.assertDictEqual(net_config.activation_config,
                             {
                                 'name': "relu"
                             })
        self.assertDictEqual(net_config.loss_config,
                             {
                                 'name': "cross_entropy",
                                 'reduction': "mean",
                                 'label_smoothing': 0.001
                             })
        self.assertDictEqual(net_config.initializer_config,
                             {
                                 'name': "glorot_uniform"
                             })
        self.assertDictEqual(net_config.optimizer_config,
                             {
                                 'name': "sgd",
                                 'lr': 1.0e-4,
                                 'weight_decay': 1.0e-6
                             })

    def test_to_dict(self):
        net_config = NetConfig(config_dict=self.config_dict)
        self.assertDictEqual(net_config.to_dict(), self.config_dict)

    def test_default_values(self):
        config_dict = self.config_dict
        config_dict["name"] = None
        config_dict.pop("activation")
        config_dict.pop("loss")
        config_dict.pop("initializer")
        config_dict.pop("optimizer")

        net_config = NetConfig(config_dict=config_dict)

        self.assertEqual(net_config.name, "Net")
        self.assertIsNone(net_config.activation_config)
        self.assertIsNone(net_config.loss_config)
        self.assertIsNone(net_config.initializer_config)
        self.assertIsNone(net_config.optimizer_config)


if __name__ == '__main__':
    unittest.main()
