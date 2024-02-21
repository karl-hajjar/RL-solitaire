import unittest
import torch

from nn.network_config import NetConfig
from nn.base_net import BaseNet


BASE_NET_CONFIG_FILE_PATH = './base_net_config.yaml'


class TestNN(unittest.TestCase):
    def setUp(self) -> None:
        self.net_config = NetConfig(config_path=BASE_NET_CONFIG_FILE_PATH)

    def test_network_attributes(self):
        net = BaseNet(config=self.net_config)
        self.assertEqual(net.name, "BasicNetwork")

        self.assertEqual(net.loss.reduction, "mean")
        self.assertEqual(net.loss.label_smoothing, 0.001)

        self.assertTrue(net.initializer_class == torch.nn.init.xavier_uniform_)
        self.assertTrue(type(net.activation) == torch.nn.ReLU)

    def test_activation_with_arguments(self):
        self.net_config.activation_config["name"] = "gelu"
        self.net_config.activation_config["approximate"] = "tanh"
        net = BaseNet(config=self.net_config)
        self.assertEqual(net.name, "BasicNetwork")

        self.assertEqual(net.loss.reduction, "mean")
        self.assertEqual(net.loss.label_smoothing, 0.001)

        self.assertTrue(net.initializer_class == torch.nn.init.xavier_uniform_)
        self.assertEqual(net.activation.approximate, "tanh")
        self.assertTrue(type(net.activation) == torch.nn.GELU)

    def test_regularization(self):
        net_config = self.net_config
        net_config.loss_config["regularization"] = {
            'name': "entropy",
            'coef': 0.0005
        }
        net = BaseNet(config=net_config)
        self.assertTrue(net.regularization)
        self.assertEqual(net.regularization_coef, 0.0005)
        self.assertEqual(net.regularization_type, "entropy")
        self.assertTrue(type(net.regularization_loss) == torch.nn.KLDivLoss)


if __name__ == '__main__':
    unittest.main()
