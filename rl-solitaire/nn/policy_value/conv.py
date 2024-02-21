import torch

from nn.blocks.residual import ResidualBlock
from nn.network_config import NetConfig
from .skeleton import BasePolicyValueNet


class ConvPolicyValueNet(BasePolicyValueNet):
    """
    A class implementing a convolutional architecture for the policy-value network, including residual blocks and
    normalization layers.
    """

    def __init__(self, config: NetConfig):
        super().__init__(config)

    # TODO: change default kernel size to 3x3 instead.
    def _build_state_embeddings(self, n_residual_blocks: int, input_dim: int, hidden_dim: int, n_layers: int = 2,
                                residual_hidden_dim: int = None, bias: bool = True, kernel_size: tuple = (5, 5)):
        self.state_embeddings = torch.nn.Sequential()
        self.state_embeddings.add_module(name="input_conv",
                                         module=torch.nn.Conv2d(in_channels=input_dim,
                                                                out_channels=hidden_dim,
                                                                padding="same",
                                                                kernel_size=kernel_size,
                                                                bias=bias))
        self.state_embeddings.add_module(name="input_activation", module=self.activation)
        self.state_embeddings.add_module(name="residual_blocks",
                                         module=self._get_residual_blocks(n_residual_blocks=n_residual_blocks,
                                                                          input_dim=hidden_dim,
                                                                          hidden_dim=residual_hidden_dim,
                                                                          n_layers=n_layers,
                                                                          bias=bias,
                                                                          kernel_size=kernel_size))
        self.state_embeddings.add_module(name="ouput_activation", module=self.activation)

    def _build_policy_head(self, n_residual_blocks: int, output_dim: int, input_dim: int, n_layers: int = 2,
                           hidden_dim: int = None, bias: bool = True, kernel_size: tuple = (5, 5)):
        self.policy_head = torch.nn.Sequential()
        self.policy_head.add_module(name="residual_blocks",
                                    module=self._get_residual_blocks(n_residual_blocks=n_residual_blocks,
                                                                     input_dim=input_dim,
                                                                     hidden_dim=hidden_dim,
                                                                     n_layers=n_layers,
                                                                     bias=bias,
                                                                     kernel_size=kernel_size))
        # (N, C, H, W) -> (N, C*H,*W)
        self.policy_head.add_module(name="flatten", module=torch.nn.Flatten(start_dim=1, end_dim=-1))
        self.policy_head.add_module(name="output_activation", module=self.activation)
        self.policy_head.add_module(name="output_linear",
                                    module=torch.nn.Linear(in_features=7 * 7 * input_dim,
                                                           out_features=output_dim,
                                                           bias=bias))

    def _build_value_head(self, n_residual_blocks: int, output_dim: int, input_dim: int, n_layers: int = 2,
                          hidden_dim: int = None, bias: bool = True, kernel_size: tuple = (5, 5)):
        self.value_head = torch.nn.Sequential()
        self.value_head.add_module(name="residual_blocks",
                                   module=self._get_residual_blocks(n_residual_blocks=n_residual_blocks,
                                                                    input_dim=input_dim,
                                                                    hidden_dim=hidden_dim,
                                                                    n_layers=n_layers,
                                                                    bias=bias,
                                                                    kernel_size=kernel_size))
        self.value_head.add_module(name="flatten", module=torch.nn.Flatten(start_dim=1, end_dim=-1))
        self.policy_head.add_module(name="output_activation", module=self.activation)
        self.value_head.add_module(name="output_linear",
                                   module=torch.nn.Linear(in_features=7 * 7 * input_dim,
                                                          out_features=output_dim,
                                                          bias=bias))

    def _get_residual_blocks(self, n_residual_blocks: int, input_dim: int, hidden_dim: int, n_layers: int = 2,
                             bias: bool = True, kernel_size: tuple = (5, 5)):
        residual_blocks = torch.nn.Sequential()
        for i in range(1, n_residual_blocks + 1):
            residual_blocks.add_module(name=f"batchnorm{i}",
                                       module=torch.nn.BatchNorm2d(num_features=input_dim))
            residual_blocks.add_module(name=f"residual{i}",
                                       module=ResidualBlock(input_dim=input_dim,
                                                            hidden_dim=hidden_dim,
                                                            activation=self.activation,
                                                            layer_type="conv",
                                                            n_layers=n_layers,
                                                            bias=bias,
                                                            kernel_size=kernel_size))
        return residual_blocks

    @staticmethod
    def _reshape_2d_input(x: torch.Tensor):
        """
        Reshapes a tensor from shape (N, W, H, C) to (N, C, W, H) for the convolution modules.
        :param x: torch.Tensor of shape (N, W, H, C).
        :return: torch.Tensor of shape (N, C, W, H).
        """
        return x.reshape(x.shape[0], x.shape[-1], x.shape[1], x.shape[2])

    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.state_embeddings(self._reshape_2d_input(x)))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.state_embeddings(self._reshape_2d_input(x)))

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Outputs the policies and values associated with a batch of states. The states are flattened along non-batch
        dimension.
        :param x: a torch.Tensor representing a state or a batch of states of shape (n_batch, state_shape).
        :return: (policies, values) of type (torch.Tensor, torch.Tensor) where policies is of shape (n_batch, N_ACTIONS)
        and values is oh shape (n_batch, 1).
        """
        x = self._reshape_2d_input(x)
        x = self.state_embeddings(x)
        # return self.policy_head(x), 2 * torch.nn.functional.sigmoid()(value_head(x))
        return self.policy_head(x), self.value_head(x)
