import torch

from nn.network_config import NetConfig
from .skeleton import BasePolicyValueNet


class FCPolicyValueNet(BasePolicyValueNet):
    """
    A class implementing a policy-value network with a fully-connected architecture.
    """

    def __init__(self, config: NetConfig):
        super().__init__(config)

    def _build_state_embeddings(self, input_dim: int, hidden_dim: int, n_layers: int, bias=True):
        self.state_embeddings = torch.nn.Sequential()
        self.state_embeddings.add_module(name="linear1",
                                         module=torch.nn.Linear(in_features=input_dim, out_features=hidden_dim,
                                                                bias=bias))
        self.state_embeddings.add_module(name=f"{self.activation}1", module=self.activation)
        for i in range(1, n_layers):
            self.state_embeddings.add_module(name=f"linear{i + 1}",
                                             module=torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim,
                                                                    bias=bias))
            self.state_embeddings.add_module(name=f"{self.activation}{i + 1}", module=self.activation)

    def _build_policy_head(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, bias=True):
        self.policy_head = torch.nn.Sequential()
        self.policy_head.add_module(name="linear1",
                                    module=torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=bias))
        self.policy_head.add_module(name=f"{self.activation}1", module=self.activation)
        for i in range(1, n_layers - 1):
            self.policy_head.add_module(name=f"linear{i + 1}",
                                        module=torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim,
                                                               bias=bias))
            self.policy_head.add_module(name=f"{self.activation}{i + 1}", module=self.activation)

        self.policy_head.add_module(name="output",
                                    module=torch.nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=bias))

    def _build_value_head(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, bias=True):
        self.value_head = torch.nn.Sequential()
        self.value_head.add_module(name="linear1",
                                   module=torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=bias))
        self.value_head.add_module(name=f"{self.activation}1", module=self.activation)
        for i in range(1, n_layers - 1):
            self.value_head.add_module(name=f"linear{i + 1}",
                                       module=torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim,
                                                              bias=bias))
            self.value_head.add_module(name=f"{self.activation}{i + 1}", module=self.activation)

        self.value_head.add_module(name="output",
                                   module=torch.nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=bias))

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Outputs the policies and values associated with a batch of states. The states are flattened along non-batch
        dimension.
        :param x: a torch.Tensor representing a state or a batch of states of shape (n_batch, state_shape).
        :return: (policies, values) of type (torch.Tensor, torch.Tensor) where policies is of shape (n_batch, N_ACTIONS)
        and values is oh shape (n_batch, 1).
        """
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.state_embeddings(x)
        return self.policy_head(x), self.value_head(x)
