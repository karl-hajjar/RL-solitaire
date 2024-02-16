import torch


class ResidualBlock(torch.nn.Module):
    """
    A class implementing a residual block, that is, some fully-connected or convolutional layers followed by a residual
    connection.
    """

    def __init__(self, n_layers: int, input_dim: int, hidden_dim: int, activation: torch.nn.Module, layer_type: str,
                 bias: bool = True, **kwargs):
        super().__init__()
        self.n_layer = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.n_layer = n_layers
        self.layer_type = layer_type
        self.bias = bias

        self.build_layers_(**kwargs)

    def build_layers_(self, **kwargs):
        if self.n_layers <= 1:
            raise ValueError(f"ResidualBlock only accepts at least 2 layers "
                             f"but the number of layers was {self.n_layer}")
        else:
            self.layers = torch.nn.Sequential()
            if self.layer_type == "linear":
                self.build_linear_layers_()

            elif self.layer_type == "conv":
                self.build_conv_layers_(**kwargs)
            else:
                raise ValueError(f"`layer_type` argument must be one of 'linear' or 'conv' but was {self.layer_type}")

    def build_linear_layers_(self):
        self.layers.add_module(name=f"linear1", module=torch.nn.Linear(in_features=self.input_dim,
                                                                       out_features=self.hidden_dim,
                                                                       bias=self.bias))
        self.layers.add_module(name=f"{self.activation}1", module=self.activation)
        for i in range(1, self.n_layers - 1):
            self.layers.add_module(name=f"linear{i + 1}",
                                   module=torch.nn.Linear(in_features=self.hidden_dim,
                                                          out_features=self.hidden_dim,
                                                          bias=self.bias))
            self.layers.add_module(name=f"{self.activation}{i + 1}", module=self.activation)

        self.layers.add_module(name=f"linear{self.n_layers}",
                               module=torch.nn.Linear(in_features=self.hidden_dim,
                                                      out_features=self.input_dim,
                                                      bias=self.bias))

    def build_conv_layers_(self, **kwargs):
        self.layers.add_module(name=f"conv1", module=torch.nn.Conv2d(in_channels=self.input_dim,
                                                                     out_channels=self.hidden_dim,
                                                                     padding="same",
                                                                     bias=self.bias, **kwargs))
        self.layers.add_module(name=f"{self.activation}1", module=self.activation)
        for i in range(1, self.n_layers - 1):
            self.layers.add_module(name=f"conv{i + 1}",
                                   module=torch.nn.Conv2d(in_channels=self.hidden_dim,
                                                          out_channels=self.hidden_dim,
                                                          padding="same",
                                                          bias=self.bias, **kwargs))
            self.layers.add_module(name=f"{self.activation}{i + 1}", module=self.activation)
        self.layers.add_module(name=f"conv{self.n_layers}",
                               module=torch.nn.Conv2d(in_channels=self.hidden_dim,
                                                      out_channels=self.hidden_dim,
                                                      padding="same",
                                                      bias=self.bias, **kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the residual block as x + Layers(x).
        :param x: Tensor of shape (batch_size, input_shape)
        :return: Tensor of shape (batch_size, input_shape)
        """
        return x + self.layers(x)
