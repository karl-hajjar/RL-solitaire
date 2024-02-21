import torch


class ResidualBlock(torch.nn.Module):
    """
    A class implementing a residual block, that is, some fully-connected or convolutional layers followed by a residual
    connection.
    """

    def __init__(self, input_dim: int, hidden_dim: int, activation: torch.nn.Module, layer_type: str, n_layers: int = 2,
                 bias: bool = True, **kwargs):
        """
        A residual connection block with a number of intermediate layers. Note that there is no activation after the
        last layer. By design, the input and output of the block have exactly the same shape.
        :param n_layers: int, number of intermediate layers in the residual block
        :param input_dim: int, the dimension (or number of channels for a conv architecture) of the input to the
        residual block.
        :param hidden_dim: int, the dimension (or number of channels for a conv architecture) of the layers which are
        neither the first or last layer in the block.
        :param activation: torch.nn.Module, the activation function.
        :param layer_type: str, either 'linear' or 'conv', representing the type of layers implemented inside the
        residual block.
        :param bias: bool, whether to use a bias term or not.
        :param kwargs: dict, additional key-word arguments to be passed in the case of convolutional blocks (such as the
        kernel size).
        """
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.n_layers = n_layers
        self.layer_type = layer_type
        self.bias = bias

        self._build_layers(**kwargs)

    def _build_layers(self, **kwargs):
        if self.n_layers <= 1:
            raise ValueError(f"ResidualBlock only accepts at least 2 layers "
                             f"but the number of layers was {self.n_layers}")
        else:
            self.layers = torch.nn.Sequential()
            if self.layer_type == "linear":
                self._build_linear_layers()

            elif self.layer_type == "conv":
                self._build_conv_layers(**kwargs)
            else:
                raise ValueError(f"`layer_type` argument must be one of 'linear' or 'conv' but was {self.layer_type}")

    def _build_linear_layers(self):
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

    def _build_conv_layers(self, **kwargs):
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
                                                      out_channels=self.input_dim,
                                                      padding="same",
                                                      bias=self.bias, **kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the residual block as x + Layers(x).
        :param x: Tensor of shape (batch_size, input_shape)
        :return: Tensor of shape (batch_size, input_shape)
        """
        return x + self.layers(x)
