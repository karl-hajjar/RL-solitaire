import math

import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.functional import softmax

from nn.network_config import NetConfig
from .skeleton import BasePolicyValueNet


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerPolicyValueNet(BasePolicyValueNet):
    """
    A class implementing a policy-value network with a Transformer encoder architecture.
    """

    def __init__(self, config: NetConfig):
        super().__init__(config)
        self._set_src_mask()

    def _build_state_embeddings(self, input_dim: int, hidden_dim: int, n_layers: int, bias=True,
                                dropout: float = 0.1, max_len: int = 5000, n_heads: int = 4,
                                feedforward_hidden_dim: int = 256):
        self.state_embeddings = torch.nn.Module()
        self.state_embeddings.add_module(name="transformer_encoder_layer",
                                         module=TransformerEncoderLayer(d_model=hidden_dim,
                                                                        nhead=n_heads,
                                                                        dim_feedforward=feedforward_hidden_dim,
                                                                        dropout=dropout,
                                                                        activation=self.activation,
                                                                        norm_first=True))
        self.state_embeddings.add_module(name="input_embedding",
                                         module=torch.nn.Linear(in_features=input_dim, out_features=hidden_dim,
                                                                bias=bias))
        self.state_embeddings.add_module(name="positional_encoder",
                                         module=PositionalEncoding(d_model=hidden_dim, dropout=dropout,
                                                                   max_len=max_len))
        self.state_embeddings.add_module(name="transformer_encoder",
                                         module=TransformerEncoder(self.state_embeddings.transformer_encoder_layer,
                                                                   n_layers))

    def _build_policy_head(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, bias=True,
                           dropout: float = 0.1, max_len: int = 5000, n_heads: int = 4,
                           feedforward_hidden_dim: int = 256):
        self.policy_head = torch.nn.Module()
        self.policy_head.add_module(name="transformer_encoder_layer",
                                    module=TransformerEncoderLayer(d_model=hidden_dim,
                                                                   nhead=n_heads,
                                                                   dim_feedforward=feedforward_hidden_dim,
                                                                   dropout=dropout,
                                                                   activation=self.activation,
                                                                   norm_first=True))
        self.policy_head.add_module(name="input_linear",
                                    module=torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=bias))
        self.policy_head.add_module(name="transformer_encoder",
                                    module=TransformerEncoder(self.policy_head.transformer_encoder_layer,
                                                              n_layers))
        self.policy_head.add_module(name="output_linear",
                                    module=torch.nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=bias))

    def _build_value_head(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, bias=True,
                          dropout: float = 0.1, max_len: int = 5000, n_heads: int = 4,
                          feedforward_hidden_dim: int = 256):
        self.value_head = torch.nn.Module()
        self.value_head.add_module(name="transformer_encoder_layer",
                                   module=TransformerEncoderLayer(d_model=hidden_dim,
                                                                  nhead=n_heads,
                                                                  dim_feedforward=feedforward_hidden_dim,
                                                                  dropout=dropout,
                                                                  activation=self.activation,
                                                                  norm_first=True))
        self.value_head.add_module(name="input_linear",
                                   module=torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False))
        self.value_head.add_module(name="transformer_encoder",
                                   module=TransformerEncoder(self.value_head.transformer_encoder_layer,
                                                             n_layers))
        self.value_head.add_module(name="output_linear",
                                   module=torch.nn.Linear(in_features=33*hidden_dim, out_features=output_dim,
                                                          bias=bias))

    def reformat_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unsqueeze the input along all dimensions except the batch and the channel dimension, that is convert an input of
        shape (N, 7, 7, N_STATE_CHANNELS) into an input of shape (N, 7*7, N_STATE_CHANNELS)
        :param x: Tensor containing the input tensor of shape (N, 7, 7, N_STATE_CHANNELS).
        :return: A Tensor which is an unsqueezed version of the input tensor, of shape (N, 7, 7, N_STATE_CHANNELS).
        """
        return self._swap_sequence_and_batch_dimensions(torch.flatten(x, start_dim=1, end_dim=2))

    @staticmethod
    def _swap_sequence_and_batch_dimensions(x: torch.Tensor) -> torch.Tensor:
        """
        Swaps the batch and sequence dimension: (N, 49, N_STATE_CHANNELS) -> (49, N, N_STATE_CHANNELS).
        :param x: Tensor of shape (N, 49, N_STATE_CHANNELS)
        :return: Tensor of shape (49, N, N_STATE_CHANNELS)
        """
        return x.reshape(x.shape[1], x.shape[0], x.shape[2])

    def get_policy(self, x: torch.Tensor):
        return softmax(self.get_policy_from_state_embeddings(self.get_state_embeddings(x)), dim=-1)

    def get_value(self, x: torch.Tensor):
        return self.get_value_from_state_embeddings(self.get_state_embeddings(x))

    def get_state_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reformat_input(x)
        x = self.state_embeddings.input_embedding(x)
        x = self.state_embeddings.positional_encoder(x)  # returns x + positional_encodings
        return self.state_embeddings.transformer_encoder(src=x, mask=self.src_mask, is_causal=False)

    def get_policy_from_state_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.policy_head.input_linear(x)
        x = self.policy_head.transformer_encoder(src=x, mask=self.src_mask, is_causal=False)
        x = self.policy_head.output_linear(x)  # shape (49, N, 4)~
        # Note: src_mask is True for indices that ARE MASKED,and we want to return the vector values at indices which
        # are not masked, hence the ~self.src_mask[0, :]
        return x[~self.src_mask[0, :], :, :].reshape(x.shape[1], -1)  # shape (N, 33*4) = (N, 132)

    def get_value_from_state_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.value_head.input_linear(x)
        x = self.value_head.transformer_encoder(src=x, mask=self.src_mask, is_causal=False)
        # Note: src_mask is True for indices that ARE MASKED,and we want to return the vector values at indices which
        # are not masked, hence the ~self.src_mask[0, :]
        x = x[~self.src_mask[0, :], :, :].reshape(x.shape[1], -1)  # shape (N, 33 * value_head_hidden_dim)
        return self.value_head.output_linear(x)  # shape (N, 1)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Outputs the policies and values associated with a batch of states.
        :param x: a torch.Tensor representing a state or a batch of states of shape (N, state_shape).
        :return: (policies, values) of type (torch.Tensor, torch.Tensor) where policies is of shape (N, N_ACTIONS)
        and values is of shape (N, 1).
        """
        x = self.get_state_embeddings(x)
        return self.get_policy_from_state_embeddings(x), self.get_value_from_state_embeddings(x)
