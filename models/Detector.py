import torch
from torch import Tensor, nn

from .TransformerModel import TransformerModel


class Detector(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        num_heads: int,
        num_layers: int,
        behavior_sequence_max_len: int,
        normal_sequence_max_len: int,
        abnormal_sequence_max_len: int,
        *,
        dtype=None,
        device=None
    ):
        super().__init__()

        self.behavior_sequence_encoder = nn.Sequential(
            TransformerModel(
                d_input=d_input,
                d_hidden=d_hidden,
                d_output=d_hidden,
                num_heads=num_heads,
                num_layers=num_layers,
                max_len=behavior_sequence_max_len
            ),
            nn.ReLU()
        )
        self.normal_encoder = nn.Sequential(
            TransformerModel(
                d_input=d_input,
                d_hidden=d_hidden,
                d_output=d_hidden,
                num_heads=num_heads,
                num_layers=num_layers,
                max_len=normal_sequence_max_len
            ),
            nn.ReLU()
        )
        self.abnormal_encoder = nn.Sequential(
            TransformerModel(
                d_input=d_input,
                d_hidden=d_hidden,
                d_output=d_hidden,
                num_heads=num_heads,
                num_layers=num_layers,
                max_len=abnormal_sequence_max_len
            ),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(d_hidden * 3, 1)

        self.to(dtype)
        self.to(device)

    def forward(self, behavior_sequence: Tensor, normal_sequence: Tensor, abnormal_sequence: Tensor) -> Tensor:
        behavior_embedding = self.behavior_sequence_encoder(behavior_sequence)
        normal_embedding = self.normal_encoder(normal_sequence)
        abnormal_embedding = self.abnormal_encoder(abnormal_sequence)

        embedding = torch.cat([behavior_embedding, normal_embedding, abnormal_embedding], dim=-1)

        return self.output_layer(embedding)
