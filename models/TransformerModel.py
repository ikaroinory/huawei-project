import torch
from torch import Tensor, nn

from .PositionEncoder import PositionEncoder


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0,
        max_len: int = 5000
    ):
        super().__init__()
        self.embedding = nn.Embedding(d_input, d_output)

        self.position_encoder = PositionEncoder(d_output, dropout, max_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_output, nhead=num_heads, dim_feedforward=d_hidden, batch_first=True),
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(d_output, d_output)

        self.d_hidden = d_output

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_hidden, device=x.device))
        x = self.position_encoder(x)

        encode = self.transformer_encoder(x)

        encode = torch.mean(encode, dim=1)

        output = self.output_layer(encode)

        return output
