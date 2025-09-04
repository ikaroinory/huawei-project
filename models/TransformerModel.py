import torch
from torch import Tensor, nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        num_heads: int,
        num_layers: int,
        max_len: int
    ):
        super().__init__()
        self.embedding = nn.Embedding(d_input, d_output)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_output, nhead=num_heads, dim_feedforward=d_hidden, batch_first=True),
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(d_output, d_output)

        self.d_hidden = d_output

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_hidden, device=x.device))

        encode = self.transformer_encoder(x)

        encode = torch.mean(encode, dim=1)

        output = self.output_layer(encode)

        return output
