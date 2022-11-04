import pytorch_lightning as pl
import torch
import torch.nn as nn

from .exogenous import ConcatExogenous, EmbeddingExogenous


class GRU(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool,
        dropout: float,
        pred_input_dim: int,
        proj_size: int = 0,
        exogenous: str = "None",
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.last_proj = None
        if proj_size > 0:
            self.last_proj = nn.Linear(hidden_size, proj_size)
        self.pred_input_dim = pred_input_dim

        # FIXME: hyperparameter
        if exogenous == "concat":
            self.ex = ConcatExogenous()
        elif exogenous == "embedding":
            self.ex = EmbeddingExogenous()
        else:
            self.ex = None

    def forward(
        self, input_tensor, spa_cate, spa_flag, spa_temp_flag, spa_temp, h_0=None
    ):
        if self.ex is not None:
            input_tensor = self.ex(
                input_tensor, spa_cate, spa_flag, spa_temp_flag, spa_temp
            )

        # N * T * C * L
        bsize, seq_len, input_channel, length = input_tensor.size()

        # N * T * C * L -> N * L * T * C -> (N*L) * T * C
        input_tensor = input_tensor.permute(0, 3, 1, 2).reshape(
            (bsize * length), seq_len, input_channel
        )

        # (N*L) * T * Cin -> (N*L) * T * Ho
        out_tensor, h_n = self.gru(input_tensor, h_0)

        if self.last_proj is not None:
            out_tensor = self.last_proj(out_tensor)

        # (N*L) * T * Ho -> N * L * T * Ho -> N * T * Ho * L
        out_tensor = out_tensor.reshape(bsize, length, seq_len, -1).permute(0, 2, 3, 1)

        # N * Co * L
        return out_tensor[:, -1]
