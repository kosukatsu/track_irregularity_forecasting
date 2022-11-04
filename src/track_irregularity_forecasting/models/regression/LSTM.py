import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .exogenous import ConcatExogenous,EmbeddingExogenous

# LSTM w/ peephole connections


class LSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.Wxi = nn.Linear(input_size, hidden_size, bias)
        self.Whi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wci = Parameter(torch.zeros(1, hidden_size))

        self.Wxf = nn.Linear(input_size, hidden_size, bias)
        self.Whf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wcf = Parameter(torch.zeros(1, hidden_size))

        self.Wxg = nn.Linear(input_size, hidden_size, bias)
        self.Whg = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wxo = nn.Linear(input_size, hidden_size, bias)
        self.Who = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wco = Parameter(torch.zeros(1, hidden_size))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x, hx):
        h, c = hx
        i = self.sigmoid(self.Wxi(x) + self.Whi(h) + self.Wci * c)
        f = self.sigmoid(self.Wxf(x) + self.Whf(h) + self.Wcf * c)
        g = self.tanh(self.Wxg(x) + self.Whg(h))
        o = self.sigmoid(self.Wxo(x) + self.Who(h) + self.Wco * c)
        ct = g * c + i * g
        ht = o * self.tanh(c)
        return (ht, ct)


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias: bool = True,
        # batch_first:bool=True,
        dropout: float = 0,
        # bidirectional:bool=False,
        proj_size: int = 0,
    ):
        super().__init__()

        self.cells = []
        for i in range(num_layers):
            if i == 0:
                i_channel = input_size
            else:
                i_channel = hidden_size
            self.cells.append(LSTMCell(i_channel, hidden_size, bias))
        self.cells = nn.ModuleList(self.cells)

        self.hidden_size=hidden_size
        self.num_layers=num_layers

        self.dropout = nn.Dropout(p=dropout)

        self.last_proj = None
        if proj_size > 0:
            self.last_proj = nn.Linear(hidden_size, proj_size)

        

    def forward(
        self,
        input,
        hx,
    ):
        N,T,C=input.shape
        
        out_tensor=torch.empty((N,T,self.hidden_size)).to(input.device)
        if (hx is None) or (hx[0] is None) or (hx[1] is None):
            hs=[torch.zeros((N,self.hidden_size)).to(input.device) for _ in range(self.num_layers)]
            cs=[torch.zeros((N,self.hidden_size)).to(input.device) for _ in range(self.num_layers)]
        else:
            hs,cs=hx

        for t in range(T):
            x=input[:,t]

            for i in range(self.num_layers):
                h,c=self.cells[i](x,(hs[i],cs[i]))
                hs[i]=h
                cs[i]=c
                x=self.dropout(h)
            out_tensor[:,t]=h
        
        if self.last_proj is not None:
            out_tensor=self.last_proj(out_tensor)
        
        return out_tensor,(hs[-1],cs[-1])

class VanillaLSTM(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        pred_input_dim: int,
        bias: bool = True,
        dropout: float = 0,
        proj_size: int = 0,
        exogenous: str = "None",
    ):
        super().__init__()
        self.lstm=LSTM(input_size,hidden_size,num_layers,bias,dropout,proj_size)
        self.pred_input_dim=pred_input_dim

        # FIXME: hyperparameter
        if exogenous == "concat":
            self.ex = ConcatExogenous()
        elif exogenous == "embedding":
            self.ex = EmbeddingExogenous()
        else:
            self.ex = None

    def forward(
        self, 
        input_tensor, 
        spa_cate, 
        spa_flag, 
        spa_temp_flag, 
        spa_temp,
        h_0=None,
        c_0=None
    ):

        if self.ex is not None:
            input_tensor=self.ex(
                input_tensor,spa_cate,spa_flag,spa_temp_flag,spa_temp
            )
        # N * T * C * L
        bsize, seq_len, input_channel, length = input_tensor.size()
        # N * T * C * L -> N * L * T * C -> (N*L) * T * C
        input_tensor = input_tensor.permute(0, 3, 1, 2).reshape(
            (bsize * length), seq_len, input_channel
        )
        if (h_0 is not None) and (c_0 is not None):
            # (N,C,L) -> (N,L,C) -> (N*L,C)
            h_0=h_0.permute(0,2,1).reshape(bsize*length,-1)
            c_0=c_0.permute(0,2,1).reshape(bsize*length,-1)


        # (N*L) * T * Cin -> (N*L) * T * Ho
        h_n,c_n = self.lstm(input_tensor,(h_0,c_0))
        
        # (N*L) * T * Ho -> N * L * T * Ho -> N * T * Ho * L
        h_n = h_n.reshape(bsize, length, seq_len, -1).permute(0, 2, 3, 1)

        # N * Co * L
        return h_n[:, -1]


    