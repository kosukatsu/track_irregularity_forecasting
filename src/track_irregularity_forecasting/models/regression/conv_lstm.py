from typing import List
from copy import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ConvLSTMCell(pl.LightningModule):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv1d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whi = nn.Conv1d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxf = nn.Conv1d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whf = nn.Conv1d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxc = nn.Conv1d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whc = nn.Conv1d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxo = nn.Conv1d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Who = nn.Conv1d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden(self, tensor, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Parameter(torch.zeros(1, hidden, shape[0]).type_as(tensor))
            self.Wcf = Parameter(torch.zeros(1, hidden, shape[0]).type_as(tensor))
            self.Wco = Parameter(torch.zeros(1, hidden, shape[0]).type_as(tensor))
        else:
            assert shape[0] == self.Wci.size()[2], "Input Height Mismatched!"
            # assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (
            torch.zeros(batch_size, hidden, shape[0]).type_as(tensor),
            torch.zeros(batch_size, hidden, shape[0]).type_as(tensor),
        )


class ConvLSTM(pl.LightningModule):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        kernel_size: int,
        last_kernel_size: int,
        pred_input_dim: int,
        output_channels: int,
        spa_cate_use: bool,
        spa_cate_timing: int,
        spa_cate_input_channel: int,
        spa_cate_uniques: List[int],
        spa_cate_output_channel: int,
        spa_flag_use: bool,
        spa_flag_timing: int,
        spa_flag_input_channel: int,
        spa_flag_output_channel: int,
        spa_temp_flag_use: bool,
        spa_temp_flag_timing: int,
        spa_temp_flag_input_channel: int,
        spa_temp_flag_output_channel: int,
        spa_temp_use: bool,
        spa_temp_timing: int,
        spa_temp_input_channel: int,
        channel_mode: str,  # increase, constant
    ):
        super().__init__()
        self.original_input_channel = input_channels

        self.input_channels = [input_channels] + copy(hidden_channels)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.pred_input_dim = pred_input_dim

        assert len(spa_cate_uniques) == spa_cate_input_channel
        self.spa_cate_use = spa_cate_use
        self.spa_cate_timing = spa_cate_timing

        self.spa_flag_use = spa_flag_use
        self.spa_flag_timing = spa_flag_timing

        self.spa_temp_flag_use = spa_temp_flag_use
        self.spa_temp_flag_timing = spa_temp_flag_timing

        self.spa_temp_use = spa_temp_use
        self.spa_temp_timing = spa_temp_timing
        self.spa_temp_input_channel = spa_temp_input_channel

        self.setEmbedding(
            spa_cate_uniques,
            spa_cate_input_channel,
            spa_cate_output_channel,
            spa_flag_input_channel,
            spa_flag_output_channel,
            spa_temp_flag_input_channel,
            spa_temp_flag_output_channel,
        )

        self.calcChannels(
            spa_cate_input_channel,
            spa_cate_output_channel,
            spa_flag_input_channel,
            spa_flag_output_channel,
            spa_temp_flag_input_channel,
            spa_temp_flag_output_channel,
            spa_temp_input_channel,
            mode=channel_mode,
        )

        # Define the ConvLSTM layer
        self.convlstmCells = []
        for i in range(self.num_layers):
            cell = ConvLSTMCell(
                self.input_channels[i],
                self.hidden_channels[i],
                self.kernel_size,
            )
            self.convlstmCells.append(cell)
        self.convlstmCells = nn.ModuleList(self.convlstmCells)

        # Prediction layer
        self.pred_conv = nn.Conv1d(
            self.pred_input_dim * self.hidden_channels[-1],
            output_channels,
            last_kernel_size,
            1,
            last_kernel_size // 2,
            bias=True,
        )

        self.tanh = nn.Tanh()

    def setEmbedding(
        self,
        spa_cate_uniques,
        spa_cate_input_channel,
        spa_cate_output_channel,
        spa_flag_input_channel,
        spa_flag_output_channel,
        spa_temp_flag_input_channel,
        spa_temp_flag_output_channel,
    ):
        if self.spa_cate_use:
            self.spa_cate_embeddings = []
            for num_unique in spa_cate_uniques:
                embedding = nn.Embedding(num_unique, spa_cate_output_channel)
                self.spa_cate_embeddings.append(embedding)
            self.spa_cate_embeddings = nn.ModuleList(self.spa_cate_embeddings)

        if self.spa_flag_use:
            self.spa_flag_embeddings = []
            for i in range(spa_flag_input_channel):
                embedding = nn.Embedding(2, spa_flag_output_channel)
                self.spa_flag_embeddings.append(embedding)
            self.spa_flag_embeddings = nn.ModuleList(self.spa_flag_embeddings)

        if self.spa_temp_flag_use:
            self.spa_temp_flag_embeddings = []
            for i in range(spa_temp_flag_input_channel):
                embedding = nn.Embedding(2, spa_temp_flag_output_channel)
                self.spa_temp_flag_embeddings.append(embedding)
            self.spa_temp_flag_embeddings = nn.ModuleList(self.spa_temp_flag_embeddings)

    def calcChannels(
        self,
        spa_cate_input_channel,
        spa_cate_output_channel,
        spa_flag_input_channel,
        spa_flag_output_channel,
        spa_temp_flag_input_channel,
        spa_temp_flag_output_channel,
        spa_temp_input_channel,
        mode,
    ):
        if mode == "constant":
            if self.spa_cate_use:
                self.input_channels[self.spa_cate_timing] += (
                    spa_cate_input_channel * spa_cate_output_channel
                )

            if self.spa_flag_use:
                self.input_channels[self.spa_flag_timing] += (
                    spa_flag_input_channel * spa_flag_output_channel
                )

            if self.spa_temp_flag_use:
                self.input_channels[self.spa_temp_flag_timing] += (
                    spa_temp_flag_input_channel * spa_temp_flag_output_channel
                )

            if self.spa_temp_use:
                self.input_channels[self.spa_temp_timing] += spa_temp_input_channel

        elif mode == "increase":
            input_channel = self.input_channels[0]
            for i in range(self.num_layers):
                self.input_channels[i] = input_channel
                if self.spa_cate_use and (i == self.spa_cate_timing):
                    self.input_channels[i] += (
                        spa_cate_input_channel * spa_cate_output_channel
                    )
                if self.spa_flag_use and (i == self.spa_flag_timing):
                    self.input_channels[i] += (
                        spa_flag_input_channel * spa_flag_output_channel
                    )
                if self.spa_temp_flag_use and (i == self.spa_temp_flag_timing):
                    self.input_channels[i] += (
                        spa_temp_flag_input_channel * spa_temp_flag_output_channel
                    )
                if self.spa_temp_use and (i == self.spa_temp_timing):
                    self.input_channels[i] += spa_temp_input_channel

                if self.input_channels[i] < self.hidden_channels[i]:
                    input_channel = self.hidden_channels[i]
                else:
                    self.hidden_channels[i] = self.input_channels[i]
                    input_channel = self.hidden_channels[i]

    def calc_spa_cate(self, input_tensor, spa_cate):
        N_batches, time_length, channel, length = input_tensor.shape
        N_batches_sc, channel_sc, length_sc = spa_cate.shape

        assert N_batches == N_batches_sc
        assert length == length_sc

        sc_embed_l = []
        for i in range(channel_sc):
            buf = self.spa_cate_embeddings[i](spa_cate[:, i, :].int())
            sc_embed_l.append(buf)
        # [tensor(N,L,E)]->tensor(N,L,EC)
        sc_embed_l = torch.cat(sc_embed_l, axis=-1)
        # tensor(N,L,EC)->tensor(N,EC,L)
        sc_embed_T = sc_embed_l.permute(0, 2, 1)
        N, EC, L = sc_embed_T.shape
        # tensor(N,EC,L)->tensor(N,1,EC,L)
        sc_embed_T = sc_embed_T.view(N, 1, EC, L)
        # tensor(N,1,EC,L)->tensor(N,T,EC,L)
        sc_embed = sc_embed_T.expand(-1, time_length, -1, -1)
        input_tensor = torch.cat([input_tensor, sc_embed], axis=2)

        return input_tensor

    def calc_spa_flag(self, input_tensor, spa_flag):
        N_batches, time_length, channel, length = input_tensor.shape
        N_batches_sf, channel_sf, length_sf = spa_flag.shape
        assert N_batches == N_batches_sf
        assert length == length_sf
        sf_embed_l = []
        for i in range(channel_sf):
            buf = self.spa_flag_embeddings[i](spa_flag[:, i, :].int())
            sf_embed_l.append(buf)
        # [tensor(N,L,E)]->tensor(N,L,ECC)
        sf_embedd_c = torch.cat(sf_embed_l, axis=-1)
        # tensor(N,L,ECC)->tensor(N,ECC,L)
        sf_embedd_T = sf_embedd_c.permute(0, 2, 1)
        N, ECC, L = sf_embedd_T.shape
        # tensor(N,ECC,L)->tensor(N,1,ECC,L)
        sf_embedd_T = sf_embedd_T.view(N, 1, ECC, L)
        # tensor(N,1,ECC,L)->tensor(N,1,ECC,L)
        sf_embedd = sf_embedd_T.expand(N_batches, time_length, ECC, L)
        input_tensor = torch.cat([input_tensor, sf_embedd], axis=2)

        return input_tensor

    def calc_spa_temp_flag(self, input_tensor, spa_temp_flag):
        N_batches, time_length, channel, length = input_tensor.shape
        N_batches_stf, time_length_stf, channel_stf, length_stf = spa_temp_flag.shape
        assert N_batches == N_batches_stf
        assert time_length == time_length_stf, "{}!={}".format(
            time_length, time_length_stf
        )
        assert length == length_stf

        stf_embedd_l = []
        for i in range(channel_stf):
            buf = self.spa_temp_flag_embeddings[i](spa_temp_flag[:, :, i, :].int())
            stf_embedd_l.append(buf)
        # [tensor(N,T,L)]->tensor(N,T,L,CC)
        stf_embed = torch.cat(stf_embedd_l, axis=-1)
        # tensor(N,T,L,CC)->tensor(N,T,CC,T)
        stf_embed_T = stf_embed.permute(0, 1, 3, 2)
        input_tensor = torch.cat([input_tensor, stf_embed_T], axis=2)
        return input_tensor

    def calc_spa_temp(self, input_tensor, spa_temp):
        input_tensor = torch.cat([input_tensor, spa_temp], axis=2)
        return input_tensor

    def forward(
        self,
        input_tensor,
        spa_cate,
        spa_flag,
        spa_temp_flag,
        spa_temp,
        h_0=None,
        c_0=None,
    ):
        if torch.isnan(input_tensor).any():
            print("track data contains null")
        if torch.isnan(spa_cate).any():
            print("spatinal categorical data contains null")
        if torch.isnan(spa_flag).any():
            print("spational flag data contains null")
        if torch.isnan(spa_temp_flag).any():
            print("spational temporal flag data contains null")
        if torch.isnan(spa_temp).any():
            print("spational temporal data contains null")
        bsize, seq_len, input_channel, length = input_tensor.size()
        internal_state = torch.Tensor(
            bsize, self.hidden_channels[-1] * self.pred_input_dim, length
        ).type_as(input_tensor)
        self.out_feature = []
        for i in range(self.num_layers):
            if self.spa_cate_use and (self.spa_cate_timing == i):
                input_tensor = self.calc_spa_cate(input_tensor, spa_cate)

            if self.spa_flag_use and (self.spa_flag_timing == i):
                input_tensor = self.calc_spa_flag(input_tensor, spa_flag)

            if self.spa_temp_flag_use and (self.spa_temp_flag_timing == i):
                input_tensor = self.calc_spa_temp_flag(input_tensor, spa_temp_flag)

            if self.spa_temp_use and (self.spa_temp_timing == i):
                input_tensor = self.calc_spa_temp(input_tensor, spa_temp)

            y = torch.Tensor(bsize, seq_len, self.hidden_channels[i], length).type_as(
                input_tensor
            )
            self.out_feature.append(input_tensor)
            (h, c) = self.convlstmCells[i].init_hidden(
                tensor=input_tensor,
                batch_size=bsize,
                hidden=self.hidden_channels[i],
                shape=(length,),
            )
            if h_0 is not None and c_0 is not None:
                (h, c) = (h_0, c_0)

            for step in range(seq_len):
                x = input_tensor[:, step, :, :]
                new_h, new_c = self.convlstmCells[i](x, h, c)
                (h, c) = (new_h, new_c)
                y[:, step, :, :] = new_h
                if i == self.num_layers - 1:
                    internal_start = step * self.hidden_channels[-1]
                    internal_stop = (step + 1) * self.hidden_channels[-1]
                    internal_state[:, internal_start:internal_stop] = new_h
            input_tensor = y
        self.out_feature.append(internal_state)
        y = self.pred_conv(internal_state)
        # Tensor(N,output_channel,L)
        return y
