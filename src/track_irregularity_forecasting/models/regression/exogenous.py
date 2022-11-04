from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class EmbeddingExogenous(nn.Module):
    def __init__(
        self,
        spa_cate_use: bool = True,
        spa_cate_input_channel: int = 1,
        spa_cate_uniques: List[int] = [
            5,
        ],
        spa_cate_output_channel: int = 4,
        spa_flag_use: bool = True,
        spa_flag_input_channel: int = 4,
        spa_flag_output_channel: int = 4,
        spa_temp_flag_use: bool = True,
        spa_temp_flag_input_channel: int = 9,
        spa_temp_flag_output_channel: int = 4,
        spa_temp_use: bool = True,
        spa_temp_input_channel: int = 6,
    ):
        super().__init__()
        self.spa_cate_use = spa_cate_use
        self.spa_cate_input_channel = spa_cate_input_channel
        self.spa_cate_output_channel = spa_cate_output_channel

        self.spa_flag_use = spa_flag_use
        self.spa_flag_input_channel = spa_flag_input_channel
        self.spa_flag_output_channel = spa_flag_output_channel

        self.spa_temp_flag_use = spa_temp_flag_use
        self.spa_temp_flag_input_channel = spa_temp_flag_input_channel
        self.spa_temp_flag_output_channel = spa_temp_flag_output_channel

        self.spa_temp_use = spa_temp_use
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

    def forward(
        self,
        input_tensor,
        spa_cate,
        spa_flag,
        spa_temp_flag,
        spa_temp,
    ):
        if self.spa_cate_use:
            input_tensor = self.calc_spa_cate(input_tensor, spa_cate)
        if self.spa_flag_use:
            input_tensor = self.calc_spa_flag(input_tensor, spa_flag)
        if self.spa_temp_flag_use:
            input_tensor = self.calc_spa_temp_flag(input_tensor, spa_temp_flag)
        if self.spa_temp_use:
            input_tensor = self.calc_spa_temp(input_tensor, spa_temp)
        return input_tensor

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


class ConcatExogenous(nn.Module):
    # FIXME: hard coding
    def __init__(
        self,
        spa_cate_use=True,
        spa_flag_use=True,
        spa_temp_flag_use=True,
        spa_temp_use=True,
        spa_cate_uniques: List[int] = [
            5,
        ],
    ):
        super().__init__()
        self.spa_cate_use = spa_cate_use
        self.spa_flag_use = spa_flag_use
        self.spa_temp_flag_use = spa_temp_flag_use
        self.spa_temp_use = spa_temp_use
        self.spa_cate_uniques = spa_cate_uniques

    def forward(self, x, spa_cate, spa_flag, spa_temp_flag, spa_temp):
        if self.spa_cate_use:
            # FIXME: hardcoding
            spa_cate_uniques = torch.Tensor(self.spa_cate_uniques).to(x.device, x.dtype)
            spa_cate_uniques = spa_cate_uniques.view(1, -1, 1)
            spa_cate_cast = spa_cate.to(x.dtype) / (spa_cate_uniques - 1)
            x = self.calc_spa(x, spa_cate_cast)
        if self.spa_flag_use:
            spa_flag_cast = spa_flag.to(x.dtype)
            x = self.calc_spa(x, spa_flag_cast)
        if self.spa_temp_flag_use:
            spa_temp_flag_cast = spa_temp_flag.to(x.dtype)
            x = self.calc_spa_temp(x, spa_temp_flag_cast)
        if self.spa_temp_use:
            x = self.calc_spa_temp(x, spa_temp)

        return x

    def calc_spa(self, x, spa_cate):
        N, T, C, L = x.shape
        N_e, C_e, L_e = spa_cate

        assert N == N_e
        assert L == L_e

        # N,C_e,L_e -> N,1,C_e,L
        spa_cate = spa_cate.view(N, 1, C_e, L_e)
        # N,1,C_e,L -> N,T,C_e,L
        spa_cate_expand = spa_cate.expand(-1, T, -1, -1)
        # concat x and spa_cate
        concat_x = torch.cat([x, spa_cate_expand], axis=2)

        return concat_x

    def calc_spa_temp(self, x, spa_temp_flag):
        N, T, C, L = x.shape
        N_e, T_e, C_e, L_e = spa_temp_flag.shape

        assert N == N_e
        assert T == T_e
        assert L == L_e

        x_concat = torch.cat([x, spa_temp_flag], axis=2)

        return x_concat
