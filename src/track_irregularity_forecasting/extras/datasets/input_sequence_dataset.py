import os
from pathlib import Path
import pickle
from glob import glob
from typing import List, Union

import torch
from torch.utils.data import Dataset
import numpy as np

from ...utils import load_pickle


class InputSequenceDataset(Dataset):
    def __init__(
        self,
        track_data_dir: Union[str, Path],
        spational_categorical_exogenous_data_dirs: Union[List[str], List[Path]],
        spational_flag_exogenous_data_dirs: Union[List[str], List[Path]],
        spatio_temporal_flag_exogenous_data_dirs: Union[List[str], List[Path]],
        spatio_temporal_exogenous_data_dirs: Union[List[str], List[Path]],
        data_class: str,
        transform=None,
    ):
        self.track_data_dir = Path(track_data_dir)
        self.spational_categorical_exogenous_data_dirs = (
            spational_categorical_exogenous_data_dirs
        )
        self.spational_flag_exogenous_data_dirs = spational_flag_exogenous_data_dirs
        self.spatio_temporal_flag_exogenous_data_dirs = (
            spatio_temporal_flag_exogenous_data_dirs
        )
        self.spatio_temporal_exogenous_data_dirs = spatio_temporal_exogenous_data_dirs
        self.data_class = data_class

        files_list = glob(str(self.track_data_dir / self.data_class / "*.pkl"))
        data_list = sorted(
            files_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        self.data_len = len(data_list)
        self.transform = transform

        self.spa_cate_datas = []
        for p in self.spational_categorical_exogenous_data_dirs:
            with open(p, "rb") as f:
                buf = pickle.load(f)
            # numpy (L,)->Tensor(1,L)
            buf = torch.from_numpy(buf).unsqueeze(0)
            self.spa_cate_datas.append(buf)
        # [Tensor(1,L),...] -> Tensor(C,L) -> Tensor(N,C,L)
        self.spa_cate_datas = (
            torch.cat(self.spa_cate_datas, dim=0)
            .unsqueeze(0)
            .expand(self.data_len, -1, -1)
        )

        self.spa_flag_datas = []
        for p in self.spational_flag_exogenous_data_dirs:
            with open(p, "rb") as f:
                buf = pickle.load(f)
                # numpy (L,C) -> Tensor (C,L)
            buf = torch.from_numpy(buf).permute(1, 0)
            self.spa_flag_datas.append(buf)
        # [Tensor(C,L),...] -> Tensor (CC,L) -> Tensor (1,CC,L) -> Tensor(N,CC,L)
        self.spa_flag_datas = (
            torch.cat(self.spa_flag_datas, dim=0)
            .unsqueeze(0)
            .expand(self.data_len, -1, -1)
        )

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx: int):
        path = Path(self.track_data_dir) / self.data_class / "{}.pkl".format(idx)
        with open(path, "rb") as f:
            track_data = pickle.load(f)

        spa_temp_flag_datas = []
        for stefdd in self.spatio_temporal_flag_exogenous_data_dirs:
            path = Path(stefdd) / self.data_class / "{}.pkl".format(idx)
            with open(path, "rb") as f:
                # array(T,L,C)
                buf = pickle.load(f)["observation"]
            spa_temp_flag_datas.append(buf)
        # [torch(T,C,L)]->torch(T,CC,L)
        spa_temp_flag_datas = torch.cat(spa_temp_flag_datas, dim=1)

        spa_temp_datas = []
        for stedd in self.spatio_temporal_exogenous_data_dirs:
            path = Path(stedd) / self.data_class / "{}.pkl".format(idx)
            with open(path, "rb") as f:
                # array(T,C,L)
                buf = pickle.load(f)["observation"]
            # buf = torch.from_numpy(buf)
            spa_temp_datas.append(buf)
        # [torch(T,C,L)]->torch(T,CC,L)
        spa_temp_datas = torch.cat(spa_temp_datas, dim=1)

        input_len = track_data["observation"].shape[0]

        return (
            track_data["observation"],
            track_data["target"],
            track_data["dates"][input_len],
            self.spa_cate_datas[idx],
            self.spa_flag_datas[idx],
            spa_temp_flag_datas,
            spa_temp_datas,
        )
