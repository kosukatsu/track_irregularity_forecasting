import os
from pathlib import Path
import pickle
from glob import glob

import torch
from torch.utils.data import Dataset


class TrackDataset(Dataset):
    """
    JR dataset.


    Args:
        data_dir (str): Directory with all the data.
        transform (callable): Optional transform to be applied on a sample.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        # files_list = os.listdir(data_dir)
        # if ".DS_Store" in files_list:
        #     files_list.remove(".DS_Store")
        files_list = glob(str(self.data_dir / "*.pkl"))
        self.data_list = sorted(
            files_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        self.data_len = len(self.data_list)
        self.transform = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        with open(data_path, "rb") as fin:
            data = pickle.load(fin)

        if self.transform:
            data = self.transform(data)

        return data
