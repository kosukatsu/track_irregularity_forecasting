from typing import List, Union, Optional
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..datasets.input_sequence_dataset import InputSequenceDataset


class InputSequenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        track_data_dir: Union[str, Path],
        spational_categorical_exogenous_data_dirs: Union[List[str], List[Path]],
        spational_flag_exogenous_data_dirs: Union[List[str], List[Path]],
        spatio_temporal_flag_exogenous_data_dirs: Union[List[str], List[Path]],
        spatio_temporal_exogenous_data_dirs: Union[List[str], List[Path]],
        batch_size: int,
    ):
        super().__init__()
        self.tdd = track_data_dir
        self.scedds = spational_categorical_exogenous_data_dirs
        self.sfedds = spational_flag_exogenous_data_dirs
        self.stfedds = spatio_temporal_flag_exogenous_data_dirs
        self.stedds = spatio_temporal_exogenous_data_dirs
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        tdd = self.tdd
        scedd = self.scedds
        sfedds = self.sfedds
        stfedds = self.stfedds
        stedds = self.stedds
        if stage in (None, "fit"):
            self.train_dataset = InputSequenceDataset(
                tdd, scedd, sfedds, stfedds, stedds, "train"
            )
            self.valid_dataset = InputSequenceDataset(
                tdd, scedd, sfedds, stfedds, stedds,  "valid"
            )
        if stage in (None, "test"):
            self.test_dataset = InputSequenceDataset(
                tdd, scedd, sfedds, stfedds, stedds,  "test"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset)
