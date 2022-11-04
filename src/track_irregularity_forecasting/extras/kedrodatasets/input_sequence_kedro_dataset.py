import os, sys
from pathlib import Path, PurePosixPath
from typing import Dict, Any, Union, List
import pickle

from kedro.io import AbstractDataSet
import numpy as np

from ...extras.datasets.input_sequence_dataset import InputSequenceDataset
from ...extras.datamodules.input_sequence_datamodule import InputSequenceDataModule
from ...utils import load_yml


class InputSequenceKedroDataSet(AbstractDataSet):
    def __init__(
        self,
        track_data_dir: Union[str, Path],
        spational_categorical_exogenous_data_dirs: Union[List[str], List[Path]],
        spational_flag_exogenous_data_dirs: Union[List[str], List[Path]],
        spatio_temporal_flag_exogenous_data_dirs: Union[List[str], List[Path]],
        spatio_temporal_exogenous_data_dirs: Union[List[str], List[Path]],
        flag_conf,
        batch_size: int,
    ):
        flag_conf = load_yml(flag_conf)
        filter_path = lambda paths, conf: list(np.array(paths)[np.array(conf)])

        self.track_data_dir = track_data_dir
        self.scedds = filter_path(
            spational_categorical_exogenous_data_dirs, flag_conf["spa_cate"]
        )
        self.sfedds = filter_path(
            spational_flag_exogenous_data_dirs, flag_conf["spa_flag"]
        )
        self.stfedds = filter_path(
            spatio_temporal_flag_exogenous_data_dirs, flag_conf["spa_temp_flag"]
        )
        self.stedds = filter_path(
            spatio_temporal_exogenous_data_dirs, flag_conf["spa_temp"]
        )
        self.batch_size = batch_size

    def _describe(self):
        return None

    def _save(self, data):
        pass

    def _load(self):
        return InputSequenceDataModule(
            self.track_data_dir,
            self.scedds,
            self.sfedds,
            self.stfedds,
            self.stedds,
            self.batch_size,
        )
