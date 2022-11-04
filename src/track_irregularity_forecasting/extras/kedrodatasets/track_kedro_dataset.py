import os, sys
from pathlib import Path, PurePosixPath
from typing import Dict, Any
import pickle

# sys.path.append(os.pardir)

from kedro.io import AbstractDataSet

from track_irregularity_forecasting.extras.datasets.track_dataset import TrackDataset


class TrackKedroDataSet(AbstractDataSet):
    def __init__(
        self, filepath: str, sec: str, bow_string: str, data_name: str
    ) -> None:
        self._filepath = (
            PurePosixPath(filepath) / sec / "track" / bow_string / data_name
        )
        self._sec = sec
        self._bow_string = bow_string
        self._data_name = data_name

    def _load(self) -> TrackDataset:
        return TrackDataset(self._filepath)

    def _save(self, data) -> None:
        file_num = len(data)
        for i in range(file_num):
            path = self._filepath / "{}.pkl".format(i)
            with open(path, "rb") as f:
                pickle.dump(data[i], f)

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix() / "0.pkl").exists()

    def _describe(self) -> Dict[str, Any]:
        return {
            "section": self._sec,
            "bow_string_m": self._bow_string,
            "data_name": self._data_name,
        }
