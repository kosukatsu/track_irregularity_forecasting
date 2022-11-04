import os
import shutil
from pathlib import Path


def _remove_directory(path):
    shutil.rmtree(path)
    os.mkdir(path)


def remove_track_series(path, section, bow_string, task):
    path = Path(path) / section / "track" / bow_string / task
    _remove_directory(path)


def remove_work(path, section):
    path = Path(path) / section / "work"
    _remove_directory(path)


def remove_ballast_age(path, section):
    path = Path(path) / section / "ballast_age"
    _remove_directory(path)


def remove_rainfall(path, section):
    path = Path(path) / section / "rainfall"
    _remove_directory(path)


def remove_tonnage(path, section):
    path = Path(path) / section / "tonnage"
    _remove_directory(path)
