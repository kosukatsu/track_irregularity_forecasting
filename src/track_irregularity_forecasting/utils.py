from pathlib import Path
import os
import pickle

import numpy as np
import torch
import yaml


def merge_dictionary(dict1, dict2):
    dict1.update(dict2)
    return dict1


def normalize_min_max_scale(torch_track_data, feature_order, min_max_dict, dim=0):
    assert torch_track_data.shape[dim] == len(feature_order)

    min = []
    max = []
    for feature in feature_order:
        min.append(min_max_dict[feature]["min"])
        max.append(min_max_dict[feature]["max"])
    min = np.array(min)
    max = np.array(max)

    shape = np.ones(len(torch_track_data.shape))
    shape[dim] = len(feature_order)

    min = min.reshape(shape.astype(int))
    max = max.reshape(shape.astype(int))

    min = torch.from_numpy(min).type_as(torch_track_data)
    max = torch.from_numpy(max).type_as(torch_track_data)

    return (2.0 * (torch_track_data - min) / (max - min)) - 1


def denormalize_min_max_scale(torch_track_data, feature_order, min_max_dict, dim=0):
    assert torch_track_data.shape[dim] == len(feature_order)

    min = []
    max = []
    for feature in feature_order:
        min.append(min_max_dict[feature]["min"])
        max.append(min_max_dict[feature]["max"])
    min = np.array(min)
    max = np.array(max)

    shape = np.ones(len(torch_track_data.shape))
    shape[dim] = len(feature_order)

    min = min.reshape(shape.astype(int))
    max = max.reshape(shape.astype(int))

    min = torch.from_numpy(min).type_as(torch_track_data)
    max = torch.from_numpy(max).type_as(torch_track_data)

    return ((torch_track_data * (max - min)) + max + min) / 2.0


# I/O
def load_yml(path):
    with open(path, mode="r") as f:
        data = yaml.safe_load(f)
    return data


def save_yml(path, data):
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    path = str(path)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(path, data):
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    path = str(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)
