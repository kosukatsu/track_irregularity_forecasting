from tqdm import tqdm
from pathlib import Path
import pickle
import os

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch


def get_dates(data):
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y/%m/%d")
    dates = data.date.unique()
    return data, dates


def group_data_by_date(data, start_distance, end_distance, feature_order):
    data = data.loc[data["distance"] >= start_distance]
    data = data.loc[data["distance"] < end_distance]
    data = data.set_index(["date", "distance"])
    data = data[feature_order]
    data = data.to_xarray().to_array().data

    # C,T,L -> T,C,L
    return data.transpose(1, 0, 2)

def split_data(data, dates, split_date=None):
    """Split data into training and validation data, using a specified split date when given.


    Args:
        data (np.array): Data represented in a 3D numpy array.
        dates (str[]): A list of unique dates for index reference.
        split_date (str): Date which specifies the split point (inclusive w.r.t training data).

    Return:
        train_data (np.array): A 3D numpy array for training.
        valid_data (np.array): A 3D numpy array for validation.
    """
    split_point = np.where(dates == split_date)[0][0] + 1
    train_data = data[:split_point, :, :]
    valid_data = data[split_point:, :, :]

    train_dates = dates[dates <= split_date]
    valid_dates = dates[dates > split_date]
    return (train_data, valid_data, train_dates, valid_dates)


def calc_max_min(data, params, feature_order):
    method = params["method"]
    config_value = params["config_value"]
    if method == "measurement":
        maxs = np.nanmax(np.nanmax(data, axis=2), axis=0)
        mins = np.nanmin(np.nanmin(data, axis=2), axis=0)
        max_min = {}
        for i, f in enumerate(feature_order):
            max_min[f] = {}
            max_min[f]["max"] = maxs[i]
            max_min[f]["min"] = mins[i]
        return max_min

    elif method == "config":
        return {f: config_value[f] for f in feature_order}


def make_time_series(data, date, t=5):
    """Make data into a time series with n time points.


    Args:
        data (np.array): Data represented in a 3D numpy array.
        t (int): Length of the time series.

    Return:
        data (np.array): Generated time series data.
    """
    time_series = sliding_window_view(data, t, axis=0).copy()
    date_series = sliding_window_view(date, t).copy()
    # N,t,L,C
    time_series = np.transpose(time_series, (0, 3, 1, 2))
    return time_series, date_series


def split_data_target(data, de_params, general_params):
    feature_order = de_params["feature_order"]
    predict_columns = general_params["predict_columns"]
    total_len = general_params["total_len"]
    input_len = general_params["input_len"]
    predict_len = total_len - input_len

    indexs = []
    for c in predict_columns:
        index = feature_order.index(c)
        indexs.append(index)

    if predict_len != 1:
        target = data[:, input_len:, indexs]
    else:
        target = data[:, -1, indexs]

    data = data[:, :input_len]

    return data, target

def save_tensor(data, target, date, path, data_class, general_params):

    data = torch.Tensor(data)
    target = torch.Tensor(target)
    path = (
        Path(path)
        / general_params["section"]
        / "track"
        / general_params["bow_string"]
        / general_params["task"]
        / data_class
    )

    if os.path.isdir(path) == False:
        os.makedirs(path)

    file_cnt = 0
    for i in tqdm(range(data.shape[0])):
        if torch.isnan(data[i]).any() or torch.isnan(target[i]).any():
            continue
        with open(path / "{}.pkl".format((file_cnt)), "wb") as f:
            pickle.dump(
                {
                    "observation": data[i].clone(),
                    "target": target[i].clone(),
                    "dates": date[i],
                },
                f,
            )
        file_cnt += 1


def dump_data_engeering_params(
    general_params,
    data_engeering_params,
):
    dump_data_engeering_params = {
        "general_params": general_params,
        "data_engeering_params": data_engeering_params,
    }
    return dump_data_engeering_params
