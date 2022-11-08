import os, sys
sys.path.append(os.pardir)

from pathlib import Path
import pickle
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from ..data_engeering.nodes import split_data, make_time_series, calc_max_min
from ...utils import load_pickle

def cvt_structure_to_array(df, start_distance, end_distance):
    df = df.loc[df["distance"] >= start_distance]
    df = df.loc[df["distance"] < end_distance]
    data_len = len(df)
    structure = df.to_numpy()[:, 1:]
    return structure.reshape(data_len, -1).argmax(axis=1)


def cvt_ijej_to_array(df, start_distance, end_distance):
    df = df.loc[df["distance"] >= start_distance]
    df = df.loc[df["distance"] < end_distance]
    return df.to_numpy()[:, 1:]


def cvt_welding_to_array(df, start_distance, end_distance):
    df["weld_r"] = df["flash_r"] + df["gas_r"] + df["enclose_r"]
    df["weld_l"] = df["flash_l"] + df["gas_l"] + df["enclose_l"]
    df = df.sort_values("distance")
    df = df.loc[df["distance"] >= start_distance]
    df = df.loc[df["distance"] < end_distance]
    return df[["weld_r", "weld_l"]].to_numpy()


def save_tensor(data, date, dir_name, path, data_class, section, bow_string, task):
    data = torch.Tensor(data)
    save_path = Path(path) / section / dir_name / data_class
    track_path = Path(path) / section / "track" / bow_string / task / data_class
    os.makedirs(save_path, exist_ok=True)
    file_cnt = 0
    track = load_pickle(track_path / "{}.pkl".format(file_cnt))
    for i in tqdm(range(data.shape[0])):
        if track["dates"][0] != date[i][0]:
            continue
        with open(save_path / "{}.pkl".format((file_cnt)), "wb") as f:
            pickle.dump(
                {
                    "observation": data[i].clone(),
                    "dates": date[i],
                },
                f,
            )
        file_cnt += 1
        if os.path.isfile(track_path / "{}.pkl".format(file_cnt)):
            track = load_pickle(track_path / "{}.pkl".format(file_cnt))
        else:
            break


def make_work_time_series(
    work_df: pd.DataFrame,
    series_len: int,
    split_date: str,
    valid_split_date: str,
    work_order: List[str],
    section: str,
    bow_string: str,
    task: str,
    start_distance: int,
    end_distance: int,
):
    num_work = len(work_order)
    work_df = work_df.sort_values(["date", "distance"])
    work_df = work_df.loc[work_df["distance"] >= start_distance]
    work_df = work_df.loc[work_df["distance"] < end_distance]
    dates = work_df["date"].dt.strftime("%Y/%m/%d").unique()
    work_data = work_df[work_order].to_numpy().reshape(len(dates), -1, num_work)
    train_valid_data, test_data, train_valid_dates, test_dates = split_data(
        work_data, dates, split_date
    )
    train_data, valid_data, train_dates, valid_dates = split_data(
        train_valid_data, train_valid_dates, valid_split_date
    )

    train_data, train_dates = make_time_series(train_data, train_dates, series_len)
    valid_data, valid_dates = make_time_series(valid_data, valid_dates, series_len)
    test_data, test_dates = make_time_series(test_data, test_dates, series_len)
    wo_split_data, wo_split_dates = make_time_series(work_data, dates, series_len)

    # convert to N,t,C,L
    train_data = train_data.transpose(0, 1, 3, 2)
    valid_data = valid_data.transpose(0, 1, 3, 2)
    test_data = test_data.transpose(0, 1, 3, 2)
    wo_split_data = wo_split_data.transpose(0, 1, 3, 2)

    save_tensor(
        train_data,
        train_dates,
        "work",
        "./data/05_model_input",
        "train",
        section,
        bow_string,
        task,
    )
    save_tensor(
        valid_data,
        valid_dates,
        "work",
        "./data/05_model_input",
        "valid",
        section,
        bow_string,
        task,
    )
    save_tensor(
        test_data,
        test_dates,
        "work",
        "./data/05_model_input",
        "test",
        section,
        bow_string,
        task,
    )
    save_tensor(
        wo_split_data,
        wo_split_dates,
        "work",
        "./data/05_model_input",
        "wo_split",
        section,
        bow_string,
        task,
    )

def make_rainfall_series(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    section: str,
    split_date: str,
    valid_split_date: str,
    params,
    series_len: int,
    bow_string,
    task,
    start_distance: int,
    end_distance: int,
):
    L = 15000
    rainfall = np.empty((L, len(df1.index), 4))
    rainfall[:11000] = df1.to_numpy().reshape((1, len(df1.index), 4))
    rainfall[11000:] = df2.to_numpy().reshape((1, len(df1.index), 4))
    rainfall = rainfall[(start_distance - 27000) : (end_distance - 27000)]
    dates = df1.index.strftime("%Y/%m/%d").to_numpy()
    
    rainfall = rainfall.transpose((1, 0, 2))
    train_valid_data, test_data, train_valid_dates, test_dates = split_data(
        rainfall, dates, split_date
    )
    train_data, valid_data, train_dates, valid_dates = split_data(
        train_valid_data, train_valid_dates, valid_split_date
    )
    max_min_dict = calc_max_min(
        train_data.transpose(0, 2, 1), params, df1.columns.to_list()
    )
    maxs = [max_min_dict[column]["max"] for column in df1.columns.to_list()]
    maxs = np.array(maxs).reshape(1, 1, -1)
    mins = [max_min_dict[column]["min"] for column in df1.columns.to_list()]
    mins = np.array(mins).reshape(1, 1, -1)

    train_data = (2.0 * (train_data - mins) / (maxs - mins)) - 1
    valid_data = (2.0 * (valid_data - mins) / (maxs - mins)) - 1
    test_data = (2.0 * (test_data - mins) / (maxs - mins)) - 1
    rainfall = (2.0 * (rainfall - mins) / (maxs - mins)) - 1

    train_data, train_dates = make_time_series(train_data, train_dates, series_len)
    valid_data, valid_dates = make_time_series(valid_data, valid_dates, series_len)
    test_data, test_dates = make_time_series(test_data, test_dates, series_len)
    wo_split_data, wo_split_dates = make_time_series(rainfall, dates, series_len)

    # convert to N,t,C,L
    train_data = train_data.transpose(0, 1, 3, 2)
    valid_data = valid_data.transpose(0, 1, 3, 2)
    test_data = test_data.transpose(0, 1, 3, 2)
    wo_split_data = wo_split_data.transpose(0, 1, 3, 2)

    save_tensor(
        train_data,
        train_dates,
        "rainfall",
        "./data/05_model_input",
        "train",
        section,
        bow_string,
        task,
    )
    save_tensor(
        valid_data,
        valid_dates,
        "rainfall",
        "./data/05_model_input",
        "valid",
        section,
        bow_string,
        task,
    )
    save_tensor(
        test_data,
        test_dates,
        "rainfall",
        "./data/05_model_input",
        "test",
        section,
        bow_string,
        task,
    )
    save_tensor(
        wo_split_data,
        wo_split_dates,
        "rainfall",
        "./data/05_model_input",
        "wo_split",
        section,
        bow_string,
        task,
    )
    return max_min_dict


def make_ballast_age_series(
    ballast_age: pd.DataFrame,
    params,
    section: int,
    split_date: str,
    valid_split_date: str,
    series_len: int,
    structure: pd.DataFrame,
    bow_string: str,
    task: str,
    start_distance: int,
    end_distance: int,
):
    dates = ballast_age.index.strftime("%Y/%m/%d").to_numpy()
    ballast_age.columns = ballast_age.columns.astype(int)
    ballast_age = ballast_age.loc[:, ballast_age.columns >= start_distance]
    ballast_age = ballast_age.loc[:, ballast_age.columns < end_distance]
    ballast_age = ballast_age.to_numpy()
    T, L = ballast_age.shape
    ballast_age = ballast_age.reshape((T, L, 1))
    structure = structure.loc[structure["distance"] >= start_distance]
    structure = structure.loc[structure["distance"] < end_distance]
    bridge = structure["bridge"].to_numpy().astype(bool)
    ballast_age[:, bridge] = 0
    train_valid_data, test_data, train_valid_dates, test_dates = split_data(
        ballast_age, dates, split_date
    )
    train_data, valid_data, train_dates, valid_dates = split_data(
        train_valid_data, train_valid_dates, valid_split_date
    )
    max_min_dict = calc_max_min(train_data.transpose(0, 2, 1), params, ["ballast_age"])

    max = max_min_dict["ballast_age"]["max"]
    min = max_min_dict["ballast_age"]["min"]
    train_data = (2.0 * (train_data - min) / (max - min)) - 1
    valid_data = (2.0 * (valid_data - min) / (max - min)) - 1
    test_data = (2.0 * (test_data - min) / (max - min)) - 1
    ballast_age = (2.0 * (ballast_age - min) / (max - min)) - 1

    train_data, train_dates = make_time_series(train_data, train_dates, series_len)
    valid_data, valid_dates = make_time_series(valid_data, valid_dates, series_len)
    test_data, test_dates = make_time_series(test_data, test_dates, series_len)
    wo_split_data, wo_split_dates = make_time_series(ballast_age, dates, series_len)

    # convert to N,t,C,L
    train_data = train_data.transpose(0, 1, 3, 2)
    valid_data = valid_data.transpose(0, 1, 3, 2)
    test_data = test_data.transpose(0, 1, 3, 2)
    wo_split_data = wo_split_data.transpose(0, 1, 3, 2)

    save_tensor(
        train_data,
        train_dates,
        "ballast_age",
        "./data/05_model_input",
        "train",
        section,
        bow_string,
        task,
    )
    save_tensor(
        valid_data,
        valid_dates,
        "ballast_age",
        "./data/05_model_input",
        "valid",
        section,
        bow_string,
        task,
    )
    save_tensor(
        test_data,
        test_dates,
        "ballast_age",
        "./data/05_model_input",
        "test",
        section,
        bow_string,
        task,
    )
    save_tensor(
        wo_split_data,
        wo_split_dates,
        "ballast_age",
        "./data/05_model_input",
        "wo_split",
        section,
        bow_string,
        task,
    )

    return max_min_dict


def make_tonnage_series(
    df1: pd.DataFrame,
    section: int,
    split_date: str,
    valid_split_date: str,
    params,
    series_len: int,
    bow_string: str,
    task: str,
    start_distance: int,
    end_distance: int,
):
    L = 15000
    dates = df1.index.strftime("%Y/%m/%d").to_numpy()
    tonnage = np.empty((len(dates), L))
    tonnage[:, :] = df1["delta_load"].to_numpy().reshape(-1, 1)
    tonnage = tonnage[:, (start_distance - 27000) : (end_distance - 27000)]

    tonnage = tonnage.reshape(len(dates), (end_distance - start_distance), 1)

    train_valid_data, test_data, train_valid_dates, test_dates = split_data(
        tonnage, dates, split_date
    )
    train_data, valid_data, train_dates, valid_dates = split_data(
        train_valid_data, train_valid_dates, valid_split_date
    )

    max_min_dict = calc_max_min(train_data.transpose(0, 2, 1), params, ["tonnage"])

    max = max_min_dict["tonnage"]["max"]
    min = max_min_dict["tonnage"]["min"]

    train_data = (2.0 * (train_data - min) / (max - min)) - 1
    valid_data = (2.0 * (valid_data - min) / (max - min)) - 1
    test_data = (2.0 * (test_data - min) / (max - min)) - 1
    tonnage = (2.0 * (tonnage - min) / (max - min)) - 1

    train_data, train_dates = make_time_series(train_data, train_dates, series_len)
    valid_data, valid_dates = make_time_series(valid_data, valid_dates, series_len)
    test_data, test_dates = make_time_series(test_data, test_dates, series_len)
    wo_split_data, wo_split_dates = make_time_series(tonnage, dates, series_len)

    # convert to N,t,C,L
    train_data = train_data.transpose(0, 1, 3, 2)
    valid_data = valid_data.transpose(0, 1, 3, 2)
    test_data = test_data.transpose(0, 1, 3, 2)
    wo_split_data = wo_split_data.transpose(0, 1, 3, 2)

    save_tensor(
        train_data,
        train_dates,
        "tonnage",
        "./data/05_model_input",
        "train",
        section,
        bow_string,
        task,
    )
    save_tensor(
        valid_data,
        valid_dates,
        "tonnage",
        "./data/05_model_input",
        "valid",
        section,
        bow_string,
        task,
    )
    save_tensor(
        test_data,
        test_dates,
        "tonnage",
        "./data/05_model_input",
        "test",
        section,
        bow_string,
        task,
    )
    save_tensor(
        wo_split_data,
        wo_split_dates,
        "tonnage",
        "./data/05_model_input",
        "wo_split",
        section,
        bow_string,
        task,
    )

    return max_min_dict


def dump_exogenous_series_params(
    general_params,
    data_engeering_params,
    preprocess_exogenous,
):
    exogenous_series_params = {
        "general_params": general_params,
        "data_engeering_params": data_engeering_params,
        "preprocess_exogenous": preprocess_exogenous,
    }
    return exogenous_series_params
