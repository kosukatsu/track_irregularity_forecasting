from glob import glob
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
import chardet


def detect_codec(filepath):
    with open(filepath, "rb") as f:
        b = f.read()
    codec = chardet.detect(b)
    return codec["encoding"]


def strToDate(input_str):
    year = int("20" + input_str[:2])
    month = int(input_str[2:4])
    day = int(input_str[4:6])
    return datetime(year, month, day)


def dump_preprocess_params(
    general_params,
    track_raw_kilotei,
    sec27_10m,
):
    preprocess_params = {
        "general_params": general_params,
        "track_raw_kilotei": track_raw_kilotei,
        "sec27_10m": sec27_10m,
    }
    return preprocess_params


def preprocess_track_degradation(params, track_raw_kilotei_params):
    filepath = params["input_directory"]
    start_distance = params["start_distance"]
    end_distance = params["end_distance"]

    filepath = Path(filepath)
    filelist = glob(str(filepath / "**" / "*.csv"), recursive=True)

    att_dist = {
        "cx": "lateral_osc",
        "cz": "vertical_osc",
        "he": "flatness",
        "ki": "gauge",
        "kl": "left_surface",
        "kr": "right_surface",
        "su": "cross_level",
        "tl": "left_alignment",
        "tr": "right_alignment",
        "ve": "velocity",
    }

    column = [
        "distance",
        "date",
        "lateral_osc",
        "vertical_osc",
        "flatness",
        "gauge",
        "left_surface",
        "right_surface",
        "cross_level",
        "left_alignment",
        "right_alignment",
        "velocity",
    ]

    type_dist = {
        "distance": int,
        "lateral_osc": float,
        "vertical_osc": float,
        "flatness": float,
        "gauge": float,
        "left_surface": float,
        "right_surface": float,
        "cross_level": float,
        "left_alignment": float,
        "right_alignment": float,
        "velocity": float,
    }

    df = pd.DataFrame(index=[], columns=column)
    df = df.astype(type_dist)
    df = df.set_index(["date", "distance"])

    for filepath in tqdm(filelist):
        filename = Path(filepath).name
        date, att = filename[2:8], filename[8:10]
        date = strToDate(date)
        att = att_dist[att]

        # load csv
        encoding = detect_codec(filepath)
        df_csv = pd.read_csv(filepath, header=None, encoding=encoding)
        df_csv = df_csv.drop(df_csv.columns[[0, 2]], axis=1)
        df_csv.columns = ["kilo_tei", "value"]

        # calc distance
        df_csv["kilo"] = (
            df_csv["kilo_tei"].str.extract("([0-9]+)K").astype(float).fillna(0)
        )
        df_csv["meter"] = (
            df_csv["kilo_tei"].str.extract("([0-9.]+)M").astype(float).fillna(0)
        )
        df_csv["distance"] = df_csv["kilo"] * 1000 + df_csv["meter"]
        df_csv = df_csv.drop(["kilo", "meter", "kilo_tei"], axis=1)

        if track_raw_kilotei_params["interpolate"] == False:
            df_csv["distance"] = np.floor(df_csv["distance"].to_numpy()).astype(int)
            df_csv = df_csv.groupby("distance").mean()
            df_csv = df_csv.reset_index()
        elif track_raw_kilotei_params["interpolate"] == True:
            df_csv = df_csv.set_index("distance")
            df_csv = pd.concat(
                [df_csv, pd.DataFrame(index=np.arange(start_distance, end_distance))]
            )
            df_csv = df_csv.sort_index()
            df_csv = df_csv.interpolate(
                **track_raw_kilotei_params["interpolate_params"]
            )
            df_csv = df_csv.loc[np.arange(start_distance, end_distance)]
            df_csv = df_csv.reset_index()
            df_csv = df_csv.rename(columns={"index": "distance"})
            df_csv = df_csv.groupby("distance").mean()
            df_csv = df_csv.reset_index()

        df_csv = df_csv.rename(columns={"value": att})
        df_csv = df_csv[df_csv["distance"] >= start_distance]
        df_csv = df_csv[df_csv["distance"] < end_distance]
        df_csv = df_csv.set_index("distance")

        df = df.reset_index()
        if (df["date"] == date).sum() == 0:
            buf = pd.DataFrame(index=[], columns=column)
            buf = buf.astype(type_dist)
            buf["distance"] = np.arange(start_distance, end_distance).astype(int)
            buf["distance"] = buf["distance"].astype(int)
            buf["date"] = date
            df = pd.concat([df, buf], axis=0)
        df["distance"] = df["distance"].astype(int)
        df = df.set_index(["date", "distance"])
        df.loc[date, att] = df_csv[att].to_numpy()

    df = df.sort_index()
    df = df.reset_index()

    return df

def interpolateMissingValue(df, date, cols_name, isSetIndex=True):
    if isSetIndex:
        df = df.reset_index()
    dates = df.date.unique()
    idx = np.where(dates == np.datetime64(date))[0].item()
    df = df.set_index(["date", "distance"])
    dT1 = dates[idx] - dates[idx - 1]
    dT2 = dates[idx + 1] - dates[idx]
    df.loc[dates[idx], cols_name] = (
        (
            df.loc[dates[idx - 1], cols_name] * dT2
            + df.loc[dates[idx + 1], cols_name] * dT1
        )
        / (dT2 + dT1)
    ).to_numpy()
    if not isSetIndex:
        df = df.reset_index()
    return df


def fill_missing(df, bow_string):
    df = interpolateMissingValue(df, "2012-12-15", "lateral_osc", isSetIndex=False)
    df = interpolateMissingValue(df, "2012-12-15", "vertical_osc", isSetIndex=False)
    df = interpolateMissingValue(df, "2014-09-16", "gauge", isSetIndex=False)
    df = interpolateMissingValue(df, "2015-07-14", "gauge", isSetIndex=False)
    df = interpolateMissingValue(df, "2018-03-27", "cross_level", isSetIndex=False)
    if bow_string == "5m":
        df = df[df.date > "2011-05-27"]
    return df
