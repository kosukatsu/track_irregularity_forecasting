from pathlib import Path
import calendar
from datetime import datetime, timedelta
from glob import glob

import pandas as pd
import numpy as np
from tqdm import tqdm

def preprocess_structure(struct_df, start_distance, end_distance):
    struct_df = struct_df.iloc[:, 2:5]

    # one hot encoding
    struct_df.columns = ["start", "end", "struct"]
    struct_df["embankment"] = 0
    struct_df.loc[struct_df["struct"] == "盛土", "embankment"] = 1
    struct_df["viaduct"] = 0
    struct_df.loc[struct_df["struct"] == "高架", "viaduct"] = 1
    struct_df["cut"] = 0
    struct_df.loc[struct_df["struct"] == "切取", "cut"] = 1
    struct_df["bridge"] = 0
    struct_df.loc[struct_df["struct"] == "橋りょう", "bridge"] = 1
    struct_df["tunnel"] = 0
    struct_df.loc[struct_df["struct"] == "ﾄﾝﾈﾙ", "tunnel"] = 1

    struct_df = struct_df.drop("struct", axis=1)

    # flatten
    def flatten(row):
        df1 = pd.Series(np.arange(row["start"], row["end"])).to_frame(name="distance")
        df2 = row.to_frame().T
        df2["key"] = 0
        df1["key"] = 0
        return df1.merge(df2, how="outer", on="key").drop(columns="key")

    struct_df = pd.concat(struct_df.apply(flatten, axis=1).values)

    struct_df = struct_df.drop(["start", "end"], axis=1)

    struct_df = struct_df[struct_df["distance"] >= start_distance]
    struct_df = struct_df[struct_df["distance"] < end_distance]

    return struct_df

def preprocess_welding(welding_df, start_distance, end_distance):
    welding_df = welding_df.drop(
        welding_df.columns[[0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], axis=1
    )
    welding_df.columns = ["rl", "distance", "kind"]
    welding_df_l = welding_df[welding_df["rl"] == "左"]
    welding_df_r = welding_df[welding_df["rl"] == "右"]
    welding_df = pd.merge(welding_df_r, welding_df_l, how="outer", on="distance")
    welding_df = welding_df.rename({"kind_x": "kind_r", "kind_y": "kind_l"}, axis=1)
    welding_df = welding_df.drop(["rl_x", "rl_y"], axis=1)

    output_df = pd.DataFrame(
        np.arange(start_distance, end_distance).astype(int), columns=["distance"]
    )
    output_df = pd.merge(output_df, welding_df, on="distance", how="left")

    jp_words = ["フラッシュ", "ガス", "エンクロ"]
    en_words = ["flash", "gas", "enclose"]
    rl = ["r", "l"]
    for fix in rl:
        for i in range(len(jp_words)):
            new_column = "{}_{}".format(en_words[i], fix)
            output_df[new_column] = 0
            output_df.loc[
                output_df["kind_{}".format(fix)] == jp_words[i], new_column
            ] = 1

    output_df = output_df.drop(["kind_r", "kind_l"], axis=1)

    return output_df


def preprocess_IJ_EJ(IJ_df, EJ_df, start_distance, end_distance):
    EJ_df = EJ_df[["中心位置"]]

    EJ_df["ej"] = 1
    EJ_df = EJ_df.rename(columns={"中心位置": "distance"})

    IJ_df = IJ_df[["中心位置"]]
    IJ_df["ij"] = 1
    IJ_df = IJ_df.rename(columns={"中心位置": "distance"})

    df = pd.DataFrame(range(start_distance, end_distance), columns=["distance"])
    df = pd.merge(df, EJ_df, on="distance", how="left")
    df = pd.merge(df, IJ_df, on="distance", how="left")

    df = df.fillna(0)
    df = df.astype(int)

    return df

def preprocess_ballast_age(
    df1, df2, df3, structure, track, start_distance, end_distance, open_date, e
):
    track["date"] = pd.to_datetime(track["date"])
    dates = track["date"].unique()
    start_date = dates[0]

    # preprocess tables
    df = df1[["位置（始点）", "位置（終点）", "作業年月日"]]
    df = df.rename(columns={"位置（始点）": "start", "位置（終点）": "end", "作業年月日": "date"})
    df["date"] = pd.to_datetime(df["date"])

    start_age = df2[["位置（始点）", "位置（終点）", "作業年月日"]]
    start_age = start_age.rename(
        columns={"位置（始点）": "start", "位置（終点）": "end", "作業年月日": "date"}
    )
    start_age["date"] = pd.to_datetime(start_age["date"])

    update_age = df3[["位置（始点）", "位置（終点）", "作業年月日"]]
    update_age = update_age.rename(
        columns={"位置（始点）": "start", "位置（終点）": "end", "作業年月日": "date"}
    )
    update_age = update_age.sort_values("date")

    structure = structure[["キロ程(始点)", "キロ程(終点)", "土木構造物"]]
    structure = structure.rename(
        columns={"キロ程(始点)": "start", "キロ程(終点)": "end", "土木構造物": "st"}
    )

    # merge tables
    df = pd.concat([df, start_age, update_age])
    df = df.drop_duplicates()
    df = df.sort_values("date")
    df = df.reset_index(drop=True)

    # adjust construction's range to structure's border
    es = np.arange(-e, e + 1)
    for i, row in tqdm(df.iterrows()):
        for e in es:
            if row["start"] + e in structure["start"].to_numpy():
                df.loc[row.name, "start"] = row["start"] + e
                break
            if row["start"] + e in structure["end"].to_numpy():
                df.loc[row.name, "start"] = row["start"] + e
                break
        for e in es:
            if row["end"] + e in structure["start"].to_numpy():
                df.loc[row.name, "end"] = row["end"] + e
                break
            if row["end"] + e in structure["end"].to_numpy():
                df.loc[row.name, "end"] = row["end"] + e
                break

    prev_df = df.loc[df["date"] < start_date, :]
    update_df = df.loc[df["date"] >= start_date, :]

    last_update_dates = pd.DataFrame(
        columns=np.arange(start_distance, end_distance), index=dates
    )
    for _, row in tqdm(prev_df.iterrows()):
        last_update_dates.loc[:, row["start"] : row["end"] - 1] = row["date"]
    for _, row in tqdm(update_df.iterrows()):
        last_update_dates.loc[
            (last_update_dates.index >= row["date"]), row["start"] : row["end"] - 1
        ] = row["date"]
    for loc in tqdm(
        last_update_dates.loc[:, last_update_dates.isnull().any()].columns.to_numpy()
    ):
        buf = structure.loc[structure["start"] <= loc]
        buf = buf.loc[buf["end"] > loc]
        if buf.loc[buf.iloc[0].name, "st"] != "橋りょう":
            last_update_dates.loc[:, loc] = pd.to_datetime(open_date)

    ballast_age = pd.DataFrame(
        index=last_update_dates.index, columns=last_update_dates.columns
    )
    for loc in tqdm(ballast_age.columns):
        ballast_age[loc] = pd.to_datetime(last_update_dates.index) - pd.to_datetime(
            last_update_dates[loc]
        )

    return ballast_age / pd.Timedelta(days=365)


def preprocess_tonnage(track, section_name):
    track["date"] = pd.to_datetime(track["date"])
    dates = track["date"].unique()

    filelist = glob("./data/01_raw/shinkansen_raw_data/荷重条件（通トン）/*.csv")
    passing_load = pd.DataFrame(columns=["year", "month", "load"])
    passing_loads = []
    for file in filelist:
        file = Path(file)
        filename = file.name
        year = int(filename[3:7])
        df = pd.read_csv(file, header=2, encoding="shift-jis")
        if "3月" in df.columns:
            df = df[
                [
                    "区間/構内間",
                    "1月",
                    "2月",
                    "3月",
                    "4月",
                    "5月",
                    "6月",
                    "7月",
                    "8月",
                    "9月",
                    "10月",
                    "11月",
                    "12月",
                ]
            ]
        else:
            df[["8月", "9月", "10月", "11月", "12月", "1月", "2月", "3月"]] = np.nan
            df = df[
                [
                    "区間/構内間",
                    "1月",
                    "2月",
                    "3月",
                    "4月",
                    "5月",
                    "6月",
                    "7月",
                    "8月",
                    "9月",
                    "10月",
                    "11月",
                    "12月",
                ]
            ]
        df = df[df["区間/構内間"] == section_name]
        df = df.drop("区間/構内間", axis=1)
        df.columns = np.arange(1, 12 + 1)
        df = df.T
        df = df.reset_index()
        df.columns = ["month", "load"]
        years = np.zeros(12, dtype=int) + year
        years[:3] += 1
        df["year"] = years
        passing_loads.append(df)
    passing_load = pd.concat(passing_loads)

    passing_load["date"] = (
        passing_load["year"].astype(str) + "/" + passing_load["month"].astype(str)
    )
    passing_load["date"] = pd.to_datetime(passing_load["date"])
    passing_load = passing_load.sort_values("date")

    passing_load["days"] = 0
    for i, row in passing_load.iterrows():
        passing_load.loc[row.name, "days"] = calendar.monthrange(
            row["year"], row["month"]
        )[1]

    passing_load["load_per_day"] = passing_load["load"] / passing_load["days"]
    passing_load = passing_load.set_index("date")

    delta_load = pd.DataFrame(index=dates, columns=["delta_load"])
    prev_date = None
    for i, (date, row) in enumerate(delta_load.iterrows()):
        year = date.year
        month = date.month
        day = date.day
        load_month = datetime(year=year, month=month, day=1)
        if month == 1:
            prev_load_month = datetime(year=year - 1, month=12, day=1)
        else:
            prev_load_month = datetime(year=year, month=month - 1, day=1)

        if load_month not in passing_load.index:
            break
        if i == 0:
            delta_load.loc[date, "delta_load"] = (
                passing_load.loc[load_month, "load_per_day"] * 10
            )
        else:
            if date.month == prev_date.month:
                delta_day = (date - prev_date) / timedelta(days=1)
                delta_load.loc[date, "delta_load"] = (
                    delta_day * passing_load.loc[load_month, "load_per_day"]
                )
            else:
                delta_day1 = (load_month - prev_date) / timedelta(days=1)
                delta_day2 = (date - load_month) / timedelta(days=1)
                delta_load.loc[date, "delta_load"] = (
                    delta_day2 * passing_load.loc[load_month, "load_per_day"]
                    + delta_day1 * passing_load.loc[prev_load_month, "load_per_day"]
                )
        prev_date = date

    return delta_load


def preprocess_rainfall(loc_name, track):
    track["date"] = pd.to_datetime(track["date"])
    dates = track["date"].unique()

    filelist = glob("./data/01_raw/shinkansen_raw_data/雨量/**/*.csv", recursive=True)

    # read rainfall files
    df = pd.DataFrame(columns=["date", "10min", "hour", "contd", "YMD"])
    table_list = []
    for file in tqdm(filelist):
        buf = pd.read_csv(file, encoding="cp932", header=1)
        buf = buf[["雨量計", "日　付", "時　分", "10分間雨量", "時雨量", "連続雨量"]]
        buf = buf.rename(
            columns={
                "雨量計": "location",
                "日　付": "YMD",
                "時　分": "HM",
                "10分間雨量": "10min",
                "時雨量": "hour",
                "連続雨量": "contd",
            }
        )
        buf = buf.loc[buf["location"] == loc_name]
        buf["date"] = buf["YMD"] + " " + buf["HM"]
        buf["date"] = pd.to_datetime(buf["date"])
        buf = buf[["date", "10min", "hour", "contd", "YMD"]]
        table_list.append(buf)
    df = pd.concat(table_list)

    # sort
    df["YMD"] = pd.to_datetime(df["YMD"])
    df = df.set_index(["YMD", "date"])
    df = df.sort_index(axis=0, level=0)
    df = df.reset_index()
    df = df.set_index("date")
    # drop duplicate
    df = df[~df.index.duplicated()]
    df = df.reset_index()
    # fill missing
    df = df.replace("－", np.nan)
    df = df.astype({"10min": float, "hour": float, "contd": float})
    df = df.interpolate()
    # calc max(10min.) and max(1hour)
    rainfall_result = pd.DataFrame(
        index=dates, columns=["max(10min.)", "max(1hour)", "max(1day)", "sum"]
    )
    buf = df[
        ((dates[0] - pd.Timedelta(days=10)) < df["date"]) & (df["date"] < dates[0])
    ]
    rainfall_result.loc[dates[0], "max(10min.)"] = buf["10min"].max()
    rainfall_result.loc[dates[0], "max(1hour)"] = buf["hour"].max()
    for i in tqdm(range(len(dates) - 1)):
        buf = df[(dates[i] < df["date"]) & (df["date"] < dates[i + 1])]
        rainfall_result.loc[dates[i + 1], "max(10min.)"] = buf["10min"].max()
        rainfall_result.loc[dates[i + 1], "max(1hour)"] = buf["hour"].max()
    # sample per 10min
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.asfreq("600S")
    df = df.replace("－", np.nan)
    df["10min"] = df["10min"].astype(float)
    df_10min = df.groupby("YMD").sum()
    # calc max(1day) and sum
    buf = df_10min[
        ((dates[0] - pd.Timedelta(days=10)) < df_10min.index)
        & (df_10min.index < dates[0])
    ]
    rainfall_result.loc[dates[0], "max(1day)"] = buf["10min"].max()
    rainfall_result.loc[dates[0], "sum"] = buf["10min"].sum()
    for i in tqdm(range(len(dates) - 1)):
        buf = df_10min[(dates[i] < df_10min.index) & (df_10min.index < dates[i + 1])]
        rainfall_result.loc[dates[i + 1], "max(1day)"] = buf["10min"].max()
        rainfall_result.loc[dates[i + 1], "sum"] = buf["10min"].sum()
    return rainfall_result


def preprocess_work(df1, df2, work_list, track, structure, df3=None, e=1):
    track["date"] = pd.to_datetime(track["date"])
    dates = track["date"].unique()

    work_list = work_list[["作業種別", "高低狂いとの関連"]]
    work_list = work_list.rename(columns={"作業種別": "work", "高低狂いとの関連": "causal"})

    df1 = df1[["作業種別", "位置（始点）", "位置（終点）", "作業年月日", "左右レール"]]
    df1 = df1.rename(
        columns={
            "作業種別": "work",
            "位置（始点）": "start",
            "位置（終点）": "end",
            "作業年月日": "date",
            "左右レール": "rail",
        }
    )
    df1["date"] = pd.to_datetime(df1["date"])

    df2 = df2[df2["線別"] == "下り本線"]
    df2 = df2[["作業種別", "位置（始点）", "位置（終点）", "作業年月日", "左右レール"]]
    df2 = df2.rename(
        columns={
            "作業種別": "work",
            "位置（始点）": "start",
            "位置（終点）": "end",
            "作業年月日": "date",
            "左右レール": "rail",
        }
    )
    df2["date"] = pd.to_datetime(df2["date"])
    df2 = df2.loc[df2["work"].isnull() == False]

    if df3 is None:
        df = pd.concat([df1, df2])
    else:
        df3 = df3[["作業種別", "位置（始点）", "位置（終点）", "作業年月日", "左右レール"]]
        df3 = df3.rename(
            columns={
                "作業種別": "work",
                "位置（始点）": "start",
                "位置（終点）": "end",
                "作業年月日": "date",
                "左右レール": "rail",
            }
        )
        df3["date"] = pd.to_datetime(df3["date"])

        df = pd.concat([df1, df2, df3])

    df = df.drop_duplicates()
    df = df.sort_values("date")
    df = df.loc[df["date"] > dates[0]]
    df = df.reset_index(drop=True)
    df = df.loc[df["date"] < dates[-1]].copy()

    df = pd.merge(df, work_list, left_on="work", right_on="work", how="left")

    causal_works = [
        "接着絶縁継目部分更換(緊張器)",
        "接着絶縁継目部分更換(常温)",
        "ﾛﾝｸﾞﾚｰﾙ更換(常温)200m以上",
        "むら直し（所長指示）",
    ]
    no_causal_works = ["バラスト止め新設", "脱防ガード更換"]
    df.loc[df["work"].isin(causal_works), "causal"] = "○"
    df.loc[df["work"].isin(no_causal_works), "causal"] = "×"
    assert df["causal"].isna().any() == False

    causal_df = df[df["causal"] != "×"].copy()

    causal_df.loc[causal_df["work"].str.contains("むら直し"), "work"] = "むら直し"
    causal_df.loc[causal_df["work"].str.contains("接着絶縁継目部分更換"), "work"] = "接着絶縁継目部分更換"
    causal_df.loc[causal_df["work"].str.contains("ﾚｰﾙ更換"), "work"] = "ﾚｰﾙ更換"
    causal_df.loc[causal_df["work"].str.contains("総つき固め"), "work"] = "総つき固め"
    causal_df.loc[:, "work"] = causal_df["work"].str.replace("ﾏｸﾗｷﾞ", "まくらぎ")

    structure = structure[["キロ程(始点)", "キロ程(終点)", "土木構造物"]]
    structure = structure.rename(
        columns={"キロ程(始点)": "start", "キロ程(終点)": "end", "土木構造物": "st"}
    )

    es = np.arange(-e, e + 1)
    for i, row in causal_df.iterrows():
        for e in es:
            if row["start"] + e in structure["start"].to_numpy():
                causal_df.loc[row.name, "start"] = row["start"] + e
                break
            if row["start"] + e in structure["end"].to_numpy():
                causal_df.loc[row.name, "start"] = row["start"] + e
                break
        for e in es:
            if row["end"] + e in structure["start"].to_numpy():
                causal_df.loc[row.name, "end"] = row["end"] + e
                break
            if row["end"] + e in structure["end"].to_numpy():
                causal_df.loc[row.name, "end"] = row["end"] + e
                break
    return causal_df


def seiya_work(causal_df, bow_string, start_distance, end_distance):
    if bow_string == "5m":
        extention = 3
    elif bow_string == "10m":
        extention = 5
    else:
        return causal_df
    causal_df["start"] = causal_df["start"] - extention
    causal_df["end"] = causal_df["end"] + extention
    causal_df.loc[causal_df["start"] < start_distance, "start"] = start_distance
    causal_df.loc[causal_df["end"] > end_distance, "end"] = end_distance
    return causal_df


def unroll_work(track, causal_df):
    track["date"] = pd.to_datetime(track["date"])
    work_result = track[["date", "distance"]].copy()
    for column in [
        "fix_uneven",
        "tamping",
        "marutai",
        "exchange_ballast",
        "exchange_rail_r",
        "exchange_rail_l",
        "disposal_mud",
        "sleeper",
        "others",
    ]:
        work_result.loc[:, column] = 0
    for i, row in tqdm(causal_df.iterrows()):
        dates = work_result.loc[row["date"] >= work_result["date"], "date"].unique()
        if len(dates) == 0:
            continue
        dates = np.sort(dates)
        date = dates[-1]
        start = row["start"]
        end = row["end"]

        if row["work"] == "むら直し":
            work_result.loc[
                (work_result["date"] == date)
                & (work_result["distance"] >= start)
                & (work_result["distance"] < end),
                "fix_uneven",
            ] = 1
        if row["work"] == "マルタイ作業":
            work_result.loc[
                (work_result["date"] == date)
                & (work_result["distance"] >= start)
                & (work_result["distance"] < end),
                "marutai",
            ] = 1
        if row["work"] == "総つき固め":
            work_result.loc[
                (work_result["date"] == date)
                & (work_result["distance"] >= start)
                & (work_result["distance"] < end),
                "tamping",
            ] = 1
        if row["work"] == "道床更換":
            work_result.loc[
                (work_result["date"] == date)
                & (work_result["distance"] >= start)
                & (work_result["distance"] < end),
                "exchange_ballast",
            ] = 1
        if row["work"] == "ﾚｰﾙ更換" and row["rail"] == "右レール":
            work_result.loc[
                (work_result["date"] == date)
                & (work_result["distance"] >= start)
                & (work_result["distance"] < end),
                "exchange_rail_r",
            ] = 1
        if row["work"] == "ﾚｰﾙ更換" and row["rail"] == "左レール":
            work_result.loc[
                (work_result["date"] == date)
                & (work_result["distance"] >= start)
                & (work_result["distance"] < end),
                "exchange_rail_l",
            ] = 1
        if row["work"] in [
            "まくらぎ更換",
            "まくらぎ新設",
            "まくらぎ浮き補修",
            "まくらぎ位置整正",
            "まくらぎ撤去",
            "まくらぎ移設",
        ]:
            work_result.loc[
                (work_result["date"] == date)
                & (work_result["distance"] >= start)
                & (work_result["distance"] < end),
                "sleeper",
            ] = 1
        if row["work"] == "簡易噴泥処理":
            work_result.loc[
                (work_result["date"] == date)
                & (work_result["distance"] >= start)
                & (work_result["distance"] < end),
                "disposal_mud",
            ] = 1
        if row["work"] in ["伸縮継目部分更換", "分岐器部分更換", "接着絶縁継目部分更換", "ﾚｰﾙ削正(一般)"]:
            work_result.loc[
                (work_result["date"] == date)
                & (work_result["distance"] >= start)
                & (work_result["distance"] < end),
                "others",
            ] = 1
    return work_result

def dump_preprocess_exogenous_params(
    preprocess_exogenous_params,
    general_params,
):
    dump_preprocess_exogenous_params = {
        "preprocess_exogenous_params": preprocess_exogenous_params,
        "general_params": general_params,
    }
    return dump_preprocess_exogenous_params
