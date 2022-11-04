from kedro.pipeline import Pipeline, node

from .nodes import (
    get_dates,
    group_data_by_date,
    split_data,
    calc_max_min,
    save_tensor,
    make_time_series,
    split_data_target,
    dump_data_engeering_params,
)


def create_data_engeering_pipeline(**kwargs):
    return Pipeline(
        [
            node(get_dates, ["data_filled_missing"], ["data_strtime", "dates"]),
            node(
                group_data_by_date,
                [
                    "data_strtime",
                    "params:general.start_distance",
                    "params:general.end_distance",
                    "params:data_engeering.feature_order",
                ],
                "data_groupped",
            ),
            node(
                split_data,
                [
                    "data_groupped",
                    "dates",
                    "params:data_engeering.test_split_date",
                ],
                [
                    "train_valid_data_splited",
                    "test_data_splited",
                    "train_valid_dates",
                    "test_dates",
                ],
            ),
            node(
                split_data,
                [
                    "train_valid_data_splited",
                    "train_valid_dates",
                    "params:data_engeering.valid_split_date",
                ],
                [
                    "train_data_splited",
                    "valid_data_splited",
                    "train_dates",
                    "valid_dates",
                ],
            ),
            node(
                calc_max_min,
                [
                    "train_data_splited",
                    "params:data_engeering.min_max_params",
                    "params:data_engeering.feature_order",
                ],
                "train_max_min",
            ),
            node(
                make_time_series,
                ["test_data_splited", "test_dates", "params:general.total_len"],
                ["test_data_time_series", "test_date_series"],
            ),
            node(
                make_time_series,
                ["valid_data_splited", "valid_dates", "params:general.total_len"],
                ["valid_data_time_series", "valid_date_series"],
            ),
            node(
                make_time_series,
                ["train_data_splited", "train_dates", "params:general.total_len"],
                ["train_data_time_series", "train_date_series"],
            ),
            node(
                split_data_target,
                {
                    "data": "test_data_time_series",
                    "de_params": "params:data_engeering",
                    "general_params": "params:general",
                },
                ["splited_test_data_time_series", "splited_test_target"],
            ),
            node(
                save_tensor,
                {
                    "data": "splited_test_data_time_series",
                    "target": "splited_test_target",
                    "date": "test_date_series",
                    "path": "params:data_engeering.sequance_datapath",
                    "data_class": "params:general.data_class.test",
                    "general_params": "params:general",
                },
                None,
            ),
            node(
                split_data_target,
                {
                    "data": "valid_data_time_series",
                    "de_params": "params:data_engeering",
                    "general_params": "params:general",
                },
                ["splited_valid_data_time_series", "splited_valid_target"],
            ),
            node(
                save_tensor,
                {
                    "data": "splited_valid_data_time_series",
                    "target": "splited_valid_target",
                    "date": "valid_date_series",
                    "path": "params:data_engeering.sequance_datapath",
                    "data_class": "params:general.data_class.valid",
                    "general_params": "params:general",
                },
                None,
            ),
            node(
                split_data_target,
                {
                    "data": "train_data_time_series",
                    "de_params": "params:data_engeering",
                    "general_params": "params:general",
                },
                ["splited_train_data_time_series", "splited_train_target"],
            ),
            node(
                save_tensor,
                {
                    "data": "splited_train_data_time_series",
                    "target": "splited_train_target",
                    "date": "train_date_series",
                    "path": "params:data_engeering.sequance_datapath",
                    "data_class": "params:general.data_class.train",
                    "general_params": "params:general",
                },
                None,
            ),
            node(
                dump_data_engeering_params,
                {
                    "general_params": "params:general",
                    "data_engeering_params": "params:data_engeering",
                },
                "data_engeering_params",
            ),
        ]
    )


def create_data_engeering_wo_split_pipeline(**kwargs):
    return Pipeline(
        [
            node(get_dates, ["data_filled_missing"], ["data_strtime", "dates"]),
            node(
                group_data_by_date,
                [
                    "data_strtime",
                    "params:general.start_distance",
                    "params:general.end_distance",
                    "params:data_engeering.feature_order",
                ],
                "data_groupped",
            ),
            node(
                calc_max_min,
                [
                    "data_groupped",
                    "params:data_engeering.min_max_params",
                    "params:data_engeering.feature_order",
                ],
                "max_min_wo_split",
            ),
            node(
                make_time_series,
                ["data_groupped", "dates", "params:general.total_len"],
                ["data_time_series", "date_series"],
            ),
            node(
                split_data_target,
                {
                    "data": "data_time_series",
                    "de_params": "params:data_engeering",
                    "general_params": "params:general",
                },
                ["splited_data_time_series", "splited_target"],
            ),
            node(
                save_tensor,
                {
                    "data": "splited_data_time_series",
                    "target": "splited_target",
                    "date": "date_series",
                    "path": "params:data_engeering.sequance_datapath",
                    "data_class": "params:general.data_class.wo_split",
                    "general_params": "params:general",
                },
                None,
            ),
        ]
    )
