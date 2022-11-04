from .nodes import (
    preprocess_welding,
    preprocess_IJ_EJ,
    preprocess_structure,
    preprocess_ballast_age,
    preprocess_tonnage,
    preprocess_rainfall,
    preprocess_work,
    seiya_work,
    unroll_work,
    dump_preprocess_exogenous_params,
)
from kedro.pipeline import Pipeline, node


def create_preprocess_sturcture(**kwargs):
    return Pipeline(
        [
            node(
                preprocess_structure,
                [
                    "structure_raw_data_sec27",
                    "params:preprocess_exogenous.start_distance_sec27",
                    "params:preprocess_exogenous.end_distance_sec27",
                ],
                "structure_preprocessed_data_sec27",
            ),
            node(
                preprocess_structure,
                [
                    "structure_raw_data_sec201",
                    "params:preprocess_exogenous.start_distance_sec201",
                    "params:preprocess_exogenous.end_distance_sec201",
                ],
                "structure_preprocessed_data_sec201",
            ),
        ]
    )


def create_preprocess_welding(**kwargs):
    return Pipeline(
        [
            node(
                preprocess_welding,
                [
                    "welding_raw_data_sec27",
                    "params:preprocess_exogenous.start_distance_sec27",
                    "params:preprocess_exogenous.end_distance_sec27",
                ],
                "welding_preprocessed_data_sec27",
            ),
            node(
                preprocess_welding,
                [
                    "welding_raw_data_sec201",
                    "params:preprocess_exogenous.start_distance_sec201",
                    "params:preprocess_exogenous.end_distance_sec201",
                ],
                "welding_preprocessed_data_sec201",
            ),
        ]
    )


def create_preprocess_IJ_EJ(**kwargs):
    return Pipeline(
        [
            node(
                preprocess_IJ_EJ,
                [
                    "IJ_raw_data_sec27",
                    "EJ_raw_data_sec27",
                    "params:preprocess_exogenous.start_distance_sec27",
                    "params:preprocess_exogenous.end_distance_sec27",
                ],
                "IJ_EJ_preprocessed_data_sec27",
            ),
            node(
                preprocess_IJ_EJ,
                [
                    "IJ_raw_data_sec201",
                    "EJ_raw_data_sec201",
                    "params:preprocess_exogenous.start_distance_sec201",
                    "params:preprocess_exogenous.end_distance_sec201",
                ],
                "IJ_EJ_preprocessed_data_sec201",
            ),
        ]
    )


def create_preprocess_rainfall_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                preprocess_rainfall,
                {
                    "loc_name": "params:preprocess_exogenous.rainfall_loc_labels.nishiya",
                    "track": "track_data_sec27_10m",
                },
                "rainfall_preprocessed_data_nishiya",
            ),
            node(
                preprocess_rainfall,
                {
                    "loc_name": "params:preprocess_exogenous.rainfall_loc_labels.kouza_shibuya",
                    "track": "track_data_sec27_10m",
                },
                "rainfall_preprocessed_data_kouza_shibuya",
            ),
            node(
                preprocess_rainfall,
                {
                    "loc_name": "params:preprocess_exogenous.rainfall_loc_labels.kakegawa",
                    "track": "track_data_sec201_10m",
                },
                "rainfall_preprocessed_data_kakegawa",
            ),
            node(
                preprocess_rainfall,
                {
                    "loc_name": "params:preprocess_exogenous.rainfall_loc_labels.fukuroi",
                    "track": "track_data_sec201_10m",
                },
                "rainfall_preprocessed_data_fukuroi",
            ),
            node(
                preprocess_rainfall,
                {
                    "loc_name": "params:preprocess_exogenous.rainfall_loc_labels.kikukawa",
                    "track": "track_data_sec201_10m",
                },
                "rainfall_preprocessed_data_kikukawa",
            ),
        ]
    )


def create_preprocess_ballast_age_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                preprocess_ballast_age,
                {
                    "df1": "ballast_raw_data_sec27_1",
                    "df2": "ballast_raw_data_sec27_2",
                    "df3": "ballast_raw_data_sec27_3",
                    "structure": "structure_raw_data_sec27",
                    "track": "track_data_sec27_10m",
                    "start_distance": "params:preprocess_exogenous.start_distance_sec27",
                    "end_distance": "params:preprocess_exogenous.end_distance_sec27",
                    "open_date": "params:general.open_date",
                    "e": "params:preprocess_exogenous.error_to_border_of_structure",
                },
                "ballast_age_preprocessed_data_sec27",
            ),
            node(
                preprocess_ballast_age,
                {
                    "df1": "ballast_raw_data_sec201_1",
                    "df2": "ballast_raw_data_sec201_2",
                    "df3": "ballast_raw_data_sec201_3",
                    "structure": "structure_raw_data_sec201",
                    "track": "track_data_sec201_10m",
                    "start_distance": "params:preprocess_exogenous.start_distance_sec201",
                    "end_distance": "params:preprocess_exogenous.end_distance_sec201",
                    "open_date": "params:general.open_date",
                    "e": "params:preprocess_exogenous.error_to_border_of_structure",
                },
                "ballast_age_preprocessed_data_sec201",
            ),
        ]
    )


def create_preprocess_tonnage_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                preprocess_tonnage,
                {
                    "track": "track_data_sec27_10m",
                    "section_name": "params:preprocess_exogenous.tonnage_section_labels.shinyoko2odawara",
                },
                "tonnage_preprocessed_data_shinyoko2odawara",
            ),
            node(
                preprocess_tonnage,
                {
                    "track": "track_data_sec201_10m",
                    "section_name": "params:preprocess_exogenous.tonnage_section_labels.shizuoka2kakegawa",
                },
                "tonnage_preprocessed_data_shizuoka2kakegawa",
            ),
            node(
                preprocess_tonnage,
                {
                    "track": "track_data_sec201_10m",
                    "section_name": "params:preprocess_exogenous.tonnage_section_labels.kakegawa",
                },
                "tonnage_preprocessed_data_kakegawa",
            ),
            node(
                preprocess_tonnage,
                {
                    "track": "track_data_sec201_10m",
                    "section_name": "params:preprocess_exogenous.tonnage_section_labels.kakegawa2hamamatsu",
                },
                "tonnage_preprocessed_data_kakegawa2hamamatsu",
            ),
        ]
    )


def create_preprocess_work_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                preprocess_work,
                {
                    "df1": "work_raw_data_sec27",
                    "df2": "work_raw_data_sec27_2",
                    "df3": "work_raw_data_append",
                    "work_list": "work_list",
                    "track": "track_data_sec27_10m",
                    "structure": "structure_raw_data_sec27",
                    "e": "params:preprocess_exogenous.error_to_border_of_structure",
                },
                "work_preprocessed_data_sec27",
            ),
            node(
                preprocess_work,
                {
                    "df1": "work_raw_data_sec201",
                    "df2": "work_raw_data_sec201_2",
                    "work_list": "work_list",
                    "track": "track_data_sec201_10m",
                    "structure": "structure_raw_data_sec201",
                    "e": "params:preprocess_exogenous.error_to_border_of_structure",
                },
                "work_preprocessed_data_sec201",
            ),
            node(
                seiya_work,
                {
                    "causal_df": "work_preprocessed_data_sec27",
                    "bow_string": "params:general.bow_string",
                    "start_distance": "params:preprocess_exogenous.start_distance_sec27",
                    "end_distance": "params:preprocess_exogenous.end_distance_sec27",
                },
                "work_seiya_data_sec27",
            ),
            node(
                unroll_work,
                {
                    "track": "data_filled_missing_sec27_10m",
                    "causal_df": "work_seiya_data_sec27",
                },
                "work_unrolled_data_sec27",
            ),
            node(
                seiya_work,
                {
                    "causal_df": "work_preprocessed_data_sec201",
                    "bow_string": "params:general.bow_string",
                    "start_distance": "params:preprocess_exogenous.start_distance_sec201",
                    "end_distance": "params:preprocess_exogenous.start_distance_sec201",
                },
                "work_seiya_data_sec201",
            ),
            node(
                unroll_work,
                {
                    "track": "data_filled_missing_sec201_10m",
                    "causal_df": "work_seiya_data_sec201",
                },
                "work_unrolled_data_sec201",
            ),
        ]
    )


def create_dump_preprocess_exogenous_params_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                dump_preprocess_exogenous_params,
                {
                    "preprocess_exogenous_params": "params:preprocess_exogenous",
                    "general_params": "params:general",
                },
                "preprocess_exogenous_params",
            ),
        ]
    )
