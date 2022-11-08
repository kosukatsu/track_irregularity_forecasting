from kedro.pipeline import Pipeline, node

from .nodes import (
    cvt_ijej_to_array,
    cvt_structure_to_array,
    cvt_welding_to_array,
    make_rainfall_series,
    make_ballast_age_series,
    make_tonnage_series,
    make_work_time_series,
    dump_exogenous_series_params,
)



def create_exogenous_series_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                cvt_ijej_to_array,
                {
                    "df": "IJ_EJ_preprocessed_data",
                    "start_distance": "params:general.start_distance",
                    "end_distance": "params:general.end_distance",
                },
                "IJ_EJ_data",
            ),
            node(
                cvt_structure_to_array,
                {
                    "df": "structure_preprocessed_data",
                    "start_distance": "params:general.start_distance",
                    "end_distance": "params:general.end_distance",
                },
                "structure_data",
            ),
            node(
                cvt_welding_to_array,
                {
                    "df": "welding_preprocessed_data",
                    "start_distance": "params:general.start_distance",
                    "end_distance": "params:general.end_distance",
                },
                "welding_data",
            ),
            node(
                make_rainfall_series,
                {
                    "df1": "rainfall_preprocessed_data_nishiya",
                    "df2": "rainfall_preprocessed_data_kouza_shibuya",
                    "section": "params:general.section",
                    "split_date": "params:data_engeering.test_split_date",
                    "valid_split_date": "params:data_engeering.valid_split_date",
                    "params": "params:preprocess_exogenous.rainfall_min_max_params",
                    "series_len": "params:general.total_len",
                    "bow_string": "params:general.bow_string",
                    "task": "params:general.task",
                    "start_distance": "params:general.start_distance",
                    "end_distance": "params:general.end_distance",
                },
                "rainfall_max_min",
            ),
            node(
                make_ballast_age_series,
                {
                    "ballast_age": "ballast_age_preprocessed_data",
                    "params": "params:preprocess_exogenous.ballast_age_min_max_params",
                    "section": "params:general.section",
                    "split_date": "params:data_engeering.test_split_date",
                    "valid_split_date": "params:data_engeering.valid_split_date",
                    "series_len": "params:general.total_len",
                    "structure": "structure_preprocessed_data",
                    "bow_string": "params:general.bow_string",
                    "task": "params:general.task",
                    "start_distance": "params:general.start_distance",
                    "end_distance": "params:general.end_distance",
                },
                "ballast_age_max_min",
            ),
            node(
                make_tonnage_series,
                {
                    "df1": "tonnage_preprocessed_data_shinyoko2odawara",
                    "section": "params:general.section",
                    "split_date": "params:data_engeering.test_split_date",
                    "valid_split_date": "params:data_engeering.valid_split_date",
                    "params": "params:preprocess_exogenous.tonnage_min_max_params",
                    "series_len": "params:general.total_len",
                    "bow_string": "params:general.bow_string",
                    "task": "params:general.task",
                    "start_distance": "params:general.start_distance",
                    "end_distance": "params:general.end_distance",
                },
                "tonnage_max_min",
            ),
            node(
                make_work_time_series,
                {
                    "work_df": "work_unrolled_data",
                    "section": "params:general.section",
                    "split_date": "params:data_engeering.test_split_date",
                    "valid_split_date": "params:data_engeering.valid_split_date",
                    "series_len": "params:general.total_len",
                    "work_order": "params:preprocess_exogenous.work_order",
                    "bow_string": "params:general.bow_string",
                    "task": "params:general.task",
                    "start_distance": "params:general.start_distance",
                    "end_distance": "params:general.end_distance",
                },
                None,
            ),
            node(
                dump_exogenous_series_params,
                {
                    "general_params": "params:general",
                    "data_engeering_params": "params:data_engeering",
                    "preprocess_exogenous": "params:preprocess_exogenous",
                },
                "make_exogenous_series_params",
            ),
        ]
    )
