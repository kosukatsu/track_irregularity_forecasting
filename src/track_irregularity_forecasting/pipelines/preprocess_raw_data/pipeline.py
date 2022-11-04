from kedro.pipeline import Pipeline, node

from .nodes import (
    preprocess_track_degradation,
    fill_5m_data_missing,
    fill_missing,
    dump_preprocess_params,
    dump_fill_5m_missing_params,
)


def create_preprocess_raw_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                preprocess_track_degradation,
                [
                    "params:track_raw_data_sec27_5m",
                    "params:track_raw_kilotei",
                ],
                "track_data_sec27_5m_missing",
            ),
            node(
                preprocess_track_degradation,
                [
                    "params:track_raw_data_sec27_10m",
                    "params:track_raw_kilotei",
                ],
                "track_data_sec27_10m",
            ),
            node(
                preprocess_track_degradation,
                [
                    "params:track_raw_data_sec27_original_wave",
                    "params:track_raw_kilotei",
                ],
                "track_data_sec27_original_missing",
            ),
            node(
                preprocess_track_degradation,
                [
                    "params:track_raw_data_sec201_5m",
                    "params:track_raw_kilotei",
                ],
                "track_data_sec201_5m_missing",
            ),
            node(
                preprocess_track_degradation,
                [
                    "params:track_raw_data_sec201_10m",
                    "params:track_raw_kilotei",
                ],
                "track_data_sec201_10m",
            ),
            node(
                preprocess_track_degradation,
                [
                    "params:track_raw_data_sec201_original_wave",
                    "params:track_raw_kilotei",
                ],
                "track_data_sec201_original_missing",
            ),
            node(
                fill_missing,
                ["track_data_sec27_10m", "params:general.bow_string"],
                "data_filled_missing_sec27_10m",
            ),
            node(
                fill_missing,
                ["track_data_sec201_10m", "params:general.bow_string"],
                "data_filled_missing_sec201_10m",
            ),
            node(
                dump_preprocess_params,
                {
                    "general_params": "params:general",
                    "track_raw_kilotei": "params:track_raw_kilotei",
                    "sec27_10m": "params:track_raw_data_sec27_10m",
                    "sec27_5m": "params:track_raw_data_sec27_5m",
                    "sec27_original": "params:track_raw_data_sec27_original_wave",
                    "sec201_10m": "params:track_raw_data_sec201_10m",
                    "sec201_5m": "params:track_raw_data_sec201_5m",
                    "sec201_original": "params:track_raw_data_sec201_original_wave",
                },
                "preprocess_raw_data_params",
            ),
        ]
    )


def create_fill_5m_data_missing_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                fill_5m_data_missing,
                [
                    "track_data_sec201_5m_missing",
                    "track_data_sec201_10m",
                    "params:fill_sec201_5m_data_missing",
                    "params:data_engeering.feature_order",
                ],
                "track_data_sec201_5m",
            ),
            node(
                fill_5m_data_missing,
                [
                    "track_data_sec201_original_missing",
                    "track_data_sec201_10m",
                    "params:fill_sec201_original_data_missing",
                    "params:data_engeering.feature_order",
                ],
                "track_data_sec201_original_wave",
            ),
            node(
                fill_5m_data_missing,
                [
                    "track_data_sec27_5m_missing",
                    "track_data_sec27_10m",
                    "params:fill_sec27_5m_data_missing",
                    "params:data_engeering.feature_order",
                ],
                "track_data_sec27_5m",
            ),
            node(
                fill_5m_data_missing,
                [
                    "track_data_sec27_original_missing",
                    "track_data_sec27_10m",
                    "params:fill_sec27_original_data_missing",
                    "params:data_engeering.feature_order",
                ],
                "track_data_sec27_original_wave",
            ),
            node(
                fill_missing,
                ["track_data_sec27_5m", "params:bow_string_list.5m"],
                "data_filled_missing_sec27_5m",
            ),
            node(
                fill_missing,
                ["track_data_sec27_original_wave", "params:bow_string_list.original"],
                "data_filled_missing_sec27_original",
            ),
            node(
                fill_missing,
                ["track_data_sec201_5m", "params:bow_string_list.5m"],
                "data_filled_missing_sec201_5m",
            ),
            node(
                fill_missing,
                ["track_data_sec201_original_wave", "params:bow_string_list.original"],
                "data_filled_missing_sec201_original",
            ),
            node(
                dump_fill_5m_missing_params,
                {
                    "data_engeering_params": "params:data_engeering",
                    "fill_sec27_5m": "params:fill_sec27_5m_data_missing",
                    "fill_sec27_original": "params:fill_sec27_original_data_missing",
                    "fill_sec201_5m": "params:fill_sec201_5m_data_missing",
                    "fill_sec201_original": "params:fill_sec201_original_data_missing",
                },
                "fill_missing_params",
            ),
        ]
    )
