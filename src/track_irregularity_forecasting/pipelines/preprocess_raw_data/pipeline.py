from kedro.pipeline import Pipeline, node

from .nodes import (
    preprocess_track_degradation,
    fill_missing,
    dump_preprocess_params,
)


def create_preprocess_raw_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                preprocess_track_degradation,
                [
                    "params:track_raw_data_sec27_10m",
                    "params:track_raw_kilotei",
                ],
                "track_data_sec27_10m",
            ),
            node(
                fill_missing,
                ["track_data_sec27_10m", "params:general.bow_string"],
                "data_filled_missing_sec27_10m",
            ),
            node(
                dump_preprocess_params,
                {
                    "general_params": "params:general",
                    "track_raw_kilotei": "params:track_raw_kilotei",
                    "sec27_10m": "params:track_raw_data_sec27_10m",
                },
                "preprocess_raw_data_params",
            ),
        ]
    )
