from kedro.pipeline import Pipeline, node

from .remove_series_node import (
    remove_track_series,
    remove_tonnage,
    remove_ballast_age,
    remove_rainfall,
    remove_work,
)


def create_remove_series_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                remove_track_series,
                {
                    "path": "params:remove_series.series_path",
                    "section": "params:general.section",
                    "bow_string": "params:general.bow_string",
                    "task": "params:general.task",
                },
                None,
            ),
            node(
                remove_tonnage,
                {
                    "path": "params:remove_series.series_path",
                    "section": "params:general.section",
                },
                None,
            ),
            node(
                remove_work,
                {
                    "path": "params:remove_series.series_path",
                    "section": "params:general.section",
                },
                None,
            ),
            node(
                remove_ballast_age,
                {
                    "path": "params:remove_series.series_path",
                    "section": "params:general.section",
                },
                None,
            ),
            node(
                remove_rainfall,
                {
                    "path": "params:remove_series.series_path",
                    "section": "params:general.section",
                },
                None,
            ),
        ]
    )
