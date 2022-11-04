"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

import track_irregularity_forecasting.pipelines.preprocess_raw_data as pr
import track_irregularity_forecasting.pipelines.data_engeering as de
import track_irregularity_forecasting.pipelines.preprocess_exogenous as pe
import track_irregularity_forecasting.pipelines.make_exogenous_series as mes
import track_irregularity_forecasting.pipelines.data_science as ds
import track_irregularity_forecasting.pipelines.remove_series as rs

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    preprocess_pipeline = pr.create_preprocess_raw_data_pipeline()
    preprocess_5m_pipeline = pr.create_fill_5m_data_missing_pipeline()

    data_engeering_pipeline = de.create_data_engeering_pipeline()

    preprocess_structure_pipeline = pe.create_preprocess_sturcture()
    preprocess_IJEJ_pipeline = pe.create_preprocess_IJ_EJ()
    preprocess_welding_pipeline = pe.create_preprocess_welding()
    preprocess_work_pipeline = pe.create_preprocess_work_pipeline()
    preprocess_rainfall_pipeline = pe.create_preprocess_rainfall_pipeline()
    preprocess_ballast_age_pipeline = pe.create_preprocess_ballast_age_pipeline()
    preprocess_tonnage_pipeline = pe.create_preprocess_tonnage_pipeline()
    preprocess_dump_exogenous_params = (
        pe.create_dump_preprocess_exogenous_params_pipeline()
    )

    train_pipeline = ds.create_pipeline_data_science()
    inference_pipeline = ds.create_pipeline_inference()
    test_pipeline = ds.create_pipeline_test()

    exogenous_series_pipeline = mes.create_exogenous_series_pipeline()

    de_wo_split_pipeline = de.create_data_engeering_wo_split_pipeline()

    remove_series_pipeline = rs.create_remove_series_pipeline()

    return {
        "__default__": preprocess_pipeline,  # dummy
        "preprocess": preprocess_pipeline,
        "preprocess_5m": preprocess_5m_pipeline,
        "data_engeering": data_engeering_pipeline,
        "preprocess_structure": preprocess_structure_pipeline
        + preprocess_dump_exogenous_params,
        "preprocess_IJEJ": preprocess_IJEJ_pipeline + preprocess_dump_exogenous_params,
        "preprocess_welding": preprocess_welding_pipeline
        + preprocess_dump_exogenous_params,
        "preprocess_work": preprocess_work_pipeline
        + preprocess_dump_exogenous_params,
        "preprocess_rainfall": preprocess_rainfall_pipeline
        + preprocess_dump_exogenous_params,
        "preprocess_tonnage": preprocess_tonnage_pipeline
        + preprocess_dump_exogenous_params,
        "preprocess_ballast_age": preprocess_ballast_age_pipeline
        + preprocess_dump_exogenous_params,
        "preprocess_exogenous": preprocess_structure_pipeline
        + preprocess_IJEJ_pipeline
        + preprocess_welding_pipeline
        + preprocess_work_pipeline
        + preprocess_tonnage_pipeline
        + preprocess_ballast_age_pipeline
        + preprocess_rainfall_pipeline
        + preprocess_dump_exogenous_params,
        "exogenous_series": exogenous_series_pipeline,
        "train": train_pipeline,
        "inference": inference_pipeline,
        "test": test_pipeline,
        "data_engeering_wo_split": de_wo_split_pipeline,
        "remove_series": remove_series_pipeline,
    }
