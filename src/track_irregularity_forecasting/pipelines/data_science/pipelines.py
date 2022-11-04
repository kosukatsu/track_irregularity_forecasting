from kedro.pipeline import Pipeline, node

from .nodes import merge_parameter, prepare_model, train, inference, test, set_seed


def create_pipeline_data_science(**kwargs):
    return Pipeline(
        [
            node(set_seed, {"seed": "params:general.seed"}, None),
            node(
                merge_parameter,
                {
                    "general_params": "params:general",
                    "param1": "model_params",
                    "param2": "diff_model_params",
                },
                "merged_model_params",
            ),
            node(
                prepare_model,
                {
                    "general_params": "params:general",
                    "model_params": "merged_model_params",
                    "ds_params": "params:ds_params",
                    "feature_order": "params:data_engeering.feature_order",
                    "min_max_dict": "train_max_min",
                    "checkpoint": "params:ds_params.checkpoint",
                },
                "model",
            ),
            node(
                train,
                {
                    "general_params": "params:general",
                    "model_params": "merged_model_params",
                    "ds_params": "params:ds_params",
                    "datamodule": "input_data",
                    "model": "model",
                },
                "trained_model",
            ),
            node(
                test,
                {
                    "train_params": "params:ds_params.trainer",
                    "datamodule": "input_data",
                    "model": "trained_model",
                },
                None,
            ),
        ]
    )


def create_pipeline_test(**kwargs):
    return Pipeline(
        [
            node(set_seed, {"seed": "params:general.seed"}, None),
            node(
                merge_parameter,
                {
                    "general_params": "params:general",
                    "param1": "model_params",
                    "param2": "diff_model_params",
                },
                "merged_model_params",
            ),
            node(
                prepare_model,
                {
                    "general_params": "params:general",
                    "model_params": "merged_model_params",
                    "ds_params": "params:ds_params",
                    "feature_order": "params:data_engeering.feature_order",
                    "min_max_dict": "train_max_min",
                    "checkpoint": "params:inference.checkpoint",
                },
                "model",
            ),
            node(
                test,
                {
                    "train_params": "params:ds_params.trainer",
                    "datamodule": "input_data",
                    "model": "model",
                },
                None,
            ),
        ]
    )


def create_pipeline_inference(**kwargs):
    return Pipeline(
        [
            node(set_seed, {"seed": "params:general.seed"}, None),
            node(
                merge_parameter,
                {
                    "general_params": "params:general",
                    "param1": "model_params",
                    "param2": "diff_model_params",
                },
                "merged_model_params",
            ),
            node(
                prepare_model,
                {
                    "model_params": "merged_model_params",
                    "general_params": "params:general",
                    "ds_params": "params:ds_params",
                    "feature_order": "params:data_engeering.feature_order",
                    "min_max_dict": "train_max_min",
                    "checkpoint": "params:inference.checkpoint",
                },
                "model",
            ),
            node(
                inference,
                {
                    "datamodule": "input_data",
                    "datamode": "params:inference.datamode",
                    "model_name": "params:general.model",
                    "task": "params:general.task",
                    "model_params": "merged_model_params",
                    "model": "model",
                    "params": "params:ds_params",
                },
                "inference_result",
            ),
        ]
    )

