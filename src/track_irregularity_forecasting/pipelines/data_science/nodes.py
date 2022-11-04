from typing import List
import random
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import mlflow.pytorch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from ...models.regression.conv_lstm import ConvLSTM
from ...models.regression.LSTM import VanillaLSTM
from ...models.regression.GRU import GRU
from ...models.regression.conv_lstm_ms import ConvLSTMMultiStep
from ...models.regression.regressor import Regressor, seiya10
from ...extras.datasets.input_sequence_dataset import InputSequenceDataset
from ...extras.datamodules.input_sequence_datamodule import InputSequenceDataModule


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge_parameter(general_params, param1, param2):
    model_name = general_params["model"]
    task = general_params["task"]
    if param2 is not None:
        param1[model_name][task].update(param2[model_name][task])
    return param1


def select_model(model_name, model_params, output_channels, model_list, pred_len):
    assert model_name in model_list, "{} not in  model_list{}".format(
        model_name, model_list
    )
    if model_name == "conv_lstm":
        model_params["output_channels"] = output_channels
        model = ConvLSTM(**model_params)

    elif model_name == "LSTM":
        model_params["proj_size"] = output_channels
        model = VanillaLSTM(**model_params)

    elif model_name == "GRU":
        model_params["proj_size"] = output_channels
        model = GRU(**model_params)

    elif model_name == "conv_lstm_ms":
        model_params["output_channels"] = output_channels
        model_params["pred_len"] = pred_len
        model = ConvLSTMMultiStep(**model_params)

    return model


def prepare_model(
    model_params,
    general_params,
    ds_params,
    feature_order: List[str],
    min_max_dict,
    checkpoint=None,
):

    model_name = general_params["model"]
    task = general_params["task"]
    optim_params = ds_params["optimizer"]
    predict_feature_order = general_params["predict_columns"]
    eval_feature_order = general_params["eval_feature_order"]
    slice_step = ds_params["slice_step"]
    model_list = general_params["model_list"]
    input_len = general_params["input_len"]
    total_len = general_params["total_len"]

    model_params = model_params[model_name][task]

    model_params["pred_input_dim"] = input_len
    pred_len = total_len - input_len
    model = select_model(
        model_name, model_params, len(predict_feature_order), model_list, pred_len
    )
    if ds_params["regression"]["post_process"] == "seiya10":
        post_process = seiya10
    else:
        post_process = None
    lightning_module = Regressor(
        model=model,
        optim_params=optim_params,
        feature_order=feature_order,
        predict_feature_order=predict_feature_order,
        eval_feature_order=eval_feature_order,
        min_max_dict=min_max_dict,
        post_process=post_process,
        input_len=input_len,
        total_len=total_len,
        slice_step=slice_step,
    )

    if checkpoint != "None" and checkpoint is not None:
        checkpoint_path = str(
            Path("./data/06_models")
            / general_params["model"]
            / general_params["task"]
            / str(general_params["run_id"])
            / checkpoint
        )
        lightning_module.load_state_dict(
            state_dict=torch.load(checkpoint_path)["state_dict"], strict=False
        )
    return lightning_module


def train(
    general_params,
    ds_params,
    model_params,
    datamodule: InputSequenceDataModule,
    model,
):
    model_name = general_params["model"]
    task = general_params["task"]

    train_params = ds_params["trainer"]
    callback_configure = ds_params["call_back"]

    model_params = model_params[model_name][task]

    # configure callback
    callbacks = []
    if callback_configure["early_stop"]:
        callbacks.append(EarlyStopping(**callback_configure["early_stop_configure"]))
    if callback_configure["model_checkpoint"]:
        callbacks.append(
            ModelCheckpoint(**callback_configure["model_checkpoint_configure"])
        )

    trainer = pl.Trainer(**train_params, callbacks=callbacks)

    mlflow.pytorch.autolog()

    trainer.fit(model, datamodule)

    return model


def test(
    train_params,
    datamodule: InputSequenceDataModule,
    model,
):
    trainer = pl.Trainer(**train_params)

    mlflow.pytorch.autolog()

    trainer.test(model, datamodule=datamodule)


def inference(
    datamodule,
    datamode,
    model_name,
    task,
    model_params,
    model,
    params,
):
    model.eval()
    if datamode == "overall":
        datatypes = ["wo_split"]
    elif datamode == "only test":
        datatypes = ["test"]
    else:
        return
    model_params = model_params[model_name][task]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    targets = []
    preds = []
    processed_preds = []
    dates = []
    for datatype in datatypes:
        tdd = datamodule.tdd
        scedd = datamodule.scedds
        sfedds = datamodule.sfedds
        stfedds = datamodule.stfedds
        stedds = datamodule.stedds
        if datatype == "train":
            print("train")
            dataset = InputSequenceDataset(tdd, scedd, sfedds, stfedds, stedds, "train")
        elif datatype == "valid":
            print("valid")
            dataset = InputSequenceDataset(tdd, scedd, sfedds, stfedds, stedds,"valid")
        elif datatype == "test":
            print("test")
            dataset = InputSequenceDataset(tdd, scedd, sfedds, stfedds, stedds,"test")
        elif datatype == "wo_split":
            print("wo_split")
            dataset = InputSequenceDataset(
                tdd, scedd, sfedds, stfedds, stedds, "wo_split"
            )
        dataloader = DataLoader(dataset, num_workers=2, pin_memory=True)
        for batch in tqdm(dataloader):
            for i in range(len(batch)):
                if "to" in dir(batch[i]):
                    batch[i] = batch[i].to(device)
            (
                observation,
                target,
                date,
                spa_cate,
                spa_flag,
                spa_temp_flag,
                spa_temp,
            ) = batch

            pred, processed_pred = model(
                observation, spa_cate, spa_flag, spa_temp_flag, spa_temp
            )
            # pred = model.denormalize_target(pred, dim=1)
            targets.append(target.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            dates.append(date)
            if params["regression"]["post_process"] is not None:
                # processed_pred = model.denormalize_target(processed_pred, dim=1)
                processed_preds.append(processed_pred.detach().cpu().numpy())
    targets = np.concatenate(targets, axis=0)
    preds = np.concatenate(preds, axis=0)
    dates = np.stack(dates)
    if params["regression"]["post_process"] is not None:
        processed_preds = np.concatenate(processed_preds, axis=0)
        output = (targets, preds, processed_preds, dates)
    else:
        output = (targets, preds, None, dates)
    return output
