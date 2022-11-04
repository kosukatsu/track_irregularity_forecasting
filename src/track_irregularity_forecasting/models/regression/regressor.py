from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np

from ...utils import normalize_min_max_scale, denormalize_min_max_scale

def seiya10(preds, target):
    if target is not None:
        assert preds.ndim == target.ndim
    if preds.ndim == 3:
        seiya10 = preds[:, :, 5:-5] - 0.5 * (preds[:, :, 10:] + preds[:, :, :-10])
        if target is not None:
            target = target[:, :, 5:-5]
    elif preds.ndim == 4:
        seiya10 = preds[:, :, :, 5:-5] - 0.5 * (
            preds[:, :, :, 10:] + preds[:, :, :, :-10]
        )
        if target is not None:
            target = target[:, :, :, 5:-5]
    return seiya10, target

class Regressor(pl.LightningModule):
    def __init__(
        self,
        model,
        optim_params,
        feature_order: List[str],
        predict_feature_order: List[str],
        eval_feature_order: List[str],
        min_max_dict,
        post_process,
        total_len,
        input_len,
        slice_step: int = 1,
    ):
        super().__init__()

        self.input_len = input_len
        self.total_len = total_len
        self.pred_len = total_len - input_len

        self.model = model
        self.optim_params = optim_params
        self.metric = self._prepare_metric()
        self.normalize = lambda torch_track_data, dim: normalize_min_max_scale(
            torch_track_data, feature_order, min_max_dict, dim
        )
        self.denormalize = lambda torch_track_data, dim: denormalize_min_max_scale(
            torch_track_data, feature_order, min_max_dict, dim
        )
        self.normalize_target = lambda torch_track_data, dim: normalize_min_max_scale(
            torch_track_data, predict_feature_order, min_max_dict, dim
        )
        self.denormalize_target = (
            lambda torch_track_data, dim: denormalize_min_max_scale(
                torch_track_data, predict_feature_order, min_max_dict, dim
            )
        )
        self.post_process = post_process
        self.slice_step = slice_step
        self.predict_feature_order = predict_feature_order

        self.criterion = torch.nn.MSELoss()

        self.eval_index = [
            predict_feature_order.index(feature) for feature in eval_feature_order
        ]

    def _calc_pred(self, x):
        if self.slice_step == 1:
            return self.model(*x)
        else:
            if self.pred_len == 1:
                normalized_observation, spa_cate, spa_flag, spa_temp_flag, spa_temp = x
                N, T, C, L = normalized_observation.shape
                predict_channel = len(self.predict_feature_order)
                pred = torch.empty(N, predict_channel, L).type_as(
                    normalized_observation
                )
                for i in range(self.slice_step):
                    obs = normalized_observation[:, :, :, i :: self.slice_step]
                    sc = spa_cate[:, :, i :: self.slice_step]
                    sf = spa_flag[:, :, i :: self.slice_step]
                    stf = spa_temp_flag[:, :, :, i :: self.slice_step]
                    st = spa_temp[:, :, :, i :: self.slice_step]
                    pred[:, :, i :: self.slice_step] = self.model(obs, sc, sf, stf, st)
                return pred

            if self.pred_len > 1:
                normalized_observation, spa_cate, spa_flag, spa_temp_flag, spa_temp = x
                N, T, C, L = normalized_observation.shape
                predict_channel = len(self.predict_feature_order)
                pred = torch.empty(N, self.pred_len, predict_channel, L).type_as(
                    normalized_observation
                )
                for i in range(self.slice_step):
                    obs = normalized_observation[:, :, :, i :: self.slice_step]
                    sc = spa_cate[:, :, i :: self.slice_step]
                    sf = spa_flag[:, :, i :: self.slice_step]
                    stf = spa_temp_flag[:, :, :, i :: self.slice_step]
                    st = spa_temp[:, :, :, i :: self.slice_step]
                    pred[:, :, :, i :: self.slice_step] = self.model(
                        obs, sc, sf, stf, st
                    )
                return pred

    def _prepare_metric(self):
        return nn.MSELoss(reduction="sum")

    def shared_step(self, batch, batch_idx):
        (
            observation,
            target,
            _,
            spa_cate,
            spa_flag,
            spa_temp_flag,
            spa_temp,
        ) = batch

        input_len = self.input_len
        observation = observation[:, :input_len]
        input_spa_temp_flag = spa_temp_flag[:, :input_len].detach()
        spa_temp = spa_temp[:, :input_len]

        normalized_observation = self.normalize(observation, dim=2)
        assert target.ndim == 3 or target.ndim == 4
        if target.ndim == 3:
            normalized_target = self.normalize_target(target, dim=1)
        elif target.ndim == 4:
            normalized_target = self.normalize_target(target, dim=2)

        preds = self._calc_pred(
            (normalized_observation, spa_cate, spa_flag, input_spa_temp_flag, spa_temp),
        )

        if self.post_process is not None:
            preds, target = self.post_process(preds, target)

        loss = self.criterion(preds, normalized_target)

        assert preds.ndim == 3 or preds.ndim == 4
        if preds.ndim == 3:
            denormalized_preds = self.denormalize_target(preds, dim=1)
        elif preds.ndim == 4:
            denormalized_preds = self.denormalize_target(preds, dim=2)

        return loss, denormalized_preds.detach(), target.detach()

    def forward(self, track, spa_cate, spa_flag, spa_temp_flag, spa_temp):
        input_len = self.input_len
        observation = track[:, :input_len]
        spa_temp_flag = spa_temp_flag[:, :input_len]
        spa_temp = spa_temp[:, :input_len]

        observation = self.normalize(observation, dim=2)

        preds1 = self._calc_pred(
            (observation, spa_cate, spa_flag, spa_temp_flag, spa_temp)
        )
        preds2 = None

        if preds1.ndim == 3:
            denormalized_preds = self.denormalize_target(preds1, dim=1)
        elif preds1.ndim == 4:
            denormalized_preds = self.denormalize_target(preds1, dim=2)

        denormalized_preds2 = None
        if self.post_process is not None:
            preds2, _ = self.post_process(preds1, None)
            assert preds2.ndim == 3 or preds2.ndim == 4
            if preds2.ndim == 3:
                denormalized_preds2 = self.denormalize_target(preds2, dim=1)
            elif preds2.ndim == 4:
                denormalized_preds2 = self.denormalize_target(preds2, dim=2)
        return denormalized_preds, denormalized_preds2

    def training_step(self, batch, batch_idx):
        loss, denorm_pred, denorm_target = self.shared_step(batch, batch_idx)
        if self.pred_len == 1:  # single step
            metrics = self._calc_metrics(denorm_pred, denorm_target)
        else:  # multi step / horizon
            metrics = self._calc_metrics_multi_step(denorm_pred, denorm_target)
        metrics.update({"loss": loss})
        self.log_dict(metrics)
        return loss

    def _calc_metrics(self, denorm_pred, denorm_target):
        losses = {}

        # N,C,L
        denorm_pred = denorm_pred[:, self.eval_index]
        denorm_target = denorm_target[:, self.eval_index]
        denorm_mse = self.metric(denorm_pred, denorm_target)
        losses["denorm_MSE"] = denorm_mse
        cnt = torch.ones_like(denorm_pred).to(int).sum()
        losses["cnt"] = cnt

        for label_th, th in {"4mm": -4.0, "6mm": -6.0}.items():
            mask = torch.lt(denorm_target, th)
            mask_pred = torch.masked_select(denorm_pred, mask)
            mask_target = torch.masked_select(denorm_target, mask)

            cnt = mask.sum()
            losses["cnt_" + label_th] = cnt
            if cnt > 0:
                losses["mask_MSE_" + label_th] = self.metric(mask_pred, mask_target)
                abs_error = torch.abs(mask_pred - mask_target)
                for label_to, tolerance in {
                    "03mm": 0.3,
                    "05mm": 0.5,
                    "10mm": 1.0,
                }.items():
                    losses["cnt_" + label_th + "_" + label_to] = (
                        abs_error < tolerance
                    ).sum()
            else:
                losses["mask_MSE_" + label_th] = cnt
                losses["cnt_" + label_th + "_03mm"] = cnt
                losses["cnt_" + label_th + "_05mm"] = cnt
                losses["cnt_" + label_th + "_10mm"] = cnt

        return losses

    def _calc_metrics_multi_step(self, denorm_pred, denorm_target):
        losses = {}
        denorm_pred = denorm_pred[:, :, self.eval_index]
        denorm_target = denorm_target[:, :, self.eval_index]
        cnt = torch.ones_like(denorm_pred).to(int).sum()
        losses["cnt"] = cnt

        losses["denorm_MSE"] = self.metric(denorm_pred, denorm_target)

        steps = ["step1", "step2", "step3", "allstep"]
        levels = ["-4mm", "-6mm", "overall"]
        tolerances = ["0.3mm", "0.5mm", "1.0mm"]
        for step in steps:
            if step == "allstep":
                tmp_pred = denorm_pred
                tmp_target = denorm_target
            else:
                steps_str_int = {"step1": 1, "step2": 2, "step3": 3}
                step_index = steps_str_int[step] - 1
                step_pred = denorm_pred[:, step_index]
                step_target = denorm_target[:, step_index]
            for level in levels:
                tmp_pred = step_pred.flatten()
                tmp_target = step_target.flatten()
                if level != "overall":
                    levels_str_int = {"-4mm": -4, "-6mm": -6}
                    level_value = levels_str_int[level]
                    tmp_pred = torch.masked_select(tmp_pred, tmp_target < level_value)
                    tmp_target = torch.masked_select(
                        tmp_target, tmp_target < level_value
                    )
                # MSE
                label = "MSE" + "_" + step + "_" + level + "_" + "cnt"
                losses[label] = torch.ones_like(tmp_target).to(int).sum()
                if len(tmp_target) == 0:
                    torch_zero = (tmp_target < level_value).sum()
                    label = "MSE" + "_" + step + "_" + level + "_" + "error"
                    losses[label] = torch_zero
                    for tolerance in tolerances:
                        label = "accuracy_" + step + "_" + level + "_" + tolerance
                        losses[label] = torch_zero
                else:
                    label = "MSE" + "_" + step + "_" + level + "_" + "error"
                    losses[label] = self.metric(tmp_pred, tmp_target)
                    # Accuracy
                    for tolerance in tolerances:
                        tolerances_str_int = {
                            "0.3mm": 0.3,
                            "0.5mm": 0.5,
                            "1.0mm": 1.0,
                        }
                        tolerance_value = tolerances_str_int[tolerance]
                        abs_error = torch.abs(tmp_pred - tmp_target)
                        label = "accuracy" + "_" + step + "_" + level + "_" + tolerance
                        losses[label] = (abs_error < tolerance_value).sum()
        return losses

    def validation_step(self, batch, batch_idx):
        loss, denorm_pred, denorm_target = self.shared_step(batch, batch_idx)
        if self.pred_len == 1:
            metrics = self._calc_metrics(denorm_pred, denorm_target)
        else:
            metrics = self._calc_metrics_multi_step(denorm_pred, denorm_target)
        metrics.update({"loss": loss})
        return metrics

    def _aggregate_metrics(self, outputs, prefix):
        losses = {}
        losses[prefix + "loss"] = torch.stack(
            [output["loss"] for output in outputs]
        ).mean()
        losses[prefix + "denorm_RMSE"] = (
            torch.stack([output["denorm_MSE"] for output in outputs]).sum()
            / torch.stack([output["cnt"] for output in outputs]).sum()
        ).sqrt()

        for mask in ["4mm", "6mm"]:
            cnt = torch.stack([output["cnt_" + mask] for output in outputs]).sum()
            losses[prefix + "_mask_RMSE_" + mask] = (
                torch.stack([output["mask_MSE_" + mask] for output in outputs]).sum()
                / cnt
            ).sqrt()

            for e in ["03mm", "05mm", "10mm"]:
                losses[prefix + "_mask_" + mask + "_accuracy_" + e] = (
                    torch.stack(
                        [output["cnt_" + mask + "_" + e] for output in outputs]
                    ).sum()
                    / cnt
                )
        return losses

    def _aggregate_metrics_multi_step(self, outputs, prefix):
        losses = {}

        # aggregate loss
        losses[prefix + "loss"] = torch.stack(
            [output["loss"] for output in outputs]
        ).mean()
        cnt = torch.stack([output["cnt"] for output in outputs]).sum()

        # aggregate overall (denormalized) RMSE
        losses[prefix + "denorm_RMSE"] = (
            torch.stack([output["denorm_MSE"] for output in outputs]).sum() / cnt
        ).sqrt()

        steps = ["step1", "step2", "step3", "allstep"]
        levels = ["-4mm", "-6mm", "overall"]
        tolerances = ["0.3mm", "0.5mm", "1.0mm"]
        for step in steps:
            for level in levels:
                output_label = "MSE_" + step + "_" + level + "_cnt"
                level_cnt = torch.stack(
                    [output[output_label] for output in outputs]
                ).sum()
                if level_cnt == 0:
                    label = prefix + "RMSE" + "_" + step + "_" + level
                    losses[label] = -1
                    for tolerance in tolerances:
                        label = (
                            prefix + "accuracy_" + step + "_" + level + "_" + tolerance
                        )
                        losses[label] = -1
                else:
                    label = prefix + "RMSE_" + step + "_" + level
                    output_label = "MSE_" + step + "_" + level + "_error"
                    losses[label] = (
                        torch.stack([output[output_label] for output in outputs]).sum()
                        / level_cnt
                    ).sqrt()
                    for tolerance in tolerances:
                        label = (
                            prefix + "accuracy_" + step + "_" + level + "_" + tolerance
                        )
                        output_label = (
                            "accuracy_" + step + "_" + level + "_" + tolerance
                        )
                        losses[label] = (
                            torch.stack(
                                [output[output_label] for output in outputs]
                            ).sum()
                            / level_cnt
                        )
        return losses

    def train_epoch_end(self, outputs):
        if self.pred_len == 1:  # single step
            losses = self._aggregate_metrics(outputs, "train_")
        else:  # multi step/horizon
            losses = self._aggregate_metrics_multi_step(outputs, "train_")
        self.log_dict(losses)
        return losses["train_loss"]

    def validation_epoch_end(self, outputs):
        if self.pred_len == 1:  # single step
            losses = self._aggregate_metrics(outputs, "val_")
        else:  # multi step/horizon
            losses = self._aggregate_metrics_multi_step(outputs, "val_")
        self.log_dict(losses)
        return losses["val_loss"]

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        if self.pred_len == 1:
            losses = self._aggregate_metrics(outputs, "test_")
        else:
            losses = self._aggregate_metrics_multi_step(outputs, "test_")
        self.log_dict(losses)
        return losses["test_loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), **(self.optim_params))