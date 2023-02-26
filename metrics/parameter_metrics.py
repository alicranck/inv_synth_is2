from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Union, Sequence, Dict, Any

import numpy as np
import librosa
import torch


class ParameterMetric(ABC):

    def __init__(self, learnable_params: Sequence):

        self.learnable_params = set(learnable_params)

    def __call__(self, prediction_params: Sequence[Dict[Any, Union[np.ndarray, torch.Tensor]]],
                 ground_truth_params: Sequence[Dict[Any, Union[np.ndarray, torch.Tensor]]]):

        prediction_params = self._collate(prediction_params)
        ground_truth_params = self._collate(ground_truth_params)

        self._verify_input(prediction_params, ground_truth_params)

        if isinstance(prediction_params, torch.Tensor) or isinstance(ground_truth_params, torch.Tensor):
            prediction_params = prediction_params.detach().cpu().numpy()
            ground_truth_params = ground_truth_params.detach().cpu().numpy()

        metric_val = self._calc_metric(prediction_params, ground_truth_params)

        return metric_val

    @staticmethod
    def _collate(param_dicts: Sequence[Dict[Any, Union[np.ndarray, torch.Tensor]]]) -> Dict[Any, np.ndarray]:

        res = defaultdict(list)
        for d in param_dicts:
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                res[k].append(v)

        for k, l in res.items():
            res[k] = np.stack(l)

        return res

    def _verify_input(self, prediction: Dict[Any, Union[np.ndarray, torch.Tensor]],
                      ground_truth: Dict[Any, Union[np.ndarray, torch.Tensor]]):

        assert set(prediction.keys()) == set(ground_truth.keys()),\
            f"Metric {self.to_str()} expected input parameter names to be identical but gut {prediction}" \
            f" and {ground_truth}"

        keys = sorted(prediction.keys())
        pred_param_shapes = [prediction[k].shape for k in keys]
        gt_param_shapes = [ground_truth[k].shape for k in keys]
        assert pred_param_shapes == gt_param_shapes, f"Metric {self.to_str()} expected param sizes to be identical but " \
                                                     f"got pred {pred_param_shapes} and gt: {gt_param_shapes} instead"

        assert isinstance(prediction[keys[0]], (torch.Tensor, np.ndarray)), f"Metric {self.to_str()} expected input pred to be" \
                                                                   f" one of ndarray, torch tensor, but got " \
                                                                   f"{type(prediction)} instead."

        assert isinstance(ground_truth[keys[0]], (torch.Tensor, np.ndarray)), f"Metric {self.to_str()} expected input pred to " \
                                                                     f"be one of ndarray, torch tensor, but got " \
                                                                     f"{type(ground_truth)} instead."
        assert np.all([len(s) == 2 for s in gt_param_shapes]), \
            f"Metric {self.to_str()} expected inputs to be 2-dimensional (param_vector_length,) " \
            f"or (batch, param_vector_length) but got input of shapes {gt_param_shapes}"

        return

    @abstractmethod
    def _calc_metric(self, prediction_params: Dict[Any, Union[np.ndarray, torch.Tensor]],
                     ground_truth_params: Dict[Any, Union[np.ndarray, torch.Tensor]]):
        pass

    @abstractmethod
    def to_str(self) -> str:
        pass

    @staticmethod
    def absolute_error(pred: np.ndarray, gt: np.ndarray, power: int = 1, reduction: str = 'mean'):
        diff = np.abs(pred - gt) ** power

        reduction_dims = tuple(range(diff.ndim))[1:] if diff.ndim > 1 else 0

        if reduction == 'mean':
            return diff.mean(axis=reduction_dims)
        elif reduction == 'sum':
            return diff.sum(axis=reduction_dims)
        else:
            raise NotImplementedError(f"Reduction {reduction} not implemented")


class ParametersMAE(ParameterMetric):

    def _calc_metric(self, prediction_params: Dict[Any, Union[np.ndarray, torch.Tensor]],
                     ground_truth_params: Dict[Any, Union[np.ndarray, torch.Tensor]]):

        all_errors = []
        for param_name, target_val in ground_truth_params.items():

            if param_name not in self.learnable_params:
                continue

            param_len = target_val.shape[-1]
            pred_val = prediction_params[param_name]

            if target_val.ndim > 1 and param_len > 1:      # Categorical value
                pred_val = self._get_ordinality(pred_val)
                target_val = self._get_ordinality(target_val)

            mae = self.absolute_error(pred_val, target_val)
            all_errors.append(mae)

        metric_val = np.mean(np.stack(all_errors).transpose(), axis=1)

        return metric_val

    @staticmethod
    def _get_ordinality(param_vals: np.ndarray) -> np.ndarray:
        ordinality = np.argmax(param_vals, axis=1, keepdims=True)
        normalized_ordinality = ordinality / (param_vals.shape[1] - 1)
        return normalized_ordinality

    def to_str(self) -> str:
        return 'ParametersMAE'


class ParametersAccuracy(ParameterMetric):

    def _calc_metric(self, prediction_params: Dict[Any, Union[np.ndarray, torch.Tensor]],
                     ground_truth_params: Dict[Any, Union[np.ndarray, torch.Tensor]]):

        all_errors = []
        for param_name, target_val in ground_truth_params.items():

            pred_val = prediction_params[param_name]
            param_len = target_val.shape[-1]

            if param_name not in self.learnable_params or param_len <= 1:
                continue

            pred_val = np.argmax(pred_val, axis=1)
            target_val = np.argmax(target_val, axis=1)

            match = pred_val == target_val

            all_errors.append(match)

        metric_val = np.mean(np.stack(all_errors).transpose(), axis=1)

        return metric_val

    def to_str(self) -> str:
        return 'ParametersAccuracy'
