from abc import abstractmethod, ABC
from typing import Union, Sequence

import numpy as np
import librosa
import torch


class AudioMetric(ABC):

    def __call__(self, prediction_audio: Union[np.ndarray, torch.Tensor],
                 ground_truth_audio: Union[np.ndarray, torch.Tensor]):

        self._verify_input(prediction_audio, ground_truth_audio)

        if isinstance(prediction_audio, torch.Tensor) or isinstance(ground_truth_audio, torch.Tensor):
            prediction_audio = prediction_audio.detach().cpu().numpy()
            ground_truth_audio = ground_truth_audio.detach().cpu().numpy()

        prediction_transform_signal = self._transform_signal(prediction_audio)
        ground_truth_transform_signal = self._transform_signal(ground_truth_audio)

        metric_val = self._calc_metric(prediction_transform_signal, ground_truth_transform_signal)

        return metric_val

    def _verify_input(self, prediction: Union[np.ndarray, torch.Tensor], ground_truth: Union[np.ndarray, torch.Tensor]):
        assert prediction.shape == ground_truth.shape, f"Metric {self.to_str()} expected inputs of equal shape but " \
                                                       f"got input a of shape {prediction.shape}, input b of shape " \
                                                       f"{ground_truth.shape}"
        assert isinstance(prediction, (torch.Tensor, np.ndarray)), f"Metric {self.to_str()} expected input pred to be" \
                                                                   f" one of ndarray, torch tensor, but got " \
                                                                   f"{type(prediction)} instead."
        assert isinstance(ground_truth, (torch.Tensor, np.ndarray)), f"Metric {self.to_str()} expected input pred to " \
                                                                     f"be one of ndarray, torch tensor, but got " \
                                                                     f"{type(ground_truth)} instead."
        assert prediction.ndim in (1, 2), f"Metric {self.to_str()} expected inputs to be 1 or 2-dimensional " \
                                          f"(audio_length,) or (batch, audio_length) but got input of shape" \
                                          f" {prediction.shape}"

        return

    def _transform_signal(self, signal: np.ndarray):
        return signal

    @abstractmethod
    def _calc_metric(self, transformed_prediction_signal: np.ndarray, transformed_ground_truth_signal: np.ndarray):
        pass

    @abstractmethod
    def to_str(self) -> str:
        pass

    @staticmethod
    def stft_transform(audio_signal: np.ndarray, n_fft: int, hop_length: int, **kwargs):
        transformed_signal = librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length, **kwargs)
        return transformed_signal

    @staticmethod
    def mel_transform(audio_signal: np.ndarray, n_fft: int, hop_length: int, sample_rate: int):
        transformed_signal = librosa.feature.melspectrogram(y=audio_signal, n_fft=n_fft, n_mels=128, fmin=50,
                                                            fmax=sample_rate // 2, hop_length=hop_length,
                                                            pad_mode='reflect', sr=sample_rate, norm='slaney',
                                                            htk=True)
        return transformed_signal

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


class LogMagSpectrogramMAE(AudioMetric):

    def _calc_metric(self, transformed_prediction_signal: np.ndarray, transformed_ground_truth_signal: np.ndarray):
        metric_val = self.absolute_error(transformed_prediction_signal, transformed_ground_truth_signal)
        return metric_val

    def _transform_signal(self, signal: np.ndarray):

        eps = 1e-4

        stft_spec = self.stft_transform(signal, n_fft=1024, hop_length=256)

        abs_stft_spec = np.abs(stft_spec)
        abs_stft_spec = np.where(abs_stft_spec > eps, abs_stft_spec, eps)

        log_abs_stft_spec = np.log10(abs_stft_spec)

        return log_abs_stft_spec

    def to_str(self) -> str:
        return 'LogMagSpectrogramMAE'


class MelMultiSpectrogramMAE(AudioMetric):

    def __init__(self, sample_rate: int, mel_fft_sizes: Sequence[int] = (2048, 1024, 512, 256, 128, 64),
                 reduction: str = 'none'):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_fft_sizes = mel_fft_sizes
        self.reduction = reduction

    def _calc_metric(self, transformed_prediction_signal: np.ndarray, transformed_ground_truth_signal: np.ndarray):

        res = {}
        for fft_size in self.mel_fft_sizes:

            pred_mel = self.mel_transform(transformed_prediction_signal, fft_size, fft_size // 2, self.sample_rate)
            gt_mel = self.mel_transform(transformed_ground_truth_signal, fft_size, fft_size // 2, self.sample_rate)

            error = self.absolute_error(pred_mel, gt_mel)

            res[str(fft_size)] = error

        if self.reduction == 'mean':
            res_vals = np.stack([v for v in res.values()]).transpose()
            res = np.mean(res_vals, axis=-1)

        return res

    def to_str(self) -> str:
        return 'MelMultiSpectrogramMAE'


class MFCCMAE(AudioMetric):

    def __init__(self, sample_rate: int, n_mfcc: int = 40):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def _calc_metric(self, transformed_prediction_signal: np.ndarray, transformed_ground_truth_signal: np.ndarray):
        metric_val = self.absolute_error(transformed_prediction_signal, transformed_ground_truth_signal)
        return metric_val

    def _transform_signal(self, signal: np.ndarray):

        mfcc_features = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=self.n_mfcc)

        return mfcc_features

    def to_str(self) -> str:
        return 'MFCCMAE'


class SpectralConvergence(AudioMetric):

    def _calc_metric(self, transformed_prediction_signal: np.ndarray, transformed_ground_truth_signal: np.ndarray):

        nom = np.linalg.norm(transformed_ground_truth_signal - transformed_prediction_signal, axis=(1, 2), ord='fro')
        denom = np.linalg.norm(transformed_ground_truth_signal, axis=(1, 2), ord='fro')

        metric_val = nom / denom

        return metric_val

    def _transform_signal(self, signal: np.ndarray):

        stft_spec = self.stft_transform(signal, n_fft=1024, hop_length=256)
        abs_stft_spec = np.abs(stft_spec)

        return abs_stft_spec

    def to_str(self) -> str:
        return 'SpectralConvergence'
