import os
from typing import Sequence

import numpy as np
import pandas as pd
import soundfile
import torch
from scipy import signal
from torch.utils.data import Dataset


class DXDataset(Dataset):

    def __init__(self, data_root: str, wavs_dir: str, split: str, operators: Sequence[int] = None):

        if operators is not None:
            self.ops_suffix = '_op' + ''.join(['{}'.format(op) for op in operators])
        else:
            self.ops_suffix = ''

        self.wavs_dir = wavs_dir

        pt_file_path = os.path.join(data_root, f'{split.lower()}_TensorDataset.pt')
        self.dataset = torch.load(pt_file_path, map_location='cpu')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        spec, params, sample_info = self.dataset[item]

        preset_id, midi_pitch, midi_velocity = sample_info
        audio = self._get_wav_file(preset_id, midi_pitch, midi_velocity)

        sample = {'audio': audio, 'spectrogram': spec, 'parameters_vector': params}
        return sample

    def _get_wav_file(self, preset_uid, midi_note, midi_velocity):
        file_path = self._get_wav_file_path(preset_uid, midi_note, midi_velocity)
        try:
            return soundfile.read(file_path)[0]
        except RuntimeError:
            raise RuntimeError("[data/dataset.py] Can't open file {}. Please pre-render audio files for this "
                               "dataset configuration.".format(file_path))

    def _get_wav_file_path(self, preset_uid, midi_note, midi_velocity):
        """ Returns the path of a wav (from dexed_presets folder). Operators"""
        filename = f"preset{preset_uid:06d}_midi{midi_note:03d}vel{midi_velocity:03d}{self.ops_suffix}.wav"
        return os.path.join(self.wavs_dir, filename)


class TALDataset(Dataset):

    def __init__(self, wavs_dir: str, csv_path: str, min_val: float = -10, truncate_spectrogram: bool = False):
        self.params_df = pd.read_csv(csv_path)
        self.wavs_dir = wavs_dir
        self.clip_spectrogram = truncate_spectrogram
        self.min_val = min_val

    def __getitem__(self, item):

        sample_row = self.params_df.iloc[item]

        params = sample_row.drop('wav_id').to_dict()

        wav_id = sample_row['wav_id']
        wav_path = os.path.join(self.wavs_dir, f'{wav_id}.wav')
        audio, sample_rate = soundfile.read(wav_path)[0]

        spectrogram = signal.stft(audio, sample_rate, window='hann', nperseg=512, noverlap=512 - 128)
        normalized_spectrogram = self._normalize_spec(spectrogram)

        sample = {'audio': audio, 'spectrogram': normalized_spectrogram, 'parameters_vector': params}

        return sample

    def _normalize_spec(self, spectrogram):

        if self.clip_spectrogram:
            spectrogram = np.abs(spectrogram[2][:-1, :-1]) ** 2
        else:
            spectrogram = np.abs(spectrogram[2]) ** 2

        # Normalize and take log
        spectrogram = spectrogram / np.max(spectrogram) + np.finfo(float).eps
        spectrogram = np.log10(spectrogram)

        # take to (-1, 1)
        spectrogram = np.maximum(spectrogram, self.min_val) / (np.abs(self.min_val) / 2) + 1

        return spectrogram

    def __len__(self):
        return len(self.params_df)



