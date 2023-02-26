import dawdreamer as daw

import numpy as np
import torch
from torch.nn.functional import softmax

from .base import *


class DawDreamerSynth(BaseSynth):

    def __init__(self, name, config):
        super().__init__(name)

        self.cfg = config

        self.engine, self.synth = self._create_daw_engine_and_synth()

    def generate_single_sound(self, params: Union[Dict, Tensor]):

        cfg = self.cfg

        # Apply parameters to engine
        self._apply_params(params)
        self.synth.clear_midi()
        self.synth.add_midi_note(cfg.midi_note, cfg.velocity, cfg.start_sec, cfg.note_duration)

        # Get sound, convert stereo to mono
        self.engine.render(cfg.note_duration + cfg.note_off)
        audio_out = self.engine.get_audio()

        audio = np.asarray(audio_out)
        audio = (audio[0] + audio[1]) / 2

        # Add fadeout
        fadeout_len = int(np.floor(cfg.sample_rate * cfg.fadeout))
        if fadeout_len > 1:  # fadeout might be disabled if too short
            fadeout = np.linspace(1.0, 0.0, fadeout_len)
            audio[-fadeout_len:] = audio[-fadeout_len:] * fadeout

        return audio

    def parse_param_vectors(self, param_vectors: Union[np.ndarray, torch.Tensor]):

        params_cardinality_dict = self.get_params_cardinality()
        n_param_values = sum([np.abs(v) for v in params_cardinality_dict.values() if v != 1])

        assert param_vectors.ndim == 2 and param_vectors.shape[1] == n_param_values, \
            f"DexedSynth: parse_param_vectors expected input prameters to be of shape (batch, n_params) but got " \
            f"{param_vectors.shape} instead"

        parsed_params = [{} for _ in param_vectors]
        offset = 0
        for param_name, param_cardinality in params_cardinality_dict.items():
            if param_cardinality == 1:  # Skip parameter
                continue
            elif param_cardinality == -1:       # Numerical value
                param_vals = param_vectors[:, offset: offset + 1]
            elif param_cardinality > 1:      # Categorical value
                param_vals = param_vectors[:, offset: offset + param_cardinality]
                param_vals = softmax(param_vals, dim=-1)
            else:
                raise ValueError(f"DexedSynth: Unexpected value {param_cardinality} in params cardinality")

            param_vals = torch.clamp(param_vals, min=0.0, max=1.0)

            for i, d in enumerate(parsed_params):
                d[param_name] = param_vals[i]

            offset += param_cardinality

        return parsed_params

    def _apply_params(self, params_dict):
        self._reset_all_params()

        for default_p_idx, default_p_val in self.cfg.default_parameters.items():
            self.synth.set_parameter(int(default_p_idx), default_p_val)

        for p_idx, p_val in params_dict.items():
            if p_val.ndim > 0 and len(p_val) > 1:
                p_val = torch.argmax(p_val) / len(p_val)
            self.synth.set_parameter(p_idx, p_val.cpu().numpy())

        return

    def _reset_all_params(self):
        for i in range(self.synth.get_plugin_parameter_size()):
            self.synth.set_parameter(i, 0)

    def _create_daw_engine_and_synth(self):

        cfg = self.cfg

        engine = daw.RenderEngine(cfg.sample_rate, cfg.block_size)
        engine.set_bpm(cfg.bpm)

        synth = engine.make_plugin_processor("base_dx_synth", cfg.plugin_path)
        graph = [
            (synth, [])
        ]
        engine.load_graph(graph)

        return engine, synth

    @abstractmethod
    def get_params_cardinality(self) -> dict:
        pass


class DexedSynth(DawDreamerSynth):

    def __init__(self, name, config):
        super().__init__(name, config)

    def get_params_cardinality(self):

        params_cardinality_list = self.cfg.param_cardinality
        params_cardinality_dict = {i: v for i, v in enumerate(params_cardinality_list)}

        return params_cardinality_dict


class TALSynth(DawDreamerSynth):

    def __init__(self, name, config):
        super().__init__(name, config)

    def get_params_cardinality(self):
        return self.cfg.param_cardinality
