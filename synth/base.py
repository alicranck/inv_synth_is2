from abc import ABC, abstractmethod, abstractproperty
from typing import Sequence, Union, Dict

import numpy as np
import torch
from torch import Tensor, softmax

import dawdreamer as daw


class BaseSynth(ABC):

    def __init__(self, name: str):
        self.name = name

    def generate_sounds(self, params: Sequence[Union[Dict, Tensor, np.ndarray]]) -> Tensor:
        sounds = [self.generate_single_sound(single_sound_params) for single_sound_params in params]
        torch_sounds = torch.stack([torch.Tensor(sound) for sound in sounds])
        return torch_sounds

    @abstractmethod
    def generate_single_sound(self, params: Union[Dict, Tensor, np.ndarray]) -> np.ndarray:
        pass
