import os
from argparse import Namespace

import yaml

from collections import defaultdict
from typing import Dict, Sequence, Union, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from tqdm import tqdm

from shlomis_new_code.metrics.audio_metrics import LogMagSpectrogramMAE, MelMultiSpectrogramMAE, MFCCMAE, SpectralConvergence
from shlomis_new_code.data.torch_datasets import DXDataset
from shlomis_new_code.metrics.parameter_metrics import ParametersMAE, ParametersAccuracy
from shlomis_new_code.models.encoder_decoder_archs.models import EncDecBaselineDS
from shlomis_new_code.synth.base import BaseSynth
from shlomis_new_code.synth.daw_synths import DexedSynth, DawDreamerSynth


def evaluate_model(model: nn.Module, dataset_to_eval: torch.utils.data.Dataset, synth: DawDreamerSynth,
                   audio_metrics: Union[Sequence[Callable], Dict[str, Callable]],
                   parameter_metrics: Union[Sequence[Callable], Dict[str, Callable]], itf: bool = False,
                   batch_size: int = 64, output_dir: str = '', device: str = 'cuda:0'):

    evaluation_dataloader = DataLoader(dataset_to_eval, batch_size=batch_size, shuffle=False)

    output_metrics = {'audio': defaultdict(list), 'parameters': defaultdict(list)}
    for i, batch in tqdm(enumerate(evaluation_dataloader), total=len(dataset_to_eval) // batch_size):
        gt_audio, gt_spec, gt_params_raw = batch['audio'].to(device), batch['spectrogram'].to(device), \
                                       batch['parameters_vector'].to(device)
        gt_params = synth.parse_param_vectors(gt_params_raw)

        if itf:
            pass  # TODO implement inference time finetuning

        with torch.no_grad():

            _, pred_params_raw = model(gt_spec)
            pred_params = synth.parse_param_vectors(pred_params_raw)
            pred_audio = synth.generate_sounds(pred_params)

            for tag, metrics_collection in zip(['audio', 'parameters'], [audio_metrics, parameter_metrics]):
                for metric in metrics_collection:
                    metric_fn = metrics_collection[metric] if isinstance(metric, str) else metric
                    metric_val = metric_fn(pred_audio, gt_audio) if tag == 'audio' \
                        else metric_fn(pred_params, gt_params)
                    output_metrics[tag][str(metric)].extend(metric_val)

    for tag, metrics_dict in output_metrics.items():
        print(f"*****{tag} metrics:*****")
        for metric_name, metric_vals in metrics_dict.items():
            print(f"{metric_name}: \n\tmean: {np.mean(metric_vals)} \n\tstd: {np.std(metric_vals)}\n\t"
                  f"max: {np.amax(metric_vals)}\n\t min: {np.amin(metric_vals)}")

        if output_dir:
            res_df = pd.DataFrame(output_metrics[tag])
            output_path = os.path.join(output_dir, f'{tag}_metrics.csv')
            res_df.to_csv(output_path)

    return output_metrics


if __name__ == '__main__':

    device = 'cuda:0'

    data_root = r'/home/shlomis/PycharmProjects/baseline/'
    wavs_dir = os.path.join(data_root, 'synth', 'dexed_presets')
    data_split = 'test'

    model_ckpt_path = '/home/shlomis/PycharmProjects/baseline/runs/Journal/1/best_checkpoint'

    synth_cfg_path = r'/home/almogelharar/almog/ai_synth/shlomis_new_code/configs/synth/dexed_base_config.yaml'
    with open(synth_cfg_path, 'r') as f:
        synth_cfg_dict = yaml.load(f, Loader=yaml.Loader)
        synth_cfg = Namespace(**synth_cfg_dict)

    dataset_to_eval = DXDataset(data_root, wavs_dir, data_split)

    model = EncDecBaselineDS(synth_cfg.param_cardinality, is_encoder_as_baseline=True, is_decoder_as_baseline=True,
                             with_discriminator=True).to(device)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device)['weights'])
    model.eval()

    synth = DexedSynth('dexed_synth', synth_cfg)

    audio_metrics = {
        'spec_mae': LogMagSpectrogramMAE(),
        'mel_spec_mae': MelMultiSpectrogramMAE(sample_rate=synth_cfg.sample_rate, reduction='mean'),
        'mfcc_mae': MFCCMAE(sample_rate=synth_cfg.sample_rate),
        'spectral_convergence_value': SpectralConvergence(),
    }

    numerical_pseudo_numerical_params = synth_cfg.numeric_learnable_params + synth_cfg.pseudo_numeric_learnable_params

    parameter_metrics = {'params_mae': ParametersMAE(numerical_pseudo_numerical_params),
                         'params_accuracy': ParametersAccuracy(synth_cfg.categorical_learnable_params)}

    evaluate_model(model, dataset_to_eval, synth, audio_metrics, parameter_metrics, itf=False, device=device)



