from collections import defaultdict
from typing import Any, Optional

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import CrossEntropyLoss, L1Loss
from torch.optim.lr_scheduler import ConstantLR, ReduceLROnPlateau

from ..synth.base import *
from shlomis_new_code.metrics.audio_metrics import *


class LitIS2Module(LightningModule):

    def __init__(self, synth_cfg, train_cfg):

        super().__init__()

        self.synth_cfg = synth_cfg
        self.cfg = train_cfg

        self.synth = self._init_synth_from_config(synth_cfg)
        self.model = self._init_model_from_config(train_cfg.model)

        self.ce = CrossEntropyLoss()
        self.l1 = L1Loss()

        self.metric_fns = [LogMagSpectrogramMAE(), MelMultiSpectrogramMAE(self.synth_cfg.sample_rate),
                           SpectralConvergence(), MFCCMAE(self.synth_cfg.sample_rate)]
        
        self.train_epoch_param_diffs = defaultdict(list)
        self.train_epoch_param_vals = defaultdict(list)
        self.val_epoch_param_diffs = defaultdict(list)
        self.val_epoch_param_vals = defaultdict(list)

        self.tb_logger = None

    def forward(self, spectrogram: torch.Tensor, *args, **kwargs) -> Any:

        assert spectrogram.ndim == 3, f'Expected tensor of dimensions [batch_size, n_fft, len]' \
                                      f' but got shape {spectrogram.shape}'

        # Run NN model and convert predicted params from (0, 1) to original range
        predicted_parameters = self.model(spectrogram)

        return predicted_parameters

    def generate_synth_sound(self, params: Sequence[Union[Dict, Tensor]]) -> torch.Tensor:
        sounds = self.synth.generate_sounds(params)
        return sounds

    def in_domain_step(self, batch, return_metrics=False):

        gt_audio, gt_spec, gt_params = batch['audio'], batch['spectrogram'], batch['parameters_vector']

        _, pred_params_raw = self.forward(gt_spec)
        pred_params = self.synth.parse_param_vectors(pred_params_raw)
        pred_audio = self.synth.generate_sounds(pred_params)

        # Calculate losses
        params_loss = self.parameters_loss(pred_params, gt_params)
        audio_loss = self.audio_loss(pred_audio, gt_audio)
        loss = self.cfg.parameters_loss_w * params_loss + self.cfg.audio_loss_w * audio_loss
        step_losses = {'audio': audio_loss, 'parameters': params_loss, 'total': loss}

        param_diffs = self._get_param_diffs(pred_params.copy(), gt_params.copy())
        step_artifacts = {'predicted_parameters': pred_params, 'param_diffs': param_diffs}

        if return_metrics:   # Calculate and log metrics
            batch_audio_metrics = self._calculate_audio_metrics(pred_audio, gt_audio)
            batch_parameter_metrics = self._calculate_parameter_metrics(pred_params, gt_params)
            step_metrics = {**batch_audio_metrics, **batch_parameter_metrics}
            
            return loss, step_losses, step_artifacts, step_metrics
            
        return loss, step_losses, step_artifacts

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:

        if batch_idx == 0:
            self._log_sounds_batch(batch, f'samples_train')

        loss, step_losses, step_artifacts = self.in_domain_step(batch)
        
        self._log_recursive(step_losses, tag='train_losses')
        
        self._accumulate_batch_values(self.train_epoch_param_vals, step_artifacts['predicted_parameters'])
        self._accumulate_batch_values(self.train_epoch_param_diffs, step_artifacts['param_diffs'])

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx) -> Optional[STEP_OUTPUT]:

        if batch_idx == 0:
            self._log_sounds_batch(batch, 'samples_validation')

        loss, step_losses, step_artifacts, step_metrics = self.in_domain_step(batch, return_metrics=True)
        
        self._log_recursive(step_losses, tag='val_losses')
        self._log_recursive(step_metrics, tag='val_metrics')
        
        self._accumulate_batch_values(self.val_epoch_param_diffs, step_artifacts['param_diffs'])
        self._accumulate_batch_values(self.val_epoch_param_vals, step_artifacts['predicted_parameters'])

        return loss

    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        gt_audio, gt_spec, gt_params = batch['audio'], batch['spectrogram'], batch['parameters_vector']

        _, pred_params_raw = self.forward(gt_spec)
        pred_params = self.synth.parse_param_vectors(pred_params_raw)

        pred_audio = self.synth.generate_sounds(pred_params)

        return pred_audio, pred_params

    def on_train_epoch_end(self) -> None:
        self._log_recursive(self.train_epoch_param_diffs, 'param_diff')
        self._log_recursive(self.train_epoch_param_vals, 'param_values')

        self.train_epoch_param_diffs = defaultdict(list)
        self.train_epoch_param_vals = defaultdict(list)

        return

    def on_validation_epoch_end(self) -> None:
        self._log_recursive(self.val_epoch_param_diffs, 'param_diff')
        self._log_recursive(self.val_epoch_param_vals, 'param_values')

        self.val_epoch_param_diffs = defaultdict(list)
        self.val_epoch_param_vals = defaultdict(list)
        
        return 

    @torch.no_grad()
    def _calculate_audio_metrics(self, target_signal: torch.Tensor, predicted_signal: torch.Tensor):

        target_signal = target_signal.float()
        predicted_signal = predicted_signal.float()

        metrics = {}
        for metric_fn in self.metric_fns:
            metrics[metric_fn.to_str()] = metric_fn(predicted_signal, target_signal)

        return metrics

    @torch.no_grad()
    def _calculate_parameter_metrics(self, target_params: dict, predicted_params: dict):

        metrics = {}
        for metric_fn in self.param_metric_fns:
            metrics[metric_fn.to_str()] = metric_fn(predicted_params, target_params)

        return metrics

    def _log_sounds_batch(self, batch, tag: str):

        gt_audio, gt_spec, gt_params = batch['audio'], batch['spectrogram'], batch['parameters_vector']

        batch_size = len(gt_audio)

        _, pred_params_raw = self.forward(gt_spec)
        pred_params = self.synth.parse_param_vectors(pred_params_raw)
        pred_audio = self.synth.generate_sounds(pred_params)

        for i in range(min(self.cfg.logging.n_images_to_log, batch_size)):
            self.tb_logger.add_audio(f'{tag}/input_{i}_target', gt_audio[i], global_step=self.current_epoch,
                                     sample_rate=self.synth_cfg.sample_rate)
            self.tb_logger.add_audio(f'{tag}/input_{i}_pred', pred_audio[i],
                                     global_step=self.current_epoch, sample_rate=self.synth_cfg.sample_rate)

            signal_vis = visualize_signal_prediction(gt_audio[i], pred_audio[i], gt_params, pred_params, db=True)
            signal_vis_t = torch.tensor(signal_vis, dtype=torch.uint8, requires_grad=False)

            self.tb_logger.add_image(f'{tag}/{256}_spec/input_{i}', signal_vis_t, global_step=self.current_epoch,
                                     dataformats='HWC')

    def _log_recursive(self, items_to_log: dict, tag: str, on_epoch=False):
        if isinstance(items_to_log, np.float) or isinstance(items_to_log, np.int):
            self.log(tag, items_to_log, on_step=True, on_epoch=on_epoch)
            return

        if type(items_to_log) == list:
            items_to_log = np.asarray(items_to_log)

        if type(items_to_log) in [torch.Tensor, np.ndarray, int, float]:
            items_to_log = items_to_log.squeeze()
            if len(items_to_log.shape) == 0 or len(items_to_log) <= 1:
                if isinstance(items_to_log, (np.ndarray, np.generic)):
                    items_to_log = torch.tensor(items_to_log)
                self.log(tag, items_to_log, batch_size=self.cfg.model.batch_size)
            elif len(items_to_log) > 1:
                self.tb_logger.add_histogram(tag, items_to_log, self.current_epoch)
            else:
                raise ValueError(f"Unexpected value to log {items_to_log}")
            return

        if not isinstance(items_to_log, dict):
            return

        if 'operation' in items_to_log:
            tag += '_' + items_to_log['operation']

        for k, v in items_to_log.items():
            self._log_recursive(v, f'{tag}/{k}', on_epoch)

        return

    @staticmethod
    def _accumulate_batch_values(accumulator: dict, batch_vals: dict):

        batch_vals_np = to_numpy_recursive(batch_vals)

        for k, v in batch_vals_np:
            accumulator[k].extend(v)

        return

    def configure_optimizers(self):

        optimizer_params = self.cfg.model.optimizer

        # Configure optimizer
        if 'optimizer' not in optimizer_params or optimizer_params['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=optimizer_params.base_lr)
        elif optimizer_params['optimizer'].lower() == 'adam_lookahead':
            base_optimizer = torch.optim.Adam(self.parameters(), lr=optimizer_params.base_lr)
            optimizer = Lookahead(base_optimizer, la_steps=optimizer_params['lookahead_steps'],
                                  la_alpha=optimizer_params['lookahead_alpha'])
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_params['optimizer']} not implemented")

        # Configure learning rate scheduler
        if 'scheduler' not in optimizer_params or optimizer_params.scheduler.lower() == 'constant':
            scheduler_config = {"scheduler": ConstantLR(optimizer)}
        elif optimizer_params.scheduler.lower() == 'reduce_on_plateau':
            scheduler_config = {"scheduler": ReduceLROnPlateau(optimizer),
                                "interval": "epoch",
                                "monitor": "val_loss",
                                "frequency": 3,
                                "strict": True}
        elif optimizer_params.scheduler.lower() == 'cosine':
            scheduler_config = {"scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.model.num_epochs),
                "interval": "epoch"}
        elif optimizer_params.scheduler.lower() == 'cyclic':
            scheduler_config = {"scheduler": torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=self.cfg.model.optimizer.base_lr, max_lr=self.cfg.model.optimizer.max_lr,
                step_size_up=self.cfg.model.optimizer.cyclic_step_size_up),
                "interval": "step"}
        else:
            raise NotImplementedError(f"Scheduler {self.optimizer_params['scheduler']} not implemented")

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
