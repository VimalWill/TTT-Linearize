"""Optional Lightning adapter; import after MAD dependencies are installed."""
from mad.model.pl_model_wrapper import PLModelWrap

from .objective import configure_objective, mad_loss


class MADObjectiveWrap(PLModelWrap):
    """Use with Lightning Trainer.fit and MAD train/validation dataloaders."""
    def __init__(self, model, mad_config, mode='task', mse_factor=1000.,
                 lm_loss_weight=1.):
        super().__init__(model, mad_config)
        self.mode, self.mse_factor = mode, mse_factor
        self.lm_loss_weight = lm_loss_weight
        configure_objective(model, mode)
        self.save_hyperparameters('mode', 'mse_factor', 'lm_loss_weight')

    def step(self, batch, batch_idx):
        inputs, targets = batch
        loss, outputs, components = mad_loss(
            self.model, inputs, targets, mode=self.mode,
            mse_factor=self.mse_factor, lm_loss_weight=self.lm_loss_weight,
            ignore_index=self.mad_config.target_ignore_index)
        self._loss_components = components
        return loss, outputs, targets

    def phase_step(self, batch, batch_idx, phase='train'):
        result = super().phase_step(batch, batch_idx, phase)
        for name, value in self._loss_components.items():
            self.log(f'{phase}/{name}', value, on_step=True, on_epoch=True,
                     sync_dist=True)
        return result
