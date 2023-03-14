import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_accuracy


class ClassificationTask(pl.LightningModule):
    def __init__(self, model, optimizer_params):
        super().__init__()
        self.save_hyperparameters()
        # self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.optimizer_params = optimizer_params

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = multiclass_accuracy(y_hat, y, 10)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_params["type"])
        return optimizer(self.model.parameters(), **self.optimizer_params["parameters"])
