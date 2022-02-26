from abc import abstractmethod
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class AbstractModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        ...

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
