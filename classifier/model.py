import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class Classifier(pl.LightningModule):
    def __init__(self,
                 base_network,
                 fully_connected_layer,
                 activation=F.log_softmax,
                 loss=F.nll_loss):
        super().__init__()
        self.model = base_network
        self.model.fc = fully_connected_layer
        self.activation = activation
        self.loss = loss

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        images, labels = batch
        predict = self.activation(self.forward(images)).flatten()
        # print(labels.shape)
        loss = self.loss(predict, labels)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
