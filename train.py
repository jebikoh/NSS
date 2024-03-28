import lightning as L
from torch import optim
from model import NeuralSuperSampling
from loss import WeightedSSIMPerceptualLoss


class NeuralSuperSamplingPL(L.LightningModule):
    def __init__(
        self,
        scale_factor,
        num_frames=5,
        weight_scale=10,
        lr=1e-4,
        perceptual_weight=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.nss = NeuralSuperSampling(scale_factor, num_frames, weight_scale)
        self.loss = WeightedSSIMPerceptualLoss(perceptual_weight)

    def training_step(self, batch, batch_idx):
        color, motion, depth, y = batch
        y_hat = self.nss(color, motion, depth)
        loss = self.loss(y, y_hat)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
