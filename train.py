import lightning as L
from torch import optim
from model import NeuralSuperSampling


class NeuralSuperSamplingPL(L.LightningModule):
    def __init__(self, scale_factor, num_frames=5, weight_scale=10, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.nss = NeuralSuperSampling(scale_factor, num_frames, weight_scale)

    def training_step(self, batch, batch_idx):
        y = self.nss(batch)
        loss = None
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
