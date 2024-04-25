import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import optim
from model import NeuralSuperSampling
import kornia as K
from loss import WeightedSSIMPerceptualLoss
from dataset import QRISPDataset
from torch.utils.data import DataLoader


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
        self.lr = lr
        self.nss = NeuralSuperSampling(scale_factor, num_frames, weight_scale)
        self.loss = WeightedSSIMPerceptualLoss(perceptual_weight)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        color, motion, depth, y = batch
        y_hat = self.nss(color, motion, depth)
        loss = self.loss(K.color.rgb_to_ycbcr(y), y_hat)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        color, motion, depth, y = batch
        y_hat = self.nss(color, motion, depth)
        loss = self.loss(K.color.rgb_to_ycbcr(y), y_hat)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        color, motion, depth, y = batch
        y_hat = self.nss(color, motion, depth)
        loss = self.loss(K.color.rgb_to_ycbcr(y), y_hat)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, color, motion, depth):
        return self.nss(color, motion, depth)


if __name__ == "__main__":
    SEQUENCE_LENGTH = 5
    BATCH_SIZE = 8
    EPOCHS = 10
    NUM_WORKERS = 11

    train_data = QRISPDataset("data/", split="train", sequence_length=SEQUENCE_LENGTH)
    val_data = QRISPDataset("data/", split="val", sequence_length=SEQUENCE_LENGTH)
    test_data = QRISPDataset("data/", split="test", sequence_length=SEQUENCE_LENGTH)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )

    model = NeuralSuperSamplingPL(
        scale_factor=2, num_frames=5, weight_scale=10, lr=1e-4, perceptual_weight=0.1
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    trainer = L.Trainer(max_epochs=EPOCHS, callbacks=[early_stop])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.test(model=model, dataloaders=test_loader)
