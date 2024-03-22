import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import kornia as K


# Dimension constants for readability
BATCH_DIM = 0
FRAME_DIM = 1
CHANNEL_DIM = 2


class NeuralSuperSampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extraction = FeatureExtraction()

    def forward(self, batch):
        # color: (B, I, 3, H, W)
        # motion: (B, I, 2, H, W)
        # depth: (B, I, 1, H, W)
        B, I, _, H, W = color.shape

        color, motion, depth = batch

        # The current frame gets converted to YCbCr before feature extraction
        # But we still need to keep an RGB copy for feature reweighting
        f1_rgb = color[:, 0, :, :, :].clone()
        color[:, 0, :, :, :] = K.color.rgb_to_ycbcr(color[:, 0, :, :, :])

        # Feature extraction
        features = self.feature_extraction(torch.cat([color, depth], dim=CHANNEL_DIM))


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction).__init__()
        self.f1_conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fp_conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, I, C, H, W = x.shape
        current_frame = x[:, 0, :, :, :]
        past_frames = x[:, 1:, :, :, :].reshape(B * (I - 1), C, H, W)

        f1_features = self.f1_conv(current_frame)
        fp_features = self.fp_conv(past_frames)

        f1_features_cat = torch.cat([f1_features, current_frame], dim=1).unsqueeze(
            FRAME_DIM
        )
        fp_features_cat = torch.cat([fp_features, past_frames], dim=2).reshape(
            B, I - 1, 8, H, W
        )

        return torch.cat([f1_features_cat, fp_features_cat], dim=FRAME_DIM)
