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
    def __init__(self, scale_factor):
        super(NeuralSuperSampling, self).__init__()
        self.scale_factor = scale_factor
        self.feature_extraction = FeatureExtraction()
        self.zero_upsample = ZeroUpsample(scale_factor)

    def forward(self, batch):
        # color: (B, I, 3, H, W)
        # motion: (B, I, 2, H, W)
        # depth: (B, I, 1, H, W)
        color, motion, depth = batch
        B, I, _, H, W = color.shape

        # The current frame gets converted to YCbCr before feature extraction
        # But we still need to keep an RGB copy for feature reweighting
        f1_rgb = color[:, 0, :, :, :].clone()
        color[:, 0, :, :, :] = K.color.rgb_to_ycbcr(color[:, 0, :, :, :])

        # Feature extraction
        features = self.feature_extraction(torch.cat([color, depth], dim=CHANNEL_DIM))
        zu_features = self.zero_upsample(features)

        # Reprojection
        bl_motion = F.interpolate(
            motion.reshape(B * I, 2, H, W),
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=False,
        )


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
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

        f1_features = self.f1_conv(current_frame).unsqueeze(FRAME_DIM)
        # Each fame is processed individually with shared weights, so we reshape
        fp_features = self.fp_conv(past_frames).reshape(B, I - 1, 8, H, W)
        features = torch.cat([f1_features, fp_features], dim=FRAME_DIM)

        # Output: (B, I, 12, H, W)
        return torch.cat([x, features], dim=CHANNEL_DIM)


class ZeroUpsample(nn.Module):
    def __init__(self, scale_factor):
        super(ZeroUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        B, I, C, H, W = x.shape

        output_w, output_h = W * self.scale_factor, H * self.scale_factor
        output = torch.zeros(B, I, C, output_h, output_w, device=x.device)

        indices_h = (
            torch.arange(0, output_h, self.scale_factor, device=x.device)
            + self.scale_factor // 2
        )
        indices_w = (
            torch.arange(0, output_w, self.scale_factor, device=x.device)
            + self.scale_factor // 2
        )
        indices_grid_h, indices_grid_w = torch.meshgrid(indices_h, indices_w)

        output[:, :, :, indices_grid_h, indices_grid_w] = x

        return output
