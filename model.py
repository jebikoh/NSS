import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import kornia as K
from util import warp


# Dimension constants for readability
BATCH_DIM = 0
FRAME_DIM = 1
CHANNEL_DIM = 2


class NeuralSuperSampling(nn.Module):
    def __init__(self, scale_factor, num_frames=5):
        super(NeuralSuperSampling, self).__init__()
        self.scale_factor = scale_factor
        self.num_frames = 5
        self.feature_extraction = FeatureExtraction()
        self.zero_upsample = ZeroUpsample(scale_factor)
        self.backward_warp = AccumulativeBackwardWarp()
        self.feature_reweighting = FeatureReweightingNetwork(self.num_frames)

    def forward(self, batch):
        # color: (B, I, 3, H, W)
        # motion: (B, I, 2, H, W)
        # depth: (B, I, 1, H, W)
        color, motion, depth = batch
        B, I, _, H, W = color.shape
        H_n, W_n = H * self.scale_factor, W * self.scale_factor

        # The current frame gets converted to YCbCr before feature extraction
        # But we still need to keep an RGB copy for feature reweighting
        f0_rgb = color[:, 0, :, :, :].clone()
        color[:, 0, :, :, :] = K.color.rgb_to_ycbcr(color[:, 0, :, :, :])

        # Feature extraction
        features = self.feature_extraction(torch.cat([color, depth], dim=CHANNEL_DIM))
        features = self.zero_upsample(features)

        # Reprojection
        motion = F.interpolate(
            motion.reshape(B * I, 2, H, W),
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=False,
        ).reshape(B, I, 2, H_n, W_n)

        # This occurs in-place
        features = self.backward_warp(features, motion)

        # Feature reweighting
        # This takes in 0-upsampled f1 RGBD and 0-upsampled warped RGBD of past frames
        f0_rgbd = self.zero_upsample(
            torch.cat(
                [
                    f0_rgb.unsqueeze(FRAME_DIM),
                    depth[:, 0, :, :, :].unsqueeze(FRAME_DIM),
                ],
                dim=CHANNEL_DIM,
            )
        )
        fp_rgbd = features[:, 1:, 0:4, :, :]
        features = self.feature_reweighting(
            torch.cat([f0_rgbd, fp_rgbd], dim=FRAME_DIM).reshape(B, I * 4, H_n, W_n),
            features,
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

        offset = self.scale_factor // 2

        output_w, output_h = W * self.scale_factor, H * self.scale_factor
        output = torch.zeros(B, I, C, output_h, output_w, device=x.device)

        indices_h = torch.arange(
            offset, output_h + offset, self.scale_factor, device=x.device
        )
        indices_w = torch.arange(
            offset, output_w + offset, self.scale_factor, device=x.device
        )
        indices_grid_h, indices_grid_w = torch.meshgrid(indices_h, indices_w)

        output[:, :, :, indices_grid_h, indices_grid_w] = x

        return output


class AccumulativeBackwardWarp(nn.Module):
    def __init__(self):
        super(AccumulativeBackwardWarp, self).__init__()

    def forward(self, features, mv):
        _, I, _, _, _ = features.shape

        for i in range(1, I):
            for j in range(i - 1, -1, -1):
                features[:, i, :, :, :] = warp(
                    features[:, i, :, :, :],
                    mv[:, j, :, :, :].squeeze(FRAME_DIM),
                )

        return features


class FeatureReweightingNetwork(nn.Module):
    def __init__(self, num_frames, scale=10):
        super(FeatureReweightingNetwork, self).__init__()
        self.scale = scale
        self.num_frames = num_frames

        self.conv1 = nn.Conv2d(4 * num_frames, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, rgbd, features):
        B, I, C, H, W = features.shape
        x = self.relu(self.conv1(rgbd))
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x))

        x = (x + 1) * self.scale
        print(features.shape)
        print(rgbd.shape)
