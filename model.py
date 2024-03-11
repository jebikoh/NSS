import torch
import torch.nn as nn
import torch.nn.functional as F
import util

PADDING = "same"
KERNEL_SIZE = 3


class FeatureExtractionNetwork(nn.Module):
    def __init__(self):
        super(FeatureExtractionNetwork, self).__init__()

        self.current_frame = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
        )

        self.past_frames = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
        )

    def forward(self, current_frame, past_frames):
        # current_frame: (B, 4, H, W)
        # past_frames: (B * (I - 1), 4, H, W)
        features_current = self.current_frame(current_frame)
        features_past = self.past_frames(past_frames)

        # (B * I, C, H, W)
        return torch.cat(
            torch.cat([current_frame, features_current], dim=1),
            torch.cat([past_frames, features_past], dim=1),
            dim=0,
        )


class ZeroUpsampling(nn.Module):
    def __init__(self, sample_factor):
        super(ZeroUpsampling, self).__init__()
        self.sample_factor = sample_factor

    def forward(self, x):
        h_scale, w_scale = self.sample_factor
        B, C, H, W = x.shape
        up = torch.zeros((B, C, H * h_scale, W * w_scale), dtype=x.dtype)
        up[:, :, ::h_scale, ::w_scale] = x

        return up


class FeatureReweightingNetwork(nn.Module):
    def __init__(self, num_frames, scale_factor=10):
        super(FeatureReweightingNetwork, self).__init__()

        self.scale_factor = scale_factor

        self.conv1 = nn.Conv2d(
            num_frames * 4, 32, kernel_size=KERNEL_SIZE, padding=PADDING
        )
        self.conv2 = nn.Conv2d(32, 32, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv3 = nn.Conv2d(32, 4, kernel_size=KERNEL_SIZE, padding=PADDING)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x))
        x = (x + 1) * self.scale_factor

        return x


class ReconstructionNetwork(nn.Module):
    def __init__(self, num_frames):
        super(ReconstructionNetwork, self).__init__()

        self.conv1 = nn.conv2d(
            num_frames * 4, 64, kernel_size=KERNEL_SIZE, padding=PADDING
        )
        self.conv2 = nn.conv2d(64, 32, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv3 = nn.conv2d(32, 64, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv4 = nn.conv2d(64, 64, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv5 = nn.conv2d(64, 128, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv6 = nn.conv2d(128, 128, kernel_size=KERNEL_SIZE, padding=PADDING)
        # The paper isn't clear on what "upsize" is
        # I think this is the most likely interpretation
        # TODO: Try upsampling (clone) instead of transposed convolution
        self.upsize6 = nn.ConvTranspose2d(
            128, 128, kernel_size=KERNEL_SIZE, stride=2, padding=1, output_padding=1
        )
        self.conv7 = nn.conv2d(128, 64, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv8 = nn.conv2d(64, 64, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.upsize8 = nn.ConvTranspose2d(
            64, 64, kernel_size=KERNEL_SIZE, stride=2, padding=1, output_padding=1
        )
        self.conv9 = nn.conv2d(64, 32, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv10 = nn.conv2d(32, 3, kernel_size=KERNEL_SIZE, padding=PADDING)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.pool(self.relu(self.conv2(x1)))
        x3 = self.relu(self.conv3(x2))
        x4 = self.pool(self.relu(self.conv4(x3)))
        x5 = self.relu(self.conv5(x4))
        x6 = self.upsize6(self.relu(self.conv6(x5)))

        x7 = torch.cat([x4, x6], dim=1)
        x7 = self.relu(self.conv7(x7))

        x8 = self.upsize8(self.relu(self.conv8(x7)))

        x9 = torch.cat([x3, x8], dim=1)
        x9 = self.relu(self.conv9(x9))

        return self.conv10(x9)


class NeuralSuperSamplingNetwork(nn.Module):
    def __init__(self, num_frames, source_resolution, target_resolution):
        super(NeuralSuperSamplingNetwork, self).__init__()
        # TODO: Implement the network
        self.num_frames = num_frames
        self.sample_factor = (
            target_resolution[0] // source_resolution[0],
            target_resolution[1] // source_resolution[1],
        )

        self.feature_extraction = FeatureExtractionNetwork()
        self.zero_upsampling = ZeroUpsampling(self.sample_factor)

    def forward(self, color, motion_vectors, depth):
        """
        Args:
            color: (B, I, 3, H, W)
            motion_vectors: (B, I, 1, H, W)
            depth: (B, I, 2, H, W)
        """
        B, I, _, H, W = color.shape
        # For now, we will process the current frame and previous frames separately
        # The can done together, but it gets confusing
        current_frame_rgb = color[:, 0, :, :, :].unsqueeze(1)
        current_frame_depth = depth[:, 0, :, :, :].unsqueeze(1)
        current_frame_motion = motion_vectors[:, -1, :, :, :].unsqueeze(1)
        current_frame_ycbcr = util.rgb_to_ycbcr(current_frame_rgb)

        past_frames_color = color[:, 1:, :, :, :].reshape(B * (I - 1), 3, H, W)
        past_frames_depth = depth[:, 1:, :, :, :].reshape(B * (I - 1), 3, H, W)
        past_frames_motion = motion_vectors[:, 1:, :, :, :].reshape(
            B * (I - 1), 3, H, W
        )

        # Feature extraction
        # (B * I, 12, H, W)
        features = self.feature_extraction(
            torch.cat([current_frame_ycbcr, current_frame_depth], dim=1),
            torch.cat([past_frames_color, past_frames_depth], dim=1),
        )

        # Zero upsampling
        # (B * I, 12, H * h_scale, W * w_scale)
        zero_upsampled_features = self.zero_upsampling(features)
        # (B, 3, H * h_scale, W * w_scale)
        zero_upsampled_rgb = self.zero_upsampling(current_frame_rgb)

        # Warping
        current_frame_motion_up = F.interpolate(
            current_frame_motion,
            scale_factor=(self.sample_factor[1], self.sample_factor[0]),
            mode="bilinear",
        )
        past_frames_motion_up = F.interpolate(
            past_frames_motion,
            scale_factor=(self.sample_factor[1], self.sample_factor[0]),
            mode="bilinear",
        )
