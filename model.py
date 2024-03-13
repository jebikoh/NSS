import torch
import torch.nn as nn
import torch.nn.functional as F

# import util

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
            [
                torch.cat([current_frame, features_current], dim=1),
                torch.cat([past_frames, features_past], dim=1),
            ],
            dim=0,
        )


class ZeroUpsampling(nn.Module):
    def __init__(self, sample_factor):
        super(ZeroUpsampling, self).__init__()
        self.sample_factor = sample_factor

    def forward(self, x):
        h_scale, w_scale = self.sample_factor
        B, C, H, W = x.shape
        up = torch.zeros(
            (B, C, H * h_scale, W * w_scale), dtype=x.dtype, device=x.device
        )
        up[:, :, ::h_scale, ::w_scale] = x

        return up


class BackwardWarp(nn.Module):
    def __init__(self):
        super(BackwardWarp, self).__init__()

    def forward(self, input, motion_vectors):
        # This is taken from here: https://discuss.pytorch.org/t/image-warping-for-backward-flow-using-forward-flow-matrix-optical-flow/99298
        # I added comments for clarity
        B, C, H, W = input.shape
        # These two lines create a grid of coordinates
        # x goes from 0 to W-1, repeated H times
        # y goes from 0 to H-1, repeated W times
        x = (
            torch.arange(0, W, dtype=input.dtype, device=input.device)
            .view(1, -1)
            .repeat(H, 1)
        )
        y = (
            torch.arange(0, H, dtype=input.dtype, device=input.device)
            .view(-1, 1)
            .repeat(1, W)
        )
        # We reshape them to input shape, repeat for batch size
        x = x.view(1, 1, H, W).repeat(B, 1, 1, 1)
        y = y.view(1, 1, H, W).repeat(B, 1, 1, 1)
        # We concat them: (B, 2, H, W)
        grid = torch.cat((x, y), 1).float()
        # We add the motion vectors to the grid
        vgrid = grid + motion_vectors
        # Now we normalize the vectors to be in the range [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        # Warp the input tensor off the grid using bilinear interpolation
        return F.grid_sample(
            input, vgrid.permute(0, 2, 3, 1), mode="bilinear", align_corners=False
        )


class FeatureReweightingNetwork(nn.Module):
    def __init__(self, num_frames, scale_factor=10):
        super(FeatureReweightingNetwork, self).__init__()

        self.scale_factor = scale_factor

        self.conv1 = nn.Conv2d(
            4 + 4 * (num_frames - 1), 32, kernel_size=KERNEL_SIZE, padding=PADDING
        )
        self.conv2 = nn.Conv2d(32, 32, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv3 = nn.Conv2d(32, 4, kernel_size=KERNEL_SIZE, padding=PADDING)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x))
        x = (x + torch.ones_like(x)) * self.scale_factor

        return x


class ReconstructionNetwork(nn.Module):
    def __init__(self, num_frames):
        super(ReconstructionNetwork, self).__init__()

        self.conv1 = nn.Conv2d(
            num_frames * 12, 64, kernel_size=KERNEL_SIZE, padding=PADDING
        )
        self.conv2 = nn.Conv2d(64, 32, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=KERNEL_SIZE, padding=PADDING)
        # The paper isn't clear on what "upsize" is
        # I think this is the most likely interpretation
        # TODO: Try upsampling (clone) instead of transposed convolution
        self.upsize6 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv7 = nn.Conv2d(128 + 64, 64, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.upsize8 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv9 = nn.Conv2d(64 + 32, 32, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv10 = nn.Conv2d(32, 3, kernel_size=KERNEL_SIZE, padding=PADDING)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        # NOTE: skip connections require PRE MAXPOOL activations
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(self.pool(x2)))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(self.pool(x4)))
        x6 = self.upsize6(self.relu(self.conv6(x5)))

        x7 = torch.cat([x4, x6], dim=1)
        x7 = self.relu(self.conv7(x7))

        x8 = self.upsize8(self.relu(self.conv8(x7)))

        x9 = torch.cat([x2, x8], dim=1)
        x9 = self.relu(self.conv9(x9))

        return self.relu(self.conv10(x9))


class NeuralSuperSamplingNetwork(nn.Module):
    def __init__(self, source_resolution, target_resolution):
        super(NeuralSuperSamplingNetwork, self).__init__()
        # TODO: Implement the network
        self.num_frames = 5
        self.sample_factor = (
            target_resolution[0] // source_resolution[0],
            target_resolution[1] // source_resolution[1],
        )

        self.feature_extraction = FeatureExtractionNetwork()
        self.zero_upsampling = ZeroUpsampling(self.sample_factor)
        self.backward_warp = BackwardWarp()
        self.feature_reweighting = FeatureReweightingNetwork(self.num_frames)
        self.reconstruction = ReconstructionNetwork(self.num_frames)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, color, motion_vectors, depth):
        """
        Args:
            color: (B, I, 3, H, W)
            motion_vectors: (B, I, 2, H, W)
            depth: (B, I, 1, H, W)
        """
        B, I, _, H, W = color.shape
        H_n, W_n = H * self.sample_factor[1], W * self.sample_factor[0]
        assert I == 5
        ## Feature extraction
        # Takes each color & depth of each frame -> generates 8-channel feature map
        # Concatenates the current frame with the feature map
        frames_color_depth = torch.cat([color, depth], dim=2)
        # These two don't share weights, so we have to split them
        # The shapes should end up: (B, 4, H, W) and (B * 4, 4, H, W)
        current_color_depth = frames_color_depth[:, 0, :, :, :].squeeze(1)
        past_color_depth = frames_color_depth[:, 1:, :, :, :].reshape(
            B * (I - 1), 4, H, W
        )
        # This method concatenates the features: (B, 12, H, W)
        features = self.feature_extraction(current_color_depth, past_color_depth)

        ## Temporal Re-projection
        # First, we zero upsample the features
        upsampled_features = self.zero_upsampling(features).reshape(B, 5, 12, H_n, W_n)
        # Then, we use bilinear interpolation to upscale the motion vectors
        upsampled_motion_vectors = F.interpolate(
            motion_vectors.reshape(B * I, 2, H, W),
            scale_factor=(self.sample_factor[1], self.sample_factor[0]),
            mode="bilinear",
        ).reshape(B, I, 2, H_n, W_n)

        # Now, we use the motion vectors to backward-warp the previous frames
        # For now, we are going to hardcode this for 4 previous frames
        accumulated_warped_features = torch.zeros(
            (B, 4, 12, H_n, W_n), device=upsampled_features.device
        )
        for i in range(1, I):
            curr_features = upsampled_features[:, i, :, :, :]
            for j in range(i - 1, -1, -1):
                curr_features = self.backward_warp(
                    curr_features,
                    upsampled_motion_vectors[:, j, :, :, :].squeeze(1),
                )
            accumulated_warped_features[:, i - 1, :, :, :] = curr_features

        ## Feature Reweighting
        upsampled_current_rgbd = upsampled_features[:, 0, 0:4, :, :].unsqueeze(1)
        warped_prev_rgbd = accumulated_warped_features[:, :, 0:4, :, :]

        reweight_input = torch.cat(
            [upsampled_current_rgbd, warped_prev_rgbd], dim=1
        ).reshape(B, I * 4, H_n, W_n)

        weight_map = self.feature_reweighting(reweight_input)
        weight_map = torch.cat(
            [
                torch.ones(
                    weight_map.shape[0],
                    1,
                    weight_map.shape[2],
                    weight_map.shape[3],
                    device=weight_map.device,
                ),
                weight_map,
            ],
            dim=1,
        )
        weighted_features = upsampled_features.reshape(
            B * I, 12, H_n, W_n
        ) * weight_map.reshape(B * I, 1, H_n, W_n)

        return self.reconstruction(weighted_features.reshape(B, I * 12, H_n, W_n))
