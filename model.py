import torch
import torch.nn as nn
import util

PADDING = "same"
KERNEL_SIZE = 3


class FeatureExtractionNetwork(nn.Module):
    def __init__(self):
        super(FeatureExtractionNetwork, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=KERNEL_SIZE, padding=PADDING)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))

        return torch.cat([x, x3], dim=1)


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
    def __init__(self, num_frames):
        super(NeuralSuperSamplingNetwork, self).__init__()
        # TODO: Implement the network
        pass
