import torch
from model import NeuralSuperSamplingNetwork

# (Batch, Frames, Channels, Height, Width)
color = torch.rand((2, 5, 3, 4, 4))
depth = torch.rand((2, 5, 1, 4, 4))
motion = torch.rand((2, 5, 2, 4, 4))

net = NeuralSuperSamplingNetwork((4, 4), (8, 8))

output = net(color, motion, depth)
