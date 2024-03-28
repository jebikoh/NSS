import torch
import torch.nn.functional as F
from model import NeuralSuperSampling, ZeroUpsample
from util import warp
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    color = torch.randn(2, 5, 3, 100, 100)
    depth = torch.randn(2, 5, 1, 100, 100)
    motion = torch.randn(2, 5, 2, 100, 100)
    batch = (color, motion, depth)

    model = NeuralSuperSampling(2)
    output = model(batch)
