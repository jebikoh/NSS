import torch
from model import NeuralSuperSampling


if __name__ == "__main__":
    color = torch.randn(2, 5, 3, 100, 100)
    depth = torch.randn(2, 5, 1, 100, 100)
    motion = torch.randn(2, 5, 2, 100, 100)
    batch = (color, motion, depth)

    model = NeuralSuperSampling((100, 100), (200, 200))
    output = model(batch)
