import torch
import torch.nn.functional as F
from model import NeuralSuperSampling, ZeroUpsample


if __name__ == "__main__":
    ones = torch.ones(1, 1, 1, 5, 5)
    zu = ZeroUpsample(3)
    print(zu(ones))
    # zu = F.interpolate(ones, scale_factor=2, mode="nearest")
    # print(zu)
    # color = torch.randn(2, 5, 3, 100, 100)
    # depth = torch.randn(2, 5, 1, 100, 100)
    # motion = torch.randn(2, 5, 2, 100, 100)
    # batch = (color, motion, depth)

    # model = NeuralSuperSampling((100, 100), (200, 200))
    # output = model(batch)
