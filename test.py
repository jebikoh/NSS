import torch
import torch.nn.functional as F
from model import NeuralSuperSampling, ZeroUpsample
from util import warp
import matplotlib.pyplot as plt
import numpy as np
import os
from util import read_motion

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

if __name__ == "__main__":
    file_path = "data/540p/motion/0009/0003.exr"
    mh, mv = read_motion(file_path, negate_mv=True)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(mv, cmap="viridis")
    plt.colorbar(label="Vertical Velocity")
    plt.title("Vertical Motion Vectors")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mh, cmap="viridis")
    plt.colorbar(label="Horizontal Velocity")
    plt.title("Horizontal Motion Vectors")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
