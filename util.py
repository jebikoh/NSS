import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def zero_upsampling(input, scale=2):
    C, H, W = input.shape
    up = torch.zeros((C, H * scale, W * scale), dtype=input.dtype)
    up[:, ::scale, ::scale] = input

    return up


def backward_warp(input, motion_vectors):
    # This is taken from here: https://discuss.pytorch.org/t/image-warping-for-backward-flow-using-forward-flow-matrix-optical-flow/99298
    # I added comments for clarity
    B, C, H, W = input.shape
    # These two lines create a grid of coordinates
    # x goes from 0 to W-1, repeated H times
    # y goes from 0 to H-1, repeated W times
    x = torch.arange(0, W, dtype=input.dtype).view(1, -1).repeat(H, 1)
    y = torch.arange(0, H, dtype=input.dtype).view(-1, 1).repeat(1, W)
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
    return F.grid_sample(input, vgrid.permute(0, 2, 3, 1), mode="bilinear")
