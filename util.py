import torch
import torch.nn.functional as F


def warp(input, flow):
    # Adapted from: https://discuss.pytorch.org/t/image-warping-for-backward-flow-using-forward-flow-matrix-optical-flow/99298
    # Flow should have horizontal in channel 0 and vertical in channel 1
    B, _, H, W = input.shape

    x = (
        torch.arange(W, dtype=input.dtype, device=input.device)
        .view(1, -1)
        .expand(H, -1)
    )
    y = (
        torch.arange(H, dtype=input.dtype, device=input.device)
        .view(-1, 1)
        .expand(-1, W)
    )
    x = x.view(1, 1, H, W).repeat(B, 1, 1, 1)
    y = y.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((x, y), 1)
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1
    vgrid[:, 1, :, :] = 2 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1

    return F.grid_sample(
        input, vgrid.permute(0, 2, 3, 1), "bilinear", align_corners=True
    )
