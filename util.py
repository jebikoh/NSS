import torch
import torch.nn.functional as F
import OpenEXR
import Imath
import numpy as np
import PIL as Image


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


def read_motion(file_path: str, negate_mv: bool = True):
    exr_file = OpenEXR.InputFile(file_path)

    header = exr_file.header()
    dw = header["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    vertical_channel = exr_file.channel("R", Imath.PixelType(Imath.PixelType.FLOAT))
    horizontal_channel = exr_file.channel("G", Imath.PixelType(Imath.PixelType.FLOAT))

    mv = np.frombuffer(vertical_channel, dtype=np.float32).reshape(size[1], size[0])
    mh = np.frombuffer(horizontal_channel, dtype=np.float32).reshape(size[1], size[0])

    if negate_mv:
        mv = -mv

    return np.stack((mh, mv), axis=0)


def read_depth(file_path: str):
    depth_img = np.array(Image.open(file_path))
    depth = (
        depth_img[:, :, 0] / 255.0
        + depth_img[:, :, 1] / 255.0**2
        + depth_img[:, :, 2] / 255.0**3
        + depth_img[:, :, 3] / 255.0**4
    )
    return depth
