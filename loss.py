import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.transforms.functional import rgb_to_grayscale
from pytorch_msssim import ssim


class NssLoss(nn.Module):
    def __init__(self, w=0.1):
        super(NssLoss, self).__init__()
        self.vgg16 = vgg16(pretrained=True).features[:23]
        for p in self.vgg16.parameters():
            p.requires_grad = False
        self.w = w

    def forward(self, im, im_hat):
        ssim_loss = 1 - ssim(im, im_hat, data_range=1.0, size_average=True)
        vgg_im = self.vgg16(im)
        vgg_im_hat = self.vgg16(im_hat)
        perceptual_loss = F.mse_loss(vgg_im, vgg_im_hat)

        return ssim_loss + self.w * perceptual_loss
