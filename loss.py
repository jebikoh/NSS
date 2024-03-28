import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as T


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.layers = {"3", "8", "15", "22"}

    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)

        loss = 0
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            y = layer(y)
            if name in self.layers:
                loss += torch.norm(x - y, p=2) ** 2
        return loss
