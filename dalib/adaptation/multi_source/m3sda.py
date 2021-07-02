import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.autograd import Function
from typing import Optional

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)


class Identity(nn.Module):
    def __init__(self, mode='p3'):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class feature_extractor(nn.Module):

    def __init__(self, backbone: nn.Module, bottleneck_dim: Optional[int] = 2048):
        super(feature_extractor, self).__init__()

        self.backbone = backbone

        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(bottleneck_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        with torch.no_grad():
            feature = self.backbone(x)
        feature = self.bottleneck(feature)
        feature = self.fc(feature)

        return feature


class predictor(nn.Module):

    def __init__(self, num_classes):
        super(predictor, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, self.num_classes)
        )

    def forward(self, feature, reverse=False):
        if reverse:
            feature = grad_reverse.grad_reverse(feature)

        feature = self.fc(feature)
        return feature