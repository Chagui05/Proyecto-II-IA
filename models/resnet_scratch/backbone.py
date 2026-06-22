import torch
from torch import nn

from .blocks import BasicBlock


class ResNetPartialBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()

        self.in_channels = base_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # conv2_x de ResNet-18: [3x3, 64; 3x3, 64] x2
        self.conv2_x = self._make_layer(
            out_channels=base_channels,
            num_blocks=2,
            stride=1,
        )

        # conv3_x de ResNet-18: [3x3, 128; 3x3, 128] x2
        self.conv3_x = self._make_layer(
            out_channels=base_channels * 2,
            num_blocks=2,
            stride=2,
        )

        self.out_channels = base_channels * 2

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ):
        layers = []

        layers.append(
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        )

        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        h = self.conv1(x)
        h = self.maxpool(h)
        h = self.conv2_x(h)
        h = self.conv3_x(h)

        return h
