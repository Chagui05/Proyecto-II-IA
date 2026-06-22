import torch
from torch import nn


class UNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=None,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        self.blocks = nn.ModuleList()

        current_channels = in_channels
        for out_ch in hidden_channels:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            current_channels = out_ch

    def forward(self, x):
        skips = []
        h = x
        # Todos los bloques menos el último guardan skip connections
        # Ej: [32, 64, 128, 256] → skips en 32, 64, 128; bottleneck en 256
        for block in self.blocks[:-1]:
            h = block(h)
            skips.append(h)

        bottleneck = self.blocks[-1](h)
        return bottleneck, skips
