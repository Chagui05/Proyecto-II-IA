import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class VAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels = 3,
        image_size = 128,
        latent_dim = 128,
        hidden_channels = None
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        self.in_channels = in_channels  # son imágenes rgb al principio
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.image_size = image_size # son imágenes de 128x128 al principio

        layers = []

        current_channels = in_channels

        for out_channels in hidden_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

            current_channels = out_channels

        self.conv_encoder = nn.Sequential(*layers)

        # Cada capa divide a la mita la resolucion
        # 128 -> 64 -> 32 -> 16 -> 8
        num_downsamplings = len(hidden_channels)
        final_size = image_size // (2 ** num_downsamplings)
        self.flatten_dim = hidden_channels[-1] * final_size * final_size  # Channels x Height x Widht

        # Para luego obtener mu y logvar
        self.fc_mu     = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):

        h = self.conv_encoder(x)

        h = torch.flatten(h, start_dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
