import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class VAEDecoder(nn.Module):
    def __init__(
        self,
        out_channels = 3,
        latent_dim = 128,
        hidden_channels = None,
        image_size = 128,
        use_sigmoid = True,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        # Depende del tamaño de la cantidad de canales podemos
        # inferir la cantidad de upsampling
        num_upsamplings = len(hidden_channels)
        self.initial_size = image_size // (2 ** num_upsamplings)
        print('initial size ', self.initial_size)

        self.initial_channels = hidden_channels[-1]
        self.flatten_dim = self.initial_channels * self.initial_size * self.initial_size

        # z -> feature map inicial
        self.fc = nn.Linear(latent_dim, self.flatten_dim)

        reversed_channels = list(reversed(hidden_channels))
        layers = []

        for i in range(len(reversed_channels) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=reversed_channels[i],
                        out_channels=reversed_channels[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(reversed_channels[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )

        # Ponemos fuera del for la ultima capa
        layers.append(
            nn.ConvTranspose2d(
                in_channels=reversed_channels[-1],
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )

        # Generalmente usaremos sigmoid debido a que las imágenes las estamos
        # transformando con ToTensor, el cuál las deja en 0 y 1
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.deconv_decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        h = self.fc(z)

        # Lo convertimos a "imagen", o mejor dicho un tensor
        # interpretable por la Conv [B, C, H, W]
        h = h.view(
            z.size(0),
            self.initial_channels,
            self.initial_size,
            self.initial_size,
        )

        x_hat = self.deconv_decoder(h)

        return x_hat
