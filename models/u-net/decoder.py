import torch
from torch import nn


class UNetDecoder(nn.Module):
    def __init__(
        self,
        out_channels=3,
        hidden_channels=None,
        use_sigmoid=True,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        # reversed: [256, 128, 64, 32]
        reversed_ch = list(reversed(hidden_channels))

        self.up_convs = nn.ModuleList()
        self.refine_convs = nn.ModuleList()

        # Cada paso: upsample + concat con skip + refinamiento
        # Ej con [256, 128, 64, 32]:
        #   i=0: bottleneck [B,256,8,8]  → up→[B,128,16,16] → cat skip[B,128,16,16] → refine→[B,128,16,16]
        #   i=1:            [B,128,16,16] → up→[B,64,32,32]  → cat skip[B,64,32,32]  → refine→[B,64,32,32]
        #   i=2:            [B,64,32,32]  → up→[B,32,64,64]  → cat skip[B,32,64,64]  → refine→[B,32,64,64]
        for i in range(len(reversed_ch) - 1):
            in_ch = reversed_ch[i]
            out_ch = reversed_ch[i + 1]

            self.up_convs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

            # Después de concatenar: out_ch (upsample) + out_ch (skip) = 2*out_ch entradas
            self.refine_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

        # Última capa: upsample a resolución original sin skip connection
        # reversed_ch[-1] → out_channels (ej: 32 → 3)
        final_layers = [
            nn.ConvTranspose2d(reversed_ch[-1], out_channels, kernel_size=4, stride=2, padding=1),
        ]
        if use_sigmoid:
            final_layers.append(nn.Sigmoid())

        self.final_conv = nn.Sequential(*final_layers)

    def forward(self, bottleneck, skips):
        h = bottleneck
        # Los skips vienen de superficial a profundo; los procesamos al revés
        reversed_skips = list(reversed(skips))

        for up_conv, refine_conv, skip in zip(self.up_convs, self.refine_convs, reversed_skips):
            h = up_conv(h)
            h = torch.cat([h, skip], dim=1)
            h = refine_conv(h)

        x_hat = self.final_conv(h)
        return x_hat
