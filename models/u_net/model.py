import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import wandb
from sklearn.manifold import TSNE
from torch import nn
from torchmetrics.functional import structural_similarity_index_measure as ssim

from .decoder import UNetDecoder
from .encoder import UNetEncoder


class UNetAutoEncoder(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 128,
        hidden_channels: list[int] | None = None,
        lr: float = 1e-3,
        loss_type: str = "l1",
        use_sigmoid: bool = True,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        self.save_hyperparameters()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.encoder = UNetEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
        )

        self.decoder = UNetDecoder(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            use_sigmoid=use_sigmoid,
        )

    def forward(self, x: torch.Tensor):
        bottleneck, skips = self.encoder(x)
        x_hat = self.decoder(bottleneck, skips)

        # El global average pooling para extraer el z
        z = self.pool(bottleneck)
        z = torch.flatten(z, start_dim=1)
        return x_hat, z

    def reconstruction_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        loss_type = self.hparams.loss_type.lower()

        if loss_type == "l1":
            return F.l1_loss(x_hat, x)

        if loss_type in ["l2", "mse"]:
            return F.mse_loss(x_hat, x)

        if loss_type == "ssim":
            return 1.0 - ssim(x_hat, x, data_range=1.0)

        if loss_type == "ssim+l1":
            ssim_loss = 1.0 - ssim(x_hat, x, data_range=1.0)
            l1_loss = F.l1_loss(x_hat, x)
            return ssim_loss + l1_loss

        raise ValueError(f"Loss no soportada: {self.hparams.loss_type}")

    def compute_loss(self, batch):
        x, _ = batch
        x_hat, z = self.forward(x)
        loss = self.reconstruction_loss(x_hat, x)
        return loss, x_hat, z

    def training_step(self, batch, batch_idx):
        loss, x_hat, z = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    # --------------------------------------- Validation --------------------------------#

    def on_validation_epoch_start(self):
        self.val_z = []
        self.val_y = []
        self.val_x = []
        self.val_x_hat = []

    def validation_step(self, batch, batch_idx):
        loss, x_hat, z = self.compute_loss(batch)
        x, y = batch

        self.log("val/loss", loss, prog_bar=False, on_step=False, on_epoch=True)

        if len(self.val_x) < 2:
            self.val_x.append(x[:8].detach().cpu())
            self.val_x_hat.append(x_hat[:8].detach().cpu())

        self.val_z.append(z.detach().cpu())
        self.val_y.append(y.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_z.clear()
            self.val_y.clear()
            self.val_x.clear()
            self.val_x_hat.clear()
            return

        if len(self.val_x) > 0 and len(self.val_x_hat) > 0:
            x = torch.cat(self.val_x, dim=0)
            x_hat = torch.cat(self.val_x_hat, dim=0)
            comparison = torch.cat([x, x_hat], dim=0)
            self._log_image_grid(
                key="val/reconstructions",
                images=comparison,
                caption="Fila 1: originales | Fila 2: reconstrucciones",
            )

        # Solo loggear cada 4 epocas el t-SNE
        if len(self.val_z) > 0 and self.current_epoch % 4 == 0:
            z_all = torch.cat(self.val_z, dim=0)
            y_all = torch.cat(self.val_y, dim=0)
            self._log_tsne(z_all, y_all)

        self.val_z.clear()
        self.val_y.clear()
        self.val_x.clear()
        self.val_x_hat.clear()

    # --------------------------------------- Testing --------------------------------#

    def on_test_epoch_start(self):
        self.test_good_x = []
        self.test_good_x_hat = []
        self.test_anom_x = []
        self.test_anom_x_hat = []

    def test_step(self, batch, batch_idx):
        loss, x_hat, z = self.compute_loss(batch)
        x, y = batch
        self.log("test/loss", loss, prog_bar=False, on_step=False, on_epoch=True)

        defect_code = y[:, 1]
        good_mask = defect_code == 0
        anom_mask = defect_code != 0

        if good_mask.any() and len(self.test_good_x) < 2:
            self.test_good_x.append(x[good_mask][:4].detach().cpu())
            self.test_good_x_hat.append(x_hat[good_mask][:4].detach().cpu())

        if anom_mask.any() and len(self.test_anom_x) < 2:
            self.test_anom_x.append(x[anom_mask][:4].detach().cpu())
            self.test_anom_x_hat.append(x_hat[anom_mask][:4].detach().cpu())

        return loss

    def on_test_epoch_end(self):
        if self.logger is None:
            return

        if len(self.test_good_x) > 0 and len(self.test_anom_x) > 0:
            good_x = torch.cat(self.test_good_x, dim=0)[:8]
            good_x_hat = torch.cat(self.test_good_x_hat, dim=0)[:8]
            anom_x = torch.cat(self.test_anom_x, dim=0)[:8]
            anom_x_hat = torch.cat(self.test_anom_x_hat, dim=0)[:8]

            comparison = torch.cat([good_x, good_x_hat, anom_x, anom_x_hat], dim=0)

            self._log_image_grid(
                key="test/good_vs_anomaly_reconstructions",
                images=comparison,
                caption=(
                    "Fila 1: good originales | "
                    "Fila 2: good reconstruidas | "
                    "Fila 3: anomalías originales | "
                    "Fila 4: anomalías reconstruidas"
                ),
                nrow=8,
            )

        self.test_good_x.clear()
        self.test_good_x_hat.clear()
        self.test_anom_x.clear()
        self.test_anom_x_hat.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    ## ------------------ Funciones Auxiliares para logging -------------------------------- ##

    def _log_image_grid(
        self, key: str, images: torch.Tensor, caption: str = "", nrow: int = 16
    ):
        if self.logger is None:
            return

        grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=False)
        grid_np = grid.detach().cpu().permute(1, 2, 0).numpy()

        self.logger.experiment.log(
            {
                key: wandb.Image(grid_np, caption=caption),
                "epoch": self.current_epoch,
            }
        )

    def _log_tsne(self, z: torch.Tensor, labels: torch.Tensor):
        if self.logger is None:
            return

        z_np = z.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        if z_np.shape[0] < 3:
            return

        perplexity = min(30, max(2, z_np.shape[0] // 3))

        z_2d = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            init="pca",
            learning_rate="auto",
        ).fit_transform(z_np)

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_np, s=12, alpha=0.8)
        ax.set_title(f"t-SNE del espacio latente - epoch {self.current_epoch}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.colorbar(scatter, ax=ax, label="Clase")
        fig.tight_layout()

        self.logger.experiment.log(
            {
                "val/latent_tsne": wandb.Image(fig),
                "epoch": self.current_epoch,
            }
        )

        plt.close(fig)
