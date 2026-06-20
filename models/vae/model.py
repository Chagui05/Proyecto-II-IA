import torch

from torchmetrics.functional import structural_similarity_index_measure as ssim # Preguntar al profe si se puede usar
import torch.nn.functional as F

import wandb
import torchvision
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from models.vae.encoder import VAEEncoder
from models.vae.decoder import VAEDecoder


import lightning as L

class VAEAutoEncoder(L.LightningModule):
    def __init__(
        self,
        in_channels: int           = 3,
        image_size: int            = 128,
        latent_dim: int            = 128,
        hidden_channels: list[int] | None = None,
        lr: float                  = 1e-3,
        beta: float                = 1e-4,
        loss_type: str             = "l1",
        use_sigmoid: bool          = True,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        # Así podemos accederlos más fácil luego
        self.save_hyperparameters()

        self.encoder = VAEEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            image_size=image_size,
        )

        self.decoder = VAEDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            image_size=image_size,
            use_sigmoid=use_sigmoid,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Aplica el reparameterization, para obtener un sample del espacio latente:
        z = mu + std * eps
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps # El eps es quien añade lo random

        return z

    def forward(self, x: torch.Tensor):
        """
        Hace una pasada completa por el Autoencoder.
        """
        # Encoder
        mu, logvar = self.encoder(x)
        # Cuello de botella
        z = self.reparameterize(mu, logvar)
        # Decoder
        x_hat = self.decoder(z)

        return x_hat, mu, logvar, z

    def reconstruction_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Elige el loss dependiendo de la config elegida.
        """
        loss_type = self.hparams.loss_type.lower()

        if loss_type == "l1":
            return F.l1_loss(x_hat, x)

        if loss_type in ["l2", "mse"]:
            return F.mse_loss(x_hat, x)

        if loss_type == "ssim":
            # SSIM se maximiza, por eso usamos 1 - SSIM como pérdida.
            return 1.0 - ssim(x_hat, x, data_range=1.0)

        if loss_type == "ssim+l1":
            ssim_loss = 1.0 - ssim(x_hat, x, data_range=1.0)
            l1_loss = F.l1_loss(x_hat, x)
            return ssim_loss + l1_loss

        raise ValueError(f"Loss no soportada: {self.hparams.loss_type}")

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL(q(z|x) || p(z))
        Esta fórmula asume:
        q(z|x) = N(mu, sigma^2)
        p(z) = N(0, I)
        """
        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1,
        )

        return kl.mean()

    def compute_loss(self, batch):
        """
        Calcula loss total, reconstruction loss y KL loss.
        Esta es la esencia de la VAE
        """
        x, _ = batch

        x_hat, mu, logvar, z = self.forward(x)

        # Aplicamos las loss especiales de un VAE
        recon = self.reconstruction_loss(x_hat, x)
        kl = self.kl_loss(mu, logvar)
        loss = recon + self.hparams.beta * kl

        return loss, recon, kl, x_hat, z

    def training_step(self, batch, batch_idx):

        loss, recon, kl, x_hat, z = self.compute_loss(batch)

        self.log("train/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/recon_loss", recon, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/kl_loss", kl, prog_bar=False, on_step=False, on_epoch=True)

        return loss

#--------------------------------------- Validation --------------------------------#


    def on_validation_epoch_start(self):
        self.val_z = []
        self.val_y = []
        self.val_x = []
        self.val_x_hat = []

    def validation_step(self, batch, batch_idx):

        loss, recon, kl, x_hat, z = self.compute_loss(batch)

        x, y = batch

        self.log("val/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/recon_loss", recon, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/kl_loss", kl, prog_bar=False, on_step=False, on_epoch=True)

        # Guardamos imagens variadas
        if len(self.val_x) < 2:
            self.val_x.append(x[:8].detach().cpu())
            self.val_x_hat.append(x_hat[:8].detach().cpu())

        # Para t-SNE usamos el z
        self.val_z.append(z.detach().cpu())

        # guardamos el label
        self.val_y.append(y.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        # Esto es para que no loggee cuando Lightning hace el
        # Sanity Check al inicio
        if self.trainer.sanity_checking:
            self.val_z.clear()
            self.val_y.clear()
            self.val_x.clear()
            self.val_x_hat.clear()
            return

        # Primero mostramos las reconstrucciones de imágenes
        if len(self.val_x) > 0 and len(self.val_x_hat) > 0:
            x = torch.cat(self.val_x, dim=0)
            x_hat = torch.cat(self.val_x_hat, dim=0)

            comparison = torch.cat([x, x_hat], dim=0)

            self._log_image_grid(
                key="val/reconstructions",
                images=comparison,
                caption="Fila 1: originales | Fila 2: reconstrucciones",
            )

        # t-SNE del espacio latente de validación
        if len(self.val_z) > 0:
            z_all = torch.cat(self.val_z, dim=0)
            y_all = torch.cat(self.val_y, dim=0)

            self._log_tsne(z_all, y_all)

        # Limpiamos todo para finalizar la función
        self.val_z.clear()
        self.val_y.clear()
        self.val_x.clear()
        self.val_x_hat.clear()


#--------------------------------------- Testing --------------------------------#
    def on_test_epoch_start(self):
        self.test_good_x = []
        self.test_good_x_hat = []
        self.test_anom_x = []
        self.test_anom_x_hat = []

    def test_step(self, batch, batch_idx):
        loss, recon, kl, x_hat, z = self.compute_loss(batch)
        x, y = batch
        self.log("test/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test/recon_loss", recon, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test/kl_loss", kl, prog_bar=False, on_step=False, on_epoch=True)

        # En test, y tiene forma [B, 2]:
        # y[:, 0] = clase del objeto
        # y[:, 1] = tipo de defecto
        defect_code = y[:, 1]

        good_mask = defect_code == 0
        anom_mask = defect_code != 0

        # Guardamos 8 good
        # Elegimos cuando < 2, para así elegir de distintos batches
        # y tener distintas clases
        if good_mask.any() and len(self.test_good_x) < 2:
            self.test_good_x.append(x[good_mask][:4].detach().cpu())
            self.test_good_x_hat.append(x_hat[good_mask][:4].detach().cpu())

        # Guardamos 8 anomalías
        if anom_mask.any() and len(self.test_anom_x) < 2:
            self.test_anom_x.append(x[anom_mask][:4].detach().cpu())
            self.test_anom_x_hat.append(x_hat[anom_mask][:4].detach().cpu())

        return loss

    def on_test_epoch_end(self):
        if self.logger is None:
            return

        # Sacamos para poder tener las reconstrucciones de testing
        if len(self.test_good_x) > 0 and len(self.test_anom_x) > 0:
            good_x = torch.cat(self.test_good_x, dim=0)
            good_x_hat = torch.cat(self.test_good_x_hat, dim=0)

            anom_x = torch.cat(self.test_anom_x, dim=0)
            anom_x_hat = torch.cat(self.test_anom_x_hat, dim=0)

            # por si algún batch tenía
            good_x = good_x[:8]
            good_x_hat = good_x_hat[:8]
            anom_x = anom_x[:8]
            anom_x_hat = anom_x_hat[:8]

            comparison = torch.cat(
                [
                    good_x,
                    good_x_hat,
                    anom_x,
                    anom_x_hat,
                ],
                dim=0,
            )

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

    # En este caso utilizaremos Adam como optimizador
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        return optimizer

    ## ------------------ Funciones Auxiliares para logging -------------------------------- ##
    def _log_image_grid(
        self,
        key: str,
        images: torch.Tensor,
        caption: str = "",
        nrow: int = 16,
    ):
        """
        Loggea una cuadricula de imágenes en WandB
        imagenes debe venir en formato [N, C, H, W]
        """
        if self.logger is None:
            return

        grid = torchvision.utils.make_grid(
            images,
            nrow=nrow,
            normalize=False,
        )

        grid_np = grid.detach().cpu().permute(1, 2, 0).numpy()

        self.logger.experiment.log({
            key: wandb.Image(grid_np, caption=caption),
            "epoch": self.current_epoch,
        })


    def _log_tsne(self, z: torch.Tensor, labels: torch.Tensor):
        """
        Aplica t-SNE al espacio latente y lo loggea como figura en wandnb
        """
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

        scatter = ax.scatter(
            z_2d[:, 0],
            z_2d[:, 1],
            c=labels_np,
            s=12,
            alpha=0.8,
        )

        ax.set_title(f"t-SNE del espacio latente - epoch {self.current_epoch}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        fig.colorbar(scatter, ax=ax, label="Clase")
        fig.tight_layout()

        self.logger.experiment.log({
            "val/latent_tsne": wandb.Image(fig),
            "epoch": self.current_epoch,
        })

        plt.close(fig)
