import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import wandb
from sklearn.manifold import TSNE
from torch import nn

from .backbone import ResNetPartialBackbone


class ResNetScratchClassifier(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.backbone = ResNetPartialBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(
            self.backbone.out_channels,
            num_classes,
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)

        # Global average pooling para obtener un embedding compacto.
        z = self.pool(features)
        z = torch.flatten(z, start_dim=1)

        logits = self.classifier(z)

        return logits, z

    def get_class_labels(self, y: torch.Tensor):
        if y.ndim == 1:
            return y.long()
        return y[:, 0].long()

    def compute_loss(self, batch):
        x, y = batch
        class_labels = self.get_class_labels(y)

        logits, z = self.forward(x)
        loss = F.cross_entropy(logits, class_labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == class_labels).float().mean()

        return loss, acc, logits, z

    def training_step(self, batch, batch_idx):
        loss, acc, logits, z = self.compute_loss(batch)

        self.log("train/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    # --------------------------------------- Validation --------------------------------#

    def on_validation_epoch_start(self):
        self.val_z = []
        self.val_y = []

    def validation_step(self, batch, batch_idx):
        loss, acc, logits, z = self.compute_loss(batch)
        x, y = batch

        self.log("val/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        self.val_z.append(z.detach().cpu())
        self.val_y.append(y.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_z.clear()
            self.val_y.clear()
            return

        # Solo loggear cada 4 épocas el t-SNE para no volver lento el entrenamiento.
        if len(self.val_z) > 0 and self.current_epoch % 4 == 0:
            z_all = torch.cat(self.val_z, dim=0)
            y_all = torch.cat(self.val_y, dim=0)
            self._log_tsne(z_all, y_all)

        self.val_z.clear()
        self.val_y.clear()

    # --------------------------------------- Testing --------------------------------#

    def test_step(self, batch, batch_idx):
        loss, acc, logits, z = self.compute_loss(batch)

        self.log("test/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # --------------------------------------- Logging --------------------------------#

    def _log_tsne(self, z: torch.Tensor, labels: torch.Tensor):
        if self.logger is None:
            return

        if labels.ndim == 2:
            labels = labels[:, 0]

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
