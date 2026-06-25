import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from sklearn.manifold import TSNE
from torch import nn

from models.resnet_scratch.backbone import ResNetPartialBackbone


class ResNetDistillClassifier(L.LightningModule):
    """
    Modelo B: student destilado mediante la técnica teacher-student.

    El student usa exactamente la misma arquitectura que el Modelo A
    (ResNetPartialBackbone: conv1, conv2_x, conv3_x + clasificador FC),
    por lo que su `forward` retorna (logits, z) con el mismo embedding `z`
    que el Modelo A. Esto lo hace compatible sin cambios con el pipeline
    de detección de anomalías (anomaly_evaluation.py / Mahalanobis).

    El destilado se hace sobre los logits del teacher (ResNet-18 fine-tuned)
    con la formulación clásica de Hinton:

        loss = alpha * KD(soft_teacher, soft_student, T) + (1 - alpha) * CE(student, y)

    donde KD es la divergencia KL entre las distribuciones suavizadas por
    la temperatura T, escalada por T^2.

    El teacher NO se registra como submódulo (se guarda en una lista para que
    PyTorch no lo incluya en el state_dict). Así el checkpoint del student
    queda limpio y se puede cargar para evaluación sin necesitar el teacher.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        lr: float = 1e-3,
        temperature: float = 4.0,
        alpha: float = 0.7,
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

        # El teacher se guarda fuera del registro de módulos de nn.Module
        # (dentro de una lista) para que no entre en el state_dict del student.
        self._teacher_ref = [None]

    # --------------------------------------- Teacher --------------------------------#

    def set_teacher(self, teacher: nn.Module):
        """Asocia un teacher ya entrenado y congelado para el destilado."""
        if teacher is not None:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
        self._teacher_ref[0] = teacher

    @property
    def teacher(self):
        return self._teacher_ref[0]

    def on_fit_start(self):
        # El teacher no es submódulo, así que hay que moverlo manualmente al device.
        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.eval()

    # --------------------------------------- Forward --------------------------------#

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)

        z = self.pool(features)
        z = torch.flatten(z, start_dim=1)

        logits = self.classifier(z)

        return logits, z

    def get_class_labels(self, y: torch.Tensor):
        if y.ndim == 1:
            return y.long()
        return y[:, 0].long()

    # --------------------------------------- Distillation loss ---------------------#

    def distillation_loss(self, student_logits, teacher_logits):
        T = self.hparams.temperature

        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)

        kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean")

        # Escalado por T^2 para mantener la magnitud de los gradientes (Hinton).
        return kd * (T * T)

    # --------------------------------------- Training --------------------------------#

    def training_step(self, batch, batch_idx):
        x, y = batch
        class_labels = self.get_class_labels(y)

        student_logits, z = self.forward(x)
        ce = F.cross_entropy(student_logits, class_labels)

        if self.teacher is not None:
            with torch.no_grad():
                teacher_logits, _ = self.teacher(x)

            kd = self.distillation_loss(student_logits, teacher_logits)
            loss = self.hparams.alpha * kd + (1 - self.hparams.alpha) * ce

            self.log("train/kd", kd, prog_bar=False, on_step=False, on_epoch=True)
        else:
            # Sin teacher el destilado se degrada a entrenamiento supervisado normal.
            loss = ce

        preds = torch.argmax(student_logits, dim=1)
        acc = (preds == class_labels).float().mean()

        self.log("train/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/ce", ce, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    # --------------------------------------- Validation --------------------------------#

    def on_validation_epoch_start(self):
        self.val_z = []
        self.val_y = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        class_labels = self.get_class_labels(y)

        student_logits, z = self.forward(x)
        loss = F.cross_entropy(student_logits, class_labels)

        preds = torch.argmax(student_logits, dim=1)
        acc = (preds == class_labels).float().mean()

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

        if len(self.val_z) > 0 and self.current_epoch % 4 == 0:
            z_all = torch.cat(self.val_z, dim=0)
            y_all = torch.cat(self.val_y, dim=0)
            self._log_tsne(z_all, y_all)

        self.val_z.clear()
        self.val_y.clear()

    # --------------------------------------- Testing --------------------------------#

    def test_step(self, batch, batch_idx):
        x, y = batch
        class_labels = self.get_class_labels(y)

        student_logits, z = self.forward(x)
        loss = F.cross_entropy(student_logits, class_labels)

        preds = torch.argmax(student_logits, dim=1)
        acc = (preds == class_labels).float().mean()

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
