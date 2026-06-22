import lightning as L
import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class ResNet18Teacher(L.LightningModule):
    """
    Teacher para el destilado del Modelo B.

    Es un ResNet-18 completo de torchvision, preentrenado en ImageNet,
    al que se le reemplaza la capa final (fc) por un clasificador de
    `num_classes` salidas y se hace fine-tuning sobre las clases sin
    defectos del dataset. Una vez entrenado sirve como teacher para el
    student (Modelo B) mediante destilado de logits (teacher-student).
    """

    # Estadísticas de normalización de ImageNet. Las imágenes vienen en
    # rango [0, 1] (ToTensor), por lo que el teacher preentrenado funciona
    # mejor si las normalizamos internamente igual que en ImageNet.
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        num_classes: int = 10,
        lr: float = 1e-4,
        pretrained: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters()

        weights = (
            torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        backbone = torchvision.models.resnet18(weights=weights)

        # Separamos el extractor de features (todo menos la fc) del clasificador.
        self.feature_dim = backbone.fc.in_features  # 512 en ResNet-18
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.register_buffer(
            "norm_mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "norm_std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor):
        x = (x - self.norm_mean) / self.norm_std

        # backbone.fc es Identity, así que esto retorna el embedding de 512-d.
        z = self.backbone(x)
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

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss(batch)

        self.log("train/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_loss(batch)

        self.log("val/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.compute_loss(batch)

        self.log("test/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
