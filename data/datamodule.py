from pathlib import Path
from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


class MVTecDataModule(L.LightningDataModule):
    def __init__(
        self,
        checkpoint_path: str,
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        super().__init__()

        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.class_to_idx = None
        self.defect_mappings = None

    def prepare_data(self):
        """
        primero validamos que el .pt con los datos preprocesados exista
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {self.checkpoint_path}")

    def setup(self, stage: Optional[str] = None):
        """
        Se encarga de construir los datasets
        """
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        self.class_to_idx = checkpoint["class_to_idx"]
        self.defect_mappings = checkpoint["defect_mappings"]

        X_train = checkpoint["X_train"].float()
        y_train = checkpoint["y_train"].long()  # recordar, esto es metadata

        X_val = checkpoint["X_val"].float()
        y_val = checkpoint["y_val"].long()  # recordar, esto es metadata

        X_test = checkpoint["X_test"].float()
        y_test = checkpoint["y_test"].long()  # recordar, esto es metadata

        # Esto elije dependiendo de en que momento se esté si en training o inferencia
        if stage == "fit" or stage is None:
            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_val, y_val)

        if stage == "test" or stage == "predict" or stage is None:
            self.test_dataset = TensorDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Para acelerar la transferencia entre cpu y gpu
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
