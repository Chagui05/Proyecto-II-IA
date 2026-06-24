import shutil
from pathlib import Path

import hydra
import lightning as L
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from data.datamodule import MVTecDataModule
from models.resnet_scratch.model import ResNetScratchClassifier
from models.u_net.model import UNetAutoEncoder


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Con esto garantizamos reproducibilidad
    L.seed_everything(cfg.seed, workers=True)

    datamodule = MVTecDataModule(
        checkpoint_path=to_absolute_path(cfg.data.checkpoint_path),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    if cfg.model.name == "u-net":
        model = UNetAutoEncoder(
            in_channels=cfg.model.in_channels,
            image_size=cfg.model.image_size,
            hidden_channels=list(cfg.model.hidden_channels),
            lr=cfg.model.lr,
            loss_type=cfg.model.loss_type,
            use_sigmoid=cfg.model.use_sigmoid,
        )

    elif cfg.model.name == "resnet_scratch":
        model = ResNetScratchClassifier(
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
            base_channels=cfg.model.base_channels,
            lr=cfg.model.lr,
        )

    else:
        raise ValueError(f"Modelo no soportado: {cfg.model.name}")

    logger = WandbLogger(
        project=cfg.logger.project,
        name=cfg.logger.name,
        log_model=cfg.logger.log_model,
    )

    early_stopping = EarlyStopping(
        monitor=cfg.trainer.early_stopping.monitor,
        mode=cfg.trainer.early_stopping.mode,
        patience=cfg.trainer.early_stopping.patience,
        min_delta=cfg.trainer.early_stopping.min_delta,
        verbose=True,
    )

    checkpoint_dir = Path(to_absolute_path(cfg.trainer.checkpoint.dirpath))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=cfg.trainer.checkpoint.monitor,
        mode=cfg.trainer.checkpoint.mode,
        save_top_k=cfg.trainer.checkpoint.save_top_k,
        filename=cfg.trainer.checkpoint.filename,
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=cfg.trainer.deterministic,
        benchmark=cfg.trainer.benchmark,
        logger=logger,
        callbacks=[
            early_stopping,
            checkpoint_callback,
        ],
    )

    trainer.fit(model, datamodule=datamodule)

    best_ckpt_path = checkpoint_callback.best_model_path

    export_path = Path(to_absolute_path(cfg.trainer.checkpoint.export_path))
    export_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(best_ckpt_path, export_path)

    print(f"Mejor checkpoint original: {best_ckpt_path}")
    print(f"Mejor checkpoint exportado en: {export_path}")

    trainer.test(model, datamodule=datamodule, ckpt_path=str(export_path))


if __name__ == "__main__":
    main()
