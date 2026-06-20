import importlib
import hydra
import lightning as L

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from lightning.pytorch.loggers import WandbLogger

from models.vae.model import VAEAutoEncoder
from data.datamodule import MVTecDataModule


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

    if cfg.model.name == "vae":
        model = VAEAutoEncoder(
            in_channels=cfg.model.in_channels,
            image_size=cfg.model.image_size,
            latent_dim=cfg.model.latent_dim,
            hidden_channels=list(cfg.model.hidden_channels),
            lr=cfg.model.lr,
            beta=cfg.model.beta,
            loss_type=cfg.model.loss_type,
            use_sigmoid=cfg.model.use_sigmoid,
        )
    elif cfg.model.name == "u-net":
        # importlib porque Python no permite importar módulos con guión via dot-notation
        unet_module = importlib.import_module("models.u-net.model")
        UNetAutoEncoder = unet_module.UNetAutoEncoder
        model = UNetAutoEncoder(
            in_channels=cfg.model.in_channels,
            image_size=cfg.model.image_size,
            hidden_channels=list(cfg.model.hidden_channels),
            lr=cfg.model.lr,
            loss_type=cfg.model.loss_type,
            use_sigmoid=cfg.model.use_sigmoid,
        )
    else:
        raise ValueError(f"Modelo no soportado: {cfg.model.name}")

    logger = WandbLogger(
        project=cfg.logger.project,
        name=cfg.logger.name,
        log_model=cfg.logger.log_model,
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
    )

    # Esta se encarga de entrenar
    trainer.fit(model, datamodule=datamodule)
    # Aquí se loggean reconstrucciones good vs anomalías.
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
