from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from data.datamodule import MVTecDataModule
from models.resnet_scratch.model import ResNetScratchClassifier
from models.u_net.model import UNetAutoEncoder
from utils.evaluation_helpers import (
    evaluate_model_with_mahalanobis,
    get_anomaly_labels,
    get_class_labels,
)


def load_model_from_cfg(cfg: DictConfig):
    checkpoint_path = to_absolute_path(cfg.evaluation.checkpoint_path)

    if cfg.model.name == "u_net":
        return UNetAutoEncoder.load_from_checkpoint(checkpoint_path)

    if cfg.model.name == "resnet_scratch":
        return ResNetScratchClassifier.load_from_checkpoint(checkpoint_path)

    raise ValueError(f"Modelo no soportado: {cfg.model.name}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed, workers=True)

    datamodule = MVTecDataModule(
        checkpoint_path=to_absolute_path(cfg.data.checkpoint_path),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    datamodule.prepare_data()
    datamodule.setup(stage=None)

    model = load_model_from_cfg(cfg)

    evaluation = evaluate_model_with_mahalanobis(
        model=model,
        val_dataloader=datamodule.val_dataloader(),
        test_dataloader=datamodule.test_dataloader(),
        device=cfg.evaluation.device,
        percentile=cfg.evaluation.percentile,
        model_name=cfg.logger.name,
    )

    output_dir = Path(to_absolute_path(cfg.evaluation.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([evaluation["results"]])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    y_test = evaluation["y_test"]
    y_true = get_anomaly_labels(y_test).numpy()
    y_class = get_class_labels(y_test).numpy()

    if y_test.ndim == 2:
        y_defect = y_test[:, 1].numpy()
    else:
        y_defect = np.zeros_like(y_true)

    predictions_df = pd.DataFrame(
        {
            "class_id": y_class,
            "defect_code": y_defect,
            "y_true": y_true,
            "y_pred": evaluation["y_pred"],
            "mahalanobis_score": evaluation["scores"],
        }
    )

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    arrays_to_save = {
        "scores": evaluation["scores"],
        "y_pred": evaluation["y_pred"],
        "y_test": evaluation["y_test"].numpy(),
        "z_test": evaluation["z_test"].numpy(),
    }

    if evaluation["reconstruction_error"] is not None:
        arrays_to_save["reconstruction_error"] = evaluation[
            "reconstruction_error"
        ].numpy()

    np.savez(
        output_dir / "evaluation_outputs.npz",
        **arrays_to_save,
    )

    print("\nMétricas finales:")
    print(metrics_df)

    print(f"\nResultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()
