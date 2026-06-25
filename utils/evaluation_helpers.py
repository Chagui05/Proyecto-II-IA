import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

"""Con los siguientes helpers podemos extraer fácilmente las etiquetas."""


def get_class_labels(y):
    if y.ndim == 1:
        return y.long()
    return y[:, 0].long()


def get_anomaly_labels(y):
    """
    Retorna:
    0 = good
    1 = anomalía
    """
    if y.ndim == 1:
        return torch.zeros_like(y).long()

    defect_code = y[:, 1]
    return (defect_code != 0).long()


"""A continuación con la siguiente función se extraen los embeddings de los modelos, cada uno de ellos determina retornarlos de la misma forma. Por lo que esta función funciona igual para cada uno"""


def extract_embeddings(model, dataloader, device="cuda"):
    model.eval()
    model.to(device)

    all_z = []
    all_y = []
    all_reconstruction_error = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            output, z = model(x)

            all_z.append(z.detach().cpu())
            all_y.append(y.detach().cpu())

            # Solo aplica para autoencoders, donde output tiene forma de imagen.
            if output.shape == x.shape:
                per_image_error = F.l1_loss(output, x, reduction="none")
                per_image_error = per_image_error.mean(dim=[1, 2, 3])
                all_reconstruction_error.append(per_image_error.detach().cpu())

    z_all = torch.cat(all_z, dim=0)
    y_all = torch.cat(all_y, dim=0)

    if len(all_reconstruction_error) > 0:
        reconstruction_error = torch.cat(all_reconstruction_error, dim=0)
    else:
        reconstruction_error = None

    return z_all, y_all, reconstruction_error


"""Ahora a continuación las funciones para el mahalanobis"""


def fit_mahalanobis_by_class(z_val, y_val, percentile=95):
    """
    Ajusta una distribución normal por clase usando embeddings de validación.
    Retorna media, precisión y threshold por clase.
    """
    class_labels = get_class_labels(y_val)

    z_np = z_val.numpy()
    class_np = class_labels.numpy()

    models = {}

    for class_id in np.unique(class_np):
        class_mask = class_np == class_id
        z_class = z_np[class_mask]

        covariance_model = LedoitWolf()
        covariance_model.fit(z_class)

        mean = covariance_model.location_
        precision = covariance_model.precision_

        distances = mahalanobis_distance(z_class, mean, precision)
        threshold = np.percentile(distances, percentile)

        models[int(class_id)] = {
            "mean": mean,
            "precision": precision,
            "threshold": threshold,
            "val_distances": distances,
        }

    return models


def mahalanobis_distance(z, mean, precision):
    diff = z - mean
    distances = np.sqrt(np.sum((diff @ precision) * diff, axis=1))
    return distances


"""Ahora una vez generado el mahalanobis con base en su mean, precision y el umbral definido se puede predecir si una imágen consiste de una anomalía o no."""


def predict_anomalies_mahalanobis(z_test, y_test, mahalanobis_models):
    z_np = z_test.numpy()
    class_labels = get_class_labels(y_test).numpy()

    scores = []
    predictions = []

    for z_i, class_id in zip(z_np, class_labels):
        class_id = int(class_id)

        model = mahalanobis_models[class_id]
        mean = model["mean"]
        precision = model["precision"]
        threshold = model["threshold"]

        distance = mahalanobis_distance(
            z_i.reshape(1, -1),
            mean,
            precision,
        )[0]

        pred = int(distance > threshold)

        scores.append(distance)
        predictions.append(pred)

    return np.array(scores), np.array(predictions)


def evaluate_anomaly_predictions(y_test, y_pred, scores=None, model_name="model"):
    y_true = get_anomaly_labels(y_test).numpy()

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred))

    print("\nReporte:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["good", "anomaly"],
            zero_division=0,
        )
    )

    return results


def evaluate_model_with_mahalanobis(
    model,
    val_dataloader,
    test_dataloader,
    device="cuda",
    percentile=95,
    model_name="model",
):
    z_val, y_val, _ = extract_embeddings(
        model,
        val_dataloader,
        device=device,
    )

    z_test, y_test, reconstruction_error = extract_embeddings(
        model,
        test_dataloader,
        device=device,
    )

    mahalanobis_models = fit_mahalanobis_by_class(
        z_val=z_val,
        y_val=y_val,
        percentile=percentile,
    )

    scores, y_pred = predict_anomalies_mahalanobis(
        z_test=z_test,
        y_test=y_test,
        mahalanobis_models=mahalanobis_models,
    )

    results = evaluate_anomaly_predictions(
        y_test=y_test,
        y_pred=y_pred,
        scores=scores,
        model_name=model_name,
    )

    return {
        "results": results,
        "scores": scores,
        "y_pred": y_pred,
        "y_test": y_test,
        "z_test": z_test,
        "reconstruction_error": reconstruction_error,
        "mahalanobis_models": mahalanobis_models,
    }


"""A continuación las dos siguientes funciones permiten graficar tanto la distribución de scores obtenidos como la matríz de confusión. Todo con base en el dataset de testing."""


def plot_score_distribution(scores, y_test, title="Distribución de scores"):
    y_true = get_anomaly_labels(y_test).numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(scores[y_true == 0], bins=40, alpha=0.6, label="good")
    plt.hist(scores[y_true == 1], bins=40, alpha=0.6, label="anomaly")
    plt.title(title)
    plt.xlabel("Score de anomalía")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_test, y_pred, title="Matriz de confusión"):
    y_true = get_anomaly_labels(y_test).numpy()
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xticks([0, 1], ["good", "anomaly"])
    plt.yticks([0, 1], ["good", "anomaly"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()
