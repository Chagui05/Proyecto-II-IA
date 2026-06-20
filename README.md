# Tarea 03: Guia de ubicacion del proyecto

Este repositorio contiene la implementacion de la Tarea 03 sobre reconstruccion/generacion de imagenes con autoencoders usando el dataset **MVTec AD**. La intencion de este README es servir como mapa para revisar donde esta cada parte del trabajo dentro del proyecto.

La tarea compara dos arquitecturas:

| Requisito | Ubicacion principal |
|---|---|
| Variational Autoencoder, VAE | `models/vae/` |
| Autoencoder tipo U-Net con skip connections | `models/u-net/` |
| Configuracion con Hydra | `conf/` |
| Entrenamiento con PyTorch Lightning | `train.py`, `models/vae/model.py`, `models/u-net/model.py`, `data/datamodule.py` |
| Registro con Weights & Biases | `train.py`, `conf/logger/wandb.yaml`, metodos de logging en `models/*/model.py` |
| Notebook final de entrega | `training.ipynb` |
| Analisis y visualizaciones exportadas | `outputs/analysis_full/` |

## Estructura del repositorio

| Ruta | Que contiene | Relacion con la especificacion |
|---|---|---|
| `train.py` | Script principal de entrenamiento. Carga Hydra, instancia el datamodule, selecciona VAE o U-Net, configura WandB y ejecuta entrenamiento/prueba con Lightning. | Integra PyTorch Lightning, Hydra, modelos y WandB. |
| `conf/` | Carpeta de configuracion Hydra. | Permite seleccionar modelo, perdida, trainer, logger y dataset sin cambiar codigo. |
| `conf/config.yaml` | Configuracion base del proyecto. | Une los grupos `data`, `model`, `trainer` y `logger`. |
| `conf/data/mvtec.yaml` | Ruta del dataset preprocesado, `batch_size` y `num_workers`. | Define como se carga MVTec AD para entrenar. |
| `conf/model/vae.yaml` | Hiperparametros del VAE: `image_size`, `latent_dim`, `hidden_channels`, `lr`, `beta`, `loss_type`. | Configura la arquitectura VAE y su funcion de perdida. |
| `conf/model/u-net.yaml` | Hiperparametros del U-Net: `image_size`, `hidden_channels`, `latent_dim`, `lr`, `loss_type`. | Configura el autoencoder U-Net con skip connections. |
| `conf/trainer/default.yaml` | Parametros de entrenamiento Lightning: epocas, acelerador, dispositivos y logging. | Centraliza la configuracion del entrenamiento. |
| `conf/logger/wandb.yaml` | Proyecto y nombre de corrida en WandB. | Configura el registro de experimentos. |
| `conf/experiments/` | Configuraciones por experimento para VAE/U-Net y perdidas L1, L2, SSIM, SSIM+L1. | Evidencia las 8 combinaciones solicitadas por la tarea. |
| `data/datamodule.py` | `MVTecDataModule`, implementado como `LightningDataModule`. | Carga tensores preprocesados y crea dataloaders de train, validation y test. |
| `data/cache/` | Archivo comprimido `mvtec_anomaly_detection.zip` con el `.pt` preprocesado y nota `LEER.md`. | Contiene el dataset preprocesado usado por el datamodule. |
| `dataset/` | Carpetas de MVTec AD para `cable`, `capsule`, `screw` y `transistor`. | Dataset original organizado por clase y defecto. |
| `models/vae/` | Implementacion modular del VAE. | Arquitectura VAE requerida. |
| `models/u-net/` | Implementacion modular del U-Net autoencoder. | Arquitectura U-Net con skip connections requerida. |
| `notebooks/preprocesamiento_tarea3.ipynb` | Preparacion del dataset y creacion de tensores. | Explica el paso de imagenes MVTec AD a datos preprocesados. |
| `notebooks/VAE.ipynb` | Desarrollo exploratorio del VAE. | Documenta la construccion inicial del modelo VAE. |
| `notebooks/u_net.ipynb` | Desarrollo exploratorio del U-Net. | Documenta la construccion inicial del modelo U-Net. |
| `notebooks/analisis_de_resultados.ipynb` | Notebook de analisis de resultados. | Contiene tablas, visualizaciones y analisis. |
| `training.ipynb` | Notebook principal de entrega. Es el mismo que `notebooks/analisis_de_resultados.ipynb`. | Presenta el flujo completo, resultados y conclusiones. |
| `outputs/analysis_full/` | Resultados exportados: resumen, AUC, curvas PR, t-SNE, reconstrucciones, violines y grids por defecto. | Evidencia visual y cuantitativa de los experimentos. |

## Donde revisar cada requisito

| Requisito de la tarea | Donde revisarlo |
|---|---|
| Uso de MVTec AD con clases `cable`, `capsule`, `screw`, `transistor` | `dataset/`, `notebooks/preprocesamiento_tarea3.ipynb`, `conf/data/mvtec.yaml` |
| Imagenes RGB de `128x128` | `conf/model/vae.yaml`, `conf/model/u-net.yaml`, notebooks de preprocesamiento/modelos |
| VAE | `models/vae/model.py`, `models/vae/encoder.py`, `models/vae/decoder.py` |
| Espacio latente `z` del VAE | `models/vae/model.py`, `models/vae/encoder.py`, `conf/model/vae.yaml` |
| Reparametrizacion del VAE | Metodo `reparameterize` en `models/vae/model.py` |
| Termino KL del VAE | Metodo `kl_loss` en `models/vae/model.py` |
| U-Net autoencoder | `models/u-net/model.py`, `models/u-net/encoder.py`, `models/u-net/decoder.py` |
| Skip connections | `models/u-net/encoder.py` guarda `skips`; `models/u-net/decoder.py` concatena skips con `torch.cat` |
| Funciones de perdida L1, L2, SSIM, SSIM+L1 | Metodo `reconstruction_loss` en `models/vae/model.py` y `models/u-net/model.py` |
| Configuraciones de los 8 experimentos | `conf/experiments/` |
| Entrenamiento con PyTorch Lightning | `train.py`, clases `VAEAutoEncoder`, `UNetAutoEncoder`, `MVTecDataModule` |
| Hydra | `conf/config.yaml` y subcarpetas de `conf/` |
| WandB | `conf/logger/wandb.yaml`, `WandbLogger` en `train.py`, logging de imagenes en `models/*/model.py` |
| Reconstrucciones normales y anomalas | `test_step` y `on_test_epoch_end` en `models/vae/model.py` y `models/u-net/model.py`; imagenes en `outputs/analysis_full/test-good_vs_anomaly_reconstructions/` |
| t-SNE del espacio latente | Metodo `_log_tsne` en `models/vae/model.py` y `models/u-net/model.py`; salidas en `outputs/analysis_full/tsne/` |
| Histogramas y comparaciones de error | `training.ipynb`, `notebooks/analisis_de_resultados.ipynb`, `outputs/analysis_full/todos/` |
| Analisis critico y conclusiones | Secciones finales de `training.ipynb` y `notebooks/analisis_de_resultados.ipynb` |

## Modelos

### VAE

El VAE esta dividido en tres archivos:

| Archivo | Contenido |
|---|---|
| `models/vae/encoder.py` | Encoder convolucional. Reduce la imagen hasta una representacion compacta y calcula `mu` y `logvar`. |
| `models/vae/decoder.py` | Decoder con capas transpuestas. Reconstruye la imagen desde el vector latente. |
| `models/vae/model.py` | `LightningModule` completo. Incluye forward pass, reparametrizacion, KL loss, perdidas de reconstruccion, pasos de train/validation/test y logging. |

### U-Net Autoencoder

El U-Net autoencoder tambien esta separado en tres archivos:

| Archivo | Contenido |
|---|---|
| `models/u-net/encoder.py` | Encoder convolucional. Guarda activaciones intermedias como skip connections. |
| `models/u-net/decoder.py` | Decoder que hace upsampling y concatena skips con las activaciones correspondientes. |
| `models/u-net/model.py` | `LightningModule` completo. Incluye forward pass, perdidas, pasos de train/validation/test, representacion latente para t-SNE y logging. |

## Configuracion Hydra

La carpeta `conf/` organiza el proyecto por grupos:

| Grupo | Archivos | Proposito |
|---|---|---|
| Base | `conf/config.yaml` | Selecciona los grupos por defecto. |
| Data | `conf/data/mvtec.yaml` | Configura la ruta del dataset preprocesado y dataloaders. |
| Model | `conf/model/vae.yaml`, `conf/model/u-net.yaml` | Define arquitectura e hiperparametros de cada modelo. |
| Trainer | `conf/trainer/default.yaml` | Define parametros de entrenamiento de Lightning. |
| Logger | `conf/logger/wandb.yaml` | Define proyecto y nombre de corrida en WandB. |
| Experiments | `conf/experiments/*.yaml` | Define combinaciones modelo/perdida para la comparacion solicitada. |

Las configuraciones de experimentos existentes cubren:

| Modelo | Perdidas configuradas |
|---|---|
| VAE | `vae_l1.yaml`, `vae_l2.yaml`, `vae_ssim.yaml`, `vae_ssim+l1.yaml` |
| U-Net | `unet_l1.yaml`, `unet_l2.yaml`, `unet_ssim.yaml`, `unet_ssim+l1.yaml` |


## Notebooks

| Notebook | Funcion |
|---|---|
| `training.ipynb` | Notebook principal de entrega. Contiene entrenamiento de los experimentos, resultados, visualizaciones y analisis critico. |
| `notebooks/analisis_de_resultados.ipynb` | Notebook de analisis con tablas, visualizaciones, metricas y conclusiones. |
| `notebooks/preprocesamiento_tarea3.ipynb` | Notebook donde se prepara MVTec AD y se generan tensores para entrenamiento. |
| `notebooks/VAE.ipynb` | Notebook de desarrollo del VAE. |
| `notebooks/u_net.ipynb` | Notebook de desarrollo del U-Net autoencoder. |

## Resultados y evidencias

| Carpeta/archivo | Evidencia |
|---|---|
| `outputs/analysis_full/analysis_summary.md` | Resumen de metricas principales por corrida. |
| `outputs/analysis_full/auc_by_run.csv` | AUC por configuracion. |
| `outputs/analysis_full/auc_by_run_class.csv` | AUC separado por clase de objeto. |
| `outputs/analysis_full/pr/` | Curvas Precision-Recall por corrida. |
| `outputs/analysis_full/val_reconstructions/` | Reconstrucciones de validacion exportadas. |
| `outputs/analysis_full/test-good_vs_anomaly_reconstructions/` | Comparaciones entre imagenes normales y anomalas. |
| `outputs/analysis_full/tsne/` | Visualizaciones t-SNE del espacio latente. |
| `outputs/analysis_full/violins/` | Graficos de violin para distribuciones de error. |
| `outputs/analysis_full/todos/` | Grids por modelo, perdida, clase y tipo de defecto. |

