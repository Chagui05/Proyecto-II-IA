# Analysis summary

Top ROC AUC by run:

| run                 |      auc |       ap |
|:--------------------|---------:|---------:|
| vae_ssim_gpu20      | 0.8259   | 0.888366 |
| vae_ssim+l1_gpu20   | 0.792018 | 0.875083 |
| u-net_ssim_gpu20    | 0.684234 | 0.77183  |
| u-net_l1_gpu20      | 0.608516 | 0.788095 |
| vae_l1_gpu20        | 0.498382 | 0.693225 |
| u-net_l2_gpu20      | 0.489286 | 0.658156 |
| vae_l2_gpu20        | 0.372024 | 0.653078 |
| u-net_ssim+l1_gpu20 | 0.3462   | 0.611383 |

Per-run/per-class AUC saved in `auc_by_run_class.csv`.
