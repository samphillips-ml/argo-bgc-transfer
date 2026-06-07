# argo-bgc-transfer

Encoder ablation study on BGC-Argo float data. Trains autoencoders on depth profiles of Temperature, Salinity, and Oxygen, then evaluates how well the frozen latent representation transfers to predicting Chlorophyll — a variable the encoder never saw.

Central question: what encoder inductive bias produces a latent space that best supports cross-variable transfer?

**Current results (val MSE, Chlorophyll probe):**

| Encoder | Val MSE |
|---|---|
| Depth-only baseline | 0.7593 |
| Raw T/S/O (no encoder) | 0.3353 |
| MLP encoder | 0.3273 |
| CNN encoder | 0.2695 |

Part of ongoing undergraduate research at UNC Charlotte under Dr. Xuyang Li.