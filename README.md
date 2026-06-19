# SatForensics — Cross-Dataset Evaluation

Cross-dataset evaluation of a frozen DINOv2 + linear probe for satellite image
manipulation detection. Part of an MSc dissertation at Newcastle University
(Project 22), supervised by Dr. Deepayan Bhowmik.

## What this repo contains

- `SatForensics_CrossDataset.ipynb` — The evaluation notebook covering:
  - RSFAKE-1M download and filtering to the official test split
  - DINOv2 feature extraction on subsampled fakes
  - Probe application and per-image score aggregation
  - Stratified heatmap visualisation
  - Confound diagnostics (brightness, std, edge density)

## Where the model lives

The trained linear probe, feature scaler, and cached features are hosted on
HuggingFace:

https://huggingface.co/OmPatil9819/satforensics-dinov2-probe

## Headline result

Pooled test patch AUROC on the Airbus validation set: **0.9593**
(See the HF model card for cross-dataset numbers.)

## Running the notebook

The notebook expects to run in Google Colab with:
- A GPU (A100 recommended; T4 also works for 2000-image subsets)
- Google Drive mounted for persistent storage
- A HuggingFace token in Colab Secrets (named `HF_TOKEN`) for dataset access

The first cell (Bootstrap) handles environment setup. After that, cells run
sequentially.

## Datasets used

- **Training**: Airbus satellite imagery + CLIP-guided SD2 + ControlNet inpainting
  (from Chapman et al. 2025, VCIP)
- **Cross-dataset evaluation**: RSFAKE-1M (Tan et al. 2025, arXiv:2505.23283)
  hosted at `huggingface.co/datasets/TZHSW/RSFAKE`

## Citation context

If using this code or the model, please cite the upstream:
- **DINOv2**: Oquab et al. (2023), arXiv:2304.07193
- **RSFAKE-1M**: Tan et al. (2025), arXiv:2505.23283
- **Chapman et al. (2025)**, VCIP — provided the training data pipeline

## Author

Om Ravindra Patil — MSc Data Science, Newcastle University, 2026.
