# Lab 4 — Conditional VAE for Video Prediction

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python Trainer.py \
    --DR {YOUR_DATASET_PATH} \
    --save_root {PATH_TO_SAVE_YOUR_CHECKPOINT} \
    --fast_train
```

`--fast_train` uses fewer dataset samples and a larger learning rate to speed up training.

## Testing

```bash
python Tester.py \
    --DR {YOUR_DATASET_PATH} \
    --save_root {PATH_TO_SAVE_YOUR_CHECKPOINT} \
    --ckpt_path {PATH_TO_YOUR_CHECKPOINT}
```

## Note: Blackwell GPUs (RTX 50-series, sm_120)

The `requirements.txt` pins `torch==2.6.0` / `torchvision==0.21.0`, whose prebuilt kernels only cover up to sm_90. On a Blackwell card (e.g. RTX 5060 Ti) you will see:

```
NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

Upgrade to a PyTorch build with CUDA 12.8 support:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```
