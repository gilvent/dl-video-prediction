# Lab 4 — Conditional VAE for Video Prediction

Pose-guided video prediction using CVAE

## Setup

```bash
pip install -r requirements.txt
```

There is PyTorch and CUDA incompatiblity issue when using RTX 50-series (sm_120), see the notes at the bottom.

## Training

### Cyclical KL

```bash
python Trainer.py \
    --DR ./dataset \
    --save_root ./runs/cyclical-kl \
    --batch_size 8 \
    --train_vi_len 16 \
    --val_vi_len 630 \
    --num_epoch 70 \
    --num_workers 4 \
    --per_save 2 \
    --lr 1e-3 \
    --lr_min 1e-5 \
    --scheduler cosine \
    --kl_anneal_type Cyclical \
    --kl_anneal_cycle 4 \
    --kl_anneal_ratio 0.4 \
    --kl_max 1.0 \
    --tfr 1.0 \
    --tfr_sde 10 \
    --tfr_d_step 0.1 \
    --tfr_d_period 6
```

### Monotonic KL

```bash
python Trainer.py \
    --DR ./dataset \
    --save_root ./runs/monotonic-kl \
    --batch_size 8 \
    --train_vi_len 16 \
    --val_vi_len 630 \
    --num_epoch 70 \
    --num_workers 4 \
    --per_save 2 \
    --lr 1e-3 \
    --lr_min 1e-5 \
    --scheduler cosine \
    --kl_anneal_type Monotonic \
    --kl_anneal_cycle 20 \
    --kl_max 0.1 \
    --tfr 1.0 \
    --tfr_sde 10 \
    --tfr_d_step 0.1 \
    --tfr_d_period 6
```

### Constant β (no annealing)

```bash
python Trainer.py \
    --DR ./dataset \
    --save_root ./runs/no-kl \
    --batch_size 8 \
    --train_vi_len 16 \
    --val_vi_len 630 \
    --num_epoch 70 \
    --num_workers 4 \
    --per_save 2 \
    --lr 1e-3 \
    --lr_min 1e-5 \
    --scheduler cosine \
    --kl_anneal_type None \
    --kl_max 0.05 \
    --tfr 1.0 \
    --tfr_sde 10 \
    --tfr_d_step 0.1 \
    --tfr_d_period 6
```

### Resume / fine-tune from a checkpoint

```bash
# Resume exactly (with optimizer, scheduler, KL annealer, TFR, history)
python Trainer.py \
    --DR ./dataset \
    --save_root ./runs/cyclical-kl \
    --ckpt_path ./runs/cyclical-kl/epoch=N.ckpt \
    --resume \
    [...same training args as the original run...]


`--ckpt_path` without `--resume` does **weights-only** restore (warm-start a new experiment from old weights; optimizer + visualization curves start fresh).

## Testing

```bash
python Tester.py \
    --DR ./dataset \
    --save_root ./runs/cyclical-kl \
    --ckpt_path ./runs/cyclical-kl/epoch=last.ckpt
```

Outputs `submission.csv` and per-sequence GIFs (`pred_seq{idx}.gif`) under `--save_root`.

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
