# Lab 4 — Conditional VAE for Video Prediction (Vanilla variant)

Per-frame stochastic CVAE following the spec's Figure 4 (no recurrence beyond the previous-frame conditioning). Branch: `vanilla-cvae`.

## Setup

```bash
pip install -r requirements.txt
```

For RTX 50-series (sm_120), see the Blackwell note at the bottom.

## Training

Canonical run (GPU-saturating batch on 16 GB):

```bash
python Trainer.py \
    --DR ./dataset \
    --save_root ./runs/vanilla-cyclical \
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
    --kl_anneal_cycle 10 \
    --kl_anneal_ratio 0.5 \
    --kl_max 1.0 \
    --tfr 1.0 \
    --tfr_sde 10 \
    --tfr_d_step 0.1 \
    --tfr_d_period 6
```

Notes on the flags:

- `--kl_anneal_type {Cyclical,Monotonic,None}` selects the β schedule. Cyclical uses `frange_cycle_linear(num_epoch, n_cycle=kl_anneal_cycle, ratio=kl_anneal_ratio)`; Monotonic linearly ramps 0→1 over `kl_anneal_cycle` epochs and holds; None uses β = 1.
- `--kl_max 1.0` is appropriate for the vanilla model (KL is against an N(0, I) prior).
- `--tfr_sde 10 --tfr_d_step 0.1 --tfr_d_period 6`: hold tfr=1.0 for 10 epochs, then drop 0.1 every 6 epochs, reaching 0 at epoch 70. The `--tfr_d_period` flag spreads the TFR decay across the full run instead of crashing it to 0 in 10 epochs — gives the model multiple plateaus to adapt at each TFR level.
- Optimizer: Adam (with `--weight_decay` opt-in, default 0). Scheduler: cosine 1e-3 → 1e-5 over `num_epoch` (`--scheduler cosine --lr_min 1e-5`); pass `--scheduler multistep` to fall back to MultiStepLR `[2, 5]` × 0.1.
- `--batch_size 8` is the sweet spot on 16 GB VRAM. Pushing to batch=16 pinned VRAM at 16 GB and triggered allocator thrashing (per-iter time blew up ~30×); batch=8 sits comfortably at ~13 GB and runs ~3 min/epoch.

### Resume / fine-tune from a checkpoint

```bash
# Resume exactly (rehydrates optimizer, scheduler, KL annealer, TFR, history)
python Trainer.py \
    --DR ./dataset \
    --save_root ./runs/vanilla-cyclical \
    --ckpt_path ./runs/vanilla-cyclical/epoch=N.ckpt \
    --resume \
    [...same training args as the original run...]

# Resume weights/history but rebuild optimizer + scheduler from current args
python Trainer.py \
    [...same as above...] --resume --reset_optim
```

`--ckpt_path` without `--resume` does **weights-only** restore (warm-start a new experiment from old weights; optimizer + history start fresh).

## Testing

```bash
python Tester.py \
    --DR ./dataset \
    --save_root ./runs/vanilla-cyclical \
    --ckpt_path ./runs/vanilla-cyclical/epoch=last.ckpt
```

Outputs `submission.csv` and per-sequence GIFs (`pred_seq{idx}.gif`) under `--save_root`. The 629-step autoregressive rollout samples z from N(0, I).

## Plots

After training, `--save_root` contains:

- `loss_curve.png` — train (per-frame mean MSE) vs val (per-frame mean MSE)
- `val_psnr.png`, `psnr_per_frame.png`, `beta_curve.png`, `tfr_curve.png`
- `history.json` — raw lists for re-plotting

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
