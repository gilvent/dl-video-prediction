import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10


def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2)
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.kl_anneal_type = args.kl_anneal_type
        self.n_iter = args.num_epoch
        self.n_cycle = max(1, args.kl_anneal_cycle)
        self.ratio = args.kl_anneal_ratio
        self.kl_max = getattr(args, 'kl_max', 1.0)
        self.current_epoch = current_epoch

        if self.kl_anneal_type == 'Cyclical':
            self.schedule = self.frange_cycle_linear(
                self.n_iter, start=0.0, stop=self.kl_max,
                n_cycle=self.n_cycle, ratio=self.ratio)
        elif self.kl_anneal_type == 'Monotonic':
            ramp = np.linspace(0.0, self.kl_max, self.n_cycle, dtype=np.float32)
            tail = np.full(max(0, self.n_iter - self.n_cycle), self.kl_max, dtype=np.float32)
            self.schedule = np.concatenate([ramp, tail])[:self.n_iter]
        elif self.kl_anneal_type in ('None', 'Without', 'none'):
            self.schedule = np.full(self.n_iter, self.kl_max, dtype=np.float32)
        else:
            raise ValueError(f"Unknown kl_anneal_type: {self.kl_anneal_type}")

    def update(self):
        self.current_epoch += 1

    def get_beta(self):
        idx = min(self.current_epoch, len(self.schedule) - 1)
        return float(self.schedule[idx])

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1):
        L = np.full(n_iter, stop, dtype=np.float32)
        period = n_iter / n_cycle
        step = (stop - start) / max(1.0, period * ratio)
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and int(i + c * period) < n_iter:
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L

    def state_dict(self):
        return {'current_epoch': self.current_epoch}

    def load_state_dict(self, state):
        self.current_epoch = state.get('current_epoch', 0)


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)

        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)

        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr,
                                     weight_decay=getattr(self.args, 'weight_decay', 0.0))
        self.scheduler  = self._build_scheduler(remaining_epochs=self.args.num_epoch)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size

        # History for plotting
        self.history = {
            'train_loss': [], 'val_loss': [], 'val_psnr': [],
            'beta': [], 'tfr': [],
        }
        self.last_per_frame_psnr = None

    def forward(self, img, label):
        pass

    def training_stage(self):
        for i in range(self.args.num_epoch):
            if self.current_epoch >= self.args.num_epoch:
                break

            train_loader = self.train_dataloader()
            epoch_losses = []

            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                adapt_TeacherForcing = random.random() < self.tfr
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss, mse_per_frame = self.training_one_step(img, label, adapt_TeacherForcing)

                epoch_losses.append(mse_per_frame)
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.2f}], beta: {:.4f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.2f}], beta: {:.4f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

            mean_train = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            self.history['train_loss'].append(mean_train)
            self.history['beta'].append(self.kl_annealing.get_beta())
            self.history['tfr'].append(self.tfr)

            val_loss, val_psnr = self.eval()
            self.history['val_loss'].append(val_loss)
            self.history['val_psnr'].append(val_psnr)

            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))

            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

            self.save_plots()

        # Final save
        self.save(os.path.join(self.args.save_root, "epoch=last.ckpt"))
        self.save_plots()

    @torch.no_grad()
    def eval(self):
        self.Generator.eval()
        val_loader = self.val_dataloader()
        losses = []
        psnrs = []
        per_frame_psnr_acc = None
        per_frame_count = 0
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, mean_psnr, per_frame = self.val_one_step(img, label)
            losses.append(float(loss))
            psnrs.append(float(mean_psnr))
            if per_frame is not None:
                if per_frame_psnr_acc is None:
                    per_frame_psnr_acc = per_frame.clone()
                else:
                    per_frame_psnr_acc = per_frame_psnr_acc + per_frame
                per_frame_count += 1
            self.tqdm_bar('val', pbar, loss, lr=self.scheduler.get_last_lr()[0])
        self.Generator.train()
        if per_frame_count > 0:
            self.last_per_frame_psnr = (per_frame_psnr_acc / per_frame_count).cpu().numpy()
        mean_loss = float(np.mean(losses)) if losses else float('nan')
        mean_psnr = float(np.mean(psnrs)) if psnrs else float('nan')
        return mean_loss, mean_psnr

    def _run_sequence(self, img, label, teacher_forcing, sample_from_posterior):
        """Unroll a video; returns (preds[T-1], mu_list, logvar_list).
        img/label shape: (T, B, C, H, W). preds aligned to img[1:T].
        """
        T = img.shape[0]
        prev = img[0]
        preds = []
        mus, logvars = [], []
        for t in range(1, T):
            label_feat = self.label_transformation(label[t])
            prev_feat = self.frame_transformation(prev)
            if sample_from_posterior:
                target_feat = self.frame_transformation(img[t])
                z, mu, logvar = self.Gaussian_Predictor(target_feat, label_feat)
                mus.append(mu)
                logvars.append(logvar)
            else:
                B, _, H, W = prev_feat.shape
                z = torch.randn(B, self.args.N_dim, H, W, device=prev_feat.device)
            fused = self.Decoder_Fusion(prev_feat, label_feat, z)
            x_hat = self.Generator(fused)
            preds.append(x_hat)
            if teacher_forcing:
                prev = img[t]
            else:
                prev = x_hat
        return preds, mus, logvars

    def training_one_step(self, img, label, adapt_TeacherForcing):
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        T, B = img.shape[0], img.shape[1]

        prev = img[0]
        mse_total = 0.0
        kl_total = 0.0
        for t in range(1, T):
            label_feat = self.label_transformation(label[t])
            prev_feat = self.frame_transformation(prev)
            target_feat = self.frame_transformation(img[t])
            z, mu, logvar = self.Gaussian_Predictor(target_feat, label_feat)
            fused = self.Decoder_Fusion(prev_feat, label_feat, z)
            x_hat = self.Generator(fused)

            mse_total = mse_total + self.mse_criterion(x_hat, img[t])
            kl_total = kl_total + kl_criterion(mu, logvar, B)

            if adapt_TeacherForcing:
                prev = img[t]
            else:
                prev = x_hat.detach()

        beta = self.kl_annealing.get_beta()
        loss = mse_total + beta * kl_total

        self.optim.zero_grad()
        loss.backward()
        self.optimizer_step()
        mse_per_frame = float(mse_total.detach().cpu()) / max(1, T - 1)
        return loss, mse_per_frame

    @torch.no_grad()
    def val_one_step(self, img, label):
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        T, B = img.shape[0], img.shape[1]
        assert B == 1, "val batch size must be 1"

        prev = img[0]
        per_frame_psnr = []
        mse_total = 0.0
        for t in range(1, T):
            label_feat = self.label_transformation(label[t])
            prev_feat = self.frame_transformation(prev)
            _, _, H, W = prev_feat.shape
            z = torch.randn(B, self.args.N_dim, H, W, device=prev_feat.device)
            fused = self.Decoder_Fusion(prev_feat, label_feat, z)
            x_hat = self.Generator(fused)
            mse_total = mse_total + self.mse_criterion(x_hat, img[t])
            psnr = Generate_PSNR(x_hat, img[t]).detach()
            per_frame_psnr.append(psnr)
            prev = x_hat
        per_frame_psnr_t = torch.stack(per_frame_psnr)
        return float(mse_total / max(1, T - 1)), float(per_frame_psnr_t.mean()), per_frame_psnr_t

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        partial = self.args.fast_partial if self.args.fast_train else self.args.partial
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train',
                                video_len=self.train_vi_len, partial=partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False

        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)
        return val_loader

    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.tfr_sde:
            period = max(1, getattr(self.args, 'tfr_d_period', 1))
            if (self.current_epoch - self.tfr_sde) % period == 0:
                self.tfr = max(0.0, self.tfr - self.tfr_d_step)

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def _build_scheduler(self, remaining_epochs):
        sched_type = getattr(self.args, 'scheduler', 'cosine')
        if sched_type == 'cosine':
            eta_min = getattr(self.args, 'lr_min', 1e-5)
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=max(1, remaining_epochs), eta_min=eta_min)
        return optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)

    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "kl_anneal": self.kl_annealing.state_dict(),
            "lr": self.scheduler.get_last_lr()[0],
            "tfr": self.tfr,
            "last_epoch": self.current_epoch,
            "history": self.history,
            "last_per_frame_psnr": self.last_per_frame_psnr,
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path is None:
            return
        checkpoint = torch.load(self.args.ckpt_path, map_location=self.args.device, weights_only=False)
        self.load_state_dict(checkpoint['state_dict'], strict=True)

        if not getattr(self.args, 'resume', False):
            print(f"Loaded weights only from {self.args.ckpt_path} (no --resume)")
            return

        try:
            self.kl_annealing.load_state_dict(checkpoint['kl_anneal'])
            self.tfr = checkpoint['tfr']
            self.current_epoch = checkpoint['last_epoch'] + 1
            if 'history' in checkpoint and checkpoint['history']:
                self.history = checkpoint['history']
            self.last_per_frame_psnr = checkpoint.get('last_per_frame_psnr', None)

            reset_optim = getattr(self.args, 'reset_optim', False)
            if reset_optim:
                # Rebuild optimizer + scheduler from current args; cover only
                # the remaining epochs so the LR curve fits the time we have
                # left. Useful for swapping scheduler shape mid-run or
                # extending num_epoch.
                self.optim = optim.Adam(self.parameters(), lr=self.args.lr,
                                        weight_decay=getattr(self.args, 'weight_decay', 0.0))
                remaining = max(1, self.args.num_epoch - self.current_epoch)
                self.scheduler = self._build_scheduler(remaining_epochs=remaining)
                print(f"Resumed weights+history from {self.args.ckpt_path} at epoch {self.current_epoch}; "
                      f"optimizer/scheduler reset (remaining={remaining}, lr={self.args.lr})")
            else:
                self.optim.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                # save() runs before scheduler.step() in training_stage, so the
                # checkpointed scheduler is one step behind. Advance once so
                # the resumed epoch trains with the correct LR.
                self.scheduler.step()
                print(f"Resumed from {self.args.ckpt_path} at epoch {self.current_epoch}")
        except KeyError as e:
            print(f"Warning: checkpoint missing key {e}; falling back to weights-only load")

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

    def save_plots(self):
        save_root = self.args.save_root
        os.makedirs(save_root, exist_ok=True)
        h = self.history
        if not h['train_loss']:
            return

        epochs = list(range(len(h['train_loss'])))

        plt.figure()
        plt.plot(epochs, h['train_loss'], label='train')
        if h['val_loss']:
            plt.plot(list(range(len(h['val_loss']))), h['val_loss'], label='val')
        plt.xlabel('epoch'); plt.ylabel('mean per-frame MSE'); plt.yscale('log')
        plt.legend(); plt.grid(alpha=0.3); plt.title('Loss curve')
        plt.savefig(os.path.join(save_root, 'loss_curve.png'), bbox_inches='tight')
        plt.close()

        if h['val_psnr']:
            plt.figure()
            plt.plot(list(range(len(h['val_psnr']))), h['val_psnr'])
            plt.xlabel('epoch'); plt.ylabel('val mean PSNR (dB)'); plt.title('Validation mean PSNR')
            plt.savefig(os.path.join(save_root, 'val_psnr.png'), bbox_inches='tight')
            plt.close()

        if h['beta']:
            plt.figure()
            plt.plot(list(range(len(h['beta']))), h['beta'])
            plt.xlabel('epoch'); plt.ylabel('beta'); plt.title('KL annealing beta')
            plt.savefig(os.path.join(save_root, 'beta_curve.png'), bbox_inches='tight')
            plt.close()

        if h['tfr']:
            plt.figure()
            plt.plot(list(range(len(h['tfr']))), h['tfr'])
            plt.xlabel('epoch'); plt.ylabel('tfr'); plt.title('Teacher forcing ratio')
            plt.savefig(os.path.join(save_root, 'tfr_curve.png'), bbox_inches='tight')
            plt.close()

        if self.last_per_frame_psnr is not None:
            plt.figure()
            plt.plot(np.arange(len(self.last_per_frame_psnr)), self.last_per_frame_psnr)
            plt.xlabel('frame'); plt.ylabel('PSNR (dB)'); plt.title('PSNR per frame (val)')
            plt.savefig(os.path.join(save_root, 'psnr_per_frame.png'), bbox_inches='tight')
            plt.close()

        with open(os.path.join(save_root, 'history.json'), 'w') as f:
            json.dump({k: v for k, v in h.items()}, f, indent=2)


def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=4)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")


    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")

    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--tfr_d_period',  type=int,   default=1,    help="Apply tfr_d_step every N epochs after tfr_sde (1 = every epoch)")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")
    parser.add_argument('--resume',        action='store_true',      help="Resume optimizer/scheduler/history from --ckpt_path")
    parser.add_argument('--reset_optim',   action='store_true',      help="With --resume: reload weights+history but rebuild optimizer & scheduler from current args (covers remaining epochs only)")

    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")

    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="Cyclical | Monotonic | None")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.5,            help="Fraction of each cycle spent ramping (Cyclical only)")
    parser.add_argument('--kl_max',             type=float, default=1.0,            help="Peak beta value of the schedule")

    # Optimizer / scheduler
    parser.add_argument('--scheduler',          type=str, default='cosine', choices=['cosine', 'multistep'])
    parser.add_argument('--lr_min',             type=float, default=1e-5,           help="Minimum LR for cosine schedule")
    parser.add_argument('--weight_decay',       type=float, default=0.0,            help="Adam weight decay")

    args = parser.parse_args()

    main(args)
