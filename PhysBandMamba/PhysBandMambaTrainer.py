"""PhysBandMamba (MidWidth-72) -- self-contained training recipe.

This file contains ONLY the training machinery that the MidWidth-72
configuration actually uses. Everything the config leaves at zero or off has
been removed rather than left behind a dead branch.

Recipe as trained:
  loss        Hybrid_Loss (0.2 * negative-Pearson + 1.0 * frequency
              cross-entropy) + 0.005 * mask-entropy
  optimiser   AdamW, lr 3e-4, weight decay 5e-4
  schedule    OneCycleLR, pct_start 0.3, div 25, final div 1e4, cosine
  epochs 30   batch 16   chunk 160 frames @ 30 fps   128x128 crops
  augment     per-sample temporal rate augmentation (halve/double, p=0.5)
              + nuisance augmentation (affine, warp), p=0.5
  selection   smoothed best epoch (window 3) on validation loss

Removed relative to the research trainer
(neural_methods/trainer/PhysBandMambaTrainer.py): the harmonic-envelope,
subharmonic, signal-quality, amplitude-profile, amplitude-correlation and
temporal-consistency auxiliary losses (all lambda = 0 here, and all rejected
by ablation); the alternative LOSS_TYPE branches (harmodiff, pearson_only,
pearson_psdkl); the video-level validation-metric branch; the watchlist /
per-condition breakdown reporting; and the tolerant checkpoint loader.

NOTE: this is the training recipe extracted for publication and reuse. The
runs that produced the shipped checkpoints were executed with the research
trainer; this file reproduces its active code path, not its logging.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm

cudnn.benchmark = False


class PhysBandMambaTrainer:
    """Minimal trainer for the MidWidth-72 recipe.

    Expects `data_loader` as a dict with "train"/"valid"/"test" keys whose
    batches are (frames [B,T,3,H,W], labels [B,T], subject_id, chunk_id).
    """

    def __init__(self, config, model, criterion, gt_hr_fn, rate_aug_fn, nuisance_aug_fn,
                 device="cuda", lambda_mask_entropy=0.005,
                 nuisance_prob=0.5, nuisance_strength=1.0,
                 nuisance_components=("affine", "warp"),
                 best_epoch_smooth_window=3):
        self.config = config
        self.model = model.to(device)
        self.criterion = criterion              # Hybrid_Loss
        self._gt_hr = gt_hr_fn                  # labels -> per-sample GT HR (bpm), cached
        self.rate_aug = rate_aug_fn             # the temporal rate augmentation below
        self.nuisance_aug = nuisance_aug_fn     # nuisance_preserving_augmentation
        self.device = device
        self.lambda_mask_entropy = lambda_mask_entropy
        self.nuisance_prob = nuisance_prob
        self.nuisance_strength = nuisance_strength
        self.nuisance_components = set(nuisance_components)
        self.best_epoch_smooth_window = best_epoch_smooth_window

        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.fs = float(config.TRAIN.DATA.FS)
        self.diff_flag = 1 if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized" else 0
        self.current_epoch = 0
        self.best_epoch = 0
        self.min_valid_loss = None
        self.valid_loss_history = []
        self.data_dict = {}

    # -- optimiser -------------------------------------------------------
    def build_optimizer(self, num_train_batches):
        self.optimizer = optim.AdamW(self.model.parameters(),
                                      lr=self.config.TRAIN.LR, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.config.TRAIN.LR, epochs=self.config.TRAIN.EPOCHS,
            steps_per_epoch=num_train_batches, pct_start=0.3, div_factor=25.0,
            final_div_factor=10000.0, anneal_strategy="cos")

    # -- loss ------------------------------------------------------------
    def _bvp_loss(self, pred_ppg, labels, fs):
        """Per-sample z-scored Hybrid_Loss, averaged over the batch. The
        z-scoring makes the loss blind to output scale (relevant context:
        the concatenated evaluation metric weights chunks by their power)."""
        pred_ppg = (pred_ppg - torch.mean(pred_ppg, axis=-1).view(-1, 1)) / \
                   (torch.std(pred_ppg, axis=-1).view(-1, 1) + 1e-8)
        N = pred_ppg.shape[0]
        loss = 0.0
        for ib in range(N):
            loss = loss + self.criterion(pred_ppg[ib], labels[ib], self.current_epoch, fs, self.diff_flag)
        return loss / N, pred_ppg

    # -- augmentation ----------------------------------------------------
    def data_augmentation(self, data, labels, index1, index2):
        """Per-sample temporal rate augmentation (RhythmMamba's scheme).

        With p=0.5 a clip is resampled: clips above 90 bpm are slowed by 2x
        (interpolating between neighbouring frames), clips below 75 bpm are
        sped up by 2x (decimate, then repeat the half-clip); 75-90 bpm is
        left alone. Indexes PER SAMPLE -- the inherited RhythmMamba trainer
        indexes the whole batch here, which silently clobbers it.
        """
        N, D, C, H, W = data.shape
        data_aug = torch.zeros_like(data)
        labels_aug = torch.zeros_like(labels)
        rand1_vals = np.random.random(N)
        for idx in range(N):
            key = (str(index1[idx]), str(index2[idx]))
            if rand1_vals[idx] < 0.5:
                if key in self.data_dict:
                    gt_hr_fft = self.data_dict[key]
                else:
                    gt_hr_fft = self._gt_hr(labels[idx], self.diff_flag, self.fs)
                    self.data_dict[key] = gt_hr_fft
                if gt_hr_fft > 90:
                    rand3 = random.randint(0, D // 2 - 1)
                    even = torch.arange(0, D, 2)
                    odd = even + 1
                    data_aug[idx, even] = data[idx, rand3 + even // 2]
                    labels_aug[idx, even] = labels[idx, rand3 + even // 2]
                    data_aug[idx, odd] = (data[idx, rand3 + odd // 2]
                                          + data[idx, rand3 + (odd // 2) + 1]) / 2
                    labels_aug[idx, odd] = (labels[idx, rand3 + odd // 2]
                                            + labels[idx, rand3 + (odd // 2) + 1]) / 2
                elif gt_hr_fft < 75:
                    data_aug[idx, :D // 2] = data[idx, ::2]
                    labels_aug[idx, :D // 2] = labels[idx, ::2]
                    data_aug[idx, D // 2:] = data_aug[idx, :D // 2]
                    labels_aug[idx, D // 2:] = labels_aug[idx, :D // 2]
                else:
                    data_aug[idx], labels_aug[idx] = data[idx], labels[idx]
            else:
                data_aug[idx], labels_aug[idx] = data[idx], labels[idx]
        return data_aug, labels_aug

    # -- training --------------------------------------------------------
    def train(self, data_loader):
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        fs = self.config.TRAIN.DATA.FS

        for epoch in range(self.max_epoch_num):
            self.current_epoch = epoch
            print(f"\n====Training Epoch: {epoch}====")
            self.model.train()
            running = []

            tbar = tqdm(data_loader["train"], ncols=80)
            for batch in tbar:
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].float(), batch[1].float()

                if self.config.TRAIN.AUG:
                    data, labels = self.data_augmentation(data, labels, batch[2], batch[3])
                data, labels = data.to(self.device), labels.to(self.device)

                if self.nuisance_prob > 0:
                    m = torch.rand(data.shape[0], device=data.device) < self.nuisance_prob
                    if m.any():
                        with torch.no_grad():
                            aug = self.nuisance_aug(data[m], strength=self.nuisance_strength,
                                                    components=self.nuisance_components)
                        data = data.clone()
                        data[m] = aug

                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(data, fs=fs, epoch=epoch, labels=labels, return_aux=True)
                loss_bvp, _ = self._bvp_loss(out['bvp'], labels, fs)
                loss = loss_bvp + self.lambda_mask_entropy * out['loss_mask_entropy']
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running.append(float(loss_bvp.item()))
                tbar.set_postfix(loss=float(loss.item()))

            print(f"[epoch {epoch}] mean loss_bvp = {np.mean(running):.4f}")
            self.save_model(epoch)

            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                self.valid_loss_history.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print(f"Update best model! Best epoch: {self.best_epoch}")

        if not self.config.TEST.USE_LAST_EPOCH:
            # Smoothed selection: the single-epoch validation minimum can land
            # mid-flicker on a bistable failure the validation split has no
            # visibility into, so prefer the centre of the best SUSTAINED
            # low-loss window instead.
            losses = self.valid_loss_history
            half = self.best_epoch_smooth_window // 2
            smoothed = [float(np.mean(losses[max(0, i - half):min(len(losses), i + half + 1)]))
                        for i in range(len(losses))]
            best = int(np.argmin(smoothed))
            print(f"[smoothed selection] raw best_epoch={self.best_epoch} -> smoothed best_epoch={best}")
            self.best_epoch = best
            self.min_valid_loss = losses[best]

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print("\n===Validating===")
        fs = self.config.VALID.DATA.FS
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(data_loader["valid"], ncols=80):
                data = batch[0].float().to(self.device)
                labels = batch[1].float().to(self.device)
                out = self.model(data, fs=fs, epoch=self.current_epoch, labels=labels, return_aux=False)
                loss_bvp, _ = self._bvp_loss(out['bvp'], labels, fs)
                losses.append(float((loss_bvp + self.lambda_mask_entropy * out['loss_mask_entropy']).item()))
        return float(np.mean(losses))

    def save_model(self, index):
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{index}.pth")
        torch.save(self.model.state_dict(), path)
