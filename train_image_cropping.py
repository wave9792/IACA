import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import csv
import time
import random
import shutil
import datetime
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F

from KUPCP_dataset import CompositionDataset
from TAD66K_dataset import ThemeDataset
from Cropping_dataset import FCDBDataset
from config_cropping import cfg
from test import evaluate_on_FCDB_and_FLMS
from CACNet import CACNet


device = torch.device(f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
SEED = 0


def set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


set_global_seed(SEED)


def create_dataloader():
    crop_dataset = FCDBDataset(split='train', keep_aspect_ratio=cfg.keep_aspect_ratio)
    crop_loader = torch.utils.data.DataLoader(
        crop_dataset,
        batch_size=cfg.crop_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
        worker_init_fn=seed_worker
    )
    print('FCDB training set has {} samples, batch_size={}, total {} batches'.format(
        len(crop_dataset), cfg.crop_batch_size, len(crop_loader)
    ))

    com_dataset = CompositionDataset(split='train', keep_aspect_ratio=cfg.keep_aspect_ratio)
    com_loader = torch.utils.data.DataLoader(
        com_dataset,
        batch_size=cfg.com_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
        worker_init_fn=seed_worker
    )
    print('KU_PCP training set has {} samples, batch_size={}, total {} batches'.format(
        len(com_dataset), cfg.com_batch_size, len(com_loader)
    ))

    theme_dataset = ThemeDataset(split='train', keep_aspect_ratio=cfg.keep_aspect_ratio)
    theme_loader = torch.utils.data.DataLoader(
        theme_dataset,
        batch_size=cfg.theme_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
        worker_init_fn=seed_worker
    )
    print('TAD66K training set has {} samples, batch_size={}, total {} batches'.format(
        len(theme_dataset), cfg.theme_batch_size, len(theme_loader)
    ))

    return crop_loader, com_loader, theme_loader


class Trainer:
    def __init__(self, model):
        self.model = model
        self.epoch = 0
        self.iters = 0
        self.max_epoch = cfg.max_epoch
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        self.optimizer, self.lr_scheduler = self.get_optimizer()
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        self.crop_loader, self.com_loader, self.theme_loader = create_dataloader()
        self.com_dataiter = iter(self.com_loader)
        self.theme_dataiter = iter(self.theme_loader)

        self.eval_results = []
        self.best_results = {
            'FCDB_iou': 0.,
            'FCDB_disp': 1.,
            'FLMS_iou': 0.,
            'FLMS_disp': 1.
        }

        self.crop_criterion = nn.SmoothL1Loss(reduction='mean')
        self.com_criterion = nn.CrossEntropyLoss()
        self.theme_criterion = nn.CrossEntropyLoss()
        self.contrastive_loss_fn = NTXentLoss(temperature=0.5)

        self.crop_weight = cfg.crop_loss_factor
        self.com_weight = cfg.com_loss_factor
        self.theme_weight = cfg.theme_loss_factor
        self.contrastive_weight = cfg.contrastive_loss_weight

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.lr_decay_epoch,
            gamma=cfg.lr_decay
        )
        return optimizer, lr_scheduler

    def _next_aux_batch(self, loader_name):
        if loader_name == 'composition':
            try:
                batch = next(self.com_dataiter)
            except StopIteration:
                self.com_dataiter = iter(self.com_loader)
                batch = next(self.com_dataiter)
            return batch

        if loader_name == 'theme':
            try:
                batch = next(self.theme_dataiter)
            except StopIteration:
                self.theme_dataiter = iter(self.theme_loader)
                batch = next(self.theme_dataiter)
            return batch

        raise ValueError(f'Unsupported loader name: {loader_name}')

    @staticmethod
    def _prepare_labels(labels):
        labels = labels.to(device)
        if labels.dim() == 2:
            labels = labels.squeeze(1)
        return labels.long()

    def run(self):
        print('======== Begin Multi-task Training (Cropping + Composition + Theme + Contrastive) ========')
        for epoch in range(1, self.max_epoch + 1):
            self.epoch = epoch
            self.train()
            if epoch % cfg.eval_freq == 0:
                self.eval()
                self.record_eval_results()
            self.lr_scheduler.step()

    def train(self):
        self.model.train()
        start = time.time()

        batch_crop_loss = 0.0
        batch_com_loss = 0.0
        batch_theme_loss = 0.0
        batch_contrastive_loss = 0.0
        batch_total_loss = 0.0

        total_batch = len(self.crop_loader)

        for batch_idx, batch_data in enumerate(self.crop_loader):
            self.iters += 1

            view1 = batch_data[0].to(device, non_blocking=True)
            view2 = batch_data[1].to(device, non_blocking=True)
            crop = batch_data[2].to(device, non_blocking=True).squeeze(1)
            width = batch_data[3].to(device, non_blocking=True)
            height = batch_data[4].to(device, non_blocking=True)

            crop[:, 0::2] = crop[:, 0::2] / width[:, None] * view1.shape[-1]
            crop[:, 1::2] = crop[:, 1::2] / height[:, None] * view1.shape[-2]

            com_batch = self._next_aux_batch('composition')
            theme_batch = self._next_aux_batch('theme')

            com_im = com_batch[0].to(device, non_blocking=True)
            com_label = self._prepare_labels(com_batch[1])

            theme_im = theme_batch[0].to(device, non_blocking=True)
            theme_label = self._prepare_labels(theme_batch[1])

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # 共享 backbone，避免 view1 重复前向
                feat_view1 = self.model.extract_features(view1)
                feat_view2 = self.model.extract_features(view2)
                feat_com = self.model.extract_features(com_im)
                feat_theme = self.model.extract_features(theme_im)

                # 对比学习
                embedding1 = self.model.forward_from_features(feat_view1, operation='embedding')
                embedding2 = self.model.forward_from_features(feat_view2, operation='embedding')
                contrastive_loss = self.contrastive_loss_fn(embedding1, embedding2)

                # 裁剪
                _, _, pre_crop = self.model.forward_from_features(feat_view1, operation='cropping')
                crop_loss = self.crop_criterion(pre_crop, crop)

                # 构图
                logits_com, _ = self.model.forward_from_features(feat_com, operation='composition')
                com_loss = self.com_criterion(logits_com, com_label)

                # 主题
                logits_theme, _ = self.model.forward_from_features(feat_theme, operation='theme')
                theme_loss = self.theme_criterion(logits_theme, theme_label)

                # 总损失
                total_loss = (
                    self.crop_weight * crop_loss +
                    self.com_weight * com_loss +
                    self.theme_weight * theme_loss +
                    self.contrastive_weight * contrastive_loss
                )

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_crop_loss += crop_loss.item()
            batch_com_loss += com_loss.item()
            batch_theme_loss += theme_loss.item()
            batch_contrastive_loss += contrastive_loss.item()
            batch_total_loss += total_loss.item()

            if batch_idx > 0 and batch_idx % cfg.display_freq == 0:
                avg_crop_loss = batch_crop_loss / (batch_idx + 1)
                avg_com_loss = batch_com_loss / (batch_idx + 1)
                avg_theme_loss = batch_theme_loss / (batch_idx + 1)
                avg_contrastive_loss = batch_contrastive_loss / (batch_idx + 1)
                avg_total_loss = batch_total_loss / (batch_idx + 1)
                cur_lr = self.optimizer.param_groups[0]['lr']

                self.writer.add_scalar('train/crop_loss', avg_crop_loss, self.iters)
                self.writer.add_scalar('train/composition_loss', avg_com_loss, self.iters)
                self.writer.add_scalar('train/theme_loss', avg_theme_loss, self.iters)
                self.writer.add_scalar('train/contrastive_loss', avg_contrastive_loss, self.iters)
                self.writer.add_scalar('train/total_loss', avg_total_loss, self.iters)
                self.writer.add_scalar('train/lr', cur_lr, self.iters)

                time_per_batch = (time.time() - start) / (batch_idx + 1)
                last_batches = ((self.max_epoch - self.epoch - 1) * total_batch + (total_batch - batch_idx - 1))
                last_time = int(last_batches * time_per_batch)
                time_str = str(datetime.timedelta(seconds=last_time))

                print(
                    f'=== epoch:{self.epoch}/{self.max_epoch}, '
                    f'step:{batch_idx}/{total_batch} | '
                    f'Crop:{avg_crop_loss:.4f} | '
                    f'Comp:{avg_com_loss:.4f} | '
                    f'Theme:{avg_theme_loss:.4f} | '
                    f'Contrast:{avg_contrastive_loss:.4f} | '
                    f'Total:{avg_total_loss:.4f} | '
                    f'lr:{cur_lr:.6f} | '
                    f'estimated last time:{time_str} ==='
                )

    def eval(self):
        fcdb_iou, fcdb_disp = evaluate_on_FCDB_and_FLMS(self.model, dataset='FCDB')
        flms_iou, flms_disp = evaluate_on_FCDB_and_FLMS(self.model, dataset='FLMS')

        self.eval_results.append([self.epoch, fcdb_iou, fcdb_disp, flms_iou, flms_disp])
        epoch_result = {
            'FCDB_iou': fcdb_iou,
            'FCDB_disp': fcdb_disp,
            'FLMS_iou': flms_iou,
            'FLMS_disp': flms_disp
        }

        for metric_name in self.best_results.keys():
            update = False
            if ('disp' not in metric_name) and (epoch_result[metric_name] > self.best_results[metric_name]):
                update = True
            elif ('disp' in metric_name) and (epoch_result[metric_name] < self.best_results[metric_name]):
                update = True

            if update:
                self.best_results[metric_name] = epoch_result[metric_name]
                checkpoint_path = os.path.join(cfg.checkpoint_dir, f'best-{metric_name}.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                print('Update best {} model, best {}={:.4f}'.format(
                    metric_name, metric_name, self.best_results[metric_name]
                ))

            if metric_name in ['FCDB_iou', 'FLMS_iou']:
                self.writer.add_scalar(f'test/{metric_name}', epoch_result[metric_name], self.epoch)
                self.writer.add_scalar(f'test/best-{metric_name}', self.best_results[metric_name], self.epoch)

        if self.epoch > 0 and self.epoch in [68, 128]:
            checkpoint_path = os.path.join(cfg.checkpoint_dir, f'epoch-{self.epoch}.pth')
            torch.save(self.model.state_dict(), checkpoint_path)

    def record_eval_results(self):
        csv_path = os.path.join(cfg.exp_path, '..', f'{cfg.exp_name}.csv')
        header = ['epoch', 'FCDB_iou', 'FCDB_disp', 'FLMS_iou', 'FLMS_disp']
        rows = [header]

        for i in range(len(self.eval_results)):
            new_results = []
            for j in range(len(self.eval_results[i])):
                if header[j] == 'epoch':
                    new_results.append(self.eval_results[i][j])
                else:
                    new_results.append(round(self.eval_results[i][j], 4))
            self.eval_results[i] = new_results

        rows += self.eval_results
        metrics = [[] for _ in header]
        for result in self.eval_results:
            for i, value in enumerate(result):
                metrics[i].append(value)

        for name, metric_values in zip(header, metrics):
            if name == 'epoch':
                continue
            index = metric_values.index(max(metric_values))
            if 'disp' in name:
                index = metric_values.index(min(metric_values))
            title = f'best {name}(epoch-{index})'
            row = [values[index] for values in metrics]
            row[0] = title
            rows.append(row)

        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        print('Save result to', csv_path)


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)

        # 相似度矩阵强制 float32，更稳
        sim_matrix = torch.matmul(z.float(), z.float().T) / self.temperature
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        positive_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ])

        return F.cross_entropy(sim_matrix, positive_indices)


if __name__ == '__main__':
    cfg.create_path()
    for file in os.listdir('./'):
        if file.endswith('.py'):
            shutil.copy(file, cfg.exp_path)
            print('backup', file)

    net = CACNet(loadweights=True).to(device)
    trainer = Trainer(net)
    trainer.run()