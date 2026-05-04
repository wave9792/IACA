# -*- coding: utf-8 -*-
import os
from paths import (
    DATA_ROOT, EXPERIMENTS_ROOT,
    FCDB_DIR, FLMS_DIR, KUPCP_DIR
)

class Config:
    data_root = str(DATA_ROOT)

    predefined_pkl = os.path.join(str(DATA_ROOT), 'pdefined_anchors.pkl')
    FCDB_dir = str(FCDB_DIR)
    FLMS_dir = str(FLMS_DIR)
    KUPCP_dir = str(KUPCP_DIR)

    image_size = (224, 224)
    data_augmentation = True
    keep_aspect_ratio = False

    backbone = 'vgg16'
    gpu_id = 0

    # ===== 4G 冒烟测试 =====
    # crop_batch_size = 1
    # com_batch_size = 1
    # theme_batch_size = 1
    # rank_batch_size = 1
    # num_workers = 0
    # display_freq = 1
    # max_epoch = 1
    # eval_freq = 1000

    # ===== 16G 正式训练 =====
    crop_batch_size = 8
    com_batch_size = 4
    theme_batch_size = 4
    rank_batch_size = 2
    num_workers = 4
    display_freq = 20
    max_epoch = 100
    eval_freq = 1

    # ---- Loss weights ----
    crop_loss_factor = 0.6
    com_loss_factor = 0.2
    theme_loss_factor = 0.2
    contrastive_loss_weight = 0.2

    # 新增：GAICD ranking loss + 区域对比损失权重
    ranking_loss_weight = 0.15
    region_contrastive_loss_weight = 0.15
    ranking_margin = 0.5

    lr_decay_epoch = [30, 60]
    lr = 1e-4
    lr_decay = 0.1
    weight_decay = 1e-3
    save_freq = max_epoch + 1
    save_image_freq = 200

    prefix = 'cropping_{}croploss_{}compositionloss_{}themeloss_{}contrastive_{}ranking_{}regioncon'.format(
        crop_loss_factor, com_loss_factor, theme_loss_factor,
        contrastive_loss_weight, ranking_loss_weight, region_contrastive_loss_weight
    )

    exp_root = str(EXPERIMENTS_ROOT)
    exp_name = prefix
    exp_path = os.path.join(exp_root, prefix)

    while os.path.exists(exp_path):
        index = os.path.basename(exp_path).split(prefix)[-1].split('repeat')[-1]
        try:
            index = int(index) + 1
        except:
            index = 1
        exp_name = prefix + ('_repeat{}'.format(index))
        exp_path = os.path.join(exp_root, exp_name)

    checkpoint_dir = os.path.join(exp_path, 'checkpoints')
    log_dir = os.path.join(exp_path, 'logs')

    def create_path(self):
        print('Create experiment directory: ', self.exp_path)
        os.makedirs(self.exp_path, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

cfg = Config()
