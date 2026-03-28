import os
import torch
from paths import KUPCP_DIR, EXPERIMENTS_ROOT

class Config:
    def __init__(self):
        self.KUPCP_dir = str(KUPCP_DIR)

        self.image_size = (224, 224)
        self.data_augmentation = True
        self.keep_aspect_ratio = False

        self.backbone = 'vgg16'
        self.gpu_id = 0
        self.num_workers = 4
        self.com_batch_size = 64

        self.epochs = 50
        self.lr = 1e-4
        self.weight_decay = 1e-4

        self.device = torch.device(
            f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu'
        )

        self.prefix = 'composition_classification'
        self.exp_root = os.path.join(str(EXPERIMENTS_ROOT), 'CompositionClassify')
        os.makedirs(self.exp_root, exist_ok=True)

        index = 1
        while True:
            exp_name = self.prefix if index == 1 else f"{self.prefix}_repeat{index}"
            exp_path = os.path.join(self.exp_root, exp_name)
            if not os.path.exists(exp_path):
                break
            index += 1

        self.exp_name = exp_name
        self.exp_path = exp_path
        self.checkpoint_dir = os.path.join(self.exp_path, 'checkpoints')
        self.log_dir = os.path.join(self.exp_path, 'logs')

    def create_path(self):
        os.makedirs(self.exp_path, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

cfg = Config()