import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 数据集根目录（本地路径）
DATA_ROOT = Path(r"E:\毕设-图像裁剪资料\裁剪数据集")

# 权重目录
WEIGHTS_ROOT = PROJECT_ROOT / "weights"

# 实验结果输出目录
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

def p(*parts) -> str:
    return str(Path(*parts))

# 各数据集路径
FCDB_DIR = DATA_ROOT / "FCDB"
FLMS_DIR = DATA_ROOT / "FLMS"
GAIC_DIR = DATA_ROOT / "GAIC"
KUPCP_DIR = DATA_ROOT / "KU_PCP"
TAD66K_DIR = DATA_ROOT / "TAD66K"

# 预训练权重
NEXTVIT_PRETRAIN = WEIGHTS_ROOT / "nextvit_large_in1k_224.pth"