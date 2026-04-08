"""
配置文件：定义所有超参数和路径
"""
import os
import torch


class Config:
    # 随机种子（确保可重复性）
    SEED = 42

    # 数据路径
    SPECIMEN_PATH = "/data/yutong/program/my_dataset_blurred"
    HABITAT_PATH = "/data/yutong/Elaeagnaceae/morethan100pic"
    OUTPUT_DIR = "outputs"

    # 图像处理
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # 训练参数
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    MARGIN = 0.5  # Triplet Loss margin

    # Triplet 采样策略
    # 注意：当前 TripletDataset negative 为随机抽样；不是真正 hard mining
    TRIPLET_SELECTION_STRATEGY = "random"  # "random" 或 "hard"
    WARN_ON_FAKE_HARD = True

    # 联合训练（classification + triplet）
    USE_CLASSIFICATION_LOSS = True
    CLASSIFICATION_WEIGHT = 1.0
    TRIPLET_WEIGHT = 1.0

    # 运行时动态写入（在 main.py 载入数据集后设置）
    NUM_CLASSES = None

    # 特征提取
    FEATURE_DIM = 512
    MODEL_TYPE = "resnet50"  # "resnet50", "vit_b16", "inception_resnet_v2"

    # 数据集划分
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 系统发育树构建
    PHYLOGENY_METHODS = ["upgma", "nj"]
    NUM_BOOTSTRAP = 100

    # 输出
    SAVE_MODEL = True
    SAVE_FEATURES = True
    SAVE_TREES = True

    @staticmethod
    def create_output_dirs():
        """创建输出目录"""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(Config.OUTPUT_DIR, "models"), exist_ok=True)
        os.makedirs(os.path.join(Config.OUTPUT_DIR, "features"), exist_ok=True)
        os.makedirs(os.path.join(Config.OUTPUT_DIR, "trees"), exist_ok=True)
        os.makedirs(os.path.join(Config.OUTPUT_DIR, "figures"), exist_ok=True)
