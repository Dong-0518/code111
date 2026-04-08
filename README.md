# 基于深度学习的植物系统发育树构建系统

## 项目简介

本项目利用深度学习技术从植物标本图像和生境图像中提取特征，通过度量学习（Triplet Loss）将图像映射到具有语义信息的特征空间，然后使用聚类算法（UPGMA、NJ法）自动生成物种的系统发育树。

## 功能特点

- ✅ 支持两种图像类型：植物标本图像和生境图像
- ✅ 多种预训练模型：ResNet50、ViT-B/16、InceptionResNetV2
- ✅ 度量学习：使用Triplet Loss学习具有区分性的特征空间
- ✅ 系统发育树构建：支持UPGMA和Neighbor-Joining方法
- ✅ 结果可视化：特征分布、距离矩阵、系统发育树
- ✅ 可重复性：固定随机种子，确保结果一致

## 环境要求

- Python >= 3.7
- PyTorch >= 1.12.0
- CUDA（可选，用于GPU加速）

## 安装

1. 克隆或下载项目到本地

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基本使用

#### 完整流程（训练+特征提取+构建树）
```bash
python main.py --mode full --image_type specimen --model_type resnet50
```

#### 仅训练模型
```bash
python main.py --mode train --image_type specimen --model_type resnet50
```

#### 仅提取特征和构建树（使用预训练模型）
```bash
python main.py --mode extract --image_type specimen --model_type resnet50 --skip_training
```

#### 仅构建系统发育树（需要先有特征文件）
```bash
python main.py --mode phylogeny --image_type specimen --model_type resnet50
```

### 2. 参数说明

- `--mode`: 运行模式
  - `full`: 完整流程（训练+特征提取+构建树）
  - `train`: 仅训练模型
  - `extract`: 仅提取特征和构建树
  - `phylogeny`: 仅构建系统发育树

- `--image_type`: 图像类型
  - `specimen`: 植物标本图像
  - `habitat`: 植物生境图像

- `--model_type`: 模型类型
  - `resnet50`: ResNet50（推荐用于标本图像）
  - `vit_b16`: Vision Transformer (ViT-B/16)（推荐用于标本图像）
  - `inception_resnet_v2`: InceptionResNetV2（推荐用于生境图像）

- `--skip_training`: 跳过训练，直接使用预训练模型

### 3. 配置修改

可以在 `config.py` 中修改以下参数：

- `SEED`: 随机种子（默认42）
- `IMAGE_SIZE`: 图像尺寸（默认224）
- `BATCH_SIZE`: 批次大小（默认32）
- `NUM_EPOCHS`: 训练轮数（默认100）
- `LEARNING_RATE`: 学习率（默认1e-4）
- `MARGIN`: Triplet Loss的margin（默认0.5）
- `FEATURE_DIM`: 特征维度（默认512）

## 项目结构

```
.
├── main.py                 # 主程序
├── config.py              # 配置文件
├── data_loader.py         # 数据加载和预处理
├── models.py              # 模型定义
├── triplet_loss.py        # Triplet Loss实现
├── trainer.py             # 训练脚本
├── feature_extractor.py   # 特征提取
├── phylogeny.py           # 系统发育树构建
├── utils.py               # 工具函数
├── requirements.txt       # 依赖包
├── README.md             # 使用说明
├── 技术方案.md            # 详细技术方案
└── outputs/              # 输出目录
    ├── models/           # 保存的模型
    ├── features/         # 提取的特征
    ├── trees/            # 系统发育树文件
    └── figures/          # 可视化结果
```

## 输出文件说明

### 模型文件
- `outputs/models/{model_type}_best.pth`: 训练好的模型权重

### 特征文件
- `outputs/features/{image_type}_{model_type}_features.npz`: 提取的特征（numpy格式）

### 系统发育树文件
- `outputs/trees/{image_type}/tree_upgma.newick`: UPGMA树的Newick格式
- `outputs/trees/{image_type}/tree_upgma.nexus`: UPGMA树的NEXUS格式
- `outputs/trees/{image_type}/tree_nj.newick`: NJ树的Newick格式
- `outputs/trees/{image_type}/tree_nj.nexus`: NJ树的NEXUS格式

### 可视化文件
- `outputs/figures/training_curve.png`: 训练曲线
- `outputs/figures/{image_type}_{model_type}_tsne.png`: 特征t-SNE可视化
- `outputs/figures/{image_type}_{model_type}_distance_matrix.png`: 距离矩阵热图
- `outputs/trees/{image_type}/tree_upgma.png`: UPGMA树可视化
- `outputs/trees/{image_type}/tree_nj.png`: NJ树可视化

## 技术细节

### 1. 数据预处理
- 图像尺寸统一为224x224
- 训练时使用数据增强：随机翻转、旋转、颜色抖动
- 使用ImageNet的均值和标准差进行归一化

### 2. 特征提取
- 使用预训练的CNN或Transformer模型提取特征
- 去除分类头，保留特征提取层
- 使用全局平均池化或最大池化
- L2归一化特征向量

### 3. 度量学习
- 使用Triplet Loss学习特征空间
- Hard negative mining：选择最难的负样本
- 特征维度：512维

### 4. 系统发育树构建
- 计算物种间的距离矩阵（欧氏距离）
- 使用UPGMA和NJ方法构建树
- 支持导出Newick和NEXUS格式

## 常见问题

### Q1: 结果每次都不一样怎么办？
A: 确保设置了随机种子。代码中已经设置了固定随机种子（SEED=42），如果结果仍然不一致，可能是由于：
- 数据加载顺序不同（已使用固定随机种子解决）
- 模型初始化不同（已使用固定随机种子解决）
- 建议对每个物种的特征进行平均，而不是使用单张图像

### Q2: 分类准确率不高怎么办？
A: 可以尝试：
- 增加训练轮数
- 调整学习率
- 使用数据增强
- 尝试不同的模型（ResNet50、ViT等）
- 增加特征维度
- 使用模型集成

### Q3: 系统发育树不合理怎么办？
A: 可以尝试：
- 使用不同的距离度量（欧氏距离、余弦距离等）
- 尝试不同的聚类方法（UPGMA、NJ）
- 检查特征质量（使用t-SNE可视化）
- 增加每个物种的图像数量

### Q4: 内存不足怎么办？
A: 可以：
- 减小批次大小（BATCH_SIZE）
- 减小图像尺寸（IMAGE_SIZE）
- 使用较小的模型
- 减少数据加载线程数（NUM_WORKERS）

## 参考文献

1. Rove-tree-11: A hierarchically structured image dataset for deep metric learning research
2. Deep Learning Derived Traits for Phylogenetic Analysis
3. Deep learning on butterfly phenotypes tests evolution's
4. A pipeline to compile expert-verified datasets of digitised herbarium specimens
5. Applying image clustering to phylogenetic analysis: A trial

## 许可证

本项目仅供学术研究使用。


