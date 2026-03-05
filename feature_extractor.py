"""
特征提取模块
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from models import create_model
from utils import set_seed, save_features, calculate_species_features

class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, model, device, model_type='resnet50'):
        self.model = model
        self.device = device
        self.model_type = model_type
        self.model.eval()
        self.model.to(device)
    
    def extract_features(self, dataloader):
        """
        提取特征
        
        Args:
            dataloader: 数据加载器
        
        Returns:
            features: 特征矩阵 (n_samples, feature_dim)
            labels: 标签列表
            image_paths: 图像路径列表
        """
        features = []
        labels = []
        image_paths = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="提取特征")
            for batch in pbar:
                if len(batch) == 3:  # (image, label, path)
                    images, batch_labels, batch_paths = batch
                    images = images.to(self.device)
                    
                    # 提取特征
                    if self.model_type == 'vit_b16':
                        # ViT需要特殊处理
                        batch_features = self.model.feature_extractor(images)
                    else:
                        batch_features = self.model.feature_extractor(images)
                    
                    features.append(batch_features.cpu().numpy())
                    labels.extend(batch_labels.numpy())
                    image_paths.extend(batch_paths)
                else:
                    # Triplet数据集
                    anchor, positive, negative, batch_labels = batch
                    anchor = anchor.to(self.device)
                    
                    anchor_feat = self.model.feature_extractor(anchor)
                    features.append(anchor_feat.cpu().numpy())
                    labels.extend(batch_labels.numpy())
        
        features = np.vstack(features)
        labels = np.array(labels)
        
        return features, labels, image_paths
    
    def extract_species_features(self, dataloader, aggregation='mean'):
        """
        提取物种级特征
        
        Args:
            dataloader: 数据加载器
            aggregation: 聚合方式 ('mean' 或 'median')
        
        Returns:
            species_features: 物种特征矩阵 (n_species, feature_dim)
            species_names: 物种名称列表
        """
        # 先提取所有图像的特征
        image_features, image_labels, image_paths = self.extract_features(dataloader)
        
        # 计算物种级特征
        species_features, species_names = calculate_species_features(
            image_features, image_labels, aggregation
        )
        
        return species_features, species_names

def load_trained_model(model_path, config, device):
    """加载训练好的模型"""
    # 创建模型
    model = create_model(
        model_type=config.MODEL_TYPE,
        feature_dim=config.FEATURE_DIM,
        pretrained=False,
        use_triplet=True
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def extract_all_features(config, dataloader, model_path=None):
    """
    提取所有特征的主函数
    
    Args:
        config: 配置对象
        dataloader: 数据加载器
        model_path: 模型路径（如果为None，则使用随机初始化的模型）
    
    Returns:
        features, labels, image_paths
    """
    set_seed(config.SEED)
    
    # 加载或创建模型
    if model_path and os.path.exists(model_path):
        print(f"加载训练好的模型: {model_path}")
        model = load_trained_model(model_path, config, config.DEVICE)
    else:
        print("使用预训练模型（未微调）")
        model = create_model(
            model_type=config.MODEL_TYPE,
            feature_dim=config.FEATURE_DIM,
            pretrained=True,
            use_triplet=True
        )
    
    # 创建特征提取器
    extractor = FeatureExtractor(model, config.DEVICE, config.MODEL_TYPE)
    
    # 提取特征
    features, labels, image_paths = extractor.extract_features(dataloader)
    
    print(f"提取了 {len(features)} 个样本的特征，特征维度: {features.shape[1]}")
    
    return features, labels, image_paths

