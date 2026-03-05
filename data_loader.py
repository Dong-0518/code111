"""
数据加载和预处理模块
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import glob

class PlantDataset(Dataset):
    """植物图像数据集"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            transform: 图像变换
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image, label, image_path
        except Exception as e:
            print(f"加载图像失败: {image_path}, 错误: {e}")
            # 返回一个黑色图像
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label, image_path

class TripletDataset(Dataset):
    """用于Triplet Loss的数据集"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # 构建标签到索引的映射
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]
        
        # 选择positive（同一物种）
        positive_indices = self.label_to_indices[anchor_label]
        positive_idx = np.random.choice(positive_indices)
        while positive_idx == idx:  # 确保不是同一个图像
            positive_idx = np.random.choice(positive_indices)
        
        # 选择negative（不同物种）
        negative_label = np.random.choice(
            [l for l in self.label_to_indices.keys() if l != anchor_label]
        )
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        
        # 加载图像
        anchor = self._load_image(anchor_path)
        positive = self._load_image(self.image_paths[positive_idx])
        negative = self._load_image(self.image_paths[negative_idx])
        
        return anchor, positive, negative, anchor_label
    
    def _load_image(self, path):
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"加载图像失败: {path}, 错误: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image

def get_transforms(mode='train', image_size=224, model_type='resnet50'):
    """
    获取图像变换
    
    Args:
        mode: 'train' 或 'test'
        image_size: 图像尺寸
        model_type: 模型类型（用于ViT的特殊处理）
    """
    if model_type == 'vit_b16':
        # ViT使用不同的归一化
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                   std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                   std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
            ])
    else:
        # ResNet等CNN模型使用ImageNet归一化
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])

def load_dataset(data_path, image_type='specimen'):
    """
    加载数据集
    
    Args:
        data_path: 数据路径
        image_type: 'specimen' 或 'habitat'
    
    Returns:
        image_paths: 图像路径列表
        labels: 标签列表
        species_names: 物种名称列表
    """
    image_paths = []
    labels = []
    species_names = []
    
    # 获取所有物种文件夹
    if os.path.isdir(data_path):
        species_dirs = [d for d in os.listdir(data_path) 
                       if os.path.isdir(os.path.join(data_path, d))]
    else:
        print(f"数据路径不存在: {data_path}")
        return [], [], []
    
    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
    
    # 遍历每个物种文件夹
    for species_idx, species_dir in enumerate(sorted(species_dirs)):
        species_path = os.path.join(data_path, species_dir)
        
        # 获取该物种的所有图像
        species_images = []
        for ext in image_extensions:
            species_images.extend(glob.glob(os.path.join(species_path, ext)))
        
        if len(species_images) == 0:
            print(f"警告: {species_dir} 文件夹中没有找到图像")
            continue
        
        # 添加到列表
        image_paths.extend(species_images)
        labels.extend([species_idx] * len(species_images))
        species_names.append(species_dir)
        
        print(f"加载物种 {species_dir}: {len(species_images)} 张图像")
    
    print(f"\n总共加载 {len(image_paths)} 张图像，{len(species_names)} 个物种")
    return image_paths, labels, species_names

def create_dataloaders(image_paths, labels, batch_size=32, train_ratio=0.7, 
                      val_ratio=0.2, test_ratio=0.1, use_triplet=False, 
                      image_size=224, num_workers=4, model_type='resnet50'):
    """
    创建数据加载器
    
    Args:
        image_paths: 图像路径列表
        labels: 标签列表
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        use_triplet: 是否使用Triplet数据集
        image_size: 图像尺寸
        num_workers: 数据加载线程数
        model_type: 模型类型（用于选择正确的图像变换）
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=(1 - train_ratio), 
        stratify=labels, random_state=42
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size), 
        stratify=y_temp, random_state=42
    )
    
    print(f"训练集: {len(X_train)} 张图像")
    print(f"验证集: {len(X_val)} 张图像")
    print(f"测试集: {len(X_test)} 张图像")
    
    # 创建数据集
    if use_triplet:
        train_dataset = TripletDataset(X_train, y_train, 
                                      transform=get_transforms('train', image_size, model_type))
        val_dataset = TripletDataset(X_val, y_val, 
                                    transform=get_transforms('test', image_size, model_type))
        # 测试集始终使用PlantDataset，以便于特征提取和评估（不需要Triplet）
        test_dataset = PlantDataset(X_test, y_test, 
                                     transform=get_transforms('test', image_size, model_type))
    else:
        train_dataset = PlantDataset(X_train, y_train, 
                                   transform=get_transforms('train', image_size, model_type))
        val_dataset = PlantDataset(X_val, y_val, 
                                 transform=get_transforms('test', image_size, model_type))
        test_dataset = PlantDataset(X_test, y_test, 
                                  transform=get_transforms('test', image_size, model_type))
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

