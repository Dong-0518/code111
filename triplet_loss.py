"""
Triplet Loss实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """Triplet Loss"""
    
    def __init__(self, margin=0.5):
        """
        Args:
            margin: 边界值
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        计算Triplet Loss
        
        Args:
            anchor: anchor特征 (batch_size, feature_dim)
            positive: positive特征 (batch_size, feature_dim)
            negative: negative特征 (batch_size, feature_dim)
        
        Returns:
            loss: Triplet Loss值
        """
        # 计算距离（使用L2距离）
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet Loss: max(0, d(a,p) - d(a,n) + margin)
        loss = torch.mean(torch.clamp(
            distance_positive - distance_negative + self.margin, 
            min=0.0
        ))
        
        return loss

class HardTripletLoss(nn.Module):
    """Hard Triplet Loss：选择最难的负样本"""
    
    def __init__(self, margin=0.5):
        super(HardTripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        计算Hard Triplet Loss
        
        Args:
            anchor: anchor特征 (batch_size, feature_dim)
            positive: positive特征 (batch_size, feature_dim)
            negative: negative特征 (batch_size, feature_dim)
        
        Returns:
            loss: Hard Triplet Loss值
        """
        # 计算所有anchor到positive的距离
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        
        # 计算所有anchor到negative的距离
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # 对于每个anchor，选择最难的negative（距离最小的）
        # 这里假设batch中的negative已经是hard negative
        loss = torch.mean(torch.clamp(
            distance_positive - distance_negative + self.margin, 
            min=0.0
        ))
        
        return loss

def select_hard_negatives(anchor_features, negative_features, k=1):
    """
    选择hard negative样本
    
    Args:
        anchor_features: anchor特征 (batch_size, feature_dim)
        negative_features: 所有negative特征 (n_negatives, feature_dim)
        k: 选择的hard negative数量
    
    Returns:
        selected_negatives: 选择的hard negative特征 (batch_size, feature_dim)
    """
    # 计算所有anchor到所有negative的距离
    distances = torch.cdist(anchor_features, negative_features, p=2)  # (batch_size, n_negatives)
    
    # 选择距离最小的k个（hard negative）
    _, indices = torch.topk(distances, k, dim=1, largest=False)
    
    # 选择对应的negative特征
    selected_negatives = negative_features[indices.squeeze()]
    
    return selected_negatives

