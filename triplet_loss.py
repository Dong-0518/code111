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

class BatchHardTripletLoss(nn.Module):
    """Batch-Hard Triplet Loss：在batch内挖掘最难正负样本"""
    
    def __init__(self, margin=0.5):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: 特征向量 (batch_size, feature_dim)，默认已L2归一化
            labels: 类别标签 (batch_size,)
        Returns:
            loss: Batch-Hard Triplet Loss
            acc: batch内triplet准确率（hard_pos < hard_neg）
        """
        if embeddings.size(0) < 2:
            return embeddings.new_tensor(0.0), embeddings.new_tensor(0.0)
        
        labels = labels.view(-1)
        dist_mat = torch.cdist(embeddings, embeddings, p=2)  # (B, B)
        
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        
        pos_mask = label_eq & (~eye)
        neg_mask = ~label_eq
        
        # 若某个样本在batch中没有正样本或负样本，跳过
        valid_pos = pos_mask.any(dim=1)
        valid_neg = neg_mask.any(dim=1)
        valid = valid_pos & valid_neg
        if not valid.any():
            return embeddings.new_tensor(0.0), embeddings.new_tensor(0.0)
        
        pos_dist = dist_mat.clone()
        pos_dist[~pos_mask] = -1e9
        hardest_pos, _ = pos_dist.max(dim=1)
        
        neg_dist = dist_mat.clone()
        neg_dist[~neg_mask] = 1e9
        hardest_neg, _ = neg_dist.min(dim=1)
        
        hardest_pos = hardest_pos[valid]
        hardest_neg = hardest_neg[valid]
        
        loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=0.0).mean()
        acc = (hardest_pos < hardest_neg).float().mean()
        return loss, acc

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

