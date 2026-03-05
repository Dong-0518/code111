"""
模型定义：特征提取器
"""
import torch
import torch.nn as nn
import timm
import torchvision.models as models
from transformers import ViTModel, ViTImageProcessor

class FeatureExtractor(nn.Module):
    """特征提取器基类"""
    
    def __init__(self, model_type='resnet50', feature_dim=512, pretrained=True):
        super(FeatureExtractor, self).__init__()
        self.model_type = model_type
        self.feature_dim = feature_dim
        
        if model_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # 移除最后的全连接层
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            # ResNet50的输出是2048维
            self.fc = nn.Linear(2048, feature_dim)
            
        elif model_type == 'inception_resnet_v2':
            # num_classes=0 会自动帮你砍掉分类头，只输出特征！
            self.backbone = timm.create_model('inception_resnet_v2', pretrained=pretrained, num_classes=0)
            # InceptionResNetV2 提取出的特征默认是 1536 维的
            self.fc = nn.Linear(1536, feature_dim)
            
        elif model_type == 'vit_b16':
            # 使用Hugging Face的ViT模型
            self.backbone = ViTModel.from_pretrained('/data/yutong/models/vit-base-patch16-224')
            # ViT的输出是768维
            self.fc = nn.Linear(768, feature_dim)
            
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def forward(self, x):
        if self.model_type == 'vit_b16':
            # ViT需要特殊处理：输入已经是归一化的tensor
            # 注意：ViT期望输入范围在[-1, 1]或[0, 1]，这里假设已经归一化
            outputs = self.backbone(pixel_values=x)
            # 使用[CLS] token的特征 (pooler_output)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                # 如果没有pooler_output，使用[CLS] token
                features = outputs.last_hidden_state[:, 0, :]
        else:
            # ResNet等CNN模型
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
        
        # 投影到目标特征维度
        features = self.fc(features)
        # L2归一化
        features = nn.functional.normalize(features, p=2, dim=1)
        
        return features

class TripletNetwork(nn.Module):
    """Triplet网络：用于度量学习"""
    
    def __init__(self, feature_extractor):
        super(TripletNetwork, self).__init__()
        self.feature_extractor = feature_extractor
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: anchor图像 (batch_size, 3, H, W)
            positive: positive图像 (batch_size, 3, H, W)
            negative: negative图像 (batch_size, 3, H, W)
        
        Returns:
            anchor_feat, positive_feat, negative_feat
        """
        anchor_feat = self.feature_extractor(anchor)
        positive_feat = self.feature_extractor(positive)
        negative_feat = self.feature_extractor(negative)
        
        return anchor_feat, positive_feat, negative_feat

class ClassificationHead(nn.Module):
    """分类头：用于分类任务"""
    
    def __init__(self, feature_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, features):
        return self.classifier(features)

def create_model(model_type='resnet50', feature_dim=512, num_classes=None, 
                pretrained=True, use_triplet=True):
    """
    创建模型
    
    Args:
        model_type: 模型类型
        feature_dim: 特征维度
        num_classes: 类别数（如果为None，则只创建特征提取器）
        pretrained: 是否使用预训练权重
        use_triplet: 是否用于Triplet Loss
    
    Returns:
        model
    """
    feature_extractor = FeatureExtractor(model_type, feature_dim, pretrained)
    
    if use_triplet:
        model = TripletNetwork(feature_extractor)
    elif num_classes is not None:
        model = nn.Sequential(
            feature_extractor,
            ClassificationHead(feature_dim, num_classes)
        )
    else:
        model = feature_extractor
    
    return model

