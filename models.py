"""
模型定义：特征提取器 + (Triplet / Classification) 组合网络
"""
import torch
import torch.nn as nn
import timm
import torchvision.models as models
from transformers import ViTModel


class FeatureExtractor(nn.Module):
    """特征提取器：输出 L2-normalized embedding（用于度量学习）"""

    def __init__(self, model_type='resnet50', feature_dim=512, pretrained=True):
        super().__init__()
        self.model_type = model_type
        self.feature_dim = feature_dim

        if model_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.fc = nn.Linear(2048, feature_dim)

        elif model_type == 'inception_resnet_v2':
            self.backbone = timm.create_model('inception_resnet_v2', pretrained=pretrained, num_classes=0)
            self.fc = nn.Linear(1536, feature_dim)

        elif model_type == 'vit_b16':
            self.backbone = ViTModel.from_pretrained('/data/yutong/models/vit-base-patch16-224')
            self.fc = nn.Linear(768, feature_dim)

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def forward(self, x):
        if self.model_type == 'vit_b16':
            outputs = self.backbone(pixel_values=x)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0, :]
        else:
            features = self.backbone(x)
            features = features.view(features.size(0), -1)

        features = self.fc(features)
        features = nn.functional.normalize(features, p=2, dim=1)
        return features


class TripletNetwork(nn.Module):
    """Triplet 网络：仅输出 embedding（向后兼容）"""

    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, anchor, positive, negative):
        anchor_feat = self.feature_extractor(anchor)
        positive_feat = self.feature_extractor(positive)
        negative_feat = self.feature_extractor(negative)
        return anchor_feat, positive_feat, negative_feat


class ClassificationHead(nn.Module):
    """分类头：embedding -> logits"""

    def __init__(self, feature_dim, num_classes, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.classifier(features)


# ======= 仓库中未发现，建议新增：Triplet + Classification 联合网络 =======
class TripletClassificationNetwork(nn.Module):
    """
    联合网络：Triplet + Classification

    forward(anchor, positive, negative) 返回：
        (anchor_feat, positive_feat, negative_feat,
         anchor_logits, positive_logits, negative_logits)
    """

    def __init__(self, feature_extractor, num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = ClassificationHead(feature_extractor.feature_dim, num_classes)

    def _encode(self, x):
        feat = self.feature_extractor(x)
        logits = self.classifier(feat)
        return feat, logits

    def forward(self, anchor, positive, negative):
        a_feat, a_log = self._encode(anchor)
        p_feat, p_log = self._encode(positive)
        n_feat, n_log = self._encode(negative)
        return a_feat, p_feat, n_feat, a_log, p_log, n_log


def create_model(model_type='resnet50', feature_dim=512, num_classes=None,
                 pretrained=True, use_triplet=True):
    """
    创建模型：

    - use_triplet=True 且 num_classes 给定：TripletClassificationNetwork（联合训练）
    - use_triplet=True 且 num_classes=None：TripletNetwork（旧行为）
    - use_triplet=False 且 num_classes 给定：普通分类网络
    - use_triplet=False 且 num_classes=None：仅 FeatureExtractor
    """
    feature_extractor = FeatureExtractor(model_type, feature_dim, pretrained)

    if use_triplet and (num_classes is not None):
        return TripletClassificationNetwork(feature_extractor, num_classes)
    if use_triplet:
        return TripletNetwork(feature_extractor)
    if num_classes is not None:
        return nn.Sequential(feature_extractor, ClassificationHead(feature_dim, num_classes))
    return feature_extractor
