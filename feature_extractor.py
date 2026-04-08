"""
特征提取模块
"""
import os
import torch
import numpy as np
from tqdm import tqdm

from models import create_model
from utils import set_seed, calculate_species_features


class FeatureExtractor:
    """特征提取器（对训练好的 TripletNetwork / TripletClassificationNetwork 取 embedding）"""

    def __init__(self, model, device, model_type='resnet50'):
        self.model = model
        self.device = device
        self.model_type = model_type
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def extract_features(self, dataloader):
        """
        提取特征

        Returns:
            features: (n_samples, feature_dim)
            labels: (n_samples,)
            image_paths: list[str]（如果 dataloader 返回 path）
        """
        features = []
        labels = []
        image_paths = []

        pbar = tqdm(dataloader, desc="提取特征")
        for batch in pbar:
            # PlantDataset: (image, label, path)
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                images, batch_labels, batch_paths = batch
                images = images.to(self.device)

                batch_features = self.model.feature_extractor(images)

                features.append(batch_features.cpu().numpy())
                labels.extend(batch_labels.numpy())
                image_paths.extend(batch_paths)
                continue

            # TripletDataset: (anchor, positive, negative, label)
            if isinstance(batch, (tuple, list)) and len(batch) == 4:
                anchor, _, _, batch_labels = batch
                anchor = anchor.to(self.device)

                anchor_feat = self.model.feature_extractor(anchor)
                features.append(anchor_feat.cpu().numpy())
                labels.extend(batch_labels.numpy())
                continue

            raise ValueError(f"未知 batch 格式: type={type(batch)}, len={len(batch) if hasattr(batch,'__len__') else 'NA'}")

        features = np.vstack(features) if len(features) > 0 else np.empty((0, 0))
        labels = np.array(labels)

        return features, labels, image_paths

    def extract_species_features(self, dataloader, aggregation='mean', all_species_names=None):
        """
        提取物种级特征

        Args:
            dataloader: 数据加载器（通常是 PlantDataset 的 test_loader）
            aggregation: 'mean' 或 'median'
            all_species_names: 可选，label->name 映射表（list[str]）

        Returns:
            species_features: (n_species, feature_dim)
            species_names: list[str]
        """
        image_features, image_labels, _ = self.extract_features(dataloader)

        species_features, species_names = calculate_species_features(
            image_features,
            image_labels,
            all_species_names=all_species_names,
            aggregation=aggregation
        )
        return species_features, species_names


def _infer_num_classes_from_state_dict(state_dict):
    # 从 classifier 最后一层权重推断 num_classes
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and k.endswith(".weight") and ("classifier" in k) and v.ndim == 2:
            return int(v.shape[0])
    return None


def load_trained_model(model_path, config, device):
    """
    加载训练好的模型（兼容 Triplet-only 与 Triplet+Classification checkpoint）
    """
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    has_classifier = any(k.startswith("classifier.") for k in state_dict.keys())

    num_classes = None
    if isinstance(checkpoint.get("model_meta"), dict):
        num_classes = checkpoint["model_meta"].get("num_classes")

    if num_classes is None and has_classifier:
        num_classes = _infer_num_classes_from_state_dict(state_dict)

    model = create_model(
        model_type=config.MODEL_TYPE,
        feature_dim=config.FEATURE_DIM,
        num_classes=num_classes if has_classifier else None,
        pretrained=False,
        use_triplet=True
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def extract_all_features(config, dataloader, model_path=None):
    """
    提取所有特征的主函数
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

    extractor = FeatureExtractor(model, config.DEVICE, config.MODEL_TYPE)
    features, labels, image_paths = extractor.extract_features(dataloader)

    print(f"提取了 {len(features)} 个样本的特征，特征维度: {features.shape[1]}")
    return features, labels, image_paths
