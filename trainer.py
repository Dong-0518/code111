"""
训练脚本：支持 “classification + triplet” 联合训练
"""
import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import create_model
from triplet_loss import TripletLoss, HardTripletLoss
from utils import set_seed


def _export_config_snapshot(config) -> Dict[str, Any]:
    """
    Config 是“类属性为主 + 实例属性覆盖”的风格；
    config.__dict__ 往往不包含全部超参，因此这里导出所有大写字段。
    """
    snapshot: Dict[str, Any] = {}
    for k in dir(config):
        if not k.isupper():
            continue
        try:
            v = getattr(config, k)
        except Exception:
            continue
        if callable(v):
            continue
        snapshot[k] = v
    return snapshot


def _infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int | None:
    """尽量从 checkpoint 权重推断 num_classes。"""
    candidates = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 2 and v.shape[0] > 1 and k.endswith(".weight") and ("classifier" in k):
            candidates.append((k, v.shape[0]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0].count("classifier"), len(x[0])), reverse=True)
    return candidates[0][1]


class Trainer:
    """训练器"""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE

        self.model.to(self.device)

        # Triplet loss（注意：hard strategy 在当前仓库仍不是真 hard negative mining）
        strategy = getattr(config, "TRIPLET_SELECTION_STRATEGY", "random")
        if strategy == "hard":
            if getattr(config, "WARN_ON_FAKE_HARD", True):
                print("[WARN] TRIPLET_SELECTION_STRATEGY='hard' 但当前 TripletDataset negative 仍为随机抽样；"
                      "HardTripletLoss 不会自动进行 hard negative mining。建议使用 'random'。")
            self.triplet_criterion = HardTripletLoss(margin=config.MARGIN)
        else:
            self.triplet_criterion = TripletLoss(margin=config.MARGIN)

        # Classification loss（模型带 classifier 且 config 启用时生效）
        self.use_cls = bool(getattr(config, "USE_CLASSIFICATION_LOSS", True))
        self.cls_weight = float(getattr(config, "CLASSIFICATION_WEIGHT", 1.0))
        self.tri_weight = float(getattr(config, "TRIPLET_WEIGHT", 1.0))
        self.cls_criterion = nn.CrossEntropyLoss(
            label_smoothing=float(getattr(config, "LABEL_SMOOTHING", 0.0))
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=float(getattr(config, "WEIGHT_DECAY", 1e-4))
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=1e-6
        )

        # history：同时记录 total/tri/cls
        self.train_total_losses = []
        self.val_total_losses = []
        self.train_triplet_losses = []
        self.val_triplet_losses = []
        self.train_cls_losses = []
        self.val_cls_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float("inf")
        self.early_stopping_patience = int(getattr(config, "EARLY_STOPPING_PATIENCE", 0))
        self.no_improve_epochs = 0

    def _forward(self, anchor, positive, negative):
        """
        兼容两类模型输出：
        - TripletNetwork: (a_feat, p_feat, n_feat)
        - TripletClassificationNetwork: (a_feat, p_feat, n_feat, a_logits, p_logits, n_logits)
        """
        out = self.model(anchor, positive, negative)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            a_feat, p_feat, n_feat = out
            return a_feat, p_feat, n_feat, None, None, None
        if isinstance(out, (tuple, list)) and len(out) == 6:
            a_feat, p_feat, n_feat, a_log, p_log, n_log = out
            return a_feat, p_feat, n_feat, a_log, p_log, n_log
        raise RuntimeError(f"Unexpected model output: type={type(out)}, len={len(out) if hasattr(out,'__len__') else 'NA'}")

    def train_epoch(self) -> Tuple[float, float, float, float]:
        self.model.train()

        total_sum = 0.0
        tri_sum = 0.0
        cls_sum = 0.0
        acc_sum = 0.0
        n = 0
        nb = 0

        pbar = tqdm(self.train_loader, desc="训练中")
        for anchor, positive, negative, labels in pbar:
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            labels = labels.to(self.device).long()

            a_feat, p_feat, n_feat, a_logits, p_logits, _ = self._forward(anchor, positive, negative)

            tri_loss = self.triplet_criterion(a_feat, p_feat, n_feat)

            if self.use_cls and (a_logits is not None) and (p_logits is not None):
                cls_loss = 0.5 * (self.cls_criterion(a_logits, labels) + self.cls_criterion(p_logits, labels))
                loss = self.tri_weight * tri_loss + self.cls_weight * cls_loss

                with torch.no_grad():
                    pred_a = a_logits.argmax(dim=1)
                    acc = (pred_a == labels).float().mean()
            else:
                # fallback：旧 triplet-only
                cls_loss = torch.tensor(0.0, device=self.device)
                loss = self.tri_weight * tri_loss
                with torch.no_grad():
                    dist_pos = F.pairwise_distance(a_feat, p_feat)
                    dist_neg = F.pairwise_distance(a_feat, n_feat)
                    acc = (dist_pos < dist_neg).float().mean()

            self.optimizer.zero_grad()
            loss.backward()
            grad_clip = float(getattr(self.config, "GRAD_CLIP_NORM", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            self.optimizer.step()

            bs = anchor.size(0)
            total_sum += float(loss.item())
            tri_sum += float(tri_loss.item())
            cls_sum += float(cls_loss.item())
            acc_sum += float(acc.item()) * bs
            n += bs
            nb += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "tri": f"{tri_loss.item():.4f}",
                "cls": f"{cls_loss.item():.4f}",
                "acc": f"{acc.item():.4f}",
            })

        return total_sum / max(1, nb), tri_sum / max(1, nb), cls_sum / max(1, nb), acc_sum / max(1, n)

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float, float]:
        self.model.eval()

        total_sum = 0.0
        tri_sum = 0.0
        cls_sum = 0.0
        acc_sum = 0.0
        n = 0
        nb = 0

        pbar = tqdm(self.val_loader, desc="验证中")
        for anchor, positive, negative, labels in pbar:
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            labels = labels.to(self.device).long()

            a_feat, p_feat, n_feat, a_logits, p_logits, _ = self._forward(anchor, positive, negative)

            tri_loss = self.triplet_criterion(a_feat, p_feat, n_feat)

            if self.use_cls and (a_logits is not None) and (p_logits is not None):
                cls_loss = 0.5 * (self.cls_criterion(a_logits, labels) + self.cls_criterion(p_logits, labels))
                loss = self.tri_weight * tri_loss + self.cls_weight * cls_loss
                pred_a = a_logits.argmax(dim=1)
                acc = (pred_a == labels).float().mean()
            else:
                cls_loss = torch.tensor(0.0, device=self.device)
                loss = self.tri_weight * tri_loss
                dist_pos = F.pairwise_distance(a_feat, p_feat)
                dist_neg = F.pairwise_distance(a_feat, n_feat)
                acc = (dist_pos < dist_neg).float().mean()

            bs = anchor.size(0)
            total_sum += float(loss.item())
            tri_sum += float(tri_loss.item())
            cls_sum += float(cls_loss.item())
            acc_sum += float(acc.item()) * bs
            n += bs
            nb += 1

        return total_sum / max(1, nb), tri_sum / max(1, nb), cls_sum / max(1, nb), acc_sum / max(1, n)

    def train(self):
        print(f"开始训练，共 {self.config.NUM_EPOCHS} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型类型: {self.config.MODEL_TYPE}")
        print(f"特征维度: {self.config.FEATURE_DIM}")
        print(f"Margin: {self.config.MARGIN}")
        print(f"联合训练: USE_CLASSIFICATION_LOSS={self.use_cls}, "
              f"CLASSIFICATION_WEIGHT={self.cls_weight}, TRIPLET_WEIGHT={self.tri_weight}")
        print(f"正则化: label_smoothing={getattr(self.config, 'LABEL_SMOOTHING', 0.0)}, "
              f"weight_decay={getattr(self.config, 'WEIGHT_DECAY', 1e-4)}, "
              f"dropout={getattr(self.config, 'CLASSIFIER_DROPOUT', 0.5)}")
        print("-" * 50)

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS}")

            tr_total, tr_tri, tr_cls, tr_acc = self.train_epoch()
            self.train_total_losses.append(tr_total)
            self.train_triplet_losses.append(tr_tri)
            self.train_cls_losses.append(tr_cls)
            self.train_accuracies.append(tr_acc)

            va_total, va_tri, va_cls, va_acc = self.validate()
            self.val_total_losses.append(va_total)
            self.val_triplet_losses.append(va_tri)
            self.val_cls_losses.append(va_cls)
            self.val_accuracies.append(va_acc)

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            print(f"训练: loss={tr_total:.4f} (tri={tr_tri:.4f}, cls={tr_cls:.4f}), acc={tr_acc:.4f}")
            print(f"验证: loss={va_total:.4f} (tri={va_tri:.4f}, cls={va_cls:.4f}), acc={va_acc:.4f}, lr={lr:.6f}")

            if va_total < self.best_val_loss:
                self.best_val_loss = va_total
                self.no_improve_epochs = 0
                if getattr(self.config, "SAVE_MODEL", True):
                    self.save_model(epoch)
                print(f"✓ 保存最佳模型 (val_loss: {va_total:.4f})")
            else:
                self.no_improve_epochs += 1
                if self.early_stopping_patience > 0 and self.no_improve_epochs >= self.early_stopping_patience:
                    print(f"触发早停：验证集 {self.no_improve_epochs} 个 epoch 无改进。")
                    break

        print("\n训练完成！")
        return self.train_total_losses, self.val_total_losses, self.train_accuracies, self.val_accuracies

    def save_model(self, epoch: int):
        model_path = os.path.join(self.config.OUTPUT_DIR, "models", f"{self.config.MODEL_TYPE}_best.pth")

        config_snapshot = _export_config_snapshot(self.config)
        state_dict = self.model.state_dict()

        has_classifier = any(k.startswith("classifier.") for k in state_dict.keys())
        num_classes = config_snapshot.get("NUM_CLASSES") or _infer_num_classes_from_state_dict(state_dict)

        torch.save({
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": {
                "train_total_losses": self.train_total_losses,
                "val_total_losses": self.val_total_losses,
                "train_triplet_losses": self.train_triplet_losses,
                "val_triplet_losses": self.val_triplet_losses,
                "train_cls_losses": self.train_cls_losses,
                "val_cls_losses": self.val_cls_losses,
                "train_accuracies": self.train_accuracies,
                "val_accuracies": self.val_accuracies,
            },
            "config": config_snapshot,
            "model_meta": {
                "has_classifier": has_classifier,
                "num_classes": num_classes,
            },
        }, model_path)


def train_model(config, train_loader, val_loader):
    """训练模型主函数"""
    set_seed(config.SEED)

    num_classes = getattr(config, "NUM_CLASSES", None)
    if num_classes is None:
        raise ValueError("config.NUM_CLASSES 未设置。请在加载数据集后设置 config.NUM_CLASSES = 物种数。")

    model = create_model(
        model_type=config.MODEL_TYPE,
        feature_dim=config.FEATURE_DIM,
        num_classes=num_classes,
        pretrained=True,
        use_triplet=True,
        classifier_hidden_dim=getattr(config, "CLASSIFIER_HIDDEN_DIM", 256),
        classifier_dropout=getattr(config, "CLASSIFIER_DROPOUT", 0.5)
    )

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

    history = {
        "train_total_losses": trainer.train_total_losses,
        "val_total_losses": trainer.val_total_losses,
        "train_triplet_losses": trainer.train_triplet_losses,
        "val_triplet_losses": trainer.val_triplet_losses,
        "train_cls_losses": trainer.train_cls_losses,
        "val_cls_losses": trainer.val_cls_losses,
        "train_accuracies": trainer.train_accuracies,
        "val_accuracies": trainer.val_accuracies,
    }

    return trainer.model, history
