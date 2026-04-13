"""
训练脚本
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from models import create_model
from triplet_loss import TripletLoss, HardTripletLoss, BatchHardTripletLoss
from utils import set_seed

class Trainer:
    """训练器"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE
        
        # 移动到设备
        self.model.to(self.device)
        
        # 损失函数
        if config.TRIPLET_SELECTION_STRATEGY == "batch_hard":
            self.criterion = BatchHardTripletLoss(margin=config.MARGIN)
        elif config.TRIPLET_SELECTION_STRATEGY == "hard":
            self.criterion = HardTripletLoss(margin=config.MARGIN)
        else:
            self.criterion = TripletLoss(margin=config.MARGIN)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.NUM_EPOCHS,
            eta_min=1e-6
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="训练中")
        for batch in pbar:
            if self.config.TRIPLET_SELECTION_STRATEGY == "batch_hard":
                # batch: (image, label, path)
                images, labels, _ = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                embeddings = self.model.feature_extractor(images)
                loss, batch_acc = self.criterion(embeddings, labels)
                correct = batch_acc.item() * images.size(0)
                total_correct += correct
                total_samples += images.size(0)
            else:
                anchor, positive, negative, labels = batch
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # 前向传播
                anchor_feat, positive_feat, negative_feat = self.model(
                    anchor, positive, negative
                )
                
                # 计算损失
                loss = self.criterion(anchor_feat, positive_feat, negative_feat)
                
                # 计算准确率 (dist(a,p) < dist(a,n))
                with torch.no_grad():
                    dist_pos = F.pairwise_distance(anchor_feat, positive_feat)
                    dist_neg = F.pairwise_distance(anchor_feat, negative_feat)
                    correct = (dist_pos < dist_neg).float().sum().item()
                    total_correct += correct
                    total_samples += anchor.size(0)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            batch_size = images.size(0) if self.config.TRIPLET_SELECTION_STRATEGY == "batch_hard" else anchor.size(0)
            current_acc = correct / batch_size
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
        
        avg_loss = total_loss / num_batches
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="验证中")
            for batch in pbar:
                if self.config.TRIPLET_SELECTION_STRATEGY == "batch_hard":
                    images, labels, _ = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    embeddings = self.model.feature_extractor(images)
                    loss, batch_acc = self.criterion(embeddings, labels)
                    correct = batch_acc.item() * images.size(0)
                    total_correct += correct
                    total_samples += images.size(0)
                else:
                    anchor, positive, negative, labels = batch
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)
                    
                    # 前向传播
                    anchor_feat, positive_feat, negative_feat = self.model(
                        anchor, positive, negative
                    )
                    
                    # 计算损失
                    loss = self.criterion(anchor_feat, positive_feat, negative_feat)
                    
                    # 计算准确率
                    dist_pos = F.pairwise_distance(anchor_feat, positive_feat)
                    dist_neg = F.pairwise_distance(anchor_feat, negative_feat)
                    correct = (dist_pos < dist_neg).float().sum().item()
                    total_correct += correct
                    total_samples += anchor.size(0)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc
    
    def train(self):
        """训练模型"""
        print(f"开始训练，共 {self.config.NUM_EPOCHS} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型类型: {self.config.MODEL_TYPE}")
        print(f"特征维度: {self.config.FEATURE_DIM}")
        print(f"Margin: {self.config.MARGIN}")
        print("-" * 50)
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS}")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # 验证
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}, 学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.config.SAVE_MODEL:
                    self.save_model(epoch)
                print(f"✓ 保存最佳模型 (验证损失: {val_loss:.4f})")
        
        print("\n训练完成！")
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def save_model(self, epoch):
        """保存模型"""
        model_path = os.path.join(
            self.config.OUTPUT_DIR, 
            "models", 
            f"{self.config.MODEL_TYPE}_best.pth"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config.__dict__
        }, model_path)
    
    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        return checkpoint['epoch']

def train_model(config, train_loader, val_loader):
    """训练模型的主函数"""
    # 设置随机种子
    set_seed(config.SEED)
    
    # 创建模型
    model = create_model(
        model_type=config.MODEL_TYPE,
        feature_dim=config.FEATURE_DIM,
        pretrained=True,
        use_triplet=True
    )
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # 训练
    train_losses, val_losses, train_accuracies, val_accuracies = trainer.train()
    
    return trainer.model, train_losses, val_losses, train_accuracies, val_accuracies
