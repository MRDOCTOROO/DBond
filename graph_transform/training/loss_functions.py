"""
损失函数模块

本模块包含多标签分类的损失函数实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np


class MultiLabelLoss(nn.Module):
    """多标签分类损失函数"""
    
    def __init__(self, config: Dict[str, Any]):
        super(MultiLabelLoss, self).__init__()
        
        self.config = config
        self.main_loss = config.get('main_loss', 'binary_cross_entropy')
        self.use_auxiliary_losses = config.get('use_auxiliary_losses', False)
        self.handle_imbalance = config.get('handle_imbalance', False)
        
        # 主要损失函数
        if self.main_loss == 'binary_cross_entropy':
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif self.main_loss == 'focal':
            self.criterion = FocalLoss(config)
        elif self.main_loss == 'dice':
            self.criterion = DiceLoss(config)
        else:
            raise ValueError(f"Unknown main loss type: {self.main_loss}")
        
        # 辅助损失权重
        if self.use_auxiliary_losses:
            self.auxiliary_weights = config.get('auxiliary_loss_weights', {})
            self.focal_loss = FocalLoss(config)
            self.dice_loss = DiceLoss(config)
        
        # 类别不平衡处理
        if self.handle_imbalance:
            self.imbalance_strategy = config.get('imbalance_strategy', 'focal')
            if self.imbalance_strategy == 'weighted':
                self._calculate_class_weights()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        Args:
            predictions: 模型预测 [batch_size, num_classes]
            targets: 真实标签 [batch_size, num_classes]
            
        Returns:
            torch.Tensor: 损失值
        """
        # 主要损失
        if self.main_loss == 'binary_cross_entropy':
            loss = self.criterion(predictions, targets)
            
            # 处理类别不平衡
            if self.handle_imbalance and self.imbalance_strategy == 'weighted':
                if hasattr(self, 'class_weights'):
                    loss = loss * self.class_weights.to(loss.device)
            
            loss = loss.mean()
        
        else:
            loss = self.criterion(predictions, targets)
        
        # 辅助损失
        if self.use_auxiliary_losses:
            auxiliary_losses = []
            
            if 'focal' in self.auxiliary_weights:
                focal_loss = self.focal_loss(predictions, targets)
                auxiliary_losses.append(self.auxiliary_weights['focal'] * focal_loss)
            
            if 'dice' in self.auxiliary_weights:
                dice_loss = self.dice_loss(predictions, targets)
                auxiliary_losses.append(self.auxiliary_weights['dice'] * dice_loss)
            
            if auxiliary_losses:
                loss = loss + sum(auxiliary_losses)
        
        return loss
    
    def _calculate_class_weights(self):
        """计算类别权重"""
        # 这里可以根据数据集统计信息计算权重
        # 简化实现，使用默认权重
        num_classes = self.config.get('num_classes', 20)
        self.class_weights = torch.ones(num_classes)


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, config: Dict[str, Any]):
        super(FocalLoss, self).__init__()
        
        self.alpha = config.get('focal_loss_alpha', 0.25)
        self.gamma = config.get('focal_loss_gamma', 2.0)
        self.reduction = 'mean'
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            inputs: 模型预测 [batch_size, num_classes]
            targets: 真实标签 [batch_size, num_classes]
            
        Returns:
            torch.Tensor: Focal Loss
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class DiceLoss(nn.Module):
    """Dice Loss用于多标签分类"""
    
    def __init__(self, config: Dict[str, Any]):
        super(DiceLoss, self).__init__()
        
        self.smooth = config.get('dice_smooth', 1e-6)
        self.reduction = 'mean'
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Dice Loss
        
        Args:
            inputs: 模型预测 [batch_size, num_classes]
            targets: 真实标签 [batch_size, num_classes]
            
        Returns:
            torch.Tensor: Dice Loss
        """
        # 应用sigmoid到预测
        inputs = torch.sigmoid(inputs)
        
        # 展平张量
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss
        elif self.reduction == 'sum':
            return dice_loss * inputs.size(0)
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self, config: Dict[str, Any]):
        super(CombinedLoss, self).__init__()
        
        self.weights = config.get('loss_weights', {'bce': 0.5, 'focal': 0.3, 'dice': 0.2})
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(config)
        self.dice_loss = DiceLoss(config)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            inputs: 模型预测 [batch_size, num_classes]
            targets: 真实标签 [batch_size, num_classes]
            
        Returns:
            torch.Tensor: 组合损失
        """
        bce = self.bce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        total_loss = (
            self.weights['bce'] * bce +
            self.weights['focal'] * focal +
            self.weights['dice'] * dice
        )
        
        return total_loss
