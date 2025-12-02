"""
训练模块

本模块包含图神经网络训练相关的类和函数。
"""

from .trainer import Trainer
from .loss_functions import MultiLabelLoss
from .metrics import MultiLabelMetrics

__all__ = ['Trainer', 'MultiLabelLoss', 'MultiLabelMetrics']
