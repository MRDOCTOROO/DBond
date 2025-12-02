"""
模型工具函数

本文件包含了图神经网络模型的辅助函数和工具类。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math


class ModelConfig:
    """模型配置类"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        # 基础配置
        self.hidden_dim = 256
        self.num_attention_heads = 8
        self.dropout = 0.1
        self.leaky_relu_slope = 0.2
        
        # 图配置
        self.use_edge_features = True
        self.edge_dim = 32
        self.concat_heads = True
        
        # 网络结构配置
        self.num_gcn_layers = 3
        self.num_gat_layers = 2
        self.num_hierarchical_layers = 2
        
        # 序列配置
        self.max_seq_len = 100
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.pad_char = "U"
        
        # 特征维度配置
        self.aa_embedding_dim = 64
        self.position_embedding_dim = 32
        self.physicochemical_dim = 32
        self.num_physicochemical_features = 4
        self.num_env_features = 5
        
        # 边配置
        self.edge_types = ['sequence', 'distance', 'functional']
        self.edge_embedding_dim = 16
        self.distance_embedding_dim = 16
        self.max_distance = 10
        
        # 输出配置
        self.num_classes = 20
        
        # 其他配置
        self.aggregator_type = 'mean'
        self.diffusion_steps = 3
        
        # 更新配置
        if config_dict:
            self.update_config(config_dict)
    
    def update_config(self, config_dict: Dict):
        """更新配置参数"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter: {key}")
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        config_dict = {}
        for attr in dir(self):
            if not attr.startswith('_'):
                value = getattr(self, attr)
                if not callable(value):
                    config_dict[attr] = value
        return config_dict


class WeightInitialization:
    """权重初始化工具类"""
    
    @staticmethod
    def xavier_uniform(module: nn.Module):
        """Xavier均匀初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
    
    @staticmethod
    def xavier_normal(module: nn.Module):
        """Xavier正态初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
    
    @staticmethod
    def kaiming_uniform(module: nn.Module):
        """Kaiming均匀初始化"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    @staticmethod
    def kaiming_normal(module: nn.Module):
        """Kaiming正态初始化"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    @staticmethod
    def orthogonal(module: nn.Module):
        """正交初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    @staticmethod
    def initialize_model(model: nn.Module, method: str = 'xavier_uniform'):
        """初始化整个模型"""
        init_methods = {
            'xavier_uniform': WeightInitialization.xavier_uniform,
            'xavier_normal': WeightInitialization.xavier_normal,
            'kaiming_uniform': WeightInitialization.kaiming_uniform,
            'kaiming_normal': WeightInitialization.kaiming_normal,
            'orthogonal': WeightInitialization.orthogonal
        }
        
        if method not in init_methods:
            raise ValueError(f"Unknown initialization method: {method}")
        
        init_method = init_methods[method]
        model.apply(init_method)


class ActivationFunctions:
    """激活函数集合"""
    
    @staticmethod
    def gelu(x: torch.Tensor) -> torch.Tensor:
        """GELU激活函数"""
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        """Swish激活函数"""
        return x * torch.sigmoid(x)
    
    @staticmethod
    def mish(x: torch.Tensor) -> torch.Tensor:
        """Mish激活函数"""
        return x * torch.tanh(F.softplus(x))
    
    @staticmethod
    def leaky_relu_custom(x: torch.Tensor, negative_slope: float = 0.2) -> torch.Tensor:
        """自定义LeakyReLU"""
        return F.leaky_relu(x, negative_slope=negative_slope)
    
    @staticmethod
    def get_activation(name: str) -> callable:
        """获取激活函数"""
        activations = {
            'relu': F.relu,
            'gelu': ActivationFunctions.gelu,
            'swish': ActivationFunctions.swish,
            'mish': ActivationFunctions.mish,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': ActivationFunctions.leaky_relu_custom
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation function: {name}")
        
        return activations[name]


class Regularization:
    """正则化工具类"""
    
    @staticmethod
    def l1_regularization(model: nn.Module, alpha: float = 0.01) -> torch.Tensor:
        """L1正则化"""
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return alpha * l1_loss
    
    @staticmethod
    def l2_regularization(model: nn.Module, alpha: float = 0.01) -> torch.Tensor:
        """L2正则化"""
        l2_loss = 0
        for param in model.parameters():
            l2_loss += torch.sum(param ** 2)
        return alpha * l2_loss
    
    @staticmethod
    def elastic_net_regularization(model: nn.Module, 
                                  l1_alpha: float = 0.01,
                                  l2_alpha: float = 0.01) -> torch.Tensor:
        """弹性网络正则化"""
        l1_loss = Regularization.l1_regularization(model, l1_alpha)
        l2_loss = Regularization.l2_regularization(model, l2_alpha)
        return l1_loss + l2_loss


class GraphUtils:
    """图处理工具类"""
    
    @staticmethod
    def build_adjacency_matrix(edge_index: torch.Tensor, 
                              num_nodes: int,
                              add_self_loops: bool = True) -> torch.Tensor:
        """构建邻接矩阵"""
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        
        row, col = edge_index
        adj[row, col] = 1.0
        adj[col, row] = 1.0  # 对称化
        
        if add_self_loops:
            adj = adj + torch.eye(num_nodes, device=edge_index.device)
        
        return adj
    
    @staticmethod
    def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
        """归一化邻接矩阵"""
        # 计算度矩阵
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
        
        # 归一化
        d_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_normalized = torch.matmul(torch.matmul(d_inv_sqrt, adj), d_inv_sqrt)
        
        return adj_normalized
    
    @staticmethod
    def compute_graph_laplacian(edge_index: torch.Tensor,
                               num_nodes: int,
                               normalized: bool = True) -> torch.Tensor:
        """计算图拉普拉斯矩阵"""
        adj = GraphUtils.build_adjacency_matrix(edge_index, num_nodes)
        
        if normalized:
            adj = GraphUtils.normalize_adjacency(adj)
        
        degree = torch.sum(adj, dim=1)
        laplacian = torch.diag(degree) - adj
        
        return laplacian
    
    @staticmethod
    def extract_subgraph(node_features: torch.Tensor,
                        edge_index: torch.Tensor,
                        subgraph_nodes: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取子图"""
        # 创建节点映射
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}
        
        # 提取节点特征
        subgraph_features = node_features[subgraph_nodes]
        
        # 提取边
        subgraph_edges = []
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src.item() in node_map and dst.item() in node_map:
                new_src = node_map[src.item()]
                new_dst = node_map[dst.item()]
                subgraph_edges.append([new_src, new_dst])
        
        if subgraph_edges:
            subgraph_edge_index = torch.tensor(subgraph_edges).t().contiguous()
        else:
            subgraph_edge_index = torch.empty(2, 0, dtype=torch.long)
        
        return subgraph_features, subgraph_edge_index


class ModelProfiler:
    """模型分析工具"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """计算模型参数数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """获取模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / (1024 ** 2)  # 转换为MB
    
    @staticmethod
    def profile_model(model: nn.Module, 
                    input_data: Dict,
                    device: torch.device) -> Dict[str, Any]:
        """分析模型性能"""
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_data)
        
        # 测量推理时间
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            for _ in range(100):
                _ = model(input_data)
            end_time.record()
            
            torch.cuda.synchronize()
            avg_time = (start_time.elapsed_time(end_time) / 100) / 1000  # 转换为秒
        
        # 内存使用
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(input_data)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        else:
            peak_memory = 0
        
        return {
            'parameters': ModelProfiler.count_parameters(model),
            'model_size_mb': ModelProfiler.get_model_size(model),
            'avg_inference_time_s': avg_time,
            'peak_memory_mb': peak_memory
        }


class CheckpointManager:
    """检查点管理工具"""
    
    @staticmethod
    def save_checkpoint(model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       loss: float,
                       metrics: Dict,
                       filepath: str,
                       is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'config': model.config.to_dict() if hasattr(model, 'config') else None
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = filepath.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_filepath)
    
    @staticmethod
    def load_checkpoint(filepath: str,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       device: torch.device = None) -> Dict:
        """加载检查点"""
        if device is None:
            device = next(model.parameters()).device
        
        checkpoint = torch.load(filepath, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint


class LearningRateScheduler:
    """学习率调度器工具"""
    
    @staticmethod
    def cosine_annealing_with_warmup(optimizer: torch.optim.Optimizer,
                                   warmup_epochs: int,
                                   max_epochs: int,
                                   min_lr: float = 0.0):
        """余弦退火与预热"""
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                # 预热阶段
                return float(current_epoch) / float(max(1, warmup_epochs))
            else:
                # 余弦退火阶段
                progress = float(current_epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    @staticmethod
    def step_with_warmup(optimizer: torch.optim.Optimizer,
                       warmup_epochs: int,
                       step_size: int,
                       gamma: float = 0.1):
        """步长衰减与预热"""
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                # 预热阶段
                return float(current_epoch) / float(max(1, warmup_epochs))
            else:
                # 步长衰减阶段
                decay_epoch = current_epoch - warmup_epochs
                return gamma ** (decay_epoch // step_size)
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
