"""
注意力权重提取工具

本模块提供从GraphTransformer模型中提取注意力权重的功能，
用于模型可解释性分析，不修改原始模型代码。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class AttentionExtractor:
    """
    注意力权重提取器
    
    从GraphTransformer模型中提取注意力权重，用于可视化分析。
    复制forward()方法中的节点准备逻辑，确保与训练时一致。
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        初始化注意力提取器
        
        Args:
            model: GraphTransformer模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def extract_attention_weights(self, batch_data: Dict) -> List[torch.Tensor]:
        """
        提取注意力权重
        
        Args:
            batch_data: 批次数据字典
            
        Returns:
            List[torch.Tensor]: 每个注意力层的权重列表
        """
        with torch.no_grad():
            # 节点特征编码
            node_features = self.model.node_encoder(batch_data)
            
            # 边特征编码
            edge_features = self.model.edge_encoder(
                batch_data['edge_index'],
                batch_data['edge_types'],
                batch_data['edge_distances'],
                batch_data.get('edge_attr')
            )
            
            # 根据真实序列长度裁剪节点特征
            seq_lens_tensor = batch_data['seq_lens'].to(device=node_features.device, dtype=torch.long)
            node_lens_tensor = batch_data.get('node_lens')
            if node_lens_tensor is not None:
                node_lens_tensor = node_lens_tensor.to(device=node_features.device, dtype=torch.long)
            else:
                node_lens_tensor = seq_lens_tensor + (1 if self.model.use_global_node else 0)
            
            batch_size = int(seq_lens_tensor.numel())
            hidden_dim = node_features.size(-1)
            max_seq_len = node_features.size(1)
            seq_positions = torch.arange(max_seq_len, device=node_features.device)
            valid_seq_mask = seq_positions.unsqueeze(0) < seq_lens_tensor.unsqueeze(1)
            
            # 处理global_node
            if self.model.use_global_node:
                # 计算global nodes
                state_raw = self.model.node_encoder._encode_state(batch_data, node_features.device)
                state_embed = self.model.node_encoder.state_encoder(state_raw)
                if not self.model.use_state_features:
                    state_embed = state_embed.new_zeros(state_embed.shape)
                env_raw = self.model.node_encoder._encode_environmental(batch_data, node_features.device)
                env_embed = self.model.node_encoder.env_encoder(env_raw)
                if not self.model.use_env_features:
                    env_embed = env_embed.new_zeros(env_embed.shape)
                global_context = torch.cat([state_embed, env_embed], dim=-1)
                global_nodes = self.model.global_node_embedding + self.model.global_node_proj(global_context)
                
                max_node_len = int(node_lens_tensor.max().item()) if batch_size > 0 else 0
                packed_nodes = node_features.new_zeros((batch_size, max_node_len, hidden_dim))
                if max_seq_len > 0:
                    packed_nodes[:, :max_seq_len] = node_features[:, :max_seq_len]
                global_positions = seq_lens_tensor.view(-1, 1, 1).expand(-1, 1, hidden_dim)
                packed_nodes.scatter_(1, global_positions, global_nodes.unsqueeze(1))
                node_positions = torch.arange(max_node_len, device=node_features.device)
                valid_node_mask = node_positions.unsqueeze(0) < node_lens_tensor.unsqueeze(1)
                node_features = packed_nodes[valid_node_mask]
            else:
                node_features = node_features[valid_seq_mask]
            
            # 通过注意力层收集权重
            attention_weights = []
            for gat_layer in self.model.gat_layers:
                weights = gat_layer.get_attention_weights(
                    node_features,
                    batch_data['edge_index'],
                    edge_features,
                )
                attention_weights.append(weights)
                node_features = gat_layer(
                    node_features,
                    batch_data['edge_index'],
                    edge_features,
                )
            
            return attention_weights
    
    def extract_attention_for_sample(self, sample_data: Dict) -> List[torch.Tensor]:
        """
        为单个样本提取注意力权重
        
        Args:
            sample_data: 单个样本数据（需要添加batch维度）
            
        Returns:
            List[torch.Tensor]: 每个注意力层的权重列表
        """
        # 添加batch维度并转换键名
        batch_data = {}
        
        # 处理序列键名（单个样本使用 'sequence'，批处理使用 'sequences'）
        if 'sequence' in sample_data and 'sequences' not in sample_data:
            batch_data['sequences'] = [sample_data['sequence']]
        
        # 处理其他字段
        for key, value in sample_data.items():
            if key == 'sequence':
                continue  # 已经处理过
            
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.unsqueeze(0).to(self.device)
            elif isinstance(value, list):
                # 列表类型保持不变（如 edge_index）
                batch_data[key] = value
            else:
                # 标量类型转换为张量
                if isinstance(value, (int, float)):
                    batch_data[key] = torch.tensor([value], dtype=torch.float32).to(self.device)
                else:
                    batch_data[key] = value
        
        # 确保必要的字段存在
        if 'seq_lens' not in batch_data and 'seq_len' in sample_data:
            batch_data['seq_lens'] = torch.tensor([sample_data['seq_len']], dtype=torch.long).to(self.device)
        
        if 'node_lens' not in batch_data and 'node_len' in sample_data:
            batch_data['node_lens'] = torch.tensor([sample_data['node_len']], dtype=torch.long).to(self.device)
        
        # 确保 charges, pep_masses, intensities 等字段存在（复数形式）
        for singular, plural in [('charge', 'charges'), ('pep_mass', 'pep_masses'), 
                                 ('intensity', 'intensities'), ('nce', 'nces'), ('rt', 'rts')]:
            if singular in sample_data and plural not in batch_data:
                batch_data[plural] = torch.tensor([sample_data[singular]], dtype=torch.float32).to(self.device)
        
        # 确保 state_vars 存在
        if 'state_vars' not in batch_data:
            if 'charge' in sample_data and 'pep_mass' in sample_data and 'intensity' in sample_data:
                state_vars = torch.tensor([
                    [sample_data['charge'], sample_data['pep_mass'], sample_data['intensity']]
                ], dtype=torch.float32).to(self.device)
                batch_data['state_vars'] = state_vars
        
        # 确保 env_vars 存在
        if 'env_vars' not in batch_data:
            if 'nce' in sample_data:
                # 获取环境变量值
                env_value = sample_data.get('env_feature_value', sample_data.get('rt', 0.0))
                env_vars = torch.tensor([
                    [sample_data['nce'], env_value]
                ], dtype=torch.float32).to(self.device)
                batch_data['env_vars'] = env_vars
        
        # 确保 secondary_envs 存在
        if 'secondary_envs' not in batch_data:
            if 'env_feature_value' in sample_data:
                batch_data['secondary_envs'] = torch.tensor([sample_data['env_feature_value']], dtype=torch.float32).to(self.device)
            elif 'rt' in sample_data:
                batch_data['secondary_envs'] = torch.tensor([sample_data['rt']], dtype=torch.float32).to(self.device)
        
        return self.extract_attention_weights(batch_data)
    
    def get_attention_weights_by_layer(self, batch_data: Dict) -> Dict[int, torch.Tensor]:
        """
        按层获取注意力权重
        
        Args:
            batch_data: 批次数据字典
            
        Returns:
            Dict[int, torch.Tensor]: 按层索引的注意力权重字典
        """
        attention_weights = self.extract_attention_weights(batch_data)
        return {i: weights for i, weights in enumerate(attention_weights)}
    
    def get_attention_statistics(self, batch_data: Dict) -> Dict[str, Any]:
        """
        获取注意力权重的统计信息
        
        Args:
            batch_data: 批次数据字典
            
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        attention_weights = self.extract_attention_weights(batch_data)
        
        stats = {
            'num_layers': len(attention_weights),
            'layer_stats': []
        }
        
        for layer_idx, weights in enumerate(attention_weights):
            layer_stat = {
                'layer_index': layer_idx,
                'shape': weights.shape,
                'mean': weights.mean().item(),
                'std': weights.std().item(),
                'min': weights.min().item(),
                'max': weights.max().item(),
                'sparsity': (weights == 0).float().mean().item(),
            }
            stats['layer_stats'].append(layer_stat)
        
        return stats


def extract_attention_weights_from_model(model: nn.Module, 
                                        batch_data: Dict, 
                                        device: torch.device) -> List[torch.Tensor]:
    """
    便捷函数：从模型中提取注意力权重
    
    Args:
        model: GraphTransformer模型
        batch_data: 批次数据字典
        device: 计算设备
        
    Returns:
        List[torch.Tensor]: 每个注意力层的权重列表
    """
    extractor = AttentionExtractor(model, device)
    return extractor.extract_attention_weights(batch_data)