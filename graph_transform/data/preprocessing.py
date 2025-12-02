"""
数据预处理模块

本模块包含序列和图数据的预处理功能。
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import re


class SequencePreprocessor:
    """序列预处理器"""
    
    def __init__(self, config):
        self.config = config
        self.alphabet = config.alphabet
        self.max_seq_len = config.max_seq_len
    
    def preprocess_sequence(self, sequence: str) -> str:
        """
        预处理氨基酸序列
        
        Args:
            sequence: 原始氨基酸序列
            
        Returns:
            str: 预处理后的序列
        """
        # 转换为大写
        sequence = sequence.upper()
        
        # 移除非标准氨基酸字符
        sequence = re.sub(f'[^{self.alphabet}]', '', sequence)
        
        # 截断到最大长度
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
        
        return sequence
    
    def encode_sequence(self, sequence: str) -> List[int]:
        """
        将序列编码为数字索引
        
        Args:
            sequence: 氨基酸序列
            
        Returns:
            List[int]: 编码后的序列
        """
        char_to_idx = {char: idx + 1 for idx, char in enumerate(self.alphabet)}
        char_to_idx['U'] = 0  # padding字符
        
        encoded = []
        for char in sequence:
            if char in char_to_idx:
                encoded.append(char_to_idx[char])
            else:
                encoded.append(0)  # 未知字符用padding
        
        return encoded
    
    def decode_sequence(self, encoded_sequence: List[int]) -> str:
        """
        将数字索引解码为序列
        
        Args:
            encoded_sequence: 编码的序列
            
        Returns:
            str: 解码后的序列
        """
        idx_to_char = {idx + 1: char for idx, char in enumerate(self.alphabet)}
        idx_to_char[0] = 'U'  # padding字符
        
        sequence = ''
        for idx in encoded_sequence:
            if idx in idx_to_char:
                sequence += idx_to_char[idx]
            else:
                sequence += 'U'  # 未知索引
        
        return sequence
    
    def batch_preprocess(self, sequences: List[str]) -> List[str]:
        """
        批量预处理序列
        
        Args:
            sequences: 序列列表
            
        Returns:
            List[str]: 预处理后的序列列表
        """
        return [self.preprocess_sequence(seq) for seq in sequences]


class DataPreprocessor:
    """通用数据预处理器"""
    
    def __init__(self, config):
        self.config = config
        self.sequence_preprocessor = SequencePreprocessor(config)
    
    def preprocess_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理单个样本
        
        Args:
            sample: 原始样本数据
            
        Returns:
            Dict[str, Any]: 预处理后的样本
        """
        processed_sample = sample.copy()
        
        # 预处理序列
        if 'sequence' in processed_sample:
            processed_sample['sequence'] = self.sequence_preprocessor.preprocess_sequence(
                processed_sample['sequence']
            )
        
        # 标准化数值特征
        numeric_features = ['charge', 'pep_mass', 'nce', 'rt', 'fbr']
        for feature in numeric_features:
            if feature in processed_sample:
                processed_sample[feature] = self._normalize_feature(
                    processed_sample[feature], feature
                )
        
        return processed_sample
    
    def _normalize_feature(self, value: float, feature_name: str) -> float:
        """
        标准化特征值
        
        Args:
            value: 特征值
            feature_name: 特征名称
            
        Returns:
            float: 标准化后的值
        """
        # 简单的标准化策略，可以根据需要调整
        normalization_ranges = {
            'charge': (1, 5),      # 电荷范围
            'pep_mass': (500, 3000),  # 肽段质量范围
            'nce': (10, 50),         # 碰撞能量范围
            'rt': (0, 100),          # 保留时间范围
            'fbr': (0, 1)            # 碎裂比例范围
        }
        
        if feature_name in normalization_ranges:
            min_val, max_val = normalization_ranges[feature_name]
            if max_val > min_val:
                return (value - min_val) / (max_val - min_val)
        
        return value
    
    def batch_preprocess(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量预处理样本
        
        Args:
            samples: 样本列表
            
        Returns:
            List[Dict[str, Any]]: 预处理后的样本列表
        """
        return [self.preprocess_sample(sample) for sample in samples]
