import torch
import torch.nn as nn
import torch.nn.functional as F

class APCModule(nn.Module):
    def __init__(self, input_dim, future_steps=3):
        super(APCModule, self).__init__()
        self.input_dim = input_dim
        self.future_steps = future_steps
         
        # APC预测器 - 预测未来的表示用于自回归学习
        self.apc_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU()
            ) for _ in range(future_steps)
        ])
            
    def compute_apc_loss(self, features):
        """
        计算自回归预测编码(APC)损失
        
        Args:
            features: 特征序列 [batch_size, seq_len, hidden_dim]
            
        Returns:
            apc_loss: APC损失
        """
        batch_size, seq_len, dim = features.shape
        total_loss = 0.0
        
        for k in range(1, self.future_steps + 1):
            if seq_len - k <= 0:
                continue
                
            # 当前时间步: [batch_size, seq_len-k, hidden_dim]
            current = features[:, :-k]
            
            # 未来时间步: [batch_size, seq_len-k, hidden_dim]
            future = features[:, k:]
            
            # 使用线性预测器预测未来
            predicted = self.apc_predictors[k-1](current)
            
            # 计算L1损失
            step_loss = F.l1_loss(predicted, future)
            total_loss += step_loss
            
        return total_loss / self.future_steps if self.future_steps > 0 else 0.0
    
    def forward(self, features):
        """
        计算HPC损失
        
        Args:
            features: 模型特征表示 [batch_size, seq_len, dim]
            
        Returns:
            apc_loss: 自回归预测编码损失
        """
        apc_loss = self.compute_apc_loss(features)
        
        return apc_loss
    
class CrossModalAPCModule(nn.Module):
    """
    跨模态APC模块：使用流式表示预测非流式的未来时间步特征
    """
    def __init__(self, input_dim, future_steps=3):
        super(CrossModalAPCModule, self).__init__()
        self.input_dim = input_dim
        self.future_steps = future_steps
         
        # 预测器 - 使用流式特征预测非流式的未来特征
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(future_steps)
        ])
            
    def compute_cross_modal_apc_loss(self, stream_features, nonstream_features):
        """
        计算跨模态APC损失：使用流式特征预测非流式的未来时间步
        
        Args:
            stream_features: 流式特征序列 [batch_size, seq_len, hidden_dim]
            nonstream_features: 非流式特征序列 [batch_size, seq_len, hidden_dim]
            
        Returns:
            apc_loss: 跨模态APC损失
        """
        batch_size, seq_len, dim = stream_features.shape
        total_loss = 0.0
        
        for k in range(1, self.future_steps + 1):
            if seq_len - k <= 0:
                continue
                
            # 当前时间步的流式特征: [batch_size, seq_len-k, hidden_dim]
            current_stream = stream_features[:, :-k]
            
            # 未来时间步的非流式特征作为目标: [batch_size, seq_len-k, hidden_dim]
            future_nonstream = nonstream_features[:, k:]
            
            # 使用流式特征预测非流式的未来特征
            predicted = self.predictors[k-1](current_stream)
            
            # 计算L1损失
            step_loss = F.l1_loss(predicted, future_nonstream.detach())
            total_loss += step_loss
            
        return total_loss / self.future_steps if self.future_steps > 0 else 0.0
    
    def forward(self, stream_features, nonstream_features):
        """
        计算跨模态APC损失
        
        Args:
            stream_features: 流式模型特征表示 [batch_size, seq_len, dim]
            nonstream_features: 非流式模型特征表示 [batch_size, seq_len, dim]
            
        Returns:
            cross_modal_apc_loss: 跨模态自回归预测编码损失
        """
        cross_modal_apc_loss = self.compute_cross_modal_apc_loss(stream_features, nonstream_features)
        
        return cross_modal_apc_loss