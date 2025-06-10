import torch
import torch.nn as nn
import torch.nn.functional as F

class HPCModule(nn.Module):
    """
    改进的混合预测编码(Hybrid Predictive Coding)模块
    适用于已经通过Conformer处理的特征
    
    Args:
        input_dim: 输入特征维度
        future_steps: 预测未来的步数
    """
    def __init__(self, input_dim, future_steps=3):
        super(HPCModule, self).__init__()
        self.input_dim = input_dim
        self.future_steps = future_steps
        
        # CPC预测器 - 预测未来的表示用于对比学习
        self.cpc_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU()
            ) for _ in range(future_steps)
        ])
        
        # APC预测器 - 预测未来的表示用于自回归学习
        self.apc_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU()
            ) for _ in range(future_steps)
        ])
            
    def compute_cpc_loss(self, features):
        batch_size, seq_len, dim = features.shape
        device = features.device
        total_loss = 0.0
        valid_steps = 0
        
        for k in range(1, self.future_steps + 1):
            if seq_len - k <= 0:
                continue
                
            # 当前和未来特征
            current = features[:, :-k]
            future = features[:, k:]
            
            # 预测未来
            predicted = self.cpc_predictors[k-1](current)
            
            # 重塑为[B*T, D]
            pred_flat = predicted.reshape(-1, dim)  # [B*T, D]
            future_flat = future.reshape(-1, dim)   # [B*T, D]
            
            # 每个样本与所有未来样本的相似度
            # [B*T, D] x [D, B*T] -> [B*T, B*T]
            similarity = torch.matmul(pred_flat, future_flat.t())
            similarity = similarity / (dim ** 0.5)  # 温度缩放
            
            # 目标：对角线是正样本
            targets = torch.arange(similarity.shape[0], device=device)
            
            # 交叉熵损失(相当于InfoNCE)
            loss = F.cross_entropy(similarity, targets)
            total_loss += loss
            valid_steps += 1
        
        # 避免除零错误
        return total_loss / valid_steps if valid_steps > 0 else 0.0
    
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
            hpc_loss: 混合预测编码总损失
            cpc_loss: 对比预测编码损失
            apc_loss: 自回归预测编码损失
        """
        cpc_loss = self.compute_cpc_loss(features)
        apc_loss = self.compute_apc_loss(features)
        
        # 组合两种损失
        hpc_loss = cpc_loss + apc_loss
        
        return hpc_loss, cpc_loss, apc_loss