import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class CPCModule(nn.Module):
    """
    Args:
        input_dim: 输入特征维度
        future_steps: 预测未来的步数
        negative_samples: 对比学习中使用的负样本数量
    """
    def __init__(self, input_dim, future_steps=3, negative_samples=10):
        super(CPCModule, self).__init__()
        self.input_dim = input_dim
        self.future_steps = future_steps
        self.negative_samples = negative_samples
        
        # CPC预测器 - 预测未来的表示用于对比学习
        self.cpc_predictors = nn.ModuleList([
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
    

    def forward(self, features):
        """
        计算CPC损失
        
        Args:
            features: 模型特征表示 [batch_size, seq_len, dim]
            
        Returns:
            cpc_loss: 对比预测编码损失
        """
        cpc_loss = self.compute_cpc_loss(features)

        return cpc_loss
