import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any, Union
from .dual_mode_cache_resnet_conformer import DualModeResNet, DualModeResNetBlock, DualModeConformerBlock
from .APC import APCModule

class DualModeResnetConformerAPC(nn.Module):
    """
    集成了HPC损失的双模态SELD模型
    """
    def __init__(
        self, 
        in_channel=7, 
        in_dim=64, 
        out_dim=39,
        att_context_size=[100, 49], 
        num_conformer_layer=8,
        encoder_dim=256,
        apc_future_steps=5,
    ):
        super().__init__()
        # 双模态ResNet
        self.resnet = DualModeResNet(DualModeResNetBlock, [2, 2, 2, 2], in_channel=in_channel)
        
        # 编码器维度
        embedding_dim = in_dim // 32 * 256
        self.encoder_dim = encoder_dim
        self.in_ch = in_channel
        self.in_dim = in_dim
        self.att_context_size = att_context_size
        self.cache_past_len = self.att_context_size[0]
        
        # 投影层
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.encoder_dim),
            nn.Dropout(p=0.05),
        )
        
        # 双模态Conformer层
        self.conformer_layers = nn.ModuleList([
            DualModeConformerBlock(
                dim=self.encoder_dim,
                dim_head=32,
                heads=8,
                ff_mult=2,
                conv_expansion_factor=2,
                conv_kernel_size=7,
                attn_dropout=0.1,
                ff_dropout=0.1,
                conv_dropout=0.1,
                att_context_size=att_context_size
            ) for _ in range(num_conformer_layer)
        ])
        
        # 时间维度池化
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
        
        # 输出层
        self.sed_out_layer = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(self.encoder_dim, 13),
            nn.Sigmoid()
        )
        
        self.out_layer = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(self.encoder_dim, out_dim),
            nn.Tanh()
        )
        
        # HPC损失模块 - 为流式和非流式表示分别创建
        self.stream_apc = APCModule(
            input_dim=self.encoder_dim,
            future_steps=apc_future_steps,
        )
        
        self.nonstream_apc = APCModule(
            input_dim=self.encoder_dim,
            future_steps=apc_future_steps,
        )

    def get_initial_cache_resnet(self, batch_size=1):
        """初始化ResNet的缓存"""
        return self.resnet.get_initial_cache(batch_size)

    def get_initial_cache_conformer(self, batch_size=1):
        """初始化Conformer的缓存"""
        caches = []
        device = next(self.parameters()).device
        for layer in self.conformer_layers:
            # 注意力缓存
            attn_cache = torch.zeros(batch_size, self.cache_past_len, self.encoder_dim, device=device)
            
            # 卷积缓存
            conv_cache = torch.zeros(batch_size, 2*self.encoder_dim, 6, device=device)
            
            caches.append((attn_cache, conv_cache))
        return caches


    def forward(self, x, mode='dual', resnet_cache=None, conformer_cache=None):
        """
        前向传播函数
        
        Args:
            x: 输入张量 [batch_size, channels, time, freq]
            mode: 
                - 'dual': 同时进行流式和非流式前向传播（训练时使用）
                - 'streaming': 仅流式前向传播
                - 'non-streaming': 仅非流式前向传播
            resnet_cache: ResNet缓存,仅流式模式需要
            conformer_cache: Conformer缓存,仅流式模式需要
                
        Returns:
            根据mode返回不同的结果:
            - 'dual': 返回{'stream_pred', 'nonstream_pred', 'distill_loss', 'hpc_losses'}
            - 'streaming' + 缓存: 返回(pred, (next_resnet_cache, next_conformer_cache))
            - 'streaming' 无缓存 或 'non-streaming': 返回pred
        """
        # 双模式 - 同时进行流式和非流式前向传播（主要用于训练）
        if mode == 'dual':
            # 非流式前向传播
            nonstream_resnet_out = self.resnet(x, mode='non-streaming')
            N, C, T, W = nonstream_resnet_out.shape
            nonstream_resnet_out = nonstream_resnet_out.permute(0, 2, 1, 3).reshape(N, T, C*W)
            
            nonstream_conformer_in = self.input_projection(nonstream_resnet_out)
            nonstream_conformer_out = nonstream_conformer_in
            
            for layer in self.conformer_layers:
                nonstream_conformer_out = layer(nonstream_conformer_out, mode='non-streaming')
            
            # 流式前向传播
            stream_resnet_out = self.resnet(x, mode='streaming')
            stream_resnet_out = stream_resnet_out.permute(0, 2, 1, 3).reshape(N, T, C*W)
            
            stream_conformer_in = self.input_projection(stream_resnet_out)
            stream_conformer_out = stream_conformer_in
            
            for layer in self.conformer_layers:
                stream_conformer_out = layer(stream_conformer_out, mode='streaming')
            
            # 计算知识蒸馏损失 - 在encoder输出层面蒸馏
            distill_loss = F.smooth_l1_loss(stream_conformer_out, nonstream_conformer_out.detach())
            
            # 计算HPC损失
            stream_apc_loss = self.stream_apc(stream_conformer_out)
            nonstream_apc_loss = self.nonstream_apc(nonstream_conformer_out)
            
            # 汇总HPC损失
            apc_losses = {
                'stream_apc_loss': stream_apc_loss,
                'nonstream_apc_loss': nonstream_apc_loss,
                'total_apc_loss': stream_apc_loss + nonstream_apc_loss
            }
            # 处理非流式输出
            nonstream_outputs = nonstream_conformer_out.permute(0, 2, 1)
            nonstream_outputs = self.t_pooling(nonstream_outputs)
            nonstream_outputs = nonstream_outputs.permute(0, 2, 1)
            
            nonstream_sed = self.sed_out_layer(nonstream_outputs)
            nonstream_doa = self.out_layer(nonstream_outputs)
            nonstream_pred = torch.cat((nonstream_sed, nonstream_doa), dim=-1)
            
            # 处理流式输出
            stream_outputs = stream_conformer_out.permute(0, 2, 1)
            stream_outputs = self.t_pooling(stream_outputs)
            stream_outputs = stream_outputs.permute(0, 2, 1)
            
            stream_sed = self.sed_out_layer(stream_outputs)
            stream_doa = self.out_layer(stream_outputs)
            stream_pred = torch.cat((stream_sed, stream_doa), dim=-1)
            
            return {
                'stream_pred': stream_pred,
                'nonstream_pred': nonstream_pred,
                'distill_loss': distill_loss,
                'apc_losses': apc_losses
            }
        
        # 仅流式模式
        elif mode == 'streaming':
            # 带缓存的流式前向传播（用于块式推理）
            if resnet_cache is not None and conformer_cache is not None:
                # ResNet前向传播
                resnet_out, next_resnet_cache = self.resnet(x, mode='streaming', caches=resnet_cache)
                N, C, T, W = resnet_out.shape
                resnet_out = resnet_out.permute(0, 2, 1, 3).reshape(N, T, C*W)
                
                # Conformer前向传播
                conformer_in = self.input_projection(resnet_out)
                conformer_out = conformer_in
                
                next_conformer_cache = []
                for i, layer in enumerate(self.conformer_layers):
                    conformer_out, next_cache = layer(
                        conformer_out, 
                        mode='streaming',
                        cache=conformer_cache[i] if conformer_cache else None
                    )
                    next_conformer_cache.append(next_cache)
                
                # 输出处理
                outputs = conformer_out.permute(0, 2, 1)
                outputs = self.t_pooling(outputs)
                outputs = outputs.permute(0, 2, 1)
                
                sed = self.sed_out_layer(outputs)
                doa = self.out_layer(outputs)
                pred = torch.cat((sed, doa), dim=-1)
                
                return pred, (next_resnet_cache, next_conformer_cache)
            
            # 无缓存的流式前向传播（用于评估或单次处理）
            else:
                resnet_out = self.resnet(x, mode='streaming')
                N, C, T, W = resnet_out.shape
                resnet_out = resnet_out.permute(0, 2, 1, 3).reshape(N, T, C*W)
                
                conformer_in = self.input_projection(resnet_out)
                conformer_out = conformer_in
                
                for layer in self.conformer_layers:
                    conformer_out = layer(conformer_out, mode='streaming')
                
                # 计算HPC损失（如果在评估时需要）
                if self.training:
                    apc_loss = self.stream_apc(conformer_out)
                
                outputs = conformer_out.permute(0, 2, 1)
                outputs = self.t_pooling(outputs)
                outputs = outputs.permute(0, 2, 1)
                
                sed = self.sed_out_layer(outputs)
                doa = self.out_layer(outputs)
                pred = torch.cat((sed, doa), dim=-1)
                
                if self.training:
                    return pred, {'apc_loss': apc_loss}
                return pred
        
        # 仅非流式模式
        elif mode == 'non-streaming':
            resnet_out = self.resnet(x, mode='non-streaming')
            N, C, T, W = resnet_out.shape
            resnet_out = resnet_out.permute(0, 2, 1, 3).reshape(N, T, C*W)
            
            conformer_in = self.input_projection(resnet_out)
            conformer_out = conformer_in
            
            for layer in self.conformer_layers:
                conformer_out = layer(conformer_out, mode='non-streaming')
            
            # 计算HPC损失（如果在评估时需要）
            if self.training:
                apc_loss = self.nonstream_apc(conformer_out)
            
            outputs = conformer_out.permute(0, 2, 1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0, 2, 1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            if self.training:
                return pred, {'apc_loss': apc_loss}
            return pred
        
        else:
            raise ValueError(f"不支持的模式: {mode}, 必须是 'dual', 'streaming' 或 'non-streaming'")
        