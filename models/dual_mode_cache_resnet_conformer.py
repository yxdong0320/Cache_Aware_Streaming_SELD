import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any, Union
from .cache_conformer import ConformerConvModule, CausalAttention, FeedForward
from .conformer import Attention as NonStreamingAttention
from .conformer import ConformerConvModule as NonStreamingConformerConvModule
from .cache_resnet import CausalConv2D, conv1x1

class DualModeConformerBlock(nn.Module):
    """
    双模态Conformer Block,同时包含流式和非流式的Conformer实现
    """
    def __init__(
        self,
        *,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        conv_dropout: float = 0.,
        att_context_size: List[int] = [100, 49]
    ):
        super().__init__()
        # 流式模式参数
        self.stream_ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.stream_attn = CausalAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            att_context_size=att_context_size
        )
        self.stream_conv = ConformerConvModule(
            dim=dim,
            causal=True,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout
        )
        self.stream_ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        
        # 非流式模式参数
        self.nonstream_ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.nonstream_attn = NonStreamingAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            max_pos_emb=att_context_size[0] + att_context_size[1]
        )
        self.nonstream_conv = NonStreamingConformerConvModule(
            dim=dim,
            causal=False,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout
        )
        self.nonstream_ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        
        # 共享的归一化层和缩放层
        self.attn_norm = nn.LayerNorm(dim)
        self.ff1_norm = nn.LayerNorm(dim)
        self.ff2_norm = nn.LayerNorm(dim)
        self.post_norm = nn.LayerNorm(dim)
        
        self.scale_ff1 = 0.5
        self.scale_ff2 = 0.5

    def forward(self, x: torch.Tensor, mode: str = 'streaming', cache: Optional[Tuple] = None, mask: Optional[torch.Tensor] = None):
        """
        前向传播函数
        
        Args:
            x: 输入张量 [batch_size, seq_len, dim]
            mode: 'streaming' 或 'non-streaming'
            cache: 流式模式下的缓存
            mask: 注意力掩码
            
        Returns:
            如果是流式模式且提供缓存，返回(output, next_cache)
            否则只返回output
        """
        if mode == 'streaming':
            # 流式模式
            # 前馈网络1
            ff1_out = self.ff1_norm(x)
            ff1_out = self.stream_ff1(ff1_out)
            ff1_out = ff1_out * self.scale_ff1 + x
            
            # 自注意力层
            if cache is None:
                attn_out = self.attn_norm(ff1_out)
                attn_out = self.stream_attn(attn_out, mask=mask)
                attn_out = attn_out + ff1_out
                
                # 卷积层
                conv_out = self.stream_conv(attn_out)
                conv_out = conv_out + attn_out
                
                # 前馈网络2
                ff2_out = self.ff2_norm(conv_out)
                ff2_out = self.stream_ff2(ff2_out)
                ff2_out = ff2_out * self.scale_ff2 + conv_out
                
                # 后处理归一化
                out = self.post_norm(ff2_out)
                return out
            else:
                attn_cache, conv_cache = cache
                attn_out = self.attn_norm(ff1_out)
                attn_out, next_attn_cache = self.stream_attn(attn_out, mask=mask, cache=attn_cache)
                attn_out = attn_out + ff1_out
                
                # 卷积层
                conv_out, next_conv_cache = self.stream_conv(attn_out, cache=conv_cache)
                conv_out = conv_out + attn_out
                
                # 前馈网络2
                ff2_out = self.ff2_norm(conv_out)
                ff2_out = self.stream_ff2(ff2_out)
                ff2_out = ff2_out * self.scale_ff2 + conv_out
                
                # 后处理归一化
                out = self.post_norm(ff2_out)
                return out, (next_attn_cache, next_conv_cache)
        else:
            # 非流式模式
            # 前馈网络1
            ff1_out = self.ff1_norm(x)
            ff1_out = self.nonstream_ff1(ff1_out)
            ff1_out = ff1_out * self.scale_ff1 + x
            
            # 自注意力层
            attn_out = self.attn_norm(ff1_out)
            attn_out = self.nonstream_attn(attn_out, mask=mask)
            attn_out = attn_out + ff1_out
            
            # 卷积层
            conv_out = self.nonstream_conv(attn_out)
            conv_out = conv_out + attn_out
            
            # 前馈网络2
            ff2_out = self.ff2_norm(conv_out)
            ff2_out = self.nonstream_ff2(ff2_out)
            ff2_out = ff2_out * self.scale_ff2 + conv_out
            
            # 后处理归一化
            out = self.post_norm(ff2_out)
            return out

class DualModeResNetBlock(nn.Module):
    expansion = 1
    """双模态ResNet块,支持流式和非流式处理"""
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DualModeResNetBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        
        # 流式卷积层
        self.stream_conv1 = conv3x3(inplanes, planes, stride, causal=True)
        self.stream_conv2 = conv3x3(planes, planes, causal=True)
        
        # 非流式卷积层
        self.nonstream_conv1 = conv3x3(inplanes, planes, stride, causal=False)
        self.nonstream_conv2 = conv3x3(planes, planes, causal=False)
        
        # 共享层
        if planes == 24:
            LN_size = 24*64
        elif planes == 48:
            LN_size = 48*16
        elif planes == 96:
            LN_size = 96*4
        elif planes == 192:
            LN_size = 192*2
        
        self.ln1 = nn.LayerNorm(LN_size)
        self.ln2 = nn.LayerNorm(LN_size)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def ln1_process(self, x):
        N, C, T, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(N, T, -1)
        x = self.ln1(x)
        x = x.reshape(N, T, C, W)
        x = x.permute(0, 2, 1, 3)
        return x

    def ln2_process(self, x):
        N, C, T, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(N, T, -1)
        x = self.ln2(x)
        x = x.reshape(N, T, C, W)
        x = x.permute(0, 2, 1, 3)
        return x

    def downsample_process(self, x):
        x = self.downsample[0](x)
        N, C, T, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(N, T, -1)
        x = self.downsample[1](x)
        x = x.reshape(N, T, C, W)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x, mode='streaming', cache=None):
        identity = x
        
        if mode == 'streaming':
            if cache is None:
                # 流式模式但无缓存（训练阶段）
                out = self.stream_conv1(x)
                out = self.ln1_process(out)
                out = self.relu(out)
                out = self.stream_conv2(out)
                out = self.ln2_process(out)
                
                if self.downsample is not None:
                    identity = self.downsample_process(x)
                    
                out += identity
                out = self.relu(out)
                return out
            else:
                # 流式模式带缓存（推理阶段）
                cache1, cache2 = cache
                
                # 验证缓存形状
                assert cache1.shape[1] == self.inplanes, f"Cache1 channel mismatch: expected {self.inplanes}, got {cache1.shape[1]}"
                assert cache2.shape[1] == self.planes, f"Cache2 channel mismatch: expected {self.planes}, got {cache2.shape[1]}"
                
                out, next_cache1 = self.stream_conv1(x, cache1)
                out = self.ln1_process(out)
                out = self.relu(out)
                out, next_cache2 = self.stream_conv2(out, cache2)
                out = self.ln2_process(out)
                
                if self.downsample is not None:
                    identity = self.downsample_process(x)
                    
                out += identity
                out = self.relu(out)
                return out, (next_cache1, next_cache2)
        else:
            # 非流式模式
            out = self.nonstream_conv1(x)
            out = self.ln1_process(out)
            out = self.relu(out)
            out = self.nonstream_conv2(out)
            out = self.ln2_process(out)
            
            if self.downsample is not None:
                identity = self.downsample_process(x)
                
            out += identity
            out = self.relu(out)
            return out

class DualModeResNet(nn.Module):
    """双模态ResNet,支持流式和非流式处理"""
    def __init__(self, block, layers, in_channel=17, zero_init_residual=False):
        super(DualModeResNet, self).__init__()
        self.inplanes = 24
        
        # 流式和非流式卷积
        self.stream_conv1 = CausalConv2D(in_channel, 24, kernel_size=3, stride=1, padding=None, bias=False)
        self.nonstream_conv1 = nn.Conv2d(in_channel, 24, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 共享层
        self.ln1 = nn.LayerNorm(64*24)
        self.relu = nn.ReLU(inplace=True)
        
        # 构建双模态层
        self.layer1 = self._make_dual_layer(block, 24, layers[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 4))
        self.layer2 = self._make_dual_layer(block, 48, layers[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 4))
        self.layer3 = self._make_dual_layer(block, 96, layers[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.layer4 = self._make_dual_layer(block, 192, layers[3])
        
        # 最终1x1卷积
        self.conv5 = conv1x1(192, 256)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, CausalConv2D)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_dual_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if planes == 48:
                LN_size = 48*16
            elif planes == 96:
                LN_size = 96*4
            elif planes == 192:
                LN_size = 192*2
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.LayerNorm(LN_size),
            )

        layers = []
        layers.append(DualModeResNetBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(DualModeResNetBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def ln1_process(self, x):
        N, C, T, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(N, T, -1)
        x = self.ln1(x)
        x = x.reshape(N, T, C, W)
        x = x.permute(0, 2, 1, 3)
        return x

    def get_initial_cache(self, batch_size=1):
        device = next(self.parameters()).device
        
        # conv1的缓存
        conv1_cache = torch.zeros(batch_size, 7, 2, 64, device=device)
        
        # 为每个layer中的每个block准备缓存
        layer_caches = []
        channels = [(24, 24), (48, 48), (96, 96), (192, 192)]
        features = [64, 16, 4, 2]
        
        for layer_idx, ((in_ch, out_ch), feat_dim) in enumerate(zip(channels, features)):
            layer_cache = []
            for block_idx in range(2):  # 每层2个block
                if block_idx == 0 and layer_idx > 0:
                    prev_ch = channels[layer_idx-1][1]
                    cache1 = torch.zeros(batch_size, prev_ch, 2, feat_dim, device=device)
                else:
                    cache1 = torch.zeros(batch_size, in_ch, 2, feat_dim, device=device)
                cache2 = torch.zeros(batch_size, out_ch, 2, feat_dim, device=device)
                layer_cache.extend([cache1, cache2])
            layer_caches.append(layer_cache)
        
        return conv1_cache, layer_caches

    def forward(self, x, mode='streaming', caches=None):
        if mode == 'streaming':
            if caches is None:
                # 流式训练模式
                x = self.stream_conv1(x)
                x = self.ln1_process(x)
                x = self.relu(x)
                
                for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                    for j, block in enumerate(layer):
                        x = block(x, mode='streaming')
                    if i < 3:
                        x = getattr(self, f'maxpool{i+1}')(x)
                        
                x = self.conv5(x)
                return x
            else:
                # 流式推理模式
                conv1_cache, layer_caches = caches
                x, next_conv1_cache = self.stream_conv1(x, conv1_cache)
                x = self.ln1_process(x)
                x = self.relu(x)
                
                next_layer_caches = []
                for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                    current_layer_caches = []
                    layer_cache = layer_caches[i]
                    
                    for j, block in enumerate(layer):
                        cache_idx = j * 2
                        block_cache = (layer_cache[cache_idx], layer_cache[cache_idx + 1])
                        x, next_block_cache = block(x, mode='streaming', cache=block_cache)
                        current_layer_caches.extend(next_block_cache)
                    
                    next_layer_caches.append(current_layer_caches)
                    
                    if i < 3:
                        x = getattr(self, f'maxpool{i+1}')(x)
                        
                x = self.conv5(x)
                return x, (next_conv1_cache, next_layer_caches)
        else:
            # 非流式模式
            x = self.nonstream_conv1(x)
            x = self.ln1_process(x)
            x = self.relu(x)
            
            for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                for j, block in enumerate(layer):
                    x = block(x, mode='non-streaming')
                if i < 3:
                    x = getattr(self, f'maxpool{i+1}')(x)
                    
            x = self.conv5(x)
            return x

def conv3x3(in_planes, out_planes, stride=1, causal=False):
    """3x3 convolution with padding"""
    if causal:
        return CausalConv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=None, bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

class DualModeResnetConformer(nn.Module):
    """
    双模态SELD模型,支持流式和非流式处理
    """
    def __init__(
        self, 
        in_channel=7, 
        in_dim=64, 
        out_dim=39,
        att_context_size=[100, 49], 
        num_conformer_layer=8,
        encoder_dim=256,
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
            - 'dual': 返回{'stream_pred', 'nonstream_pred', 'distill_loss'}
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
                'distill_loss': distill_loss
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
                
                outputs = conformer_out.permute(0, 2, 1)
                outputs = self.t_pooling(outputs)
                outputs = outputs.permute(0, 2, 1)
                
                sed = self.sed_out_layer(outputs)
                doa = self.out_layer(outputs)
                pred = torch.cat((sed, doa), dim=-1)
                
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
            
            outputs = conformer_out.permute(0, 2, 1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0, 2, 1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            return pred
        
        else:
            raise ValueError(f"不支持的模式: {mode}, 必须是 'dual', 'streaming' 或 'non-streaming'")
        
class DualModeResnetConformer_Distill_on_Result(nn.Module):
    """
    双模态SELD模型,支持流式和非流式处理
    """
    def __init__(
        self, 
        in_channel=7, 
        in_dim=64, 
        out_dim=39,
        att_context_size=[100, 49], 
        num_conformer_layer=8,
        encoder_dim=256,
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
            - 'dual': 返回{'stream_pred', 'nonstream_pred', 'distill_loss'}
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
            # distill_loss = F.smooth_l1_loss(stream_conformer_out, nonstream_conformer_out.detach())
            
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

            distill_loss = F.smooth_l1_loss(stream_pred, nonstream_pred.detach())
            
            return {
                'stream_pred': stream_pred,
                'nonstream_pred': nonstream_pred,
                'distill_loss': distill_loss
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
                
                outputs = conformer_out.permute(0, 2, 1)
                outputs = self.t_pooling(outputs)
                outputs = outputs.permute(0, 2, 1)
                
                sed = self.sed_out_layer(outputs)
                doa = self.out_layer(outputs)
                pred = torch.cat((sed, doa), dim=-1)
                
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
            
            outputs = conformer_out.permute(0, 2, 1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0, 2, 1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            return pred
        
        else:
            raise ValueError(f"不支持的模式: {mode}, 必须是 'dual', 'streaming' 或 'non-streaming'")
