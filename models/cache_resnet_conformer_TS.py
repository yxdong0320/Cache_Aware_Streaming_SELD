import torch
import torch.nn as nn

from .cache_resnet import resnet18_nopool
from .cache_conformer_TS import ConformerBlock
from .resnet_conformer_audio_T import ResnetConformer_sed_doa_nopool_original

import pdb

class ResnetConformer_sed_doa_nopool(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, 
                 att_context_size = [100,49], 
                 num_conformer_layer = 8,
                 encoder_dim = 256): # 7,64,39,[100,49],8,256
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256
        self.encoder_dim = encoder_dim
        self.in_ch = in_channel
        self.in_dim = in_dim
        self.att_context_size = att_context_size
        self.cache_past_len = self.att_context_size[0]
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = num_conformer_layer
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim = self.encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1,
                att_context_size = att_context_size
            ) for _ in range(num_layers)]
        )
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
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
    def forward(self, x, resnet_cache=None, conformer_cache=None):
        if resnet_cache is None and conformer_cache is None:
        # if cache is None:
            conv_outputs = self.resnet(x)
            N, C, T, W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)
            
            conformer_outputs = self.input_projection(conv_outputs)
            
            for layer in self.conformer_layers:
                conformer_outputs = layer(conformer_outputs)
                
            outputs = conformer_outputs.permute(0,2,1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0,2,1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            return pred
        else:
            layer_caches = conformer_cache
            # Streaming inference mode
            conv_outputs, next_resnet_cache = self.resnet(x, resnet_cache)
            N,C,T,W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)
            
            conformer_outputs = self.input_projection(conv_outputs)
            
            next_layer_caches = []
            for i, layer in enumerate(self.conformer_layers):
                conformer_outputs, next_cache = layer(
                    conformer_outputs,
                    cache=layer_caches[i] if layer_caches else None
                )
                next_layer_caches.append(next_cache)
                
            outputs = conformer_outputs.permute(0,2,1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0,2,1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            return pred, (next_resnet_cache, next_layer_caches)
    
class ResnetConformer_sed_doa_nopool_TS_hidstate_att_loss(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, 
                 att_context_size = [100,49], 
                 num_conformer_layer = 8,
                 encoder_dim = 256,
                 use_hidden_distill=True,
                 use_attn_distill=True): # 添加控制知识蒸馏类型的参数
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        self.teacher_model = ResnetConformer_sed_doa_nopool_original(in_channel=7, in_dim=64, out_dim=39)
        embedding_dim = in_dim // 32 * 256
        self.encoder_dim = encoder_dim
        self.in_ch = in_channel
        self.in_dim = in_dim
        self.att_context_size = att_context_size
        self.cache_past_len = self.att_context_size[0]
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = num_conformer_layer
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim = self.encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1,
                att_context_size = att_context_size
            ) for _ in range(num_layers)]
        )
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
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
        
        # 设置知识蒸馏参数
        self.use_hidden_distill = use_hidden_distill
        self.use_attn_distill = use_attn_distill
        self.num_layers = num_layers
        
        # 冻结教师模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
    def forward(self, x, resnet_cache=None, conformer_cache=None):
        if resnet_cache is None and conformer_cache is None:
            # 教师模型前向传播
            with torch.no_grad():
                # 教师模型输出和中间状态收集
                teacher_outputs = []
                teacher_attns = []
                
                # 获取教师模型的输入特征
                t_conv_outputs = self.teacher_model.resnet(x)
                N, C, T, W = t_conv_outputs.shape
                t_conv_outputs = t_conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C*W)
                t_conformer_outputs = self.teacher_model.input_projection(t_conv_outputs)
                
                # 收集教师模型各层特征
                for i, layer in enumerate(self.teacher_model.conformer_layers):
                    # 存储每层的输出
                    t_conformer_outputs, attn_weights = layer(t_conformer_outputs, return_attention=True)
                    teacher_outputs.append(t_conformer_outputs)
                    teacher_attns.append(attn_weights)
                
                # 教师模型最终输出
                t_outputs = t_conformer_outputs.permute(0, 2, 1)
                t_outputs = self.teacher_model.t_pooling(t_outputs)
                t_outputs = t_outputs.permute(0, 2, 1)
                t_sed = self.teacher_model.sed_out_layer(t_outputs)
                t_doa = self.teacher_model.out_layer(t_outputs)
                target_ts = torch.cat((t_sed, t_doa), dim=-1)
            
            # 学生模型前向传播
            student_outputs = []
            student_attns = []
            
            # 获取学生模型的输入特征
            conv_outputs = self.resnet(x)
            N, C, T, W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C*W)
            conformer_outputs = self.input_projection(conv_outputs)
            
            # 收集学生模型各层特征
            for i, layer in enumerate(self.conformer_layers):
                conformer_outputs, attn_weights = layer(conformer_outputs, return_attention=True)
                student_outputs.append(conformer_outputs)
                student_attns.append(attn_weights)
            
            # 学生模型最终输出处理
            outputs = conformer_outputs.permute(0, 2, 1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0, 2, 1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            # 根据参数决定返回什么内容
            result = [pred, target_ts]
            
            if self.use_hidden_distill:
                result.append((teacher_outputs, student_outputs))
                
            if self.use_attn_distill:
                result.append((teacher_attns, student_attns))
                
            return tuple(result)
        else:
            # 流式推理模式，不返回蒸馏所需的特征
            with torch.no_grad():
                target_ts = self.teacher_model(x)
                
            layer_caches = conformer_cache
            
            # 学生模型流式推理
            conv_outputs, next_resnet_cache = self.resnet(x, resnet_cache)
            N, C, T, W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C*W)
            
            conformer_outputs = self.input_projection(conv_outputs)
            
            next_layer_caches = []
            for i, layer in enumerate(self.conformer_layers):
                conformer_outputs, next_cache = layer(
                    conformer_outputs,
                    cache=layer_caches[i] if layer_caches else None
                )
                next_layer_caches.append(next_cache)
                
            outputs = conformer_outputs.permute(0, 2, 1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0, 2, 1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            return pred, target_ts, (next_resnet_cache, next_layer_caches)

    def get_initial_cache_resnet(self, batch_size=1):
        """初始化ResNet的卷积缓存"""
        device = next(self.parameters()).device
        
        conv1_cache = torch.zeros(batch_size, self.in_ch, 2, self.in_dim, device=device)
        
        layer_caches = []
        channels = [(24, 24), (48, 48), (96, 96), (192, 192)]
        features = [64, 16, 4, 2]
        
        for layer_idx, ((in_ch, out_ch), feat_dim) in enumerate(zip(channels, features)):
            current_layer_caches = []  # 当前layer的所有cache
            num_blocks = 2
            
            for block_idx in range(num_blocks):
                # 为每个block创建两个cache
                if block_idx == 0 and layer_idx > 0:
                    prev_ch = channels[layer_idx-1][1]
                    cache1 = torch.zeros(batch_size, prev_ch, 2, feat_dim, device=device)
                else:
                    cache1 = torch.zeros(batch_size, in_ch, 2, feat_dim, device=device)
                cache2 = torch.zeros(batch_size, out_ch, 2, feat_dim, device=device)
                current_layer_caches.extend([cache1, cache2])
                
            layer_caches.append(current_layer_caches)  # 将当前layer的所有cache作为一个整体添加
        
        return (conv1_cache, layer_caches)

    def get_initial_cache_conformer(self, batch_size=1):
        caches = []
        device = next(self.parameters()).device  # 确保缓存在正确设备上
        for layer in self.conformer_layers:
            # attention缓存维度: [B, cache_len, D=256]
            attn_cache = torch.zeros(batch_size, self.cache_past_len, self.encoder_dim, device=device)
            
            # convolution缓存维度: [B, D=256, kernel_size-1=6]
            conv_cache = torch.zeros(batch_size, 2*self.encoder_dim, 6, device=device)
            
            caches.append((attn_cache, conv_cache))
        return caches
    
class ResnetConformer_sed_doa_nopool_Cache_TS_hidstate_att_loss(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, 
                 att_context_size = [100,49], 
                 num_conformer_layer = 8,
                 encoder_dim = 256,
                 T_att_context_size = [100,49],
                 use_hidden_distill=True,
                 use_attn_distill=True): # 添加控制知识蒸馏类型的参数
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        self.teacher_model = ResnetConformer_sed_doa_nopool(in_channel=7, in_dim=64, out_dim=39,
                                                            att_context_size=T_att_context_size,
                                                            num_conformer_layer=num_conformer_layer,
                                                            encoder_dim=encoder_dim)
        embedding_dim = in_dim // 32 * 256
        self.encoder_dim = encoder_dim
        self.in_ch = in_channel
        self.in_dim = in_dim
        self.att_context_size = att_context_size
        self.cache_past_len = self.att_context_size[0]
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = num_conformer_layer
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim = self.encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1,
                att_context_size = att_context_size
            ) for _ in range(num_layers)]
        )
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
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
        
        # 设置知识蒸馏参数
        self.use_hidden_distill = use_hidden_distill
        self.use_attn_distill = use_attn_distill
        self.num_layers = num_layers
        
        # 冻结教师模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
    def forward(self, x, resnet_cache=None, conformer_cache=None):
        if resnet_cache is None and conformer_cache is None:
            # 教师模型前向传播
            with torch.no_grad():
                # 教师模型输出和中间状态收集
                teacher_outputs = []
                teacher_attns = []
                
                # 获取教师模型的输入特征
                t_conv_outputs = self.teacher_model.resnet(x)
                N, C, T, W = t_conv_outputs.shape
                t_conv_outputs = t_conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C*W)
                t_conformer_outputs = self.teacher_model.input_projection(t_conv_outputs)
                
                # 收集教师模型各层特征
                for i, layer in enumerate(self.teacher_model.conformer_layers):
                    # 存储每层的输出
                    t_conformer_outputs, attn_weights = layer(t_conformer_outputs, return_attention=True)
                    teacher_outputs.append(t_conformer_outputs)
                    teacher_attns.append(attn_weights)
                
                # 教师模型最终输出
                t_outputs = t_conformer_outputs.permute(0, 2, 1)
                t_outputs = self.teacher_model.t_pooling(t_outputs)
                t_outputs = t_outputs.permute(0, 2, 1)
                t_sed = self.teacher_model.sed_out_layer(t_outputs)
                t_doa = self.teacher_model.out_layer(t_outputs)
                target_ts = torch.cat((t_sed, t_doa), dim=-1)
            
            # 学生模型前向传播
            student_outputs = []
            student_attns = []
            
            # 获取学生模型的输入特征
            conv_outputs = self.resnet(x)
            N, C, T, W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C*W)
            conformer_outputs = self.input_projection(conv_outputs)
            
            # 收集学生模型各层特征
            for i, layer in enumerate(self.conformer_layers):
                conformer_outputs, attn_weights = layer(conformer_outputs, return_attention=True)
                student_outputs.append(conformer_outputs)
                student_attns.append(attn_weights)
            
            # 学生模型最终输出处理
            outputs = conformer_outputs.permute(0, 2, 1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0, 2, 1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            # 根据参数决定返回什么内容
            result = [pred, target_ts]
            
            if self.use_hidden_distill:
                result.append((teacher_outputs, student_outputs))
                
            if self.use_attn_distill:
                result.append((teacher_attns, student_attns))
                
            return tuple(result)
        else:
            # 流式推理模式，不返回蒸馏所需的特征
            with torch.no_grad():
                target_ts = self.teacher_model(x)
                
            layer_caches = conformer_cache
            
            # 学生模型流式推理
            conv_outputs, next_resnet_cache = self.resnet(x, resnet_cache)
            N, C, T, W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C*W)
            
            conformer_outputs = self.input_projection(conv_outputs)
            
            next_layer_caches = []
            for i, layer in enumerate(self.conformer_layers):
                conformer_outputs, next_cache = layer(
                    conformer_outputs,
                    cache=layer_caches[i] if layer_caches else None
                )
                next_layer_caches.append(next_cache)
                
            outputs = conformer_outputs.permute(0, 2, 1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0, 2, 1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            return pred, target_ts, (next_resnet_cache, next_layer_caches)

    def get_initial_cache_resnet(self, batch_size=1):
        """初始化ResNet的卷积缓存"""
        device = next(self.parameters()).device
        
        conv1_cache = torch.zeros(batch_size, self.in_ch, 2, self.in_dim, device=device)
        
        layer_caches = []
        channels = [(24, 24), (48, 48), (96, 96), (192, 192)]
        features = [64, 16, 4, 2]
        
        for layer_idx, ((in_ch, out_ch), feat_dim) in enumerate(zip(channels, features)):
            current_layer_caches = []  # 当前layer的所有cache
            num_blocks = 2
            
            for block_idx in range(num_blocks):
                # 为每个block创建两个cache
                if block_idx == 0 and layer_idx > 0:
                    prev_ch = channels[layer_idx-1][1]
                    cache1 = torch.zeros(batch_size, prev_ch, 2, feat_dim, device=device)
                else:
                    cache1 = torch.zeros(batch_size, in_ch, 2, feat_dim, device=device)
                cache2 = torch.zeros(batch_size, out_ch, 2, feat_dim, device=device)
                current_layer_caches.extend([cache1, cache2])
                
            layer_caches.append(current_layer_caches)  # 将当前layer的所有cache作为一个整体添加
        
        return (conv1_cache, layer_caches)

    def get_initial_cache_conformer(self, batch_size=1):
        caches = []
        device = next(self.parameters()).device  # 确保缓存在正确设备上
        for layer in self.conformer_layers:
            # attention缓存维度: [B, cache_len, D=256]
            attn_cache = torch.zeros(batch_size, self.cache_past_len, self.encoder_dim, device=device)
            
            # convolution缓存维度: [B, D=256, kernel_size-1=6]
            conv_cache = torch.zeros(batch_size, 2*self.encoder_dim, 6, device=device)
            
            caches.append((attn_cache, conv_cache))
        return caches
    
class ResnetConformer_sed_doa_nopool_TS_hidstate_att_srd_loss(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, 
                 att_context_size = [100,49], 
                 num_conformer_layer = 8,
                 encoder_dim = 256,
                 use_hidden_distill=True,
                 use_attn_distill=True,
                 use_srd_distill=False):  # 新增SRD蒸馏参数
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        self.teacher_model = ResnetConformer_sed_doa_nopool_original(in_channel=7, in_dim=64, out_dim=39)
        embedding_dim = in_dim // 32 * 256
        self.encoder_dim = encoder_dim
        self.in_ch = in_channel
        self.in_dim = in_dim
        self.att_context_size = att_context_size
        self.cache_past_len = self.att_context_size[0]
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = num_conformer_layer
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim = self.encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1,
                att_context_size = att_context_size
            ) for _ in range(num_layers)]
        )
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
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
        
        # 设置知识蒸馏参数
        self.use_hidden_distill = use_hidden_distill
        self.use_attn_distill = use_attn_distill
        self.use_srd_distill = use_srd_distill
        self.num_layers = num_layers
        
        # **新增：如果使用SRD，添加特征适配模块**
        if self.use_srd_distill:
            self.feature_adapter = nn.Sequential(
                nn.Linear(self.encoder_dim, self.encoder_dim),
                nn.LayerNorm(self.encoder_dim),  # 使用LayerNorm而不是BatchNorm1d
                nn.ReLU(),
                nn.Dropout(0.1)  # 添加dropout防止过拟合
            )
        
        # 冻结教师模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
    def forward(self, x, resnet_cache=None, conformer_cache=None):
        if resnet_cache is None and conformer_cache is None:
            # 教师模型前向传播
            with torch.no_grad():
                teacher_outputs = []
                teacher_attns = []
                
                t_conv_outputs = self.teacher_model.resnet(x)
                N, C, T, W = t_conv_outputs.shape
                t_conv_outputs = t_conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C*W)
                t_conformer_outputs = self.teacher_model.input_projection(t_conv_outputs)
                
                for i, layer in enumerate(self.teacher_model.conformer_layers):
                    t_conformer_outputs, attn_weights = layer(t_conformer_outputs, return_attention=True)
                    teacher_outputs.append(t_conformer_outputs)
                    teacher_attns.append(attn_weights)
                
                # 教师模型最终特征处理
                t_outputs = t_conformer_outputs.permute(0, 2, 1)
                t_outputs = self.teacher_model.t_pooling(t_outputs)
                t_outputs = t_outputs.permute(0, 2, 1)
                
                # **保存教师的最终特征用于SRD**
                teacher_final_features = t_outputs
                
                t_sed = self.teacher_model.sed_out_layer(t_outputs)
                t_doa = self.teacher_model.out_layer(t_outputs)
                target_ts = torch.cat((t_sed, t_doa), dim=-1)
            
            # 学生模型前向传播
            student_outputs = []
            student_attns = []
            
            conv_outputs = self.resnet(x)
            N, C, T, W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C*W)
            conformer_outputs = self.input_projection(conv_outputs)
            
            for i, layer in enumerate(self.conformer_layers):
                conformer_outputs, attn_weights = layer(conformer_outputs, return_attention=True)
                student_outputs.append(conformer_outputs)
                student_attns.append(attn_weights)
            
            # 学生模型最终特征处理
            outputs = conformer_outputs.permute(0, 2, 1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0, 2, 1)
            
            # **保存学生的最终特征用于SRD**
            student_final_features = outputs
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            # 根据参数决定返回什么内容
            result = [pred, target_ts]
            
            if self.use_hidden_distill:
                result.append((teacher_outputs, student_outputs))
                
            if self.use_attn_distill:
                result.append((teacher_attns, student_attns))
                
            # **新增：SRD相关输出**
            if self.use_srd_distill:
                # 学生特征通过适配模块映射到教师特征空间
                adapted_student_features = self.feature_adapter(student_final_features)
                
                # 学生适配特征和教师特征分别通过教师的分类头
                student_sed_cross = self.teacher_model.sed_out_layer(adapted_student_features)
                student_doa_cross = self.teacher_model.out_layer(adapted_student_features)
                student_cross_output = torch.cat((student_sed_cross, student_doa_cross), dim=-1)
                
                # 教师特征通过教师的分类头（即target_ts）
                teacher_cross_output = target_ts
                
                result.append((student_cross_output, teacher_cross_output))
                
            return tuple(result)
        else:
            # 流式推理模式保持不变
            with torch.no_grad():
                target_ts = self.teacher_model(x)
                
            layer_caches = conformer_cache
            
            conv_outputs, next_resnet_cache = self.resnet(x, resnet_cache)
            N, C, T, W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C*W)
            
            conformer_outputs = self.input_projection(conv_outputs)
            
            next_layer_caches = []
            for i, layer in enumerate(self.conformer_layers):
                conformer_outputs, next_cache = layer(
                    conformer_outputs,
                    cache=layer_caches[i] if layer_caches else None
                )
                next_layer_caches.append(next_cache)
                
            outputs = conformer_outputs.permute(0, 2, 1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0, 2, 1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            return pred, target_ts, (next_resnet_cache, next_layer_caches)
    
    def get_initial_cache_resnet(self, batch_size=1):
        """初始化ResNet的卷积缓存"""
        device = next(self.parameters()).device
        
        conv1_cache = torch.zeros(batch_size, self.in_ch, 2, self.in_dim, device=device)
        
        layer_caches = []
        channels = [(24, 24), (48, 48), (96, 96), (192, 192)]
        features = [64, 16, 4, 2]
        
        for layer_idx, ((in_ch, out_ch), feat_dim) in enumerate(zip(channels, features)):
            current_layer_caches = []  # 当前layer的所有cache
            num_blocks = 2
            
            for block_idx in range(num_blocks):
                # 为每个block创建两个cache
                if block_idx == 0 and layer_idx > 0:
                    prev_ch = channels[layer_idx-1][1]
                    cache1 = torch.zeros(batch_size, prev_ch, 2, feat_dim, device=device)
                else:
                    cache1 = torch.zeros(batch_size, in_ch, 2, feat_dim, device=device)
                cache2 = torch.zeros(batch_size, out_ch, 2, feat_dim, device=device)
                current_layer_caches.extend([cache1, cache2])
                
            layer_caches.append(current_layer_caches)  # 将当前layer的所有cache作为一个整体添加
        
        return (conv1_cache, layer_caches)

    def get_initial_cache_conformer(self, batch_size=1):
        caches = []
        device = next(self.parameters()).device  # 确保缓存在正确设备上
        for layer in self.conformer_layers:
            # attention缓存维度: [B, cache_len, D=256]
            attn_cache = torch.zeros(batch_size, self.cache_past_len, self.encoder_dim, device=device)
            
            # convolution缓存维度: [B, D=256, kernel_size-1=6]
            conv_cache = torch.zeros(batch_size, 2*self.encoder_dim, 6, device=device)
            
            caches.append((attn_cache, conv_cache))
        return caches