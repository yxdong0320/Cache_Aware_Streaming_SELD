import torch
import torch.nn as nn

from .cache_resnet import resnet18_nopool
from .cache_conformer import ConformerBlock
from .APC import APCModule

# from cache_resnet import resnet18_nopool
# from cache_conformer import ConformerBlock
import pdb

class ResnetConformer_sed_doa_nopool(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, 
                 att_context_size = [100,49], 
                 num_conformer_layer = 8,
                 encoder_dim = 256,
                 apc_future_steps = 3): # 7,64,39,[100,49],8,256
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
        self.apc = APCModule(
            input_dim=self.encoder_dim,
            future_steps=apc_future_steps,
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
            apc_loss = self.apc.compute_apc_loss(conformer_outputs)
            outputs = conformer_outputs.permute(0,2,1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0,2,1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            return pred, apc_loss
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
    