# from sympy import N
import torch
import torch.nn as nn
import torch.nn.init as init
import pdb
from typing import Union
import torch.nn.functional as F

class CausalConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, 
                 dilation=1, groups=1, bias=True):
        # 在宽度维度使用padding以保持维度不变
        padding = (0, kernel_size//2) if isinstance(kernel_size, int) else (0, kernel_size[1]//2)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, 
                        dilation, groups, bias)
        self.kernel_size_val = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.cache_len = self.kernel_size_val - 1  # 需要缓存的帧数

    def forward(self, x, cache=None):
        """
        x: [B, C, T, W]
        cache: [B, C, cache_len, W] 或 None
        """
        batch_size, channels, time_steps, width = x.shape
        
        if cache is None:
            # 训练模式：在时间维度上做padding
            padding_size = self.cache_len
            x = F.pad(x, (0, 0, padding_size, 0))  # 只在时间维度左侧padding
        else:
            # 推理模式：使用cache
            # pdb.set_trace()
            # assert cache.shape == (batch_size, channels, self.cache_len, width)
            x = torch.cat([cache, x], dim=2)
            
        # 应用卷积
        output = super().forward(x)
        
        if cache is None:
            return output
        else:
            # 更新cache为最后cache_len帧
            next_cache = x[:, :, -self.cache_len:, :]
            return output, next_cache

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return CausalConv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=None, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution (doesn't need to be causal)"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes  # 添加这行来记录输入通道数
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        if planes == 24:
            LN_size = 24*64
        elif planes == 48:
            LN_size = 48*16
        elif planes == 96:
            LN_size = 96*4
        elif planes == 192:
            LN_size = 192*2
        self.ln1 = nn.LayerNorm(LN_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.ln2 = nn.LayerNorm(LN_size)
        self.downsample = downsample
        self.stride = stride

    def ln1_process(self, x):
        N, C, T, W = x.shape
        x = x.permute(0,2,1,3).reshape(N, T, -1)
        x = self.ln1(x)
        x = x.reshape(N, T, C, W)
        x = x.permute(0,2,1,3)
        return x

    def ln2_process(self, x):
        N, C, T, W = x.shape
        x = x.permute(0,2,1,3).reshape(N, T, -1)
        x = self.ln2(x)
        x = x.reshape(N, T, C, W)
        x = x.permute(0,2,1,3)
        return x

    def downsample_process(self, x):
        x = self.downsample[0](x)
        N, C, T, W = x.shape
        x = x.permute(0,2,1,3).reshape(N, T, -1)
        x = self.downsample[1](x)
        x = x.reshape(N, T, C, W)
        x = x.permute(0,2,1,3)
        return x

    def forward(self, x, cache=None):
        if cache is None:
            identity = x
            out = self.conv1(x)
            out = self.ln1_process(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.ln2_process(out)
            
            if self.downsample is not None:
                identity = self.downsample_process(x)
                
            out += identity
            out = self.relu(out)
            return out
        else:
            identity = x
            cache1, cache2 = cache
            
            # 验证缓存形状
            assert cache1.shape[1] == self.inplanes, f"Cache1 channel mismatch: expected {self.inplanes}, got {cache1.shape[1]}"
            assert cache2.shape[1] == self.planes, f"Cache2 channel mismatch: expected {self.planes}, got {cache2.shape[1]}"
            
            out, next_cache1 = self.conv1(x, cache1)
            out = self.ln1_process(out)
            out = self.relu(out)
            out, next_cache2 = self.conv2(out, cache2)
            out = self.ln2_process(out)
            
            if self.downsample is not None:
                identity = self.downsample_process(x)
                
            out += identity
            out = self.relu(out)
            return out, (next_cache1, next_cache2)

  
class ResNet_nopool(nn.Module):

    def __init__(self, block, layers, in_channel=17, zero_init_residual=False):
        super(ResNet_nopool, self).__init__()
        # ResNet_nopool(BasicBlock, [2, 2, 2, 2], **kwargs) in_chennel = 7
        # input size: [32,7,500,64]
        self.inplanes = 24
        # self.conv1 = nn.Conv2d(in_channel, 24, kernel_size=(3, 3), stride=1, padding=(2, 1),
                            #    bias=False, dilation=(2, 1))
        self.conv1 = CausalConv2D(in_channel, 24, kernel_size=3, stride=1, padding=None,
                               bias=False)
        self.ln1 = nn.LayerNorm(64*24)
        # self.bn1 = nn.GroupNorm(1, 24)
        # self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 24, layers[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 4))
        self.layer2 = self._make_layer(block, 48, layers[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 4))
        self.layer3 = self._make_layer(block, 96, layers[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.layer4 = self._make_layer(block, 192, layers[3])
        self.conv5 = conv1x1(192, 256)

        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, CausalConv2D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)  # Initialize weight to 1
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)  # Initialize weight to 1
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        # BasicBlock,24,2
        # BasicBlock,48,2
        # BasicBlock,96,2
        # BasicBlock,192,2
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
                # nn.BatchNorm2d(planes * block.expansion),
                # nn.GroupNorm(1, planes * block.expansion), #[32, 48, 500, 16] [32, 96, 500, 4] [32, 192, 500, 2]
                nn.LayerNorm(LN_size),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def ln1_process(self, x):
        N, C, T, W = x.shape
        x = x.permute(0,2,1,3).reshape(N, T, -1)
        x = self.ln1(x)
        x = x.reshape(N, T, C, W)
        x = x.permute(0,2,1,3)
        return x

    def get_initial_cache(self, batch_size=1):
        device = next(self.parameters()).device
        
        # conv1的缓存
        conv1_cache = torch.zeros(batch_size, 7, 2, 64, device=device)  # 初始输入通道是7
        
        # 为每个layer中的每个block准备缓存
        layer_caches = []
        channels = [(24, 24), (48, 48), (96, 96), (192, 192)]  # (in_channels, out_channels)
        features = [64, 16, 4, 2]  # 特征维度
        
        for layer_idx, ((in_ch, out_ch), feat_dim) in enumerate(zip(channels, features)):
            layer_cache = []
            for block_idx in range(2):  # 每层2个block
                if block_idx == 0 and layer_idx > 0:
                    # 第一个block需要处理通道数变化
                    prev_ch = channels[layer_idx-1][1]
                    cache1 = torch.zeros(batch_size, prev_ch, 2, feat_dim, device=device)
                else:
                    cache1 = torch.zeros(batch_size, in_ch, 2, feat_dim, device=device)
                cache2 = torch.zeros(batch_size, out_ch, 2, feat_dim, device=device)
                layer_cache.extend([cache1, cache2])
            layer_caches.append(layer_cache)
        
        return conv1_cache, layer_caches

    def forward(self, x, caches=None):
        if caches is None:
            x = self.conv1(x)
            x = self.ln1_process(x)
            x = self.relu(x)
            
            for i, layer in enumerate([self.layer1, self.layer2, 
                                    self.layer3, self.layer4]):
                x = layer(x)  # 这里没有缓存，直接调用
                if i < 3:
                    x = getattr(self, f'maxpool{i+1}')(x)
                    
            x = self.conv5(x)
            return x
        else:
            conv1_cache, layer_caches = caches
            x, next_conv1_cache = self.conv1(x, conv1_cache)
            x = self.ln1_process(x)
            x = self.relu(x)
            
            next_layer_caches = []
            for i, layer in enumerate([self.layer1, self.layer2, 
                                    self.layer3, self.layer4]):
                if isinstance(layer, nn.Sequential):
                    layer_cache = layer_caches[i]
                    current_layer_caches = []  # 为当前layer创建新的cache列表
                    
                    for j, block in enumerate(layer):
                        cache_idx = j * 2
                        block_cache = (layer_cache[cache_idx], layer_cache[cache_idx + 1])
                        x, next_block_cache = block(x, block_cache)
                        current_layer_caches.extend(next_block_cache)  # 将block的cache添加到当前layer的cache列表
                    
                    next_layer_caches.append(current_layer_caches)  # 将当前layer的所有cache作为一个整体添加
                else:
                    x, next_cache = layer(x, layer_caches[i])
                    next_layer_caches.append([next_cache])  # 保持一致的列表结构
                    
                if i < 3:
                    x = getattr(self, f'maxpool{i+1}')(x)
                    
            x = self.conv5(x)
            return x, (next_conv1_cache, next_layer_caches)

def resnet18_nopool(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet_nopool(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


if __name__ == "__main__":
    input = torch.randn(32,7,500,64)
    model = resnet18_nopool(in_channel=7)
    # print(model)
    out = model(input)
    # with open('/disk6/yxdong/Dcase2023/Samsung-SELD_v1/models/model_structure.txt', 'w') as f:
    #     print(model, file=f)
    print(out.shape) #torch.Size([32, 256, 500, 2])