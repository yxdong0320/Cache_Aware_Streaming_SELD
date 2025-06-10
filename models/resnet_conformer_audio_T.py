import torch
import torch.nn as nn

from .resnet import resnet18_nopool
from .conformer_T import ConformerBlock

layer_resnet = ['conv1', 'bn1', 'relu', 'layer1', 'layer1.0', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.1', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'maxpool1', 'layer2', 'layer2.0', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.1', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu', 'layer2.1.conv2', 'layer2.1.bn2', 'maxpool2', 'layer3', 'layer3.0', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.downsample', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.1', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2', 'layer3.1.bn2', 'maxpool3', 'layer4', 'layer4.0', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.downsample', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.1', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'conv5']

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResnetConformer_sed_doa_nopool_original(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim):
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256
        encoder_dim = 256
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = 8
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim = encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1
            ) for _ in range(num_layers)]
        )
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
        self.sed_out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, 13),
            nn.Sigmoid()
        )
        self.out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim),
            nn.Tanh()
        ) 
        
    def forward(self, x, return_hidden_states=False, return_attention=False):
        # 获取输入特征
        conv_outputs = self.resnet(x)
        N, C, T, W = conv_outputs.shape
        conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C*W)
        
        # 投影到encoder维度
        conformer_outputs = self.input_projection(conv_outputs)
        
        # 存储中间状态
        hidden_states = []
        attention_maps = []
        
        # 通过conformer层
        for layer in self.conformer_layers:
            if return_attention:
                conformer_outputs, attn_weights = layer(conformer_outputs, return_attention=True)
                hidden_states.append(conformer_outputs)
                attention_maps.append(attn_weights)
            else:
                conformer_outputs = layer(conformer_outputs)
                if return_hidden_states:
                    hidden_states.append(conformer_outputs)
        
        # 输出处理
        outputs = conformer_outputs.permute(0, 2, 1)
        outputs = self.t_pooling(outputs)
        outputs = outputs.permute(0, 2, 1)
        
        sed = self.sed_out_layer(outputs)
        doa = self.out_layer(outputs)
        pred = torch.cat((sed, doa), dim=-1)
        
        # 根据需要返回不同内容
        if return_hidden_states and return_attention:
            return pred, hidden_states, attention_maps
        elif return_hidden_states:
            return pred, hidden_states
        elif return_attention:
            return pred, attention_maps
        else:
            return pred