import torch
import torch.nn as nn

from .casual_resnet import resnet18_nopool
from .casual_conformer_chunk_limited import ConformerBlock
from .mix import Mixup
from .resnet_conformer_audio import ResnetConformer_sed_doa_nopool_original
# import pdb

layer_resnet = ['conv1', 'bn1', 'relu', 'layer1', 'layer1.0', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.1', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'maxpool1', 'layer2', 'layer2.0', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.1', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu', 'layer2.1.conv2', 'layer2.1.bn2', 'maxpool2', 'layer3', 'layer3.0', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.downsample', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.1', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2', 'layer3.1.bn2', 'maxpool3', 'layer4', 'layer4.0', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.downsample', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.1', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'conv5']

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResnetConformer_sed_doa_nopool(nn.Module):
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
    def forward(self, x):
        conv_outputs = self.resnet(x)
        N,C,T,W = conv_outputs.shape
        conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)

        conformer_outputs = self.input_projection(conv_outputs) #torch.Size([32, 500, 256])
        # pdb.set_trace()
        #conformer_outputs = conv_outputs
        for layer in self.conformer_layers:
            conformer_outputs = layer(conformer_outputs) #torch.Size([32, 500, 256])
        outputs = conformer_outputs.permute(0,2,1)
        outputs = self.t_pooling(outputs)
        outputs = outputs.permute(0,2,1)
        sed = self.sed_out_layer(outputs)
        doa = self.out_layer(outputs)
        pred = torch.cat((sed, doa), dim=-1)
        return pred
    

class ResnetConformer_sed_doa_nopool_Mixup(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim):
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        self.ManifoldMixup_resnet = Mixup(self.resnet, mixed_layers=layer_resnet)
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

    def forward(self, x, target=None):
        if target == None:
            conv_outputs = self.resnet(x)
            N,C,T,W = conv_outputs.shape
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
            conv_outputs, target, weights = self.ManifoldMixup_resnet(x, target)
            N,C,T,W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)

            conformer_outputs = self.input_projection(conv_outputs)
            #conformer_outputs = conv_outputs
            for layer in self.conformer_layers:
                conformer_outputs = layer(conformer_outputs)
            outputs = conformer_outputs.permute(0,2,1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0,2,1)
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            return pred, target, weights[0]
        
class ResnetConformer_sed_doa_nopool_Mixup_TS(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim):
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        self.ManifoldMixup_resnet = Mixup(self.resnet, mixed_layers=layer_resnet)
        self.teacher_model = ResnetConformer_sed_doa_nopool_original(in_channel=7, in_dim=64, out_dim=39)
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
        for param in self.teacher_model.parameters():
            param.requires_grad = False


    def forward(self, x, target=None):
        if target == None:
            target_ts = self.teacher_model(x)
            conv_outputs = self.resnet(x)
            N,C,T,W = conv_outputs.shape
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

            return pred, target_ts
        else:
            target_ts = self.teacher_model(x)
            conv_outputs, target, weights = self.ManifoldMixup_resnet(x, target)
            N,C,T,W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)

            conformer_outputs = self.input_projection(conv_outputs)
            #conformer_outputs = conv_outputs
            for layer in self.conformer_layers:
                conformer_outputs = layer(conformer_outputs)
            outputs = conformer_outputs.permute(0,2,1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0,2,1)
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            return pred, target, weights[0], target_ts

if __name__ == "__main__":
    input = torch.randn(32,7,500,64)
    model = ResnetConformer_sed_doa_nopool(7, 64, 39)
    # print(model)
    out = model(input)
    with open('/disk6/yxdong/Dcase2023/Samsung-SELD_v1/models/model_structure.txt', 'w') as f:
        print(model, file=f)
    print(out.shape) #torch.Size([32, 256, 500, 2])