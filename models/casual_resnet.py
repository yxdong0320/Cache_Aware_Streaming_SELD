# from sympy import N
import torch
import torch.nn as nn
import torch.nn.init as init
import pdb
from typing import Union
import torch.nn.functional as F

# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 causal convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride,
#                      padding=(2, 1), bias=False, dilation=(2, 1))

class CausalConv2D(nn.Conv2d):
    """
    A causal version of nn.Conv2d where each location in the 2D matrix would have no access to locations on its right or down
    All arguments are the same as nn.Conv2d except padding which should be set as None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        if padding is not None:
            raise ValueError("Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1

        padding = 0
        super(CausalConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(
        self, x,
    ):
        x = F.pad(x, pad=(self._left_padding, self._right_padding, self._left_padding, self._right_padding))
        x = super().forward(x)
        return x

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
        # planes: 24 48 96 192 
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        if planes == 24:
            LN_size = 24*64
        elif planes == 48:
            LN_size = 48*16
        elif planes == 96:
            LN_size = 96*4
        elif planes == 192:
            LN_size = 192*2
        # self.bn1 = nn.GroupNorm(1, planes)
        self.ln1 = nn.LayerNorm(LN_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.GroupNorm(1, planes)
        self.ln2 = nn.LayerNorm(LN_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x) 
        N, C, T, W = out.shape
        out = out.permute(0,2,1,3).reshape(N, T, -1) 
        out = self.ln1(out) # [32, 24, 500, 64]*2 [32, 48, 500, 16]*2 [32, 96, 500, 4]*2 [32, 192, 500, 2]*2
        out = out.reshape(N, T, C, W)
        out = out.permute(0,2,1,3)
        out = self.relu(out)

        out = self.conv2(out)
        N, C, T, W = out.shape
        out = out.permute(0,2,1,3).reshape(N, T, -1)
        out = self.ln2(out) # [32, 24, 500, 64]*2 
        out = out.reshape(N, T, C, W)
        out = out.permute(0,2,1,3)

        if self.downsample is not None:
            identity = self.downsample[0](x)
            N ,C, T, W = identity.shape
            identity = identity.permute(0,2,1,3).reshape(N, T, -1)
            identity = self.downsample[1](identity) #[32, 48, 500, 16] [32, 96, 500, 4] [32, 192, 500, 2]
            identity = identity.reshape(N, T, C, W)
            identity = identity.permute(0,2,1,3)
            
        out += identity
        out = self.relu(out)

        return out
    
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
        # pdb.set_trace()
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

    def forward(self, x):
        x = self.conv1(x) # torch.Size([32, 24, 500, 64])
        N, C, T, W = x.shape # 32, 24, 500, 64
        x = x.permute(0,2,1,3).reshape(N, T, -1) # torch.Size([32, 500, 1536])
        x = self.ln1(x) # torch.Size([32, 500, 1536])
        x = x.reshape(N, T, C, W)
        x = x.permute(0,2,1,3)  # torch.Size([32, 24, 500, 64])

        x = self.relu(x)

        x = self.layer1(x) # torch.Size([32, 24, 500, 64])
        x = self.maxpool1(x)
        x = self.layer2(x) # torch.Size([32, 48, 500, 16])
        x = self.maxpool2(x)
        x = self.layer3(x) # torch.Size([32, 96, 500, 4])
        x = self.maxpool3(x)
        x = self.layer4(x) # torch.Size([32, 192, 500, 2])

        x = self.conv5(x)

        return x

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
    with open('/disk6/yxdong/Dcase2023/Samsung-SELD_v1/models/model_structure.txt', 'w') as f:
        print(model, file=f)
    print(out.shape) #torch.Size([32, 256, 500, 2])