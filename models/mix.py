import os, shutil, argparse

import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import yaml

# from models.resnet_conformer_2 import ResnetConformer_nopool, ResnetConformer_sed_doa_nopool

from models.resnet import resnet18, resnet18_nopool, BasicBlock


# 直接在resnet上面做,不调用整个模型
class Mixup(nn.Module):
    def __init__(self, model, mix_probability=1.1, alpha=1, batch_first=True, mixed_layers=[]):
        super().__init__()
        self.model = model
        self.mix_probability = mix_probability
        self.alpha = alpha
        self.batch_first = batch_first
      
        self.module_list = []
        for n, m in self.model.named_modules():   
            #if n in mixed_layers:
            #if 'conv' in n:
            if n[:-1] == 'layer':
            #if 'bn2' in n:
                self.module_list.append(m)
        #print(self.module_list)
        #print(len(self.module_list))             

    def forward(self, x, label):
        batch_size = x.shape[0]
        labels = [label]
        weights = []
        
        if np.random.uniform(0, 1) < self.mix_probability:
            self.indices = torch.randperm(batch_size)
            self.lam = np.random.beta(self.alpha, self.alpha)
            weights.append(self.lam)
            labels.append(label[self.indices])
            
            # mixed_layer_idx = np.random.randint(-1, len(self.module_list))
            mixed_layer_idx = -1

            if mixed_layer_idx == -1:
                x = (1 - self.lam) * x + self.lam * x[self.indices]
                y = self.model(x)
            else:
                modifier_hook = self.module_list[mixed_layer_idx].register_forward_hook(self.hook_modify)
                y = self.model(x)
                modifier_hook.remove()
        else:
            y = self.model(x)
            weights.append(1)
            labels.append(label)

        
        return y, labels, weights

    def hook_modify(self, module, input, output):
        if isinstance(output, torch.Tensor):
            if self.batch_first:
                output = (1 - self.lam) * output + self.lam * output[self.indices]
            else:
                output = (1 - self.lam) * output + self.lam * output[:, self.indices]
        else:
            raise NotImplementedError('unknown output for module')
        return output

if __name__ == '__main__':

    #layers = ['conv1', 'bn1', 'relu', 'layer1', 'layer1.0', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.1', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'maxpool1', 'layer2', 'layer2.0', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.1', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu', 'layer2.1.conv2', 'layer2.1.bn2', 'maxpool2', 'layer3', 'layer3.0', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.downsample', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.1', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2', 'layer3.1.bn2', 'maxpool3', 'layer4', 'layer4.0', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.downsample', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.1', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'conv5']
    layers = ['layer1.0.bn2', 'layer1.1.bn2', 'layer2.0.bn2', 'layer2.1.bn2', 'layer3.0.bn2', 'layer3.1.bn2', 'layer4.0.bn2', 'layer4.1.bn2']
    
    resnet = resnet18_nopool(in_channel=9)
    model = ManifoldMixup(resnet,mixed_layers = layers)

    # 这里把需要把参数穿进去，考虑到真实情况下有可能在第一层，所以x和vx两个都需要传入
    # 或者直接在resnet——conformer网络里面添加？y

    checkout_x = torch.randn(9, 9, 500, 64)
    checkout_label = torch.ones(9, 2, 500, 64)

    checkout_y, checkout_label, weights = model(
        checkout_x, checkout_label)

    print(checkout_y.shape)
    print(weights)