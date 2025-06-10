import torch
import torch.nn as nn

from .resnet import resnet18_nopool
from .conformer import ConformerBlock
from .CPC import CPCModule

class ResnetConformer_sed_doa_nopool(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, cpc_future_steps = 10, negative_samples = 10):
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
        self.cpc = CPCModule(
            input_dim=encoder_dim,
            future_steps=cpc_future_steps,
            negative_samples = negative_samples
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

        cpc_loss = self.cpc.compute_cpc_loss(conformer_outputs)
        outputs = conformer_outputs.permute(0,2,1)
        outputs = self.t_pooling(outputs)
        outputs = outputs.permute(0,2,1)
        sed = self.sed_out_layer(outputs)
        doa = self.out_layer(outputs)
        pred = torch.cat((sed, doa), dim=-1)
        return pred, cpc_loss