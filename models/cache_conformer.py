import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
import pdb

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()
    
class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()
    
class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)
        self.cache_drop_size = None
        self.padding = padding

    def update_cache(self, x, cache=None):
        if cache is None:
            x = F.pad(x, self.padding)
            return x, None
        else:
            # pdb.set_trace()
            # 在时间维度上拼接缓存
            x = torch.cat([cache, x], dim=-1)
            if self.cache_drop_size:
                next_cache = x[..., :-self.cache_drop_size]
            else:
                next_cache = x[..., -self.kernel_size + 1:]
            return x, next_cache

    def forward(self, x, cache=None):
        x, next_cache = self.update_cache(x, cache)
        x = self.conv(x)
        if cache is None:
            return x
        else:
            return x, next_cache

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class CausalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., max_pos_emb=512, att_context_size=[100,49]):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.max_pos_emb = max_pos_emb
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
        self.dropout = nn.Dropout(dropout)
        self.cache_drop_size = None
        self.max_cache_len = att_context_size[0]
        self.att_context_size = att_context_size

    # def forward(self, x, mask=None, cache=None):
    #     n, device, h = x.shape[-2], x.device, self.heads
        
    #     # 区分训练和推理模式
    #     is_training = cache is None
        
    #     if not is_training:
    #         # 推理模式：使用cache
    #         context = torch.cat([cache, x], dim=1) if cache is not None else x
    #         if self.cache_drop_size:
    #             next_cache = context[:, :-self.cache_drop_size, :]
    #         else:
    #             next_cache = context[:, -self.max_cache_len:, :]
    #     else:
    #         # 训练模式：使用完整序列
    #         context = x
    #         next_cache = None

    #     context_len = context.shape[1]
    #     # n, device, h = context.shape[-2], context.device, self.heads
    #     q = self.to_q(x)
    #     # q = self.to_q(context)
    #     k, v = self.to_kv(context).chunk(2, dim=-1)
        
    #     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
    #     dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
    #     # 相对位置编码
    #     seq_q = torch.arange(n, device=device)
    #     seq_k = torch.arange(context_len, device=device)
    #     dist = rearrange(seq_q, 'i -> i ()') - rearrange(seq_k, 'j -> () j')
    #     dist = dist.clamp(-self.max_pos_emb, self.max_pos_emb) + self.max_pos_emb
    #     rel_pos_emb = self.rel_pos_emb(dist)
    #     pos_attn = einsum('b h n d, n j d -> b h n j', q, rel_pos_emb) * self.scale
        
    #     dots = dots + pos_attn

    #     # 只在训练模式下应用chunk mask
    #     if is_training:
    #         chunk_size = self.att_context_size[1] + 1
    #         left_chunks_num = self.att_context_size[0] // chunk_size if self.att_context_size[0] >= 0 else 0
            
    #         # 创建chunk mask
    #         chunk_idx = torch.arange(0, n, dtype=torch.int, device=device)
    #         chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
    #         diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
    #         chunked_limited_mask = torch.logical_and(
    #             torch.le(diff_chunks, left_chunks_num),
    #             torch.ge(diff_chunks, 0)
    #         )
    #         att_mask = torch.ones(1, n, n, dtype=torch.bool, device=device)
    #         att_mask = torch.logical_and(att_mask, chunked_limited_mask.unsqueeze(0))
    #         att_mask = att_mask.squeeze(0)

    #         # 应用mask
    #         dots.masked_fill_(~att_mask, float('-inf'))

    #     attn = dots.softmax(dim=-1)
    #     out = einsum('b h i j, b h j d -> b h i d', attn, v)
    #     out = rearrange(out, 'b h n d -> b n (h d)')
    #     out = self.to_out(out)
    #     out = self.dropout(out)

    #     # out = out[:,-self.att_context_size[1] - 1:, :]
        
    #     if not is_training:
    #         return out, next_cache
    #     return out

    def forward(self, x, mask=None, cache=None):
        # n, device, h = x.shape[-2], x.device, self.heads
        
        # 区分训练和推理模式
        is_training = cache is None
        
        if not is_training:
            # 推理模式：使用cache
            context = torch.cat([cache, x], dim=1) if cache is not None else x
            if self.cache_drop_size:
                next_cache = context[:, :-self.cache_drop_size, :]
            else:
                next_cache = context[:, -self.max_cache_len:, :]
        else:
            # 训练模式：使用完整序列
            context = x
            next_cache = None

        context_len = context.shape[1]
        n, device, h = context.shape[-2], context.device, self.heads
        # q = self.to_q(x)
        q = self.to_q(context)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # 相对位置编码
        seq_q = torch.arange(n, device=device)
        seq_k = torch.arange(context_len, device=device)
        dist = rearrange(seq_q, 'i -> i ()') - rearrange(seq_k, 'j -> () j')
        dist = dist.clamp(-self.max_pos_emb, self.max_pos_emb) + self.max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist)
        pos_attn = einsum('b h n d, n j d -> b h n j', q, rel_pos_emb) * self.scale
        
        dots = dots + pos_attn

        # 只在训练模式下应用chunk mask
        if is_training:
            chunk_size = self.att_context_size[1] + 1
            left_chunks_num = self.att_context_size[0] // chunk_size if self.att_context_size[0] >= 0 else 0
            
            # 创建chunk mask
            chunk_idx = torch.arange(0, n, dtype=torch.int, device=device)
            chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
            diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
            chunked_limited_mask = torch.logical_and(
                torch.le(diff_chunks, left_chunks_num),
                torch.ge(diff_chunks, 0)
            )
            att_mask = torch.ones(1, n, n, dtype=torch.bool, device=device)
            att_mask = torch.logical_and(att_mask, chunked_limited_mask.unsqueeze(0))
            att_mask = att_mask.squeeze(0)

            # 应用mask
            dots.masked_fill_(~att_mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)

        # out = out[:,-self.att_context_size[1] - 1:, :]
        
        if not is_training:
            out = out[:,-self.att_context_size[1] - 1:, :]
            return out, next_cache
        else:
            return out

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = True,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = (kernel_size - 1, 0) if causal else calc_same_padding(kernel_size)
        
        self.norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, inner_dim * 2, 1)
        self.glu = GLU(dim=1)
        self.depth_conv = DepthWiseConv1d(
            inner_dim, inner_dim,
            kernel_size=kernel_size,
            padding=padding
        )
        self.batch_norm = nn.BatchNorm1d(inner_dim) if not causal else nn.Identity()
        self.swish = Swish()
        self.conv2 = nn.Conv1d(inner_dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cache=None):
        # pdb.set_trace()
        x = self.norm(x)
        x = rearrange(x, 'b n c -> b c n')
        
        x = self.conv1(x)
        x = self.glu(x)
        
        if cache is None:
            x = self.depth_conv(x)
        else:
            x, next_cache = self.depth_conv(x, cache)
            
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.conv2(x)
        x = rearrange(x, 'b c n -> b n c')
        x = self.dropout(x)
        
        if cache is None:
            return x
        else:
            return x, next_cache

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.,
        ff_dropout=0.,
        conv_dropout=0.,
        att_context_size = [100,49]
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = CausalAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            att_context_size = att_context_size
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=True,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None, cache=None):
        if cache is None:
            x = self.ff1(x) + x
            # pdb.set_trace()
            x = self.attn(x, mask=mask) + x
            x = self.conv(x) + x
            x = self.ff2(x) + x
            x = self.post_norm(x)
            return x
        else:
            attn_cache, conv_cache = cache
            x = self.ff1(x) + x
            attn_out, next_attn_cache = self.attn(x, mask=mask, cache=attn_cache)
            # pdb.set_trace()
            x = attn_out + x
            conv_out, next_conv_cache = self.conv(x, cache=conv_cache)
            x = conv_out + x
            x = self.ff2(x) + x
            x = self.post_norm(x)
            return x, (next_attn_cache, next_conv_cache)

if __name__ == "__main__":
    input = torch.randn(32, 500, 256)
    model = ConformerBlock(
                dim = 256,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1
            )
    # print(model)
    out = model(input)
    print(out.shape) #torch.Size([32, 500, 256])