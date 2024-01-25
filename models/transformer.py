# Description: Transformer block for the CoatNet model
import math

import torch
import torch.nn as nn
import torch.nn.init as init

class RelativeAttention(nn.Module):
    def __init__(self, hidden_dim, height, width, head_dim, mem_eff, qkv_bias=False):
        super(RelativeAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = hidden_dim // head_dim
        self.head_dim = head_dim
        self.height = height
        self.width = width
        
        assert self.heads * head_dim == hidden_dim, "Hidden dimension needs to be divisible by number of heads"
        
        if mem_eff:
            self.rel_bias = nn.Parameter(torch.empty([1, (2*height)*(2*width)]))
            self.bmap = lambda x: self.map_bias(x, height, width)
        else:
            self.rel_bias = nn.Parameter(torch.empty([1, height*width, height*width]))
            self.bmap = lambda x: x
        
        init.kaiming_uniform_(self.rel_bias, a=math.sqrt(5))
        
        self.query = nn.Linear(head_dim, head_dim, bias=qkv_bias)
        self.key = nn.Linear(head_dim, head_dim, bias=qkv_bias)
        self.value = nn.Linear(head_dim, head_dim, bias=qkv_bias)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        
    @staticmethod
    def map_bias(p, h, w):
        def indexing_vectorized(x):
            m, n = x//(h*w), x%(h*w)
            i, j = m//w, m%w
            ip, jp = n//w, n%w
            return (i-ip+h)*(2*w) + (j-jp+w)
        idxs = torch.arange((h**2)*(w**2), device=p.device)
        idxs = indexing_vectorized(idxs)
        return p[0][idxs].view(1, h*w, h*w)
    
    def forward(self, x):
        N = x.shape[0]
        x = x.reshape(N, self.height*self.width, self.heads, self.head_dim)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attention = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        bias = self.bmap(self.rel_bias).unsqueeze(0)
        attention = torch.softmax((attention+bias) / (self.head_dim ** 0.5), dim=-1)
        
        out = torch.einsum("nhqv,nvhd->nqhd", [attention, v]).reshape(N, self.height*self.width, -1)
        return self.fc_out(out), attention
    
class FFN(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, expansion_factor*hidden_dim)
        self.fc2 = nn.Linear(expansion_factor*hidden_dim, hidden_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
    
class Transformer(nn.Module):
    def __init__(self, hidden_dim, height, width, head_dim, mem_eff, expansion_factor, 
                 qkv_bias=False, stochastic_rate=0.2):
        super(Transformer, self).__init__()
        self.attn = RelativeAttention(hidden_dim, height, width, head_dim, mem_eff, qkv_bias)
        self.ffn = FFN(hidden_dim, expansion_factor)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.stochastic_rate = stochastic_rate
    
    def stochastic_depth(self, layer, x):
        if not self.training:
            if torch.rand(1) < self.stochastic_rate:
                return layer(x)
            else:
                return x
        else:
            return layer(x) * self.stochastic_rate
        
    def forward(self, x):
        # x += self.attn(self.ln1(x))[0]
        x = self.stochastic_depth(lambda m: m + self.attn(self.ln1(m))[0], x)
        # x += self.ffn(self.ln2(x))
        x = self.stochastic_depth(lambda m: m + self.ffn(self.ln2(m)), x)
        return x
    
class TFMRel(nn.Module):
    def __init__(self, L, D, D_in, height, width, head_dim=32, mem_eff=True, expansion_factor=4, 
                 qkv_bias=False, stochastic_rate=0.2) -> None:
        super(TFMRel, self).__init__()
        self.D = D
        self.L = L
        self.height = height
        self.width = width
        
        self.pool = nn.MaxPool2d(2, 2)
        self.proj = nn.Conv2d(D_in, D, 1)
        self.transformers = nn.Sequential(*[Transformer(D, height, width, head_dim, mem_eff, 
                                                        expansion_factor, qkv_bias, 
                                                        stochastic_rate) for i in range(L)])
    
    def forward(self, x):
        x = self.pool(x)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = self.transformers(x.view(x.shape[0], self.height*self.width, self.D))
        x = x.permute(0, 2, 1).view(x.shape[0], self.D, self.height, self.width)
        return x