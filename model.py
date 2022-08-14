import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.autograd import Variable

import numpy as np
import math
import time

from torch.nn import TransformerEncoderLayer

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 1024):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return x

class AttentionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        head_dim,
        mask = False,
        device = 'cuda'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(input_dim, head_dim)
        self.k_proj = nn.Linear(input_dim, head_dim)
        self.v_proj = nn.Linear(input_dim, head_dim)
        
        self.mask = mask
        self.device = device
        
    def get_qkv(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        return q, k, v
    
    def _generate_mask(self, dim):
        mask = (torch.triu(torch.ones(dim, dim)) == 1).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        
        return mask
    
    def attention(self, x, mask=False):
        
        # get q, k, v
        q, k, v = self.get_qkv(x) # (batch_size x seq_len x head_dim)

        # compute attn
        attention = q @ torch.transpose(k, -1, -2) # (batch_size x seq_len x seq_len)
        
        #if mask: apply to attn
        if mask:
            mask = self._generate_mask(attention.shape[-1]).to(self.device)
            attention = attention + mask
        
        #scale and normalize attn
        attention /= np.sqrt(self.head_dim)
        attention = F.softmax(attention, dim=-1) # (batch_size x seq_len x seq_len)

        #get new tokens
        new_tokens = attention @ v # (batch_size x seq_len x head_dim)
        
        return new_tokens
    
    def forward(self, x):
        return self.attention(x, self.mask)

class MultiheadAttentionLayer(nn.Module):
    def __init__(
        self,
        n_heads = 4,
        input_dim = 512,
        attn_dim = 128,
        dropout = 0,
        mask = False,
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.attn_dim = attn_dim
        self.input_dim = input_dim
        self.mask = mask


        self.dropout0 = nn.Dropout(p=dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.attn_head_dim = self.attn_dim // self.n_heads
        self.mha_mlp = nn.Linear(attn_dim, input_dim)
        self.ln0 = nn.LayerNorm(attn_dim)
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        
        self.attn_heads = nn.ModuleList([
            AttentionHead(self.input_dim, self.attn_head_dim, self.mask)
            for _ in range(self.n_heads)
        ])
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim*4, input_dim)
        )
        
        self.final_ln = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        
        attn_outputs = [
            head(x)
            for head in self.attn_heads
        ]
        
        concat = torch.concat(attn_outputs, axis=-1)
        mha_output = self.mha_mlp(self.ln0(concat))
        attn_output = self.ln1(self.dropout0(mha_output) + x)

        mlp_out = self.mlp(attn_output)
        out = self.ln2(self.dropout1(mlp_out) + attn_output)
        
        return out

class CLIPFormer(nn.Module):
    def __init__(
        self, 
        input_dim = 512,
        attn_dim = 128,
        n_layers = 4,
        n_heads = 4,
        out_dim = 512,
        seq_len = 1024,
        dropout = 0,
        mask = False,
        using_tel = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.p_dropout = dropout
        self.out_dim = out_dim
        self.seq_len = seq_len
        
        self.mask = mask

        if using_tel:
            self.blocks = nn.ModuleList([
                TransformerEncoderLayer(self.input_dim, n_heads, activation=nn.GELU(), batch_first = True)
                for _ in range(n_layers)
            ])
            
        else:
            self.blocks = nn.ModuleList([
                MultiheadAttentionLayer(n_heads = self.n_heads, input_dim = self.input_dim, attn_dim = self.attn_dim, mask = self.mask)
                for _ in range(n_layers)
            ])
        
        self.out_proj = nn.Linear(input_dim, out_dim)
        
        self.pos_encoding = PositionalEncoder(input_dim, seq_len)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.blocks.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def forward(self, tokens, mask=None):
        
        emb = self.pos_encoding(tokens)
        for block in self.blocks:
            emb = block(emb)
            
        emb = self.out_proj(emb)
        
        return emb
    
    @property
    def _num_params(self):
        total_params = 0
        for p in list(self.parameters()):
            layer_params = 1
            
            for s in list(p.size()):
                layer_params *= s
                
            total_params += layer_params
            
        return total_params