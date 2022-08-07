import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding):
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class AttentionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        head_dim,
        mask = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(input_dim, head_dim)
        self.k_proj = nn.Linear(input_dim, head_dim)
        self.v_proj = nn.Linear(input_dim, head_dim)
        
        self.mask = mask
        
    def get_qkv(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        return q, k, v
    
    def _generate_square_subsequent_mask(self, dim):
        mask = (torch.triu(torch.ones(dim, dim)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask
    
    def attention(self, x, mask=False):
        
        # get q, k, v
        q, k, v = self.get_qkv(x) # (batch_size x seq_len x head_dim)

        # compute attn
        attention = q @ torch.transpose(k, -1, -2) # (batch_size x seq_len x seq_len)
        
        #if mask: apply to attn
        if mask:
            mask = self._generate_square_subsequent_mask(attention.shape[-1])
            attention = attention + mask
        
        #scale and normalize attn
        attention /= np.sqrt(self.head_dim)
        attention = F.softmax(attention, dim=1) # (batch_size x seq_len x seq_len)
        
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
        mask = False,
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.attn_dim = attn_dim
        self.input_dim = input_dim
        self.mask = mask
        
        self.attn_head_dim = self.attn_dim // self.n_heads
        self.mha_mlp = nn.Linear(attn_dim, input_dim)
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        
        self.attn_heads = nn.ModuleList([
            AttentionHead(self.input_dim, self.attn_head_dim, self.mask)
            for _ in range(self.n_heads)
        ])
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        
        self.final_ln = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        
        attn_outputs = [
            head(x)
            for head in self.attn_heads
        ]
        
        concat = torch.concat(attn_outputs, axis=-1)
        mha_output = self.mha_mlp(concat)
        attn_output = self.ln1(mha_output + x)

        mlp_out = self.mlp(attn_output)
        out = self.ln2(mlp_out + attn_output)
        
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
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        self.out_dim = out_dim
        self.dropout = dropout
        self.seq_len = seq_len
        
        self.mask = mask
        
        self.blocks = nn.ModuleList([
            MultiheadAttentionLayer(self.n_heads, self.input_dim, self.attn_dim, self.mask)
            for _ in range(n_layers)
        ])
        
        self.out_proj = nn.Linear(input_dim, out_dim)
        
        self.pos_encoding = PositionalEncoding(input_dim, dropout, seq_len)
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