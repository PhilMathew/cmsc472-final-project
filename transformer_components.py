import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0):
        super(MLP, self).__init__(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )


class ViTEmbedding(nn.Module):
    def __init__(self, in_channels, seq_length, patch_size, hidden_size):
        super(ViTEmbedding, self).__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        # TODO Add the masking
        self.patch_embeddings = nn.Conv1d(in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, (seq_length // patch_size) + 1, hidden_size))
    
    def forward(self, x):
        patch_emb = self.patch_embeddings(x).permute(0, 2, 1)
        
        cls_tokens = torch.repeat_interleave(self.cls_token, x.shape[0], dim=0)
        emb = torch.cat([cls_tokens, patch_emb], dim=1)
        
        emb += self.position_embeddings
        
        return emb


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attn_linear = nn.Linear(hidden_size, 3 * hidden_size * num_heads)
        self.attn_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(num_heads * hidden_size, hidden_size)
    
    def forward(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.attn_linear(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(b, self.num_heads, n, self.hidden_size) for t in qkv]
        
        # sum over last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            mask = mask.expand(1, 1, -1, -1).permute(2, 0, 1, 3).repeat(1, energy.shape[1], energy.shape[2], 1) # gets mask into correct shape
            energy.masked_fill(~mask, fill_value)
        
        scaling = self.hidden_size**(1/2)
        attn = F.softmax(energy, dim=-1) / scaling
        attn = self.attn_drop(attn)
        
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', attn, v).reshape(b, n, self.num_heads * self.hidden_size)
        out = self.projection(out)

        return out


class ViTBlock(nn.Module):
    def __init__(self, hidden_size=768, attn_dropout_p=0, forward_expansion=1, mlp_drop_p=0, **kwargs):
        super(ViTBlock, self).__init__()
        self.hidden_size = hidden_size
        
        # Residual block #1
        self.ln1 = nn.LayerNorm(hidden_size)
        self.multi_head_attn = MultiHeadAttention(hidden_size, **kwargs)
        self.drop1 = nn.Dropout(attn_dropout_p)
        
        # Residual block #2
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, forward_expansion * hidden_size, dropout=mlp_drop_p)
        self.drop2 = nn.Dropout(attn_dropout_p)
    
    def forward(self, x, mask=None):
        # Block #1 forward
        res1 = x
        x = self.ln1(x)
        x = self.multi_head_attn(x, mask=mask)
        x = self.drop1(x)
        x += res1
        
        # Block #2 forward
        res2 = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = self.drop2(x)
        x += res2
        
        return x
    

class ViTEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super(ViTEncoder, self).__init__(*[ViTBlock(**kwargs) for _ in range(depth)])
    
    def forward(self, x, mask=None):
        for module in self:
            x = module(x, mask=mask)
        
        return x


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size=768, n_classes=2):
        super(ClassificationHead, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.layer_norm(x)
        x = self.fc(x)
        
        return x

