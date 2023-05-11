import torch
from torch import nn
from transformer_components import ViTEmbedding, ViTEncoder, ClassificationHead


class ViTClassifier(nn.Module):
    def __init__(self, in_channels=12, patch_size=20, hidden_size=768, seq_length=5000, depth=12, n_classes=2, **kwargs):
        super(ViTClassifier, self).__init__()
        
        self.patch_embedding = ViTEmbedding(in_channels, seq_length, patch_size, hidden_size)
        self.encoder = ViTEncoder(depth, hidden_size=hidden_size, **kwargs)
        self.clf_head = ClassificationHead(hidden_size, n_classes)
        
    def forward(self, x, mask=None):
        x = self.patch_embedding(x)
        x = self.encoder(x, mask=mask)
        x = self.clf_head(x)

        return x
