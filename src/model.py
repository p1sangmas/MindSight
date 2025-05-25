import torch
import torch.nn as nn
import torchvision.models as models
from einops.layers.torch import Rearrange

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base_model = models.efficientnet_b0(pretrained=True)
        self.backbone = nn.Sequential(*list(base_model.features.children()))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.project = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.ReLU(),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.transformer = nn.Sequential(
            TransformerBlock(dim=256, heads=4, mlp_dim=512),
            TransformerBlock(dim=256, heads=4, mlp_dim=512),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.project(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        return self.head(x)
