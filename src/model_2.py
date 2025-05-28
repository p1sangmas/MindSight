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

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ImprovedEmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base_model = models.efficientnet_b0(pretrained=True)
        
        # Extract features from multiple layers for feature pyramid
        self.layer1 = nn.Sequential(*list(base_model.features[:2]))
        self.layer2 = nn.Sequential(*list(base_model.features[2:4]))
        self.layer3 = nn.Sequential(*list(base_model.features[4:]))
        
        # Channel sizes for efficientnet_b0
        self.spatial_attention = SpatialAttention(1280)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.project = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.ReLU(),
            Rearrange('b c h w -> b (h w) c'),
        )

        # Enhanced transformer with more heads
        self.transformer = nn.Sequential(
            TransformerBlock(dim=256, heads=8, mlp_dim=512, dropout=0.2),
            TransformerBlock(dim=256, heads=8, mlp_dim=512, dropout=0.2),
            TransformerBlock(dim=256, heads=8, mlp_dim=512, dropout=0.2),
        )
        
        # Deeper MLP head with dropout for better regularization
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Multi-scale feature extraction
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        # Apply spatial attention
        x3 = self.spatial_attention(x3)
        x3 = self.pool(x3)
        
        x = self.project(x3)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        return self.head(x)
