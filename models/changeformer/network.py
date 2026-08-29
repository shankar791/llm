"""
ChangeFormer (MiT-B0 Siamese Architecture for Bi-Temporal Change Detection).
Standalone PyTorch implementation isolated under models/changeformer/.
Reference: Bandara & Patel, "A Transformer-Based Siamese Network for Change Detection" (IGARSS 2022).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping convolutions."""
    def __init__(self, img_size: int = 256, patch_size: int = 7, stride: int = 4,
                 in_chans: int = 3, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class EfficientAttention(nn.Module):
    """Multi-head attention with spatial reduction."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 sr_ratio: int = 1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MixFFN(nn.Module):
    """Mix-FeedForward Network with depthwise 3x3 conv."""
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, sr_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class MiTEncoder(nn.Module):
    """Mix Transformer (MiT-B0) Encoder backbone."""
    def __init__(self, in_chans: int = 3, embed_dims: List[int] = [32, 64, 160, 256],
                 num_heads: List[int] = [1, 2, 5, 8], mlp_ratios: List[int] = [4, 4, 4, 4],
                 depths: List[int] = [2, 2, 2, 2], sr_ratios: List[int] = [8, 4, 2, 1]):
        super().__init__()
        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.block1 = nn.ModuleList([TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratio=sr_ratios[0]) for _ in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.block2 = nn.ModuleList([TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratio=sr_ratios[1]) for _ in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.block3 = nn.ModuleList([TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratio=sr_ratios[2]) for _ in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.block4 = nn.ModuleList([TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratio=sr_ratios[3]) for _ in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        outs.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())

        # Stage 2
        x, H, W = self.patch_embed2(outs[-1])
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        outs.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())

        # Stage 3
        x, H, W = self.patch_embed3(outs[-1])
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        outs.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())

        # Stage 4
        x, H, W = self.patch_embed4(outs[-1])
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        outs.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())

        return outs


class ChangeDecoder(nn.Module):
    """Multi-Scale Difference Feature Fusion & Change Mask Decoder."""
    def __init__(self, in_channels: List[int] = [32, 64, 160, 256], embedding_dim: int = 128,
                 num_classes: int = 2):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Conv2d(in_channels[0], embedding_dim, 1), nn.BatchNorm2d(embedding_dim), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv2d(in_channels[1], embedding_dim, 1), nn.BatchNorm2d(embedding_dim), nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Conv2d(in_channels[2], embedding_dim, 1), nn.BatchNorm2d(embedding_dim), nn.ReLU())
        self.mlp4 = nn.Sequential(nn.Conv2d(in_channels[3], embedding_dim, 1), nn.BatchNorm2d(embedding_dim), nn.ReLU())

        self.fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(embedding_dim, num_classes, 1)
        )

    def forward(self, feats_t0: List[torch.Tensor], feats_t1: List[torch.Tensor]) -> torch.Tensor:
        # Multi-scale absolute difference
        diff1 = torch.abs(feats_t0[0] - feats_t1[0])
        diff2 = torch.abs(feats_t0[1] - feats_t1[1])
        diff3 = torch.abs(feats_t0[2] - feats_t1[2])
        diff4 = torch.abs(feats_t0[3] - feats_t1[3])

        target_size = diff1.shape[2:]  # H/4, W/4

        p1 = self.mlp1(diff1)
        p2 = F.interpolate(self.mlp2(diff2), size=target_size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.mlp3(diff3), size=target_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.mlp4(diff4), size=target_size, mode="bilinear", align_corners=False)

        fused = torch.cat([p1, p2, p3, p4], dim=1)
        return self.fuse(fused)


class ChangeFormerB0(nn.Module):
    """
    Complete Standalone ChangeFormer Network with Siamese MiT-B0 Backbone.
    Accepts two input images (T0, T1), extracts Siamese multi-scale representations,
    computes feature differences, and outputs a 2-class change prediction logit map.
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 2, embed_dims: List[int] = [32, 64, 160, 256]):
        super().__init__()
        self.encoder = MiTEncoder(in_chans=in_chans, embed_dims=embed_dims)
        self.decoder = ChangeDecoder(in_channels=embed_dims, embedding_dim=128, num_classes=num_classes)

    def forward(self, img_t0: torch.Tensor, img_t1: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for bi-temporal image pair.
        Args:
            img_t0: Tensor of shape (B, 3, H, W)
            img_t1: Tensor of shape (B, 3, H, W)
        Returns:
            Logits tensor of shape (B, 2, H, W)
        """
        H, W = img_t0.shape[2:]
        feats_t0 = self.encoder(img_t0)
        feats_t1 = self.encoder(img_t1)
        logits_low = self.decoder(feats_t0, feats_t1)
        # Upsample directly to full input resolution
        logits = F.interpolate(logits_low, size=(H, W), mode="bilinear", align_corners=False)
        return logits
