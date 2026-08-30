"""
ChangeFormer (Official Architecture for Bi-Temporal Change Detection).
Standalone PyTorch implementation isolated under models/changeformer/.
Reference: Bandara & Patel, "A Transformer-Based Siamese Network for Change Detection" (IGARSS 2022).
"""
from __future__ import annotations
import math
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dNormActivation(nn.Module):
    """Standard Convolution with optional Norm and Activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, deconv: bool = False):
        super().__init__()
        if deconv:
            self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                             stride=stride, padding=padding)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


class DenseBlock(nn.Module):
    """Residual Dense Block used in progressive upsampling."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv2dNormActivation(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return out + residual


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping convolutions."""
    def __init__(self, patch_size: int = 7, stride: int = 4, in_chans: int = 3, embed_dim: int = 64):
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
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, sr_ratio: int = 1):
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


class DWConv(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixFFN(nn.Module):
    """Mix-FeedForward Network with depthwise 3x3 conv."""
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
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


class MLP(nn.Module):
    """Linear MLP projection for decoder stages."""
    def __init__(self, input_dim: int = 2048, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class MiTEncoder(nn.Module):
    """Mix Transformer Encoder backbone."""
    def __init__(self, in_chans: int = 3, embed_dims: List[int] = [64, 128, 320, 512],
                 num_heads: List[int] = [1, 2, 5, 8], mlp_ratios: List[int] = [4, 4, 4, 4],
                 depths: List[int] = [3, 3, 4, 3], sr_ratios: List[int] = [8, 4, 2, 1]):
        super().__init__()
        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(7, 2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(7, 2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(7, 2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

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


class ChangeFormerDecoder(nn.Module):
    """Official ChangeFormer Difference Fusion & Progressive Upsampling Decoder."""
    def __init__(self, in_channels: List[int] = [64, 128, 320, 512], embedding_dim: int = 256,
                 num_classes: int = 2):
        super().__init__()
        self.linear_c1 = MLP(input_dim=in_channels[0], embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=in_channels[1], embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=in_channels[2], embed_dim=embedding_dim)
        self.linear_c4 = MLP(input_dim=in_channels[3], embed_dim=embedding_dim)

        self.diff_c1 = nn.Sequential(nn.Conv2d(embedding_dim * 2, embedding_dim, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(embedding_dim), nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1))
        self.diff_c2 = nn.Sequential(nn.Conv2d(embedding_dim * 2, embedding_dim, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(embedding_dim), nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1))
        self.diff_c3 = nn.Sequential(nn.Conv2d(embedding_dim * 2, embedding_dim, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(embedding_dim), nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1))
        self.diff_c4 = nn.Sequential(nn.Conv2d(embedding_dim * 2, embedding_dim, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(embedding_dim), nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1))

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.convd2x = Conv2dNormActivation(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1, deconv=True)
        self.dense_2x = nn.Sequential(DenseBlock(embedding_dim, embedding_dim))
        self.convd1x = Conv2dNormActivation(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1, deconv=True)
        self.dense_1x = nn.Sequential(DenseBlock(embedding_dim, embedding_dim))
        self.change_probability = Conv2dNormActivation(embedding_dim, num_classes, kernel_size=3, stride=1, padding=1)

        # Multi-scale auxiliary heads
        self.make_pred_c1 = nn.Sequential(nn.Conv2d(embedding_dim, num_classes, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(num_classes), nn.Conv2d(num_classes, num_classes, 3, padding=1))
        self.make_pred_c2 = nn.Sequential(nn.Conv2d(embedding_dim, num_classes, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(num_classes), nn.Conv2d(num_classes, num_classes, 3, padding=1))
        self.make_pred_c3 = nn.Sequential(nn.Conv2d(embedding_dim, num_classes, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(num_classes), nn.Conv2d(num_classes, num_classes, 3, padding=1))
        self.make_pred_c4 = nn.Sequential(nn.Conv2d(embedding_dim, num_classes, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(num_classes), nn.Conv2d(num_classes, num_classes, 3, padding=1))

    def forward(self, feats_t0: List[torch.Tensor], feats_t1: List[torch.Tensor]) -> torch.Tensor:
        c1_0, c2_0, c3_0, c4_0 = feats_t0
        c1_1, c2_1, c3_1, c4_1 = feats_t1

        B, _, H1, W1 = c1_0.shape

        # Linear projections
        _c1_0 = self.linear_c1(c1_0).permute(0, 2, 1).reshape(B, -1, H1, W1)
        _c2_0 = self.linear_c2(c2_0).permute(0, 2, 1).reshape(B, -1, c2_0.shape[2], c2_0.shape[3])
        _c3_0 = self.linear_c3(c3_0).permute(0, 2, 1).reshape(B, -1, c3_0.shape[2], c3_0.shape[3])
        _c4_0 = self.linear_c4(c4_0).permute(0, 2, 1).reshape(B, -1, c4_0.shape[2], c4_0.shape[3])

        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(B, -1, H1, W1)
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(B, -1, c2_1.shape[2], c2_1.shape[3])
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(B, -1, c3_1.shape[2], c3_1.shape[3])
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(B, -1, c4_1.shape[2], c4_1.shape[3])

        # Multi-scale feature difference
        _c1 = self.diff_c1(torch.cat([_c1_0, _c1_1], dim=1))
        _c2 = self.diff_c2(torch.cat([_c2_0, _c2_1], dim=1))
        _c3 = self.diff_c3(torch.cat([_c3_0, _c3_1], dim=1))
        _c4 = self.diff_c4(torch.cat([_c4_0, _c4_1], dim=1))

        _c2 = F.interpolate(_c2, size=(H1, W1), mode="bilinear", align_corners=False)
        _c3 = F.interpolate(_c3, size=(H1, W1), mode="bilinear", align_corners=False)
        _c4 = F.interpolate(_c4, size=(H1, W1), mode="bilinear", align_corners=False)

        # Fuse 4 scales
        _c = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))

        # Progressive upsampling (H/4 -> H/2 -> H)
        x = self.convd2x(_c)
        x = self.dense_2x(x)
        x = self.convd1x(x)
        x = self.dense_1x(x)
        logits = self.change_probability(x)
        return logits


class ChangeFormer(nn.Module):
    """
    Official ChangeFormer Siamese Architecture.
    Matches state_dict naming for official ChangeFormer pretrained checkpoints.
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 2,
                 embed_dims: List[int] = [64, 128, 320, 512],
                 depths: List[int] = [3, 3, 4, 3]):
        super().__init__()
        self.Tenc_x2 = MiTEncoder(in_chans=in_chans, embed_dims=embed_dims, depths=depths)
        self.TDec_x2 = ChangeFormerDecoder(in_channels=embed_dims, embedding_dim=256, num_classes=num_classes)

    def forward(self, img_t0: torch.Tensor, img_t1: torch.Tensor) -> torch.Tensor:
        feats_t0 = self.Tenc_x2(img_t0)
        feats_t1 = self.Tenc_x2(img_t1)
        logits = self.TDec_x2(feats_t0, feats_t1)
        return logits


class ChangeFormerModel(nn.Module):
    """Container wrapping ChangeFormer to match CD_model prefix in official checkpoints."""
    def __init__(self, in_chans: int = 3, num_classes: int = 2,
                 embed_dims: List[int] = [64, 128, 320, 512],
                 depths: List[int] = [3, 3, 4, 3]):
        super().__init__()
        self.CD_model = ChangeFormer(in_chans=in_chans, num_classes=num_classes,
                                     embed_dims=embed_dims, depths=depths)

    def forward(self, img_t0: torch.Tensor, img_t1: torch.Tensor) -> torch.Tensor:
        return self.CD_model(img_t0, img_t1)
