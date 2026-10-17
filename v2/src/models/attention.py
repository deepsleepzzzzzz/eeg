"""注意力模块：none / se / additive / self。"""
from __future__ import annotations

import torch
import torch.nn as nn


class IdentityAttention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SEAttention(nn.Module):
    """Squeeze-and-Excitation 通道注意力。"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class AdditiveAttention(nn.Module):
    """Bahdanau 式加性空间注意力（用于 CNN 特征图）。

    score = v^T tanh(W x)，再对空间维 softmax，得到注意力图。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.W = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.v = nn.Conv2d(channels, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.v(torch.tanh(self.W(x)))
        b, _, h, w = e.shape
        alpha = torch.softmax(e.view(b, -1), dim=1).view(b, 1, h, w)
        return x * alpha


class SelfAttention(nn.Module):
    """空间自注意力。每个位置对所有位置做缩放点积，再残差加回。

    本仓库的两层池化会把 6×6 压成 1×1，所以这个模块用在卷积之前。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h * w <= 1:
            return x
        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(x).flatten(2)
        v = self.v(x).flatten(2).transpose(1, 2)
        attn = torch.softmax((q @ k) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


def build_attention(kind: str, channels: int) -> nn.Module:
    k = str(kind).lower()
    if k in ("none", "identity", "off"):
        return IdentityAttention()
    if k == "se":
        return SEAttention(channels)
    if k == "additive":
        return AdditiveAttention(channels)
    if k in ("self", "self_attention", "sa"):
        return SelfAttention(channels)
    raise ValueError(f"未知 attention: {kind}")
