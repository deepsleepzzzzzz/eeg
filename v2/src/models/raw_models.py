"""原始波形基线：1D-CNN / WDCNN / ResNet1D。

用途：主张"本文以 1/552 参数达到同级水平"必须有**同一把尺子**量出来的基线。
文献报告的基线值口径不同（划分、窗长、噪声参考功率都不一样），不能直接引用。
本模块按 DDGF-Net (Machines 2026) Table 3 给出的结构实现，超参走协议，不写死。

输入：原始波形 [B, C, T]（C=3 通道，T=窗长 2048）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RawCNN1D(nn.Module):
    """普通三层 1D-CNN：最朴素的深度基线。"""

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.2,
                 conv_channels=(32, 64, 128), kernel_size: int = 7):
        super().__init__()
        conv_channels = [int(c) for c in conv_channels]
        blocks, ch = [], int(in_channels)
        for i, c in enumerate(conv_channels):
            k = int(kernel_size) if i == 0 else 3
            blocks += [
                nn.Conv1d(ch, c, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            ch = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(ch, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class WDCNN(nn.Module):
    """WDCNN（Zhang et al., Sensors 2017）：第一层宽核 + 四层 3×1 卷积。

    DDGF-Net Table 3 记为 5 层宽核 1D-CNN（16→32→64→64→64），0.048 M 参数。
    """

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.2,
                 first_kernel: int = 64, first_stride: int = 16, channels=(16, 32, 64, 64, 64),
                 fc_hidden: int = 100):
        super().__init__()
        c1, c2, c3, c4, c5 = [int(c) for c in channels]
        k, s = int(first_kernel), int(first_stride)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=k, stride=s, padding=k // 2, bias=False),
            nn.BatchNorm1d(c1), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, 3, padding=1, bias=False), nn.BatchNorm1d(c2), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, 3, padding=1, bias=False), nn.BatchNorm1d(c3), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(c3, c4, 3, padding=1, bias=False), nn.BatchNorm1d(c4), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(c4, c5, 3, padding=1, bias=False), nn.BatchNorm1d(c5), nn.ReLU(inplace=True), nn.MaxPool1d(2),
        )
        # 原版 WDCNN 固定输入长度后用全连接；这里用自适应池化以适配任意窗长
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(c5, int(fc_hidden)), nn.ReLU(inplace=True),
            nn.Linear(int(fc_hidden), num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class _BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + idt)


class ResNet1D(nn.Module):
    """ResNet-18 的 1D 版本（stem 7×1，四段 64→128→256→512）。"""

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.2,
                 stages=(64, 128, 256, 512), blocks_per_stage: int = 2, stem_kernel: int = 7):
        super().__init__()
        stages = [int(c) for c in stages]
        k = int(stem_kernel)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stages[0], kernel_size=k, stride=2, padding=k // 2, bias=False),
            nn.BatchNorm1d(stages[0]), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        layers, ch = [], stages[0]
        for i, out_ch in enumerate(stages):
            for b in range(int(blocks_per_stage)):
                layers.append(_BasicBlock1D(ch, out_ch, stride=(2 if (b == 0 and i > 0) else 1)))
                ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(ch, num_classes),
        )

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


RAW_ARCHS = {"raw_cnn": RawCNN1D, "wdcnn": WDCNN, "resnet1d": ResNet1D}


def build_raw_model(arch: str, num_classes: int, in_channels: int, model_cfg: dict,
                    dropout: float) -> nn.Module:
    arch = str(arch).lower()
    if arch not in RAW_ARCHS:
        raise ValueError(f"未知原始波形架构: {arch}（可选 {sorted(RAW_ARCHS)}）")
    kw = {k: v for k, v in (model_cfg or {}).items()
          if k in {"conv_channels", "kernel_size", "first_kernel", "first_stride", "channels",
                   "fc_hidden", "stages", "blocks_per_stage", "stem_kernel"}}
    return RAW_ARCHS[arch](num_classes=num_classes, in_channels=int(in_channels),
                           dropout=float(dropout), **kw)
