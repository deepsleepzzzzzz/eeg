"""有向加权转移图上的 GCN。输入是 [B, K, N, N] 转移矩阵，不是 CNN 的图像。"""
from __future__ import annotations

import torch
import torch.nn as nn


def _row_normalize(adj: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    a = adj + eye
    deg = a.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return a / deg


class RelLayer(nn.Module):
    """各时间步用原始转移权重做消息，再拼起来，不把概率矩阵改成对称归一化。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        steps: int,
        channels: int,
        graph_attention: bool = False,
    ):
        super().__init__()
        self.steps = steps
        self.graph_attention = bool(graph_attention)
        self.lins = nn.ModuleList(
            nn.Linear(in_features, out_features, bias=False) for _ in range(steps)
        )
        self.mix = nn.Linear(out_features * steps, out_features)
        self.channels = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(out_features)
        if self.graph_attention:
            self.att = nn.ModuleList(
                nn.Linear(out_features * 2, 1, bias=False) for _ in range(steps)
            )

    def _message(self, adj: torch.Tensor, wh: torch.Tensor, att: nn.Linear) -> torch.Tensor:
        b, c, n, f = wh.shape
        hi = wh.unsqueeze(3).expand(b, c, n, n, f)
        hj = wh.unsqueeze(2).expand(b, c, n, n, f)
        logits = att(torch.cat([hi, hj], dim=-1)).squeeze(-1)
        logits = torch.nn.functional.leaky_relu(logits, 0.2)
        logits = logits + torch.log(adj.clamp_min(1e-8))
        alpha = torch.softmax(logits, dim=-1)
        return torch.matmul(alpha, wh)

    def forward(self, mats: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # mats: [B, C, S, N, N], h: [B, C, N, F]
        parts = []
        for si, lin in enumerate(self.lins):
            wh = lin(h)
            if self.graph_attention:
                parts.append(self._message(mats[:, :, si], wh, self.att[si]))
            else:
                parts.append(torch.matmul(mats[:, :, si], wh))
        msg = self.norm(self.mix(torch.cat(parts, dim=-1)))
        mixed = self.channels(msg.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return mixed


class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.norm(torch.matmul(adj, self.lin(h)))


class TransitionGCN(nn.Module):
    """同一幅值节点上的多时间步有向图。

    relational：K 必须能被 steps 整除。每导一张图，各时间步是不同关系。
    independent：每张转移矩阵各自一张图，最后再拼起来。
    """

    def __init__(
        self,
        num_classes: int,
        num_graphs: int,
        num_nodes: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        pool: str = "flat",
        layout: str = "relational",
        steps: int = 3,
        graph_attention: bool = False,
    ):
        super().__init__()
        self.num_graphs = int(num_graphs)
        self.num_nodes = int(num_nodes)
        self.hidden = int(hidden)
        self.pool = str(pool).lower()
        self.layout = str(layout).lower()
        self.steps = int(steps)
        self.graph_attention = bool(graph_attention)
        if layers < 1:
            raise ValueError("gcn_layers must be >= 1")
        if self.graph_attention and self.layout != "relational":
            raise ValueError("graph attention is only implemented for relational GCN")
        if self.layout not in ("relational", "independent"):
            raise ValueError(f"unknown gcn_layout: {layout}")
        if self.pool not in ("flat", "mean", "nodes"):
            raise ValueError(f"unknown gcn_pool: {pool}")
        if self.layout == "relational":
            if self.num_graphs % self.steps != 0:
                raise ValueError(
                    f"relational GCN needs channels*steps graphs, got {self.num_graphs} graphs and {self.steps} steps"
                )
            self.num_channels = self.num_graphs // self.steps
            in_features = self.num_nodes * (1 + self.steps)
            self.rel = nn.ModuleList()
            fin = in_features
            for _ in range(int(layers)):
                self.rel.append(
                    RelLayer(
                        fin,
                        self.hidden,
                        self.steps,
                        self.num_channels,
                        graph_attention=self.graph_attention,
                    )
                )
                fin = self.hidden
            out_dim = self.num_channels * self.num_nodes * self.hidden
        else:
            self.num_channels = self.num_graphs
            in_features = self.num_nodes * 2
            convs = []
            fin = in_features
            for _ in range(int(layers)):
                convs.append(GraphConv(fin, self.hidden))
                fin = self.hidden
            self.convs = nn.ModuleList(convs)
            if self.pool == "mean":
                out_dim = self.hidden
            elif self.pool == "nodes":
                out_dim = self.num_graphs * self.num_nodes * self.hidden
            else:
                out_dim = self.num_graphs * self.hidden
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(out_dim, int(num_classes)),
        )

    def _relational(self, adj: torch.Tensor) -> torch.Tensor:
        b, k, n, _ = adj.shape
        c, s = self.num_channels, self.steps
        mats = adj.view(b, c, s, n, n)
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).view(1, 1, n, n)
        ident = eye.expand(b, c, n, n)
        outgoing = mats.permute(0, 1, 3, 2, 4).reshape(b, c, n, s * n)
        h = torch.cat([ident, outgoing], dim=-1)
        for layer in self.rel:
            prev = h
            h = self.dropout(torch.relu(layer(mats, h)))
            if prev.shape[-1] == h.shape[-1]:
                h = h + prev
        return h.flatten(1)

    def _independent(self, adj: torch.Tensor) -> torch.Tensor:
        b, k, n, _ = adj.shape
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).view(1, 1, n, n)
        a = _row_normalize(adj, eye)
        h = torch.cat([eye.expand(b, k, n, n), adj], dim=-1)
        for conv in self.convs:
            h = self.dropout(torch.relu(conv(a, h)))
        if self.pool == "nodes":
            return h.flatten(1)
        if self.pool == "flat":
            return h.mean(dim=2).flatten(1)
        return h.mean(dim=2).mean(dim=1)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        if adj.ndim != 4:
            raise ValueError(f"GCN expects [B,K,N,N], got {tuple(adj.shape)}")
        b, k, n, n2 = adj.shape
        if k != self.num_graphs or n != self.num_nodes or n != n2:
            raise ValueError(
                f"expected [B,{self.num_graphs},{self.num_nodes},{self.num_nodes}], got {tuple(adj.shape)}"
            )
        g = self._relational(adj) if self.layout == "relational" else self._independent(adj)
        return self.head(g)

    def embed(self, adj: torch.Tensor) -> torch.Tensor:
        """分类头之前的向量，供 t-SNE。"""
        if adj.ndim != 4:
            raise ValueError(f"GCN expects [B,K,N,N], got {tuple(adj.shape)}")
        return self._relational(adj) if self.layout == "relational" else self._independent(adj)
