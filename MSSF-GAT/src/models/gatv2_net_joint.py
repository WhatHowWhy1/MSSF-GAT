

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class JointTextGATv2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_input_features: int,
        hidden_dim: int = 64,
        num_gat_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.3,
        mlp_hidden_dim: int = 128,
        use_raw_readout: bool = True,
        use_gnn: bool = True,
    ) -> None:
        super().__init__()

        if use_gnn and num_gat_layers < 1:
            raise ValueError("当 use_gnn=True 时，num_gat_layers 必须 >= 1")

        if (not use_gnn) and (not use_raw_readout):
            raise ValueError("use_gnn=False 且 use_raw_readout=False 会导致没有任何图级特征可用")

        self.num_classes = num_classes
        self.num_input_features = num_input_features
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.gat_heads = gat_heads
        self.dropout = dropout
        self.mlp_hidden_dim = mlp_hidden_dim
        self.use_raw_readout = use_raw_readout
        self.use_gnn = use_gnn

        # 1) 输入投影
        self.input_proj = nn.Linear(num_input_features, hidden_dim)

        # 2) GATv2 编码层（可选）
        self.gat_layers = nn.ModuleList()
        if self.use_gnn:
            for _ in range(num_gat_layers):
                self.gat_layers.append(
                    GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        heads=gat_heads,
                        concat=False,
                        dropout=dropout,
                        add_self_loops=True,
                    )
                )

        # 3) 分类头输入维度
        if self.use_gnn and self.use_raw_readout:
            final_in_dim = 4 * hidden_dim   # deep(2h) + raw(2h)
        elif self.use_gnn and (not self.use_raw_readout):
            final_in_dim = 2 * hidden_dim   # deep only
        elif (not self.use_gnn) and self.use_raw_readout:
            final_in_dim = 2 * hidden_dim   # raw only
        else:
            raise ValueError("非法配置")

        self.classifier = nn.Sequential(
            nn.Linear(final_in_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Step 1. 输入投影
        h0 = self.input_proj(x)  # [N, hidden_dim]
        h0 = F.relu(h0)
        h0 = F.dropout(h0, p=self.dropout, training=self.training)

        # 情况 A：去掉结构-语义融合编码，只保留 raw branch
        if not self.use_gnn:
            raw_mean = global_mean_pool(h0, batch)   # [B, hidden_dim]
            raw_max = global_max_pool(h0, batch)     # [B, hidden_dim]
            graph_feat = torch.cat([raw_mean, raw_max], dim=1)  # [B, 2*hidden_dim]
            logits = self.classifier(graph_feat)
            return logits

        # 情况 B：正常走 GATv2
        h = h0
        for gat in self.gat_layers:
            h = gat(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        deep_mean = global_mean_pool(h, batch)
        deep_max = global_max_pool(h, batch)
        deep_graph_feat = torch.cat([deep_mean, deep_max], dim=1)

        if self.use_raw_readout:
            raw_mean = global_mean_pool(h0, batch)
            raw_max = global_max_pool(h0, batch)
            raw_graph_feat = torch.cat([raw_mean, raw_max], dim=1)
            graph_feat = torch.cat([deep_graph_feat, raw_graph_feat], dim=1)
        else:
            graph_feat = deep_graph_feat

        logits = self.classifier(graph_feat)
        return logits