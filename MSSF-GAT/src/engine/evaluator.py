"""
evaluator.py

功能简介：
用于图级多标签分类任务的评估模块。

提供功能：
1. logits -> sigmoid 概率 -> 二值预测
2. 计算论文风格的 Precision / Recall / F1
3. 计算 micro-F1 / macro-F1
4. 对一个 DataLoader 上的模型进行完整评估

适用任务：
- 图级多标签分类
- 配合 BCEWithLogitsLoss 使用


"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import Tensor
from torch_geometric.loader import DataLoader


def logits_to_predictions(logits: Tensor, threshold: float = 0.5) -> Tensor:
    """
    将 logits 转换为 0/1 多标签预测。

    参数：
        logits: [B, C]
        threshold: 二值化阈值

    返回：
        preds: [B, C]
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return preds


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算多标签分类的 Precision / Recall / F1 / micro-F1 / macro-F1。

    参数：
        y_true: [num_samples, num_classes]
        y_pred: [num_samples, num_classes]

    返回：
        dict:
            precision
            recall
            f1
            micro_f1
            macro_f1
    """
    # 论文风格整体指标：采用 micro 聚合
    precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # 常见多标签评估指标
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    在一个 dataloader 上评估模型。

    参数：
        model: 图分类模型
        dataloader: PyG DataLoader
        device: 运行设备
        threshold: 多标签二值化阈值

    返回：
        dict:
            precision
            recall
            f1
            micro_f1
            macro_f1
    """
    model.eval()

    all_logits: List[Tensor] = []
    all_targets: List[Tensor] = []

    for batch in dataloader:
        batch = batch.to(device)

        logits = model(batch.x, batch.edge_index, batch.batch)   # [B, C]
        targets = batch.y

        # 兼容 [B, 1, C] 的情况
        if targets.dim() == 3 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if len(all_logits) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
        }

    logits_tensor = torch.cat(all_logits, dim=0)    # [num_graphs, C]
    targets_tensor = torch.cat(all_targets, dim=0)  # [num_graphs, C]

    preds_tensor = logits_to_predictions(logits_tensor, threshold=threshold)

    y_true = targets_tensor.numpy()
    y_pred = preds_tensor.numpy()

    metrics = compute_multilabel_metrics(y_true, y_pred)
    return metrics