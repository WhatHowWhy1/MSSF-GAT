


from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from src.engine.evaluator import evaluate_model
from src.models.gat_net import WideDeepGAT
from src.models.gcn_net import WideDeepGCN
from src.models.gcn_net_joint import JointTextGCN
from src.models.gat_net_joint import JointTextGAT
from src.models.gatv2_net_joint import JointTextGATv2


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_dataset_dirs(config: Dict[str, Any], dataset_name: str) -> Tuple[str, str, str, str]:
    if dataset_name == "ecore":
        features_dir = config["paths"]["ecore_features_dir"]
        pyg_dir = config["paths"]["ecore_pyg_dir"]
    elif dataset_name == "uml":
        features_dir = config["paths"]["uml_features_dir"]
        pyg_dir = config["paths"]["uml_pyg_dir"]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    checkpoints_dir = config["checkpoints"]["save_dir"]
    results_dir = config["results"]["save_dir"]
    return features_dir, pyg_dir, checkpoints_dir, results_dir


def build_device(config: Dict[str, Any]) -> torch.device:
    device_name = config.get("project", {}).get("device", "cpu")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pyg_datasets(pyg_dir: str):
    train_path = Path(pyg_dir) / "train_dataset.pt"
    val_path = Path(pyg_dir) / "val_dataset.pt"
    test_path = Path(pyg_dir) / "test_dataset.pt"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"找不到 PyG 数据集文件: {p}")

    train_dataset = torch.load(train_path, weights_only=False)
    val_dataset = torch.load(val_path, weights_only=False)
    test_dataset = torch.load(test_path, weights_only=False)

    return train_dataset, val_dataset, test_dataset


def infer_num_node_types_and_num_classes(features_dir: str) -> Tuple[int, int]:
    node_type_vocab_path = Path(features_dir) / "node_type_vocab.json"
    graph_label_matrix_path = Path(features_dir) / "graph_label_matrix.npy"

    if not node_type_vocab_path.exists():
        raise FileNotFoundError(f"找不到 {node_type_vocab_path}")
    if not graph_label_matrix_path.exists():
        raise FileNotFoundError(f"找不到 {graph_label_matrix_path}")

    with open(node_type_vocab_path, "r", encoding="utf-8") as f:
        node_type_vocab = json.load(f)

    graph_label_matrix = np.load(graph_label_matrix_path)

    num_node_types = len(node_type_vocab)
    num_classes = int(graph_label_matrix.shape[1])

    return num_node_types, num_classes


def infer_num_input_features_from_dataset(train_dataset) -> int:
    if len(train_dataset) == 0:
        raise ValueError("训练集为空，无法推断输入特征维度")
    return int(train_dataset[0].x.shape[1])


def build_model(
    model_cfg: Dict[str, Any],
    num_node_types: int,
    num_classes: int,
    num_input_features: int,
) -> nn.Module:
    model_name = model_cfg.get("name", "WideDeepGCN")

    if model_name == "WideDeepGAT":
        return WideDeepGAT(
            num_node_types=num_node_types,
            num_classes=num_classes,
            num_input_features=num_input_features,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_gat_layers=model_cfg.get("num_gat_layers", 2),
            gat_heads=model_cfg.get("gat_heads", 4),
            dropout=model_cfg.get("dropout", 0.3),
            mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", 128),
        )

    if model_name == "WideDeepGCN":
        return WideDeepGCN(
            num_node_types=num_node_types,
            num_classes=num_classes,
            num_input_features=num_input_features,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_gcn_layers=model_cfg.get("num_gcn_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
            mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", 128),
        )

    if model_name == "JointTextGCN":
        return JointTextGCN(
            num_classes=num_classes,
            num_input_features=num_input_features,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_gcn_layers=model_cfg.get("num_gcn_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
            mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", 128),
        )

    if model_name == "JointTextGATv2":
        return JointTextGATv2(
            num_classes=num_classes,
            num_input_features=num_input_features,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_gat_layers=model_cfg.get("num_gat_layers", 2),
            gat_heads=model_cfg.get("gat_heads", 4),
            dropout=model_cfg.get("dropout", 0.3),
            mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", 128),
            use_raw_readout=model_cfg.get("use_raw_readout", True),
            use_gnn=model_cfg.get("use_gnn", True),
        )

    if model_name == "JointTextGAT":
        return JointTextGAT(
            num_classes=num_classes,
            num_input_features=num_input_features,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_gat_layers=model_cfg.get("num_gat_layers", 2),
            gat_heads=model_cfg.get("gat_heads", 4),
            dropout=model_cfg.get("dropout", 0.3),
            mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", 128),
            use_raw_readout=model_cfg.get("use_raw_readout", True),
        )

    raise ValueError(f"不支持的模型名: {model_name}")


def compute_pos_weight_from_dataset(train_dataset) -> torch.Tensor:
    if len(train_dataset) == 0:
        raise ValueError("训练集为空，无法计算 pos_weight")

    ys = []
    for data in train_dataset:
        y = data.y
        if y.dim() == 2 and y.size(0) == 1:
            y = y.squeeze(0)
        ys.append(y.cpu())

    y_mat = torch.stack(ys, dim=0).float()
    pos_counts = y_mat.sum(dim=0)
    total_count = y_mat.size(0)
    neg_counts = total_count - pos_counts

    pos_counts = torch.clamp(pos_counts, min=1.0)
    pos_weight = neg_counts / pos_counts
    return pos_weight


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.batch)
        targets = batch.y

        if targets.dim() == 3 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_graphs += batch_size

    return total_loss / max(total_graphs, 1)


def search_best_threshold(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold_candidates=None,
):
    if threshold_candidates is None:
        threshold_candidates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    best_threshold = threshold_candidates[0]
    best_metrics = None
    best_micro = -1.0

    for th in threshold_candidates:
        metrics = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            threshold=th,
        )
        if metrics["micro_f1"] > best_micro:
            best_micro = metrics["micro_f1"]
            best_threshold = th
            best_metrics = metrics

    return best_threshold, best_metrics


def run_training(config: Dict[str, Any], dataset_name: str) -> None:
    features_dir, pyg_dir, checkpoints_dir, results_dir = resolve_dataset_dirs(config, dataset_name)
    device = build_device(config)

    ensure_dir(checkpoints_dir)
    ensure_dir(results_dir)

    train_dataset, val_dataset, test_dataset = load_pyg_datasets(pyg_dir)

    num_node_types, num_classes = infer_num_node_types_and_num_classes(features_dir)
    num_input_features = infer_num_input_features_from_dataset(train_dataset)

    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    active_feature_type = config.get("features", {}).get("active_feature_type", "merged")

    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    num_epochs = int(train_cfg.get("num_epochs", 30))
    threshold_candidates = train_cfg.get(
        "threshold_candidates",
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    )
    max_pos_weight = float(train_cfg.get("max_pos_weight", 30.0))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 8))
    early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 1e-4))

    model = build_model(
        model_cfg=model_cfg,
        num_node_types=num_node_types,
        num_classes=num_classes,
        num_input_features=num_input_features,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    raw_pos_weight = compute_pos_weight_from_dataset(train_dataset)
    clipped_pos_weight = torch.clamp(raw_pos_weight, max=max_pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=clipped_pos_weight)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("=" * 80)
    print(f"[数据集] {dataset_name}")
    print(f"[设备] {device}")
    print(f"[节点特征来源] {active_feature_type}")
    print(f"[训练图数] {len(train_dataset)}")
    print(f"[验证图数] {len(val_dataset)}")
    print(f"[测试图数] {len(test_dataset)}")
    print(f"[num_node_types] {num_node_types}")
    print(f"[num_input_features] {num_input_features}")
    print(f"[num_classes] {num_classes}")
    print("-" * 80)
    print(f"模型名称: {model_cfg.get('name', 'WideDeepGCN')}")
    print("模型参数：")
    print(f"hidden_dim = {model_cfg.get('hidden_dim', 64)}")

    model_name = model_cfg.get("name", "WideDeepGCN")
    if model_name in ["WideDeepGAT", "JointTextGAT", "JointTextGATv2"]:
        print(f"num_gat_layers = {model_cfg.get('num_gat_layers', 2)}")
        print(f"gat_heads = {model_cfg.get('gat_heads', 4)}")
    else:
        print(f"num_gcn_layers = {model_cfg.get('num_gcn_layers', 2)}")

    print(f"dropout = {model_cfg.get('dropout', 0.3)}")
    print(f"mlp_hidden_dim = {model_cfg.get('mlp_hidden_dim', 128)}")
    if model_name in ["JointTextGAT", "JointTextGATv2"]:
        print(f"use_raw_readout = {model_cfg.get('use_raw_readout', True)}")
    if model_name == "JointTextGATv2":
        print(f"use_gnn = {model_cfg.get('use_gnn', True)}")

    print("-" * 80)
    print("训练参数：")
    print(f"batch_size = {batch_size}")
    print(f"lr = {lr}")
    print(f"weight_decay = {weight_decay}")
    print(f"num_epochs = {num_epochs}")
    print(f"threshold_candidates = {threshold_candidates}")
    print(f"max_pos_weight = {max_pos_weight}")
    print(f"early_stopping_patience = {early_stopping_patience}")
    print(f"early_stopping_min_delta = {early_stopping_min_delta}")
    print("-" * 80)
    print(f"raw_pos_weight min = {raw_pos_weight.min().item():.4f}")
    print(f"raw_pos_weight max = {raw_pos_weight.max().item():.4f}")
    print(f"clipped_pos_weight min = {clipped_pos_weight.min().item():.4f}")
    print(f"clipped_pos_weight max = {clipped_pos_weight.max().item():.4f}")
    print("=" * 80)

    best_val_micro_f1 = -1.0
    best_threshold = threshold_candidates[0]
    best_model_path = str(Path(checkpoints_dir) / f"best_{dataset_name}_{model_name}.pt")

    history = []
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        epoch_best_threshold, val_metrics = search_best_threshold(
            model=model,
            dataloader=val_loader,
            device=device,
            threshold_candidates=threshold_candidates,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "best_threshold": epoch_best_threshold,
        }
        history.append(record)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} | "
            f"val_precision={val_metrics['precision']:.6f} | "
            f"val_recall={val_metrics['recall']:.6f} | "
            f"val_f1={val_metrics['f1']:.6f} | "
            f"val_micro_f1={val_metrics['micro_f1']:.6f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.6f} | "
            f"best_th={epoch_best_threshold:.2f}"
        )

        improvement = val_metrics["micro_f1"] - best_val_micro_f1

        if improvement > early_stopping_min_delta:
            best_val_micro_f1 = val_metrics["micro_f1"]
            best_threshold = epoch_best_threshold
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print("-" * 80)
            print(
                f"[Early Stopping] 连续 {epochs_without_improvement} 个 epoch "
                f"未超过最优验证集 micro-F1，提前停止训练。"
            )
            print("-" * 80)
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=best_threshold,
    )

    print("=" * 80)
    print("[最佳模型测试结果]")
    print(f"best_val_micro_f1 = {best_val_micro_f1:.6f}")
    print(f"best_threshold    = {best_threshold:.2f}")
    print(f"test_precision    = {test_metrics['precision']:.6f}")
    print(f"test_recall       = {test_metrics['recall']:.6f}")
    print(f"test_f1           = {test_metrics['f1']:.6f}")
    print(f"test_micro_f1     = {test_metrics['micro_f1']:.6f}")
    print(f"test_macro_f1     = {test_metrics['macro_f1']:.6f}")
    print("=" * 80)

    history_path = Path(results_dir) / f"{dataset_name}_{model_name}_train_history.json"
    test_result_path = Path(results_dir) / f"{dataset_name}_{model_name}_test_result.json"

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(test_result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "active_feature_type": active_feature_type,
                "best_val_micro_f1": best_val_micro_f1,
                "best_threshold": best_threshold,
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_micro_f1": test_metrics["micro_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
                "best_model_path": best_model_path,
                "num_node_types": num_node_types,
                "num_input_features": num_input_features,
                "num_classes": num_classes,
                "max_pos_weight": max_pos_weight,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "use_raw_readout": model_cfg.get("use_raw_readout", True)
                if model_name in ["JointTextGAT", "JointTextGATv2"]
                else None,
                "use_gnn": model_cfg.get("use_gnn", True)
                if model_name == "JointTextGATv2"
                else None,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="训练图级多标签分类模型")
    parser.add_argument("--dataset", type=str, required=True, choices=["ecore", "uml"])
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    run_training(config, args.dataset)


if __name__ == "__main__":
    main()