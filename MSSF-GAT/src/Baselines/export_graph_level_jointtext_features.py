
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_dataset_dirs(config: Dict[str, Any], dataset_name: str) -> Tuple[str, str]:
    if dataset_name == "ecore":
        pyg_dir = config["paths"]["ecore_pyg_dir"]
    elif dataset_name == "uml":
        pyg_dir = config["paths"]["uml_pyg_dir"]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    results_dir = config["results"]["save_dir"]
    return pyg_dir, results_dir


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


def graph_mean_max_pool(data) -> Tuple[np.ndarray, np.ndarray]:
    x = data.x
    if x.ndim != 2:
        raise ValueError(f"期望 data.x 为二维张量，实际形状: {tuple(x.shape)}")
    if x.size(0) == 0:
        raise ValueError("发现空图，无法导出图级特征")

    mean_feat = x.mean(dim=0)
    max_feat = x.max(dim=0).values
    graph_feat = torch.cat([mean_feat, max_feat], dim=0)

    y = data.y
    if y.dim() == 2 and y.size(0) == 1:
        y = y.squeeze(0)

    return graph_feat.cpu().numpy().astype(np.float32), y.cpu().numpy().astype(np.float32)


def convert_dataset_to_arrays(dataset) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    for data in dataset:
        feat, label = graph_mean_max_pool(data)
        xs.append(feat)
        ys.append(label)

    X = np.stack(xs, axis=0).astype(np.float32)
    Y = np.stack(ys, axis=0).astype(np.float32)
    return X, Y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 joint_text PyG 数据集中导出 FFNN/SVM 可用的图级特征（mean+max pooling）"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["ecore", "uml"])
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    active_feature_type = config.get("features", {}).get("active_feature_type", "merged")
    if active_feature_type != "joint_text":
        raise ValueError(
            f"当前配置 features.active_feature_type = {active_feature_type}，"
            "本脚本要求使用 joint_text 特征。"
        )

    pyg_dir, results_dir = resolve_dataset_dirs(config, args.dataset)
    train_dataset, val_dataset, test_dataset = load_pyg_datasets(pyg_dir)

    export_dir = Path(results_dir) / f"{args.dataset}_graph_level_joint_text"
    ensure_dir(str(export_dir))

    X_train, y_train = convert_dataset_to_arrays(train_dataset)
    X_val, y_val = convert_dataset_to_arrays(val_dataset)
    X_test, y_test = convert_dataset_to_arrays(test_dataset)

    np.save(export_dir / "X_train.npy", X_train)
    np.save(export_dir / "y_train.npy", y_train)
    np.save(export_dir / "X_val.npy", X_val)
    np.save(export_dir / "y_val.npy", y_val)
    np.save(export_dir / "X_test.npy", X_test)
    np.save(export_dir / "y_test.npy", y_test)

    meta = {
        "dataset": args.dataset,
        "active_feature_type": active_feature_type,
        "graph_feature_type": "mean_max_pooling_over_joint_text_node_features",
        "num_train_graphs": int(X_train.shape[0]),
        "num_val_graphs": int(X_val.shape[0]),
        "num_test_graphs": int(X_test.shape[0]),
        "input_node_feature_dim": int(X_train.shape[1] // 2),
        "graph_feature_dim": int(X_train.shape[1]),
        "num_classes": int(y_train.shape[1]),
    }
    with open(export_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"[数据集] {args.dataset}")
    print(f"[节点特征来源] {active_feature_type}")
    print(f"[导出目录] {export_dir}")
    print("-" * 80)
    print(f"X_train shape = {tuple(X_train.shape)}")
    print(f"y_train shape = {tuple(y_train.shape)}")
    print(f"X_val   shape = {tuple(X_val.shape)}")
    print(f"y_val   shape = {tuple(y_val.shape)}")
    print(f"X_test  shape = {tuple(X_test.shape)}")
    print(f"y_test  shape = {tuple(y_test.shape)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
