from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_dataset_dirs(config: Dict[str, Any], dataset_name: str) -> Tuple[Path, Path]:
    if dataset_name == "ecore":
        cleaned_dir = Path(config["paths"]["ecore_cleaned_dir"])
        features_dir = Path(config["paths"]["ecore_features_dir"])
    elif dataset_name == "uml":
        cleaned_dir = Path(config["paths"]["uml_cleaned_dir"])
        features_dir = Path(config["paths"]["uml_features_dir"])
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    return cleaned_dir, features_dir


def load_nodes_clean(nodes_clean_path: Path) -> pd.DataFrame:
    if not nodes_clean_path.exists():
        raise FileNotFoundError(f"找不到 nodes_clean.csv: {nodes_clean_path}")
    df = pd.read_csv(nodes_clean_path, encoding="utf-8-sig")
    required_cols = {"node_global_id", "graph_id", "node_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"nodes_clean.csv 缺少必要字段: {missing}")
    return df


def build_node_type_vocab(nodes_df: pd.DataFrame) -> List[str]:
    return sorted(nodes_df["node_type"].astype(str).str.strip().unique().tolist())


def build_one_hot_features(nodes_df: pd.DataFrame, node_type_vocab: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    type_to_idx = {t: i for i, t in enumerate(node_type_vocab)}
    num_nodes = len(nodes_df)
    num_types = len(node_type_vocab)
    features = np.zeros((num_nodes, num_types), dtype=np.float32)
    rows = []
    for row_idx, (_, row) in enumerate(nodes_df.iterrows()):
        node_type = str(row["node_type"]).strip()
        col_idx = type_to_idx[node_type]
        features[row_idx, col_idx] = 1.0
        rows.append(
            {
                "feature_row_index": row_idx,
                "node_global_id": int(row["node_global_id"]),
                "graph_id": row["graph_id"],
                "node_type": node_type,
            }
        )
    return features, pd.DataFrame(rows, columns=["feature_row_index", "node_global_id", "graph_id", "node_type"])


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(str(path.parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_index_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(str(path.parent))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_npy(arr: np.ndarray, path: Path) -> None:
    ensure_dir(str(path.parent))
    np.save(path, arr)


def copy_required_label_files(features_dir: Path) -> None:
    """
    为 type-only 特征目录自动补齐 build_pyg_dataset.py / trainer.py 需要的标签文件。
    默认从同级目录 05_features 复制 graph_label_matrix.npy 和 label_index.csv。
    """
    base_dir = features_dir.parent / "05_features"
    if features_dir.name == "05_features":
        return

    required_files = ["graph_label_matrix.npy", "label_index.csv"]
    for filename in required_files:
        src = base_dir / filename
        dst = features_dir / filename
        if dst.exists():
            continue
        if not src.exists():
            raise FileNotFoundError(
                f"找不到需要复制的标签文件: {src}。\n"
                f"请先确保原始 joint_text/默认特征目录中已经生成该文件。"
            )
        ensure_dir(str(dst.parent))
        shutil.copy2(src, dst)


def print_feature_summary(dataset_name: str, nodes_df: pd.DataFrame, node_type_vocab: List[str], features: np.ndarray, feature_index_df: pd.DataFrame, features_dir: Path) -> None:
    print("=" * 80)
    print(f"[数据集] {dataset_name}")
    print(f"[输出目录] {features_dir}")
    print(f"节点总数: {len(nodes_df)}")
    print(f"node_type 种类数: {len(node_type_vocab)}")
    print(f"特征矩阵形状: {features.shape}")
    print("-" * 80)
    print("前 10 个 node_type:")
    for t in node_type_vocab[:10]:
        print(f"  - {t}")
    print("-" * 80)
    print("feature_index 前 5 行:")
    print(feature_index_df.head(5).to_string(index=False))
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 node_type 构建 one-hot 节点特征（ablation 版本）")
    parser.add_argument("--dataset", type=str, required=True, choices=["ecore", "uml"])
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    cleaned_dir, features_dir = resolve_dataset_dirs(config, args.dataset)
    nodes_clean_path = cleaned_dir / "nodes_clean.csv"

    node_type_vocab_path = features_dir / "node_type_vocab.json"
    node_type_features_path = features_dir / "node_type_features.npy"
    node_type_feature_index_path = features_dir / "node_type_feature_index.csv"

    nodes_df = load_nodes_clean(nodes_clean_path)
    node_type_vocab = build_node_type_vocab(nodes_df)
    features, feature_index_df = build_one_hot_features(nodes_df, node_type_vocab)

    save_json(node_type_vocab, node_type_vocab_path)
    save_npy(features, node_type_features_path)
    save_index_csv(feature_index_df, node_type_feature_index_path)
    copy_required_label_files(features_dir)

    print_feature_summary(args.dataset, nodes_df, node_type_vocab, features, feature_index_df, features_dir)


if __name__ == "__main__":
    main()
