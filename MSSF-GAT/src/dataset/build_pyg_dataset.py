"""
build_pyg_dataset.py

功能简介：
将清洗后的图数据、标签矩阵和节点特征组装为 PyTorch Geometric 数据集。

支持的节点特征来源：
1. type
   - node_type_features.npy
2. name
   - node_name_features.npy
3. merged
   - merged_features.npy
4. joint_text
   - joint_text_features.npy

输入：
- data/{dataset}/03_cleaned_csv/nodes_clean.csv
- data/{dataset}/03_cleaned_csv/edges_clean.csv
- data/{dataset}/03_cleaned_csv/graphs_clean.csv
- data/{dataset}/04_split/train_graph_ids.txt
- data/{dataset}/04_split/val_graph_ids.txt
- data/{dataset}/04_split/test_graph_ids.txt
- data/{dataset}/05_features/graph_label_matrix.npy
- data/{dataset}/05_features/label_index.csv
- 对应的节点特征文件

输出：
- data/{dataset}/06_pyg_dataset/train_dataset.pt
- data/{dataset}/06_pyg_dataset/val_dataset.pt
- data/{dataset}/06_pyg_dataset/test_dataset.pt

运行方式：
python src/dataset/build_pyg_dataset.py --dataset ecore
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_dataset_dirs(config: Dict[str, Any], dataset_name: str) -> Tuple[str, str, str, str]:
    if dataset_name == "ecore":
        cleaned_dir = config["paths"]["ecore_cleaned_dir"]
        split_dir = config["paths"]["ecore_split_dir"]
        features_dir = config["paths"]["ecore_features_dir"]
        pyg_dir = config["paths"]["ecore_pyg_dir"]
    elif dataset_name == "uml":
        cleaned_dir = config["paths"]["uml_cleaned_dir"]
        split_dir = config["paths"]["uml_split_dir"]
        features_dir = config["paths"]["uml_features_dir"]
        pyg_dir = config["paths"]["uml_pyg_dir"]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    return cleaned_dir, split_dir, features_dir, pyg_dir


def read_graph_id_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_feature_matrix_and_index(features_dir: Path, active_feature_type: str) -> Tuple[np.ndarray, pd.DataFrame]:
    if active_feature_type == "type":
        feature_path = features_dir / "node_type_features.npy"
        index_path = features_dir / "node_type_feature_index.csv"
    elif active_feature_type == "name":
        feature_path = features_dir / "node_name_features.npy"
        index_path = features_dir / "node_name_feature_index.csv"
    elif active_feature_type == "merged":
        feature_path = features_dir / "merged_features.npy"
        index_path = features_dir / "merged_feature_index.csv"
    elif active_feature_type == "joint_text":
        feature_path = features_dir / "joint_text_features.npy"
        index_path = features_dir / "joint_text_feature_index.csv"
    else:
        raise ValueError(f"不支持的 active_feature_type: {active_feature_type}")

    if not feature_path.exists():
        raise FileNotFoundError(f"找不到特征文件: {feature_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"找不到特征索引文件: {index_path}")

    feature_matrix = np.load(feature_path)
    feature_index_df = pd.read_csv(index_path, encoding="utf-8-sig")
    return feature_matrix, feature_index_df


def check_alignment(nodes_df: pd.DataFrame, feature_matrix: np.ndarray, feature_index_df: pd.DataFrame) -> None:
    if len(nodes_df) != len(feature_matrix):
        raise ValueError(
            f"nodes_clean.csv 行数({len(nodes_df)}) 与特征矩阵行数({len(feature_matrix)}) 不一致"
        )
    if len(nodes_df) != len(feature_index_df):
        raise ValueError(
            f"nodes_clean.csv 行数({len(nodes_df)}) 与特征索引行数({len(feature_index_df)}) 不一致"
        )

    if "node_global_id" not in feature_index_df.columns:
        raise ValueError("特征索引文件缺少 node_global_id 列")

    if not np.array_equal(
        nodes_df["node_global_id"].to_numpy(),
        feature_index_df["node_global_id"].to_numpy(),
    ):
        raise ValueError("nodes_clean.csv 与特征索引文件的 node_global_id 顺序不一致")


def load_label_matrix(features_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    label_matrix_path = features_dir / "graph_label_matrix.npy"
    label_index_path = features_dir / "label_index.csv"

    if not label_matrix_path.exists():
        raise FileNotFoundError(f"找不到标签矩阵文件: {label_matrix_path}")
    if not label_index_path.exists():
        raise FileNotFoundError(f"找不到标签索引文件: {label_index_path}")

    label_matrix = np.load(label_matrix_path)
    label_index_df = pd.read_csv(label_index_path, encoding="utf-8-sig")
    return label_matrix, label_index_df


def build_graph_label_lookup(label_matrix: np.ndarray, label_index_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    if len(label_matrix) != len(label_index_df):
        raise ValueError("label_matrix 与 label_index.csv 行数不一致")

    graph_to_y = {}
    for i, row in label_index_df.iterrows():
        graph_id = row["graph_id"]
        graph_to_y[graph_id] = label_matrix[i].astype(np.float32)
    return graph_to_y


def build_single_graph_data(
    graph_id: str,
    nodes_sub: pd.DataFrame,
    edges_sub: pd.DataFrame,
    full_feature_matrix: np.ndarray,
    graph_to_y: Dict[str, np.ndarray],
) -> Data:
    # node_global_id 仍然用于构图时建立 old_id -> local_id 映射
    node_ids = nodes_sub["node_global_id"].tolist()
    old_to_local = {nid: i for i, nid in enumerate(node_ids)}

    feature_row_indices = nodes_sub.index.to_list()

    x = full_feature_matrix[feature_row_indices]
    x = torch.tensor(x, dtype=torch.float)

    edge_pairs = []
    for _, row in edges_sub.iterrows():
        src = row["src_global_id"]
        dst = row["dst_global_id"]
        if src in old_to_local and dst in old_to_local:
            edge_pairs.append([old_to_local[src], old_to_local[dst]])

    if len(edge_pairs) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    if graph_id not in graph_to_y:
        raise ValueError(f"graph_id 不在标签矩阵中: {graph_id}")

    y = torch.tensor(graph_to_y[graph_id], dtype=torch.float).unsqueeze(0)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )
    data.graph_id = graph_id
    data.num_nodes = x.size(0)
    return data


def build_dataset_for_split(
    graph_ids: List[str],
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    full_feature_matrix: np.ndarray,
    graph_to_y: Dict[str, np.ndarray],
) -> List[Data]:
    dataset = []
    for graph_id in graph_ids:
        nodes_sub = nodes_df[nodes_df["graph_id"] == graph_id]
        edges_sub = edges_df[edges_df["graph_id"] == graph_id]

        if len(nodes_sub) == 0:
            continue

        data = build_single_graph_data(
            graph_id=graph_id,
            nodes_sub=nodes_sub,
            edges_sub=edges_sub,
            full_feature_matrix=full_feature_matrix,
            graph_to_y=graph_to_y,
        )
        dataset.append(data)

    return dataset


def save_dataset(dataset: List[Data], path: Path) -> None:
    ensure_dir(str(path.parent))
    torch.save(dataset, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 PyG 图数据集")
    parser.add_argument("--dataset", type=str, required=True, choices=["ecore", "uml"])
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    cleaned_dir, split_dir, features_dir, pyg_dir = resolve_dataset_dirs(config, args.dataset)

    cleaned_dir = Path(cleaned_dir)
    split_dir = Path(split_dir)
    features_dir = Path(features_dir)
    pyg_dir = Path(pyg_dir)

    active_feature_type = config.get("features", {}).get("active_feature_type", "merged")

    nodes_path = cleaned_dir / "nodes_clean.csv"
    edges_path = cleaned_dir / "edges_clean.csv"
    graphs_path = cleaned_dir / "graphs_clean.csv"

    train_ids_path = split_dir / "train_graph_ids.txt"
    val_ids_path = split_dir / "val_graph_ids.txt"
    test_ids_path = split_dir / "test_graph_ids.txt"

    nodes_df = pd.read_csv(nodes_path, encoding="utf-8-sig")
    edges_df = pd.read_csv(edges_path, encoding="utf-8-sig")
    graphs_df = pd.read_csv(graphs_path, encoding="utf-8-sig")

    feature_matrix, feature_index_df = load_feature_matrix_and_index(features_dir, active_feature_type)
    check_alignment(nodes_df, feature_matrix, feature_index_df)

    label_matrix, label_index_df = load_label_matrix(features_dir)
    graph_to_y = build_graph_label_lookup(label_matrix, label_index_df)

    train_graph_ids = read_graph_id_list(train_ids_path)
    val_graph_ids = read_graph_id_list(val_ids_path)
    test_graph_ids = read_graph_id_list(test_ids_path)

    train_dataset = build_dataset_for_split(
        train_graph_ids, nodes_df, edges_df, feature_matrix, graph_to_y
    )
    val_dataset = build_dataset_for_split(
        val_graph_ids, nodes_df, edges_df, feature_matrix, graph_to_y
    )
    test_dataset = build_dataset_for_split(
        test_graph_ids, nodes_df, edges_df, feature_matrix, graph_to_y
    )

    save_dataset(train_dataset, pyg_dir / "train_dataset.pt")
    save_dataset(val_dataset, pyg_dir / "val_dataset.pt")
    save_dataset(test_dataset, pyg_dir / "test_dataset.pt")

    print("=" * 80)
    print(f"[数据集] {args.dataset}")
    print(f"[节点特征来源] {active_feature_type}")
    print(f"总图数: {len(graphs_df)}")
    print(f"train 图数: {len(train_dataset)}")
    print(f"val 图数: {len(val_dataset)}")
    print(f"test 图数: {len(test_dataset)}")
    print("-" * 80)

    sample_dataset = train_dataset if len(train_dataset) > 0 else val_dataset if len(val_dataset) > 0 else test_dataset
    if len(sample_dataset) > 0:
        sample = sample_dataset[0]
        print("第一个图样例:")
        print(f"graph_id = {sample.graph_id}")
        print(f"x shape = {tuple(sample.x.shape)}")
        print(f"edge_index shape = {tuple(sample.edge_index.shape)}")
        print(f"y shape = {tuple(sample.y.shape)}")
        print(f"num_nodes = {sample.num_nodes}")
    print("=" * 80)


if __name__ == "__main__":
    main()