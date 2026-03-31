"""
label_encoder.py

功能简介：
将 graphs_clean.csv 中的多标签 tags 编码为 multi-hot 标签向量。

输入：
- graphs_clean.csv
- tag_vocab.json

输出：
- graph_label_matrix.npy
- label_index.csv

说明：
- 每一行标签向量对应一张图
- 标签顺序严格按照 tag_vocab.json
- 后续 build_pyg_dataset.py 可直接读取这两个文件

运行方式：
python src/dataset/label_encoder.py --dataset ecore
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


# =========================================================
# 配置读取
# =========================================================
def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_dataset_dirs(config: Dict[str, Any], dataset_name: str) -> Tuple[str, str]:
    if dataset_name == "ecore":
        cleaned_dir = config["paths"]["ecore_cleaned_dir"]
        features_dir = config["paths"]["ecore_features_dir"]
    elif dataset_name == "uml":
        cleaned_dir = config["paths"]["uml_cleaned_dir"]
        features_dir = config["paths"]["uml_features_dir"]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    return cleaned_dir, features_dir


# =========================================================
# 数据读取
# =========================================================
def parse_tags(tags_value: Any) -> List[str]:
    """
    兼容：
    1. JSON list 字符串: ["a", "b"]
    2. 普通分隔字符串: a|b / a,b / a;b
    3. 单个字符串
    """
    if pd.isna(tags_value):
        return []

    if isinstance(tags_value, list):
        return [str(x).strip() for x in tags_value if str(x).strip()]

    s = str(tags_value).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass

    for sep in ["|", ",", ";"]:
        if sep in s:
            return [x.strip() for x in s.split(sep) if x.strip()]

    return [s]


def load_graphs_and_vocab(graphs_clean_path: str, tag_vocab_path: str) -> Tuple[pd.DataFrame, List[str]]:
    if not Path(graphs_clean_path).exists():
        raise FileNotFoundError(f"找不到 graphs_clean.csv: {graphs_clean_path}")

    if not Path(tag_vocab_path).exists():
        raise FileNotFoundError(f"找不到 tag_vocab.json: {tag_vocab_path}")

    graphs_df = pd.read_csv(graphs_clean_path, encoding="utf-8-sig")

    with open(tag_vocab_path, "r", encoding="utf-8") as f:
        tag_vocab = json.load(f)

    required_cols = {"graph_id", "tags"}
    missing = required_cols - set(graphs_df.columns)
    if missing:
        raise ValueError(f"graphs_clean.csv 缺少必要字段: {missing}")

    if not isinstance(tag_vocab, list):
        raise ValueError("tag_vocab.json 必须是标签列表")

    return graphs_df, tag_vocab


# =========================================================
# 编码
# =========================================================
def build_graph_label_matrix(graphs_df: pd.DataFrame, tag_vocab: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    返回：
    - label_matrix: shape = [num_graphs, num_tags]
    - label_index_df: 每行标签向量对应哪个 graph_id
    """
    tag_to_idx = {tag: i for i, tag in enumerate(tag_vocab)}

    num_graphs = len(graphs_df)
    num_tags = len(tag_vocab)

    label_matrix = np.zeros((num_graphs, num_tags), dtype=np.float32)

    label_rows = []

    for row_idx, (_, row) in enumerate(graphs_df.iterrows()):
        graph_id = row["graph_id"]
        tags = parse_tags(row["tags"])

        for tag in tags:
            if tag in tag_to_idx:
                label_matrix[row_idx, tag_to_idx[tag]] = 1.0

        label_rows.append(
            {
                "label_row_index": row_idx,
                "graph_id": graph_id,
                "tags": json.dumps(tags, ensure_ascii=False),
            }
        )

    label_index_df = pd.DataFrame(
        label_rows,
        columns=["label_row_index", "graph_id", "tags"]
    )

    return label_matrix, label_index_df


# =========================================================
# 保存
# =========================================================
def save_npy(arr: np.ndarray, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    np.save(path, arr)


def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    df.to_csv(path, index=False, encoding="utf-8-sig")


# =========================================================
# 打印摘要
# =========================================================
def print_label_summary(
    dataset_name: str,
    graphs_df: pd.DataFrame,
    tag_vocab: List[str],
    label_matrix: np.ndarray,
    label_index_df: pd.DataFrame,
) -> None:
    print("=" * 80)
    print(f"[数据集] {dataset_name}")
    print(f"图总数: {len(graphs_df)}")
    print(f"标签种类数: {len(tag_vocab)}")
    print(f"标签矩阵形状: {label_matrix.shape}")
    print("-" * 80)

    print("前 20 个标签:")
    for tag in tag_vocab[:20]:
        print(f"  - {tag}")
    print("-" * 80)

    print("每张图标签数分布（Top 10）:")
    label_count_per_graph = label_matrix.sum(axis=1)
    print(pd.Series(label_count_per_graph).value_counts().head(10).to_string())
    print("-" * 80)

    print("label_index 前 5 行:")
    print(label_index_df.head(5).to_string(index=False))
    print("-" * 80)

    print("前 5 张图的标签向量非零位置:")
    for i in range(min(5, label_matrix.shape[0])):
        nz = np.where(label_matrix[i] > 0)[0].tolist()
        print(f"row {i}: nonzero_cols = {nz}")

    print("=" * 80)


# =========================================================
# 主函数
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="将图标签编码为 multi-hot 向量")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ecore", "uml"],
        help="选择数据集：ecore 或 uml"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="default_config.yaml 路径"
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    cleaned_dir, features_dir = resolve_dataset_dirs(config, args.dataset)

    graphs_clean_path = str(Path(cleaned_dir) / "graphs_clean.csv")
    tag_vocab_path = str(Path(cleaned_dir) / "tag_vocab.json")

    graph_label_matrix_path = str(Path(features_dir) / "graph_label_matrix.npy")
    label_index_path = str(Path(features_dir) / "label_index.csv")

    graphs_df, tag_vocab = load_graphs_and_vocab(graphs_clean_path, tag_vocab_path)
    label_matrix, label_index_df = build_graph_label_matrix(graphs_df, tag_vocab)

    save_npy(label_matrix, graph_label_matrix_path)
    save_csv(label_index_df, label_index_path)

    print_label_summary(
        dataset_name=args.dataset,
        graphs_df=graphs_df,
        tag_vocab=tag_vocab,
        label_matrix=label_matrix,
        label_index_df=label_index_df,
    )


if __name__ == "__main__":
    main()