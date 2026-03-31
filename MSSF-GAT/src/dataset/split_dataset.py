"""
split_dataset.py

功能简介：
对 graphs_clean.csv 做多标签分层划分（train / val / test），
尽量保持各标签在不同子集中的分布一致。

输入：
- graphs_clean.csv
- tag_vocab.json

输出：
- split_train_val_test.json
- split_summary.json
- label_coverage_report.csv
- train_graph_ids.txt
- val_graph_ids.txt
- test_graph_ids.txt


使用方法：
python src/dataset/split_dataset.py --dataset ecore
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
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# =========================================================
# 配置
# =========================================================
def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_dataset_dirs(config: Dict[str, Any], dataset_name: str) -> Tuple[str, str]:
    if dataset_name == "ecore":
        cleaned_dir = config["paths"]["ecore_cleaned_dir"]
        split_dir = config["paths"]["ecore_split_dir"]
    elif dataset_name == "uml":
        cleaned_dir = config["paths"]["uml_cleaned_dir"]
        split_dir = config["paths"]["uml_split_dir"]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    return cleaned_dir, split_dir


# =========================================================
# 数据读取
# =========================================================
def parse_tags(tags_value: Any) -> List[str]:
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


def load_graphs_and_vocab(cleaned_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    graphs_path = Path(cleaned_dir) / "graphs_clean.csv"
    vocab_path = Path(cleaned_dir) / "tag_vocab.json"

    if not graphs_path.exists():
        raise FileNotFoundError(f"找不到 graphs_clean.csv: {graphs_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"找不到 tag_vocab.json: {vocab_path}")

    graphs_df = pd.read_csv(graphs_path, encoding="utf-8-sig")
    with open(vocab_path, "r", encoding="utf-8") as f:
        tag_vocab = json.load(f)

    if "graph_id" not in graphs_df.columns or "tags" not in graphs_df.columns:
        raise ValueError("graphs_clean.csv 必须包含 graph_id 和 tags 列")

    return graphs_df, tag_vocab


# =========================================================
# 标签编码
# =========================================================
def build_multihot_matrix(graphs_df: pd.DataFrame, tag_vocab: List[str]) -> np.ndarray:
    tag_to_idx = {tag: i for i, tag in enumerate(tag_vocab)}
    y = np.zeros((len(graphs_df), len(tag_vocab)), dtype=np.int64)

    for row_idx, (_, row) in enumerate(graphs_df.iterrows()):
        tags = parse_tags(row["tags"])
        for tag in tags:
            if tag in tag_to_idx:
                y[row_idx, tag_to_idx[tag]] = 1

    return y


# =========================================================
# 多标签分层划分
# =========================================================
def multilabel_split(
    graphs_df: pd.DataFrame,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    两阶段划分：
    1. train vs temp
    2. temp 再切成 val vs test
    """
    n = len(graphs_df)
    indices = np.arange(n)

    temp_ratio = val_ratio + test_ratio
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=temp_ratio,
        random_state=random_seed
    )
    train_idx, temp_idx = next(msss1.split(indices.reshape(-1, 1), y))

    y_temp = y[temp_idx]
    temp_indices = indices[temp_idx]

    val_ratio_in_temp = val_ratio / (val_ratio + test_ratio)
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=(1 - val_ratio_in_temp),
        random_state=random_seed
    )
    val_sub_idx, test_sub_idx = next(msss2.split(temp_indices.reshape(-1, 1), y_temp))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    return train_idx, val_idx, test_idx


# =========================================================
# 覆盖检查
# =========================================================
def compute_label_coverage(
    y: np.ndarray,
    tag_vocab: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    rows = []

    train_sum = y[train_idx].sum(axis=0)
    val_sum = y[val_idx].sum(axis=0)
    test_sum = y[test_idx].sum(axis=0)
    total_sum = y.sum(axis=0)

    for i, tag in enumerate(tag_vocab):
        rows.append(
            {
                "tag": tag,
                "total_count": int(total_sum[i]),
                "train_count": int(train_sum[i]),
                "val_count": int(val_sum[i]),
                "test_count": int(test_sum[i]),
                "missing_in_train": int(train_sum[i] == 0),
                "missing_in_val": int(val_sum[i] == 0),
                "missing_in_test": int(test_sum[i] == 0),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# 保存
# =========================================================
def save_json(obj: Any, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_graph_id_txt(graph_ids: List[str], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        for gid in graph_ids:
            f.write(f"{gid}\n")


# =========================================================
# 打印摘要
# =========================================================
def print_split_summary(
    dataset_name: str,
    graphs_df: pd.DataFrame,
    tag_vocab: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    coverage_df: pd.DataFrame,
) -> None:
    print("=" * 80)
    print(f"[数据集] {dataset_name}")
    print(f"总图数: {len(graphs_df)}")
    print(f"标签种类数: {len(tag_vocab)}")
    print("-" * 80)
    print(f"train: {len(train_idx)}")
    print(f"val:   {len(val_idx)}")
    print(f"test:  {len(test_idx)}")
    print("-" * 80)

    missing_train = int(coverage_df["missing_in_train"].sum())
    missing_val = int(coverage_df["missing_in_val"].sum())
    missing_test = int(coverage_df["missing_in_test"].sum())

    print(f"train 中缺失的标签数: {missing_train}")
    print(f"val 中缺失的标签数:   {missing_val}")
    print(f"test 中缺失的标签数:  {missing_test}")
    print("-" * 80)

    print("前 20 个标签覆盖情况:")
    print(coverage_df.head(20).to_string(index=False))
    print("=" * 80)


# =========================================================
# 主函数
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="多标签分层划分 train / val / test")
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
    cleaned_dir, split_dir = resolve_dataset_dirs(config, args.dataset)

    train_ratio = config.get("split", {}).get("train_ratio", 0.72)
    val_ratio = config.get("split", {}).get("val_ratio", 0.08)
    test_ratio = config.get("split", {}).get("test_ratio", 0.20)
    random_seed = config.get("split", {}).get("random_seed", 42)

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(f"train/val/test 比例之和必须为 1.0，当前为 {total_ratio}")

    graphs_df, tag_vocab = load_graphs_and_vocab(cleaned_dir)
    y = build_multihot_matrix(graphs_df, tag_vocab)

    train_idx, val_idx, test_idx = multilabel_split(
        graphs_df=graphs_df,
        y=y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    coverage_df = compute_label_coverage(
        y=y,
        tag_vocab=tag_vocab,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    train_graph_ids = graphs_df.iloc[train_idx]["graph_id"].tolist()
    val_graph_ids = graphs_df.iloc[val_idx]["graph_id"].tolist()
    test_graph_ids = graphs_df.iloc[test_idx]["graph_id"].tolist()

    split_json = {
        "train_graph_ids": train_graph_ids,
        "val_graph_ids": val_graph_ids,
        "test_graph_ids": test_graph_ids,
    }

    split_summary = {
        "dataset": args.dataset,
        "num_graphs_total": int(len(graphs_df)),
        "num_graphs_train": int(len(train_idx)),
        "num_graphs_val": int(len(val_idx)),
        "num_graphs_test": int(len(test_idx)),
        "num_tags": int(len(tag_vocab)),
        "missing_tags_in_train": int(coverage_df["missing_in_train"].sum()),
        "missing_tags_in_val": int(coverage_df["missing_in_val"].sum()),
        "missing_tags_in_test": int(coverage_df["missing_in_test"].sum()),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_seed": random_seed,
    }

    split_json_path = str(Path(split_dir) / "split_train_val_test.json")
    split_summary_path = str(Path(split_dir) / "split_summary.json")
    coverage_csv_path = str(Path(split_dir) / "label_coverage_report.csv")

    train_txt_path = str(Path(split_dir) / "train_graph_ids.txt")
    val_txt_path = str(Path(split_dir) / "val_graph_ids.txt")
    test_txt_path = str(Path(split_dir) / "test_graph_ids.txt")

    save_json(split_json, split_json_path)
    save_json(split_summary, split_summary_path)
    save_csv(coverage_df, coverage_csv_path)

    save_graph_id_txt(train_graph_ids, train_txt_path)
    save_graph_id_txt(val_graph_ids, val_txt_path)
    save_graph_id_txt(test_graph_ids, test_txt_path)

    print_split_summary(
        dataset_name=args.dataset,
        graphs_df=graphs_df,
        tag_vocab=tag_vocab,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        coverage_df=coverage_df,
    )


if __name__ == "__main__":
    main()