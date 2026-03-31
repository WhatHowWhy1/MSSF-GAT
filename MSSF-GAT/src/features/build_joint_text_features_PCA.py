"""
build_joint_text_features.py

功能简介：
为软件模型图中的每个节点构造 joint_text 特征，并保存为节点级特征矩阵。

joint_text 的形式为：
    name: {node_name}; type: {node_type}

随后：
1. 使用 Sentence-BERT 对 joint_text 做文本编码
2. 仅使用训练集节点对应的唯一 joint_text embedding 拟合 PCA
3. 使用训练集拟合得到的 PCA 将 train / val / test / full 的 joint_text embedding
   统一映射到低维空间
4. 保存所有 clean 后节点的最终特征矩阵与索引文件

严格性说明：
- 本脚本不会再在“全体节点文本”上 fit PCA
- PCA 只在训练集节点文本上拟合
- 验证集和测试集只做 transform
- 这样可以避免特征层面的信息泄露

输入：
- data/{dataset}/03_cleaned_csv/nodes_clean.csv
- data/{dataset}/04_split/train_graph_ids.txt
- data/{dataset}/04_split/val_graph_ids.txt
- data/{dataset}/04_split/test_graph_ids.txt

输出：
- data/{dataset}/05_features/joint_text_features.npy
- data/{dataset}/05_features/joint_text_feature_index.csv
- data/{dataset}/05_features/joint_text_vocab.json
- data/{dataset}/05_features/joint_text_pca_meta.json

运行方式：
python src/features/04_build_joint_text_features.py --dataset uml
python src/features/04_build_joint_text_features.py --dataset ecore
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


# =========================================================
# 配置读取
# =========================================================
def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_dataset_dirs(config: Dict[str, Any], dataset_name: str) -> Tuple[str, str, str]:
    if dataset_name == "ecore":
        cleaned_dir = config["paths"]["ecore_cleaned_dir"]
        split_dir = config["paths"]["ecore_split_dir"]
        features_dir = config["paths"]["ecore_features_dir"]
    elif dataset_name == "uml":
        cleaned_dir = config["paths"]["uml_cleaned_dir"]
        split_dir = config["paths"]["uml_split_dir"]
        features_dir = config["paths"]["uml_features_dir"]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    return cleaned_dir, split_dir, features_dir


# =========================================================
# 工具函数
# =========================================================
def read_graph_id_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 split 文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def split_camel_case(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return text


def normalize_name_text(
    name: str,
    text_lowercase: bool,
    replace_underscore_with_space: bool,
    split_camel: bool,
) -> str:
    if name is None:
        return ""

    s = str(name).strip()
    if replace_underscore_with_space:
        s = s.replace("_", " ")
    if split_camel:
        s = split_camel_case(s)
    s = re.sub(r"\s+", " ", s).strip()

    if text_lowercase:
        s = s.lower()

    return s


def normalize_type_text(
    node_type: str,
    text_lowercase: bool,
    replace_underscore_with_space: bool,
    split_camel: bool,
) -> str:
    if node_type is None:
        return ""

    s = str(node_type).strip()
    if replace_underscore_with_space:
        s = s.replace("_", " ")
    if split_camel:
        s = split_camel_case(s)
    s = re.sub(r"\s+", " ", s).strip()

    if text_lowercase:
        s = s.lower()

    return s


def build_joint_text(
    node_name: str,
    node_type: str,
    missing_name_token: str,
    text_template: str,
    text_lowercase: bool,
    replace_underscore_with_space: bool,
    split_camel: bool,
) -> str:
    name_text = normalize_name_text(
        node_name,
        text_lowercase=text_lowercase,
        replace_underscore_with_space=replace_underscore_with_space,
        split_camel=split_camel,
    )
    type_text = normalize_type_text(
        node_type,
        text_lowercase=text_lowercase,
        replace_underscore_with_space=replace_underscore_with_space,
        split_camel=split_camel,
    )

    if name_text == "" or name_text == "[MISSING]":
        name_text = missing_name_token

    return text_template.format(name=name_text, type=type_text)


# =========================================================
# 主逻辑
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="构建严格版 joint_text 节点特征（PCA 仅在训练集拟合）")
    parser.add_argument("--dataset", type=str, required=True, choices=["ecore", "uml"])
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    cleaned_dir, split_dir, features_dir = resolve_dataset_dirs(config, args.dataset)

    cleaned_dir = Path(cleaned_dir)
    split_dir = Path(split_dir)
    features_dir = Path(features_dir)
    ensure_dir(str(features_dir))

    feat_cfg = config.get("features", {}).get("joint_text_feature", {})
    encoder_name = feat_cfg.get("encoder_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    batch_size = int(feat_cfg.get("batch_size", 64))
    normalize_embeddings = bool(feat_cfg.get("normalize_embeddings", True))
    reduced_dim = int(feat_cfg.get("reduced_dim", 64))
    reduction_method = feat_cfg.get("reduction_method", "pca").lower()
    missing_name_token = feat_cfg.get("missing_name_token", "[MISSING]")
    text_lowercase = bool(feat_cfg.get("text_lowercase", False))
    replace_underscore_with_space = bool(feat_cfg.get("replace_underscore_with_space", True))
    split_camel = bool(feat_cfg.get("split_camel_case", True))
    text_template = feat_cfg.get("text_template", "name: {name}; type: {type}")

    if reduction_method != "pca":
        raise ValueError(f"当前严格版脚本仅支持 reduction_method='pca'，当前为: {reduction_method}")

    nodes_path = cleaned_dir / "nodes_clean.csv"
    train_ids_path = split_dir / "train_graph_ids.txt"
    val_ids_path = split_dir / "val_graph_ids.txt"
    test_ids_path = split_dir / "test_graph_ids.txt"

    if not nodes_path.exists():
        raise FileNotFoundError(f"找不到节点文件: {nodes_path}")

    df_nodes = pd.read_csv(nodes_path, encoding="utf-8-sig")
    train_graph_ids = set(read_graph_id_list(train_ids_path))
    val_graph_ids = set(read_graph_id_list(val_ids_path))
    test_graph_ids = set(read_graph_id_list(test_ids_path))

    required_cols = {"node_global_id", "graph_id", "node_type", "node_name"}
    missing_cols = required_cols - set(df_nodes.columns)
    if missing_cols:
        raise ValueError(f"nodes_clean.csv 缺少必要字段: {missing_cols}")

    # 构造 joint_text
    df_nodes = df_nodes.copy()
    df_nodes["joint_text"] = df_nodes.apply(
        lambda row: build_joint_text(
            node_name=row["node_name"],
            node_type=row["node_type"],
            missing_name_token=missing_name_token,
            text_template=text_template,
            text_lowercase=text_lowercase,
            replace_underscore_with_space=replace_underscore_with_space,
            split_camel=split_camel,
        ),
        axis=1,
    )

    # 标记 split
    def infer_split(graph_id: str) -> str:
        if graph_id in train_graph_ids:
            return "train"
        if graph_id in val_graph_ids:
            return "val"
        if graph_id in test_graph_ids:
            return "test"
        return "unknown"

    df_nodes["split"] = df_nodes["graph_id"].map(infer_split)

    unknown_count = int((df_nodes["split"] == "unknown").sum())
    if unknown_count > 0:
        raise ValueError(f"存在 {unknown_count} 个节点所属 graph_id 不在 train/val/test split 中")

    # 全部唯一文本（用于统一编码缓存）
    all_unique_texts = df_nodes["joint_text"].drop_duplicates().tolist()

    # 仅训练集唯一文本（用于 PCA 拟合）
    train_unique_texts = (
        df_nodes[df_nodes["split"] == "train"]["joint_text"]
        .drop_duplicates()
        .tolist()
    )

    if len(train_unique_texts) == 0:
        raise ValueError("训练集 joint_text 为空，无法拟合 PCA")

    # 编码器
    model = SentenceTransformer(encoder_name)

    # 编码全体唯一文本
    all_raw_embeddings = model.encode(
        all_unique_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype(np.float32)

    # 为了只用 train 拟合 PCA，需要从全体编码里抽取 train 对应行
    text_to_index = {text: i for i, text in enumerate(all_unique_texts)}
    train_indices = [text_to_index[text] for text in train_unique_texts]
    train_raw_embeddings = all_raw_embeddings[train_indices]

    raw_dim = int(all_raw_embeddings.shape[1])
    if reduced_dim > raw_dim:
        raise ValueError(f"reduced_dim={reduced_dim} 不能大于原始维度 raw_dim={raw_dim}")

    # 仅在训练集 embedding 上拟合 PCA
    pca = PCA(n_components=reduced_dim, random_state=42)
    pca.fit(train_raw_embeddings)

    # 对全体唯一文本做 transform
    all_reduced_embeddings = pca.transform(all_raw_embeddings).astype(np.float32)

    explained_variance_ratio_sum = float(np.sum(pca.explained_variance_ratio_))

    # 回填到所有节点
    joint_text_to_reduced = {
        text: all_reduced_embeddings[idx]
        for idx, text in enumerate(all_unique_texts)
    }

    node_features = np.stack(
        [joint_text_to_reduced[text] for text in df_nodes["joint_text"].tolist()],
        axis=0,
    ).astype(np.float32)

    # 保存
    feature_path = features_dir / "joint_text_features.npy"
    index_path = features_dir / "joint_text_feature_index.csv"
    vocab_path = features_dir / "joint_text_vocab.json"
    meta_path = features_dir / "joint_text_pca_meta.json"

    np.save(feature_path, node_features)

    index_df = df_nodes[
        ["node_global_id", "graph_id", "node_type", "node_name", "joint_text", "split"]
    ].copy()
    index_df.insert(0, "feature_row_index", np.arange(len(index_df), dtype=np.int64))
    index_df.to_csv(index_path, index=False, encoding="utf-8-sig")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(all_unique_texts, f, ensure_ascii=False, indent=2)

    meta = {
        "dataset": args.dataset,
        "encoder_name": encoder_name,
        "normalize_embeddings": normalize_embeddings,
        "raw_dim": raw_dim,
        "reduced_dim": reduced_dim,
        "reduction_method": "pca",
        "pca_fit_scope": "train_only",
        "num_nodes": int(len(df_nodes)),
        "num_train_nodes": int((df_nodes["split"] == "train").sum()),
        "num_val_nodes": int((df_nodes["split"] == "val").sum()),
        "num_test_nodes": int((df_nodes["split"] == "test").sum()),
        "num_all_unique_joint_texts": int(len(all_unique_texts)),
        "num_train_unique_joint_texts": int(len(train_unique_texts)),
        "explained_variance_ratio_sum": explained_variance_ratio_sum,
        "text_template": text_template,
        "missing_name_token": missing_name_token,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print("=" * 80)
    print(f"[数据集] {args.dataset}")
    print(f"节点总数: {len(df_nodes)}")
    print(f"训练节点数: {(df_nodes['split'] == 'train').sum()}")
    print(f"验证节点数: {(df_nodes['split'] == 'val').sum()}")
    print(f"测试节点数: {(df_nodes['split'] == 'test').sum()}")
    print(f"全部唯一 joint_text 数量: {len(all_unique_texts)}")
    print(f"训练集唯一 joint_text 数量: {len(train_unique_texts)}")
    print(f"encoder_name: {encoder_name}")
    print(f"原始 embedding 维度: {raw_dim}")
    print(f"最终 joint_text 特征维度: {reduced_dim}")
    print(f"降维方式: pca")
    print(f"PCA 拟合范围: train_only")
    print(f"explained_variance_ratio_sum: {explained_variance_ratio_sum:.6f}")
    print("-" * 80)

    print("joint_text_feature_index 前 5 行:")
    print(index_df.head(5).to_string(index=False))
    print("-" * 80)

    print("前 3 行 joint_text 特征前 8 维预览:")
    for i in range(min(3, node_features.shape[0])):
        print(f"row {i}: {node_features[i][:8]}")
    print("=" * 80)


if __name__ == "__main__":
    main()