"""
03_clean_tags_uml.py

功能简介：
对 UML 数据集的 graphs_raw.csv 中的 tags 做清洗，并同步生成 clean 版数据。

这是 UML 专用版本，主要用于：
1. 清洗 tags（小写、去空白、去多余引号）
2. 去除单图内部重复标签
3. 应用 UML 专用 TAG_MERGE_MAP 做低风险归一化
4. 统计清洗前后标签频次
5. 删除出现次数 < min_tag_frequency 的标签
6. 删除裁剪后无任何标签的图
7. 同步过滤 nodes / edges
8. 输出清洗后各个 tag 的名称和数量

注意：
- 本文件不再重复对 node_name 做 ecore 风格的数字处理


输入：
- data/uml/02_extracted_csv/nodes_raw.csv
- data/uml/02_extracted_csv/edges_raw.csv
- data/uml/02_extracted_csv/graphs_raw.csv

输出：
- data/uml/03_cleaned_csv/nodes_clean.csv
- data/uml/03_cleaned_csv/edges_clean.csv
- data/uml/03_cleaned_csv/graphs_clean.csv
- data/uml/03_cleaned_csv/tag_vocab.json
- data/uml/03_cleaned_csv/tag_merge_map.json
- data/uml/03_cleaned_csv/tag_stats_before_prune.csv
- data/uml/03_cleaned_csv/tag_stats_after_prune.csv

运行方式：
python src/data_prep/03_clean_tags_uml.py --dataset uml
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import yaml


# =========================================================
# 配置读取
# =========================================================
def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================================================
# UML 专用 merge map
# =========================================================
TAG_MERGE_MAP = {
    # 拼写修复
    "videgame": "videogame",

    # 单复数 / 低风险变体
    "libraries": "library",
    "shopping-carts": "shopping-cart",
    "checkouts": "checkout",
    "courses": "course",
    "students": "student",
    "trains": "train",
    "cars": "car",
    "services": "service",
    "events": "event",
    "questions": "question",
    "answers": "answer",
    "vehicles": "vehicle",

    # 低风险风格统一
    "boardgame": "board-game",
    "machinelearning": "machine-learning",

    # 疑似明显拼写/格式问题
    "real-state": "real-estate",
    "-state": "real-estate",
}


# =========================================================
# 路径工具
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_dataset_dirs(config: Dict[str, Any], dataset_name: str) -> Tuple[str, str]:
    if dataset_name != "uml":
        raise ValueError("03_clean_tags_uml.py 仅支持 uml 数据集")

    extracted_dir = config["paths"]["uml_extracted_dir"]
    cleaned_dir = config["paths"]["uml_cleaned_dir"]
    return extracted_dir, cleaned_dir


# =========================================================
# tags 解析与清洗
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


def strip_outer_quotes(text: str) -> str:
    """
    去掉首尾多余引号：
    例如：
    '"graphics card"' -> graphics card
    "'abc'" -> abc
    """
    s = text.strip()
    while len(s) >= 2 and (
        (s.startswith('"') and s.endswith('"')) or
        (s.startswith("'") and s.endswith("'"))
    ):
        s = s[1:-1].strip()
    return s


def normalize_tag(tag: str) -> str:
    s = str(tag).strip().lower()
    s = strip_outer_quotes(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def merge_tag(tag: str) -> str:
    tag = normalize_tag(tag)
    return TAG_MERGE_MAP.get(tag, tag)


def clean_and_merge_tags(raw_tags: Any) -> List[str]:
    """
    单图内标签清洗流程：
    1. parse
    2. normalize
    3. merge map
    4. 去空
    5. 去重（保持顺序）
    """
    tags = parse_tags(raw_tags)

    cleaned: List[str] = []
    for t in tags:
        nt = merge_tag(t)
        if nt:
            cleaned.append(nt)

    seen = set()
    result = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


# =========================================================
# 统计
# =========================================================
def count_tag_frequency(graphs_df: pd.DataFrame, tag_col: str = "tags_merged") -> Counter:
    counter = Counter()
    for tags in graphs_df[tag_col]:
        for t in tags:
            counter[t] += 1
    return counter


def counter_to_df(counter: Counter) -> pd.DataFrame:
    rows = [{"tag": tag, "count": count} for tag, count in counter.most_common()]
    return pd.DataFrame(rows, columns=["tag", "count"])


# =========================================================
# 主清洗逻辑
# =========================================================
def clean_tags_pipeline(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    graphs_df: pd.DataFrame,
    min_tag_frequency: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    返回：
    - nodes_clean_df
    - edges_clean_df
    - graphs_clean_df
    - tag_stats_before_df
    - tag_stats_after_df
    - stats
    """
    work_df = graphs_df.copy()

    # 1. 清洗与合并 tags
    work_df["tags_merged"] = work_df["tags"].apply(clean_and_merge_tags)

    # 2. 合并后频次统计
    tag_counter_before = count_tag_frequency(work_df, tag_col="tags_merged")
    tag_stats_before_df = counter_to_df(tag_counter_before)

    # 3. 低频标签裁剪
    valid_tags: Set[str] = {tag for tag, cnt in tag_counter_before.items() if cnt >= min_tag_frequency}

    def prune_tags(tag_list: List[str]) -> List[str]:
        return [t for t in tag_list if t in valid_tags]

    work_df["tags_pruned"] = work_df["tags_merged"].apply(prune_tags)

    # 4. 删除没有任何剩余标签的图
    before_graph_count = len(work_df)
    work_df = work_df[work_df["tags_pruned"].map(len) > 0].copy()
    after_graph_count = len(work_df)

    # 5. 统计裁剪后频次
    tag_counter_after = count_tag_frequency(work_df, tag_col="tags_pruned")
    tag_stats_after_df = counter_to_df(tag_counter_after)

    # 6. 生成最终 graphs_clean.csv
    if "num_nodes" not in work_df.columns:
        # 兼容异常情况
        work_df["num_nodes"] = work_df["graph_id"].map(
            nodes_df.groupby("graph_id").size().to_dict()
        ).fillna(0).astype(int)

    if "category" not in work_df.columns:
        work_df["category"] = ""

    work_df["tags"] = work_df["tags_pruned"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    graphs_clean_df = work_df[["graph_id", "num_nodes", "category", "tags"]].copy()

    # 7. 同步过滤 nodes / edges
    valid_graph_ids = set(graphs_clean_df["graph_id"].tolist())

    nodes_clean_df = nodes_df[nodes_df["graph_id"].isin(valid_graph_ids)].copy()
    edges_clean_df = edges_df[edges_df["graph_id"].isin(valid_graph_ids)].copy()

    stats = {
        "graphs_before": before_graph_count,
        "graphs_after": after_graph_count,
        "graphs_removed": before_graph_count - after_graph_count,
        "num_tags_before": len(tag_counter_before),
        "num_tags_after": len(tag_counter_after),
        "num_tags_removed": len(tag_counter_before) - len(tag_counter_after),
        "min_tag_frequency": min_tag_frequency,
        "nodes_before": len(nodes_df),
        "nodes_after": len(nodes_clean_df),
        "edges_before": len(edges_df),
        "edges_after": len(edges_clean_df),
        "tag_assignments_before": int(sum(tag_counter_before.values())),
        "tag_assignments_after": int(sum(tag_counter_after.values())),
    }

    return (
        nodes_clean_df,
        edges_clean_df,
        graphs_clean_df,
        tag_stats_before_df,
        tag_stats_after_df,
        stats,
    )


# =========================================================
# 保存
# =========================================================
def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(obj: Any, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# 打印报告
# =========================================================
def print_cleaning_summary(
    dataset_name: str,
    stats: Dict[str, Any],
    graphs_clean_df: pd.DataFrame,
    tag_stats_before_df: pd.DataFrame,
    tag_stats_after_df: pd.DataFrame,
) -> None:
    print("=" * 80)
    print(f"[数据集] {dataset_name}")
    print(f"min_tag_frequency = {stats['min_tag_frequency']}")
    print("-" * 80)
    print(f"图数量: {stats['graphs_before']} -> {stats['graphs_after']} (删除 {stats['graphs_removed']})")
    print(f"标签种类数: {stats['num_tags_before']} -> {stats['num_tags_after']} (删除 {stats['num_tags_removed']})")
    print(f"标签赋值总数: {stats['tag_assignments_before']} -> {stats['tag_assignments_after']}")
    print(f"节点数: {stats['nodes_before']} -> {stats['nodes_after']}")
    print(f"边数: {stats['edges_before']} -> {stats['edges_after']}")
    print("-" * 80)

    if not tag_stats_before_df.empty:
        print("裁剪前 Top 20 tags:")
        print(tag_stats_before_df.head(20).to_string(index=False))
        print("-" * 80)

    if not tag_stats_after_df.empty:
        print("裁剪后 Top 20 tags:")
        print(tag_stats_after_df.head(20).to_string(index=False))
        print("-" * 80)

    if not graphs_clean_df.empty:
        print("graphs_clean.csv 前 5 行示例:")
        print(graphs_clean_df.head(5).to_string(index=False))

    print("=" * 80)


# =========================================================
# 主函数
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="UML 专用 tags 清洗，并同步过滤 nodes/edges")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["uml"],
        help="仅支持 uml"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="default_config.yaml 路径"
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    extracted_dir, cleaned_dir = resolve_dataset_dirs(config, args.dataset)

    min_tag_frequency = config.get("tag_cleaning", {}).get("min_tag_frequency", 5)

    nodes_raw_path = str(Path(extracted_dir) / "nodes_raw.csv")
    edges_raw_path = str(Path(extracted_dir) / "edges_raw.csv")
    graphs_raw_path = str(Path(extracted_dir) / "graphs_raw.csv")

    nodes_clean_path = str(Path(cleaned_dir) / "nodes_clean.csv")
    edges_clean_path = str(Path(cleaned_dir) / "edges_clean.csv")
    graphs_clean_path = str(Path(cleaned_dir) / "graphs_clean.csv")

    tag_vocab_path = str(Path(cleaned_dir) / "tag_vocab.json")
    tag_merge_map_path = str(Path(cleaned_dir) / "tag_merge_map.json")
    tag_stats_before_path = str(Path(cleaned_dir) / "tag_stats_before_prune.csv")
    tag_stats_after_path = str(Path(cleaned_dir) / "tag_stats_after_prune.csv")

    nodes_df = pd.read_csv(nodes_raw_path, encoding="utf-8-sig")
    edges_df = pd.read_csv(edges_raw_path, encoding="utf-8-sig")
    graphs_df = pd.read_csv(graphs_raw_path, encoding="utf-8-sig")

    (
        nodes_clean_df,
        edges_clean_df,
        graphs_clean_df,
        tag_stats_before_df,
        tag_stats_after_df,
        stats,
    ) = clean_tags_pipeline(
        nodes_df=nodes_df,
        edges_df=edges_df,
        graphs_df=graphs_df,
        min_tag_frequency=min_tag_frequency,
    )

    # 保存 clean 数据
    save_csv(nodes_clean_df, nodes_clean_path)
    save_csv(edges_clean_df, edges_clean_path)
    save_csv(graphs_clean_df, graphs_clean_path)

    # 保存 tag 统计
    save_csv(tag_stats_before_df, tag_stats_before_path)
    save_csv(tag_stats_after_df, tag_stats_after_path)

    # 保存 tag vocab 和 merge map
    final_tags = tag_stats_after_df["tag"].tolist() if not tag_stats_after_df.empty else []
    save_json(final_tags, tag_vocab_path)
    save_json(TAG_MERGE_MAP, tag_merge_map_path)

    print_cleaning_summary(
        dataset_name=args.dataset,
        stats=stats,
        graphs_clean_df=graphs_clean_df,
        tag_stats_before_df=tag_stats_before_df,
        tag_stats_after_df=tag_stats_after_df,
    )


if __name__ == "__main__":
    main()