"""
03_clean_tags.py

功能简介：
对 graphs_raw.csv 中的 tags 做清洗，并同步生成 clean 版数据；
同时对 nodes_raw.csv 中的 node_name 做基础语义清洗。

输入：
- nodes_raw.csv
- edges_raw.csv
- graphs_raw.csv

输出：
- nodes_clean.csv
- edges_clean.csv
- graphs_clean.csv
- tag_vocab.json
- tag_merge_map.json
- tag_stats_before_prune.csv
- tag_stats_after_prune.csv

清洗步骤：
1. tags 标准化（小写、去空格）
2. 应用 TAG_MERGE_MAP
3. 统计合并后标签频次
4. 删除出现次数 < min_tag_frequency 的标签
5. 删除裁剪后无任何标签的图
6. 同步过滤 nodes / edges
7. 对 node_name 做基础清洗：
   - 若 node_name 为纯数字 且 node_type == EGenericType，则替换为 "generic type"
   - 若 node_name 为纯数字 且 node_type != EGenericType，则清空为 ""

python src/data_prep/03_clean_tags.py --dataset ecore
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
# 保守版 merge map
# =========================================================
TAG_MERGE_MAP = {
    # 1. 单复数统一
    "classes": "class",
    "expressions": "expression",
    "components": "component",
    "transformations": "transformation",
    "widgets": "widget",
    "templates": "template",
    "graphicaleditors": "graphicaleditor",
    "patterns": "pattern",
    "statements": "statement",
    "petrinets": "petrinet",
    "types": "type",
    "activities": "activity",
    "entities": "entity",
    "games": "game",
    "sensors": "sensor",
    "animals": "animal",
    "features": "feature",
    "ports": "port",
    "actuators": "actuator",
    "robots": "robot",
    "planes": "plane",
    "actions": "action",
    "rules": "rule",
    "concerns": "concern",
    "aspects": "aspect",
    "messages": "message",
    "tasks": "task",
    "assignments": "assignment",
    "commands": "command",

    # 2. 拼写错误修复
    "datatastructure": "datastructure",
    "struture": "structure",
    "optimisation": "optimization",
    "videgame": "videogame",
    "hairdresses": "hairdresser",
    "hairdessers": "hairdresser",
    "publishsuscribe": "publishsubscribe",

    # 3. 低风险缩写/变体
    "codegen": "codegeneration",
    "bpmn2": "bpmn",
    "javatransform": "java-transformation",
}


# =========================================================
# 路径工具
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_dataset_dirs(config: Dict[str, Any], dataset_name: str) -> Tuple[str, str]:
    if dataset_name == "ecore":
        extracted_dir = config["paths"]["ecore_extracted_dir"]
        cleaned_dir = config["paths"]["ecore_cleaned_dir"]
    elif dataset_name == "uml":
        extracted_dir = config["paths"]["uml_extracted_dir"]
        cleaned_dir = config["paths"]["uml_cleaned_dir"]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
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


def normalize_tag(tag: str) -> str:
    return str(tag).strip().lower()


def merge_tag(tag: str) -> str:
    tag = normalize_tag(tag)
    return TAG_MERGE_MAP.get(tag, tag)


def clean_and_merge_tags(raw_tags: Any) -> List[str]:
    tags = parse_tags(raw_tags)
    cleaned = [merge_tag(t) for t in tags if normalize_tag(t)]

    # 去重，保持顺序
    seen = set()
    result = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


# =========================================================
# node_name 清洗
# =========================================================
PURE_NUMBER_PATTERN = re.compile(r"^\d+$")


def is_pure_number(text: Any) -> bool:
    if pd.isna(text):
        return False
    s = str(text).strip()
    return bool(PURE_NUMBER_PATTERN.fullmatch(s))


def clean_node_name_by_type(node_type: Any, node_name: Any) -> Tuple[str, str]:
    """
    返回：
    - cleaned_name
    - action_tag 用于统计
      可取值：
      * "keep"
      * "replace_generic_type"
      * "clear_numeric_name"
    """
    if pd.isna(node_name):
        return "", "keep"

    raw_name = str(node_name).strip()
    node_type = "" if pd.isna(node_type) else str(node_type).strip()

    if raw_name == "":
        return "", "keep"

    if is_pure_number(raw_name):
        if node_type == "EGenericType":
            return "generic type", "replace_generic_type"
        else:
            return "", "clear_numeric_name"

    return raw_name, "keep"


def clean_nodes_dataframe(nodes_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    对 nodes_df 的 node_name 做清洗：
    - 数字名 + EGenericType -> generic type
    - 数字名 + 其他类型 -> 空字符串
    """
    work_df = nodes_df.copy()

    cleaned_names: List[str] = []
    replace_generic_type_count = 0
    clear_numeric_name_count = 0

    for _, row in work_df.iterrows():
        cleaned_name, action = clean_node_name_by_type(
            node_type=row.get("node_type", ""),
            node_name=row.get("node_name", "")
        )
        cleaned_names.append(cleaned_name)

        if action == "replace_generic_type":
            replace_generic_type_count += 1
        elif action == "clear_numeric_name":
            clear_numeric_name_count += 1

    work_df["node_name"] = cleaned_names

    # 若存在 is_name_missing，则同步更新
    if "is_name_missing" in work_df.columns:
        work_df["is_name_missing"] = work_df["node_name"].apply(lambda x: 1 if str(x).strip() == "" else 0)

    stats = {
        "replace_generic_type_count": replace_generic_type_count,
        "clear_numeric_name_count": clear_numeric_name_count,
    }

    return work_df, stats


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
    work_df["tags"] = work_df["tags_pruned"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    graphs_clean_df = work_df[["graph_id", "num_nodes", "category", "tags"]].copy()

    # 7. 同步过滤 nodes / edges
    valid_graph_ids = set(graphs_clean_df["graph_id"].tolist())

    nodes_filtered_df = nodes_df[nodes_df["graph_id"].isin(valid_graph_ids)].copy()
    edges_clean_df = edges_df[edges_df["graph_id"].isin(valid_graph_ids)].copy()

    # 8. 对 nodes 做 node_name 清洗
    nodes_clean_df, node_clean_stats = clean_nodes_dataframe(nodes_filtered_df)

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
        "replace_generic_type_count": node_clean_stats["replace_generic_type_count"],
        "clear_numeric_name_count": node_clean_stats["clear_numeric_name_count"],
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
    print(f"节点数: {stats['nodes_before']} -> {stats['nodes_after']}")
    print(f"边数: {stats['edges_before']} -> {stats['edges_after']}")
    print("-" * 80)
    print(f"纯数字 node_name 替换为 'generic type' 的数量: {stats['replace_generic_type_count']}")
    print(f"纯数字 node_name 被清空的数量: {stats['clear_numeric_name_count']}")
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
    parser = argparse.ArgumentParser(description="清洗 tags，并同步清洗 node_name")
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
    extracted_dir, cleaned_dir = resolve_dataset_dirs(config, args.dataset)

    # 从配置读取阈值；默认 5
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