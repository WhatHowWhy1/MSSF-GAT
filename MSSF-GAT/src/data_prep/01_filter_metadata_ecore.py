"""
01_filter_metadata.py

功能简介：
从 ModelSet 元数据数据库中筛选出可用于多标签图分类的模型，并输出：
1. valid_graph_ids.csv
2. non_english_graph_ids.csv

最终版筛选策略：
- 必须存在 metadata.json 且能成功解析
- 必须至少包含一个有效 tag
- 如果 language 明确为非英文，则剔除
- 如果 language 为英文或未知（unknown），则保留

额外说明：
- 本版本会一并提取 category，并写入 valid_graph_ids.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_json_loads(text: Any) -> Dict[str, Any]:
    if text is None:
        return {}
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def normalize_tags(raw_tags: Any) -> List[str]:
    if raw_tags is None:
        return []

    if isinstance(raw_tags, list):
        results = []
        for item in raw_tags:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                results.append(s)
        return results

    if isinstance(raw_tags, str):
        raw_tags = raw_tags.strip()
        if not raw_tags:
            return []

        if raw_tags.startswith("[") and raw_tags.endswith("]"):
            try:
                arr = json.loads(raw_tags)
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except Exception:
                pass

        if "," in raw_tags:
            return [x.strip() for x in raw_tags.split(",") if x.strip()]

        return [raw_tags]

    return []


def extract_tags(metadata: Dict[str, Any]) -> List[str]:
    candidate_keys = ["tags", "tag", "labels", "modelset_tags"]
    for key in candidate_keys:
        if key in metadata:
            tags = normalize_tags(metadata.get(key))
            if tags:
                return tags
    return []


def normalize_language(raw_lang: Any) -> Optional[str]:
    if raw_lang is None:
        return None

    if isinstance(raw_lang, list):
        vals = [str(v).strip().lower() for v in raw_lang if str(v).strip()]
        if not vals:
            return None
        raw = vals[0]
    else:
        raw = str(raw_lang).strip().lower()

    if not raw:
        return None

    if raw in {"en", "english"}:
        return "english"

    return raw


def extract_language(metadata: Dict[str, Any]) -> Optional[str]:
    candidate_keys = ["language", "lang", "model_language"]
    for key in candidate_keys:
        if key in metadata:
            lang = normalize_language(metadata.get(key))
            if lang is not None:
                return lang
    return None


def normalize_category(raw_cat: Any) -> str:
    if raw_cat is None:
        return ""

    if isinstance(raw_cat, list):
        vals = [str(v).strip() for v in raw_cat if str(v).strip()]
        return vals[0] if vals else ""

    return str(raw_cat).strip()


def extract_category(metadata: Dict[str, Any]) -> str:
    candidate_keys = ["category", "categories"]
    for key in candidate_keys:
        if key in metadata:
            cat = normalize_category(metadata.get(key))
            if cat:
                return cat
    return ""


def find_dataset_db(raw_data_root: str, dataset_name: str) -> str:
    root = Path(raw_data_root)
    if not root.exists():
        raise FileNotFoundError(f"raw_data_root 不存在: {raw_data_root}")

    datasets_root = root / "datasets"
    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets 目录不存在: {datasets_root}")

    all_db_files = [p for p in datasets_root.rglob("*.db") if p.is_file()]
    if not all_db_files:
        raise FileNotFoundError(f"在 {datasets_root} 下没有找到任何 .db 文件")

    dataset_name = dataset_name.lower()

    if dataset_name == "ecore":
        preferred = [p for p in all_db_files if "ecore" in p.name.lower() or "ecore" in str(p).lower()]
    elif dataset_name == "uml":
        preferred = [p for p in all_db_files if "genmymodel" in p.name.lower() or "uml" in p.name.lower() or "genmymodel" in str(p).lower()]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    if preferred:
        preferred = sorted(preferred, key=lambda x: len(str(x)))
        return str(preferred[0])

    return str(sorted(all_db_files, key=lambda x: len(str(x)))[0])


def read_joined_metadata(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        model_df = pd.read_sql_query(
            """
            SELECT
                id AS graph_id,
                repo,
                filename
            FROM models
            """,
            conn,
        )

        metadata_df = pd.read_sql_query(
            """
            SELECT
                id AS graph_id,
                metadata AS metadata_text,
                json AS metadata_json
            FROM metadata
            """,
            conn,
        )

        merged = model_df.merge(metadata_df, on="graph_id", how="left")
        return merged
    finally:
        conn.close()


def filter_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    valid_records = []
    non_english_records = []

    stats = {
        "total_models": 0,
        "missing_metadata_json": 0,
        "missing_tags": 0,
        "missing_category": 0,
        "language_english": 0,
        "language_non_english": 0,
        "language_unknown": 0,
        "valid_models": 0,
    }

    for _, row in df.iterrows():
        stats["total_models"] += 1

        metadata = safe_json_loads(row.get("metadata_json"))
        if not metadata:
            stats["missing_metadata_json"] += 1
            continue

        tags = extract_tags(metadata)
        if not tags:
            stats["missing_tags"] += 1
            continue

        category = extract_category(metadata)
        if not category:
            stats["missing_category"] += 1

        language = extract_language(metadata)
        if language == "english":
            language_status = "english"
            stats["language_english"] += 1
        elif language is None:
            language_status = "unknown"
            stats["language_unknown"] += 1
        else:
            language_status = "non_english"
            stats["language_non_english"] += 1

        record = {
            "graph_id": row.get("graph_id"),
            "repo": row.get("repo", ""),
            "filename": row.get("filename", ""),
            "category": category,
            "language_status": language_status,
            "tags_count": len(tags),
            "tags": json.dumps(tags, ensure_ascii=False),
        }

        if language_status == "non_english":
            non_english_records.append(record)
            continue

        valid_records.append(record)
        stats["valid_models"] += 1

    valid_df = pd.DataFrame(valid_records)
    non_english_df = pd.DataFrame(non_english_records)
    return valid_df, non_english_df, stats


def save_csv(df: pd.DataFrame, output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")


def print_validation_info(
    dataset_name: str,
    db_path: str,
    valid_output_csv: str,
    non_english_output_csv: str,
    stats: Dict[str, int],
    valid_df: pd.DataFrame,
    non_english_df: pd.DataFrame,
) -> None:
    print("=" * 80)
    print(f"[数据集] {dataset_name}")
    print(f"[数据库] {db_path}")
    print(f"[输出文件] {valid_output_csv}")
    print(f"[非英文输出] {non_english_output_csv}")
    print("-" * 80)
    print(f"总模型数: {stats['total_models']}")
    print(f"metadata_json 缺失/无法解析: {stats['missing_metadata_json']}")
    print(f"无有效 tags: {stats['missing_tags']}")
    print(f"缺失 category 的样本数: {stats['missing_category']}")
    print(f"language = english: {stats['language_english']}")
    print(f"language = non_english: {stats['language_non_english']}")
    print(f"language = unknown: {stats['language_unknown']}")
    print(f"最终保留模型数(tags非空 且 非明确非英文): {stats['valid_models']}")
    print("-" * 80)

    if not valid_df.empty:
        print("前 5 条有效样本示例:")
        print(valid_df.head(5).to_string(index=False))
    else:
        print("没有筛出任何有效样本，请检查 metadata 字段结构。")

    print("-" * 80)

    if not non_english_df.empty:
        print("前 5 条明确非英文样本示例:")
        print(non_english_df.head(5).to_string(index=False))
    else:
        print("没有发现明确标记为非英文的样本。")

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="筛选 ModelSet 中 tags 非空且非明确非英文的模型")
    parser.add_argument("--dataset", type=str, required=True, choices=["ecore", "uml"], help="选择数据集：ecore 或 uml")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="default_config.yaml 路径")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    raw_data_root = config["paths"]["raw_data_root"]

    if not Path(raw_data_root).exists():
        raise FileNotFoundError(f"配置文件中的 raw_data_root 不存在: {raw_data_root}")

    if args.dataset == "ecore":
        output_dir = config["paths"]["ecore_filtered_dir"]
    else:
        output_dir = config["paths"]["uml_filtered_dir"]

    valid_output_csv = str(Path(output_dir) / "valid_graph_ids.csv")
    non_english_output_csv = str(Path(output_dir) / "non_english_graph_ids.csv")

    db_path = find_dataset_db(raw_data_root, args.dataset)
    merged_df = read_joined_metadata(db_path)
    valid_df, non_english_df, stats = filter_models(merged_df)

    save_csv(valid_df, valid_output_csv)
    save_csv(non_english_df, non_english_output_csv)

    print_validation_info(
        dataset_name=args.dataset,
        db_path=db_path,
        valid_output_csv=valid_output_csv,
        non_english_output_csv=non_english_output_csv,
        stats=stats,
        valid_df=valid_df,
        non_english_df=non_english_df,
    )


if __name__ == "__main__":
    main()