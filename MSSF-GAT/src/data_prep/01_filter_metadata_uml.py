"""
01_filter_metadata_uml.py

功能简介：
针对 UML 数据集，从 uml.db 中读取模型元数据，筛选出：
1. tags 非空
2. 非明确非英文（保留 english 和 unknown）

这是 UML 专用版本，目的是兼容 uml.db 与 ecore.db 不同的表结构。
与 ecore 版不同之处在于：
- 不假设 models 表一定有 repo 列
- 会自动检查 models 表列结构
- 会优先从 models.metadata_json 读取元数据
- 若 models 中没有 metadata_json，会尝试从常见 metadata 表中查找

输入：
- data/raw/modelset/modelset/datasets/dataset.uml/data/uml.db

输出：
- data/uml/01_filtered_metadata/valid_graph_ids.csv
- data/uml/01_filtered_metadata/non_english_graph_ids.csv

运行方式：
python src/data_prep/01_filter_metadata_uml.py --dataset uml
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


# =========================================================
# 配置
# =========================================================
def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def choose_best_db(dataset_dir: Path) -> Path:
    """
    在 dataset.genmymodel 下自动挑选最可能包含模型元数据的数据库。
    规则：
    1. 必须包含 models 表
    2. 优先选择 models 表中带有 metadata_json / tags / category / language 相关列的库
    3. 如果有多个候选，选分数最高的
    """
    db_files = list(dataset_dir.rglob("*.db"))
    if len(db_files) == 0:
        raise FileNotFoundError(f"在 {dataset_dir} 下找不到任何 .db 文件")

    scored = []

    for db_path in db_files:
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cur.fetchall()}

            score = 0
            detail = {"tables": list(tables), "model_cols": []}

            if "models" in tables:
                score += 10
                cur.execute("PRAGMA table_info(models);")
                model_cols = [row[1] for row in cur.fetchall()]
                detail["model_cols"] = model_cols

                # 强相关字段
                for col in ["metadata_json", "tags", "category", "language", "filename", "path", "file"]:
                    if col in model_cols:
                        score += 3

                # 常见 metadata 辅助表
                for t in ["metadata", "model_metadata", "metadatas", "annotations"]:
                    if t in tables:
                        score += 2

            conn.close()
            scored.append((score, db_path, detail))
        except Exception:
            continue

    if len(scored) == 0:
        raise FileNotFoundError(f"在 {dataset_dir} 下没有可读取的 .db 文件")

    scored.sort(key=lambda x: x[0], reverse=True)

    print("[候选数据库评分]")
    for score, path, detail in scored:
        print(f"score={score:02d} | {path}")
        if detail["model_cols"]:
            print(f"  models cols = {detail['model_cols']}")

    best_score, best_db, _ = scored[0]
    if best_score == 0:
        raise ValueError(
            f"在 {dataset_dir} 下找到了 .db 文件，但没有发现明显包含 metadata 的数据库。"
        )

    print(f"[自动选择数据库] {best_db}")
    return best_db


def resolve_paths(config: Dict[str, Any], dataset_name: str) -> Tuple[Path, Path, Path]:
    if dataset_name != "uml":
        raise ValueError("该脚本仅支持 uml 数据集")

    raw_root = Path(config["paths"]["raw_data_root"])
    output_dir = Path(config["paths"]["uml_filtered_dir"])

    dataset_dir = raw_root / "datasets" / "dataset.genmymodel"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"找不到 UML 数据目录: {dataset_dir}")

    db_path = choose_best_db(dataset_dir)

    valid_path = output_dir / "valid_graph_ids.csv"
    non_english_path = output_dir / "non_english_graph_ids.csv"
    return db_path, valid_path, non_english_path

# =========================================================
# SQLite 辅助
# =========================================================
def get_table_names(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cur.fetchall()]


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name});")
    return [row[1] for row in cur.fetchall()]


# =========================================================
# 元数据解析
# =========================================================
def safe_json_loads(s: Any) -> Optional[Any]:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def recursive_find_key(obj: Any, candidate_keys: List[str]) -> Any:
    """
    在嵌套 dict / list 中递归查找第一个命中的 key。
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in candidate_keys:
                return v
        for _, v in obj.items():
            found = recursive_find_key(v, candidate_keys)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = recursive_find_key(item, candidate_keys)
            if found is not None:
                return found

    return None


def normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        out = []
        for x in value:
            if isinstance(x, str):
                x = x.strip()
                if x:
                    out.append(x)
            elif isinstance(x, dict):
                # 兼容 [{"name": "..."}]、{"tag": "..."} 一类
                for key in ["name", "tag", "label", "value"]:
                    if key in x and str(x[key]).strip():
                        out.append(str(x[key]).strip())
                        break
        return out

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # 尝试解析 json list
        if s.startswith("[") and s.endswith("]"):
            arr = safe_json_loads(s)
            if isinstance(arr, list):
                return normalize_tags(arr)

        # 普通分隔
        for sep in ["|", ",", ";"]:
            if sep in s:
                return [t.strip() for t in s.split(sep) if t.strip()]

        return [s]

    return []


def parse_metadata(metadata_json: Any) -> Tuple[List[str], Optional[str], str]:
    """
    返回：
    - tags
    - category
    - language_status: english / non_english / unknown
    """
    obj = safe_json_loads(metadata_json)
    if obj is None:
        return [], None, "unknown"

    tags_raw = recursive_find_key(obj, ["tags", "tag"])
    category_raw = recursive_find_key(obj, ["category", "categories"])
    language_raw = recursive_find_key(obj, ["language", "lang"])

    tags = normalize_tags(tags_raw)

    category = None
    if category_raw is not None:
        if isinstance(category_raw, str):
            category = category_raw.strip() or None
        elif isinstance(category_raw, list) and len(category_raw) > 0:
            category = str(category_raw[0]).strip()

    language_status = "unknown"
    if language_raw is not None:
        lang = str(language_raw).strip().lower()
        if lang in {"english", "en"}:
            language_status = "english"
        elif lang in {"unknown", "", "none", "null"}:
            language_status = "unknown"
        else:
            language_status = "non_english"

    return tags, category, language_status


# =========================================================
# 读取 UML metadata
# =========================================================
def read_models_table(conn: sqlite3.Connection) -> pd.DataFrame:
    if "models" not in get_table_names(conn):
        raise ValueError("uml.db 中不存在 models 表")

    model_cols = get_table_columns(conn, "models")

    if "id" not in model_cols:
        raise ValueError("models 表缺少 id 列")

    select_cols = ["id AS graph_id"]

    # repo 在 uml 中可能不存在
    if "repo" in model_cols:
        select_cols.append("repo")
    else:
        select_cols.append("NULL AS repo")

    # filename/path/file 三选一
    filename_col = None
    for cand in ["filename", "path", "file"]:
        if cand in model_cols:
            filename_col = cand
            break

    if filename_col is None:
        select_cols.append("NULL AS filename")
    else:
        select_cols.append(f"{filename_col} AS filename")

    # 如果 models 自带 metadata_json，直接读
    if "metadata_json" in model_cols:
        select_cols.append("metadata_json")
    else:
        select_cols.append("NULL AS metadata_json")

    sql = f"""
        SELECT
            {", ".join(select_cols)}
        FROM models
    """
    return pd.read_sql_query(sql, conn)


def attach_metadata_from_aux_table(
    conn: sqlite3.Connection,
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    如果 models 表里没有 metadata_json，则尝试从常见 metadata 表中补充。
    """
    if "metadata_json" in model_df.columns and model_df["metadata_json"].notna().any():
        return model_df

    tables = set(get_table_names(conn))
    candidate_tables = ["metadata", "model_metadata", "metadatas", "annotations"]

    chosen_table = None
    join_col = None
    json_col = None

    for t in candidate_tables:
        if t not in tables:
            continue
        cols = get_table_columns(conn, t)

        for jc in ["model_id", "graph_id", "id"]:
            if jc not in cols:
                continue
            for mc in ["metadata_json", "json", "data", "content"]:
                if mc in cols:
                    chosen_table = t
                    join_col = jc
                    json_col = mc
                    break
            if chosen_table is not None:
                break
        if chosen_table is not None:
            break

    if chosen_table is None:
        # 找不到辅助表，就保留空 metadata_json
        return model_df

    sql = f"""
        SELECT
            {join_col} AS join_id,
            {json_col} AS metadata_json_aux
        FROM {chosen_table}
    """
    aux_df = pd.read_sql_query(sql, conn)

    # graph_id 通常和 models.id 对应
    merged = model_df.merge(
        aux_df,
        how="left",
        left_on="graph_id",
        right_on="join_id",
    )
    merged["metadata_json"] = merged["metadata_json"].where(
        merged["metadata_json"].notna(),
        merged["metadata_json_aux"]
    )
    merged = merged.drop(columns=[c for c in ["join_id", "metadata_json_aux"] if c in merged.columns])
    return merged


def read_joined_metadata(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"找不到数据库文件: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        model_df = read_models_table(conn)
        merged_df = attach_metadata_from_aux_table(conn, model_df)
    finally:
        conn.close()

    return merged_df


# =========================================================
# 过滤逻辑
# =========================================================
def build_output_rows(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    valid_rows: List[Dict[str, Any]] = []
    non_english_rows: List[Dict[str, Any]] = []

    total_models = len(merged_df)
    metadata_missing = 0
    tags_empty = 0
    category_missing = 0
    english_count = 0
    non_english_count = 0
    unknown_count = 0

    for _, row in merged_df.iterrows():
        graph_id = row["graph_id"]
        repo = row["repo"] if "repo" in row else None
        filename = row["filename"] if "filename" in row else None
        metadata_json = row["metadata_json"] if "metadata_json" in row else None

        if metadata_json is None or pd.isna(metadata_json):
            metadata_missing += 1

        tags, category, language_status = parse_metadata(metadata_json)

        if language_status == "english":
            english_count += 1
        elif language_status == "non_english":
            non_english_count += 1
        else:
            unknown_count += 1

        if not tags:
            tags_empty += 1

        if category is None:
            category_missing += 1

        out = {
            "graph_id": graph_id,
            "repo": repo,
            "filename": filename,
            "category": category,
            "language_status": language_status,
            "tags_count": len(tags),
            "tags": json.dumps(tags, ensure_ascii=False),
        }

        # 只过滤“明确非英文”
        if language_status == "non_english":
            non_english_rows.append(out)
            continue

        if len(tags) == 0:
            continue

        valid_rows.append(out)

    stats = {
        "total_models": total_models,
        "metadata_missing": metadata_missing,
        "tags_empty": tags_empty,
        "category_missing": category_missing,
        "english_count": english_count,
        "non_english_count": non_english_count,
        "unknown_count": unknown_count,
        "valid_count": len(valid_rows),
    }

    valid_df = pd.DataFrame(valid_rows)
    non_english_df = pd.DataFrame(non_english_rows)
    return valid_df, non_english_df, stats


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(str(path.parent))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def print_summary(
    dataset_name: str,
    db_path: Path,
    valid_path: Path,
    non_english_path: Path,
    valid_df: pd.DataFrame,
    non_english_df: pd.DataFrame,
    stats: Dict[str, int],
) -> None:
    print("=" * 80)
    print(f"[数据集] {dataset_name}")
    print(f"[数据库] {db_path}")
    print(f"[输出文件] {valid_path}")
    print(f"[非英文输出] {non_english_path}")
    print("-" * 80)
    print(f"总模型数: {stats['total_models']}")
    print(f"metadata_json 缺失/无法解析: {stats['metadata_missing']}")
    print(f"无有效 tags: {stats['tags_empty']}")
    print(f"缺失 category 的样本数: {stats['category_missing']}")
    print(f"language = english: {stats['english_count']}")
    print(f"language = non_english: {stats['non_english_count']}")
    print(f"language = unknown: {stats['unknown_count']}")
    print(f"最终保留模型数(tags非空 且 非明确非英文): {stats['valid_count']}")
    print("-" * 80)

    print("前 5 条有效样本示例:")
    if len(valid_df) > 0:
        print(valid_df.head(5).to_string(index=False))
    else:
        print("(空)")
    print("-" * 80)

    print("前 5 条明确非英文样本示例:")
    if len(non_english_df) > 0:
        print(non_english_df.head(5).to_string(index=False))
    else:
        print("(空)")
    print("=" * 80)


# =========================================================
# 主函数
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="UML 专用 metadata 过滤")
    parser.add_argument("--dataset", type=str, required=True, choices=["uml"])
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    db_path, valid_path, non_english_path = resolve_paths(config, args.dataset)

    merged_df = read_joined_metadata(db_path)
    valid_df, non_english_df, stats = build_output_rows(merged_df)

    save_csv(valid_df, valid_path)
    save_csv(non_english_df, non_english_path)

    print_summary(
        dataset_name=args.dataset,
        db_path=db_path,
        valid_path=valid_path,
        non_english_path=non_english_path,
        valid_df=valid_df,
        non_english_df=non_english_df,
        stats=stats,
    )


if __name__ == "__main__":
    main()