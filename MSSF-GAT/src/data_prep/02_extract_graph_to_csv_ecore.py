"""
02_extract_graph_to_csv.py

功能简介：
根据 01_filter_metadata.py 生成的 valid_graph_ids.csv，
从 ModelSet 的 graph/*.json 中提取图结构信息，生成三张原始表：

1. nodes_raw.csv
2. edges_raw.csv
3. graphs_raw.csv

当前版本：
- 支持从 valid_graph_ids.csv 读取 category
- 生成的 graphs_raw.csv 中 category 不再为空
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_dataset_dirs(config: Dict[str, Any], dataset_name: str) -> Tuple[str, str]:
    if dataset_name == "ecore":
        return config["paths"]["ecore_filtered_dir"], config["paths"]["ecore_extracted_dir"]
    elif dataset_name == "uml":
        return config["paths"]["uml_filtered_dir"], config["paths"]["uml_extracted_dir"]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def load_valid_graphs(valid_csv_path: str) -> pd.DataFrame:
    if not Path(valid_csv_path).exists():
        raise FileNotFoundError(f"找不到 valid_graph_ids.csv: {valid_csv_path}")

    df = pd.read_csv(valid_csv_path, encoding="utf-8-sig")
    required_cols = {"graph_id", "filename", "tags", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"valid_graph_ids.csv 缺少必要字段: {missing}")

    if "repo" not in df.columns:
        df["repo"] = ""

    return df


def get_graph_root(raw_data_root: str) -> Path:
    graph_root = Path(raw_data_root) / "graph"
    if not graph_root.exists():
        raise FileNotFoundError(f"graph 目录不存在: {graph_root}")
    return graph_root


def normalize_rel_path(p: Path, root: Path) -> str:
    rel = p.relative_to(root).as_posix()
    return rel.lower()


def build_graph_json_index(graph_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    all_json_files: List[Path] = []
    scan_errors: List[str] = []

    def onerror(err: OSError) -> None:
        scan_errors.append(str(err))

    for root, _, files in os.walk(graph_root, onerror=onerror):
        try:
            root_path = Path(root)
            for fname in files:
                try:
                    if fname.lower().endswith(".json"):
                        p = root_path / fname
                        if p.is_file():
                            all_json_files.append(p)
                except Exception as e:
                    scan_errors.append(f"[file_error] {root}\\{fname} -> {e}")
        except Exception as e:
            scan_errors.append(f"[dir_error] {root} -> {e}")

    by_rel_path: Dict[str, List[Path]] = {}
    by_name: Dict[str, List[Path]] = {}
    by_stem: Dict[str, List[Path]] = {}
    by_full_stem: Dict[str, List[Path]] = {}

    for p in all_json_files:
        try:
            rel_key = normalize_rel_path(p, graph_root)
            by_rel_path.setdefault(rel_key, []).append(p)

            name_key = p.name.lower()
            by_name.setdefault(name_key, []).append(p)

            stem_key = p.stem.lower()
            by_stem.setdefault(stem_key, []).append(p)

            full_name = p.name.lower()
            if full_name.endswith(".json"):
                tmp = full_name[:-5]
            else:
                tmp = full_name

            for suffix in [".ecore", ".xmi", ".uml"]:
                if tmp.endswith(suffix):
                    tmp = tmp[: -len(suffix)]
                    break

            by_full_stem.setdefault(tmp, []).append(p)

        except Exception as e:
            scan_errors.append(f"[index_error] {p} -> {e}")

    return {
        "by_rel_path": by_rel_path,
        "by_name": by_name,
        "by_stem": by_stem,
        "by_full_stem": by_full_stem,
        "all_files": all_json_files,
        "scan_errors": scan_errors,
    }


def build_match_keys(row: pd.Series) -> Dict[str, str]:
    graph_id = str(row["graph_id"]).strip().replace("\\", "/")
    filename = str(row.get("filename", "")).strip().replace("\\", "/")

    graph_id_base = Path(graph_id).name.lower()
    graph_id_stem = Path(graph_id).stem.lower()

    filename_base = Path(filename).name.lower()
    filename_stem = Path(filename).stem.lower()

    return {
        "graph_id_json_rel": f"{graph_id}.json".lower(),
        "graph_id_replace_json_rel": str(Path(graph_id).with_suffix(".json")).replace("\\", "/").lower(),
        "graph_id_base_json": f"{graph_id_base}.json".lower(),
        "graph_id_stem_json": f"{graph_id_stem}.json".lower(),
        "graph_id_base": graph_id_base,
        "graph_id_stem": graph_id_stem,
        "filename_json_rel": f"{filename}.json".lower(),
        "filename_replace_json_rel": str(Path(filename).with_suffix(".json")).replace("\\", "/").lower(),
        "filename_base_json": f"{filename_base}.json".lower(),
        "filename_stem_json": f"{filename_stem}.json".lower(),
        "filename_base": filename_base,
        "filename_stem": filename_stem,
    }


def choose_best_candidate(candidates: List[Path], row: pd.Series) -> Path:
    repo = str(row.get("repo", "")).lower().strip()
    if repo:
        repo_hits = [p for p in candidates if repo in str(p).lower()]
        if repo_hits:
            candidates = repo_hits
    return sorted(candidates, key=lambda x: len(str(x)))[0]


def resolve_graph_json_path(
    graph_root: Path,
    graph_index: Dict[str, Dict[str, List[Path]]],
    row: pd.Series,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    keys = build_match_keys(row)

    by_rel_path = graph_index["by_rel_path"]
    by_name = graph_index["by_name"]
    by_stem = graph_index["by_stem"]
    by_full_stem = graph_index["by_full_stem"]

    debug_info = {
        "keys": keys,
        "matched_rule": None,
    }

    for k in ["graph_id_json_rel", "graph_id_replace_json_rel", "filename_json_rel", "filename_replace_json_rel"]:
        rel_key = keys[k]
        if rel_key in by_rel_path:
            debug_info["matched_rule"] = f"by_rel_path:{k}"
            return choose_best_candidate(by_rel_path[rel_key], row), debug_info

    for k in ["graph_id_base_json", "graph_id_stem_json", "filename_base_json", "filename_stem_json"]:
        name_key = keys[k]
        if name_key in by_name:
            debug_info["matched_rule"] = f"by_name:{k}"
            return choose_best_candidate(by_name[name_key], row), debug_info

    for k in ["graph_id_stem", "filename_stem"]:
        stem_key = keys[k]
        if stem_key in by_stem:
            debug_info["matched_rule"] = f"by_stem:{k}"
            return choose_best_candidate(by_stem[stem_key], row), debug_info

    for k in ["graph_id_stem", "filename_stem"]:
        stem_key = keys[k]
        if stem_key in by_full_stem:
            debug_info["matched_rule"] = f"by_full_stem:{k}"
            return choose_best_candidate(by_full_stem[stem_key], row), debug_info

    return None, debug_info


def safe_read_json(json_path: Path) -> Optional[Dict[str, Any]]:
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_node_type(node_obj: Dict[str, Any]) -> str:
    for key in ["eClass", "type", "node_type"]:
        val = node_obj.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return "UNKNOWN_TYPE"


def extract_node_name(node_obj: Dict[str, Any]) -> str:
    for key in ["name", "label", "id"]:
        val = node_obj.get(key)
        if val is not None:
            s = str(val).strip()
            if s:
                return s
    return ""


def parse_tags(tags_str: Any) -> str:
    if pd.isna(tags_str):
        return "[]"

    if isinstance(tags_str, str):
        tags_str = tags_str.strip()
        if not tags_str:
            return "[]"
        if tags_str.startswith("[") and tags_str.endswith("]"):
            return tags_str
        return json.dumps([tags_str], ensure_ascii=False)

    if isinstance(tags_str, list):
        return json.dumps(tags_str, ensure_ascii=False)

    return "[]"


def parse_category_from_row(row: pd.Series) -> str:
    val = row.get("category", "")
    if pd.isna(val):
        return ""
    return str(val).strip()


def extract_graphs_to_csv(
    valid_df: pd.DataFrame,
    raw_data_root: str,
    default_edge_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    graph_root = get_graph_root(raw_data_root)
    graph_index = build_graph_json_index(graph_root)

    nodes_records: List[Dict[str, Any]] = []
    edges_records: List[Dict[str, Any]] = []
    graphs_records: List[Dict[str, Any]] = []

    node_global_id_counter = 0
    missing_examples: List[Dict[str, Any]] = []

    stats: Dict[str, Any] = {
        "valid_graph_ids_input": len(valid_df),
        "graph_json_total_indexed": len(graph_index["all_files"]),
        "graph_scan_error_count": len(graph_index.get("scan_errors", [])),
        "graph_json_found": 0,
        "graph_json_missing": 0,
        "graph_json_parse_failed": 0,
        "graphs_extracted": 0,
        "total_nodes": 0,
        "total_edges": 0,
        "missing_examples": missing_examples,
    }

    for _, row in valid_df.iterrows():
        graph_id = str(row["graph_id"])
        json_path, debug_info = resolve_graph_json_path(graph_root, graph_index, row)

        if json_path is None:
            stats["graph_json_missing"] += 1
            if len(missing_examples) < 5:
                missing_examples.append(
                    {
                        "graph_id": graph_id,
                        "repo": row.get("repo", ""),
                        "filename": row.get("filename", ""),
                        "keys": debug_info["keys"],
                    }
                )
            continue

        graph_obj = safe_read_json(json_path)
        if graph_obj is None:
            stats["graph_json_parse_failed"] += 1
            continue

        stats["graph_json_found"] += 1

        nodes = graph_obj.get("nodes", [])
        links = graph_obj.get("links", [])

        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(links, list):
            links = []

        local_to_global: Dict[int, int] = {}

        for local_idx, node_obj in enumerate(nodes):
            if not isinstance(node_obj, dict):
                node_obj = {}

            node_type = extract_node_type(node_obj)
            node_name = extract_node_name(node_obj)
            is_name_missing = 1 if node_name == "" else 0

            global_id = node_global_id_counter
            node_global_id_counter += 1
            local_to_global[local_idx] = global_id

            nodes_records.append(
                {
                    "node_global_id": global_id,
                    "graph_id": graph_id,
                    "node_type": node_type,
                    "node_name": node_name,
                    "is_name_missing": is_name_missing,
                }
            )

        edge_count_this_graph = 0
        for link_obj in links:
            if not isinstance(link_obj, dict):
                continue

            src_local = link_obj.get("source")
            dst_local = link_obj.get("target")

            if src_local not in local_to_global or dst_local not in local_to_global:
                continue

            src_global = local_to_global[src_local]
            dst_global = local_to_global[dst_local]

            edge_type = default_edge_type
            for key in ["edge_type", "type", "relation", "label"]:
                if key in link_obj and str(link_obj[key]).strip():
                    edge_type = str(link_obj[key]).strip()
                    break

            edges_records.append(
                {
                    "graph_id": graph_id,
                    "src_global_id": src_global,
                    "dst_global_id": dst_global,
                    "edge_type": edge_type,
                }
            )
            edge_count_this_graph += 1

        graphs_records.append(
            {
                "graph_id": graph_id,
                "num_nodes": len(nodes),
                "category": parse_category_from_row(row),
                "tags": parse_tags(row["tags"]),
            }
        )

        stats["graphs_extracted"] += 1
        stats["total_nodes"] += len(nodes)
        stats["total_edges"] += edge_count_this_graph

    nodes_df = pd.DataFrame(nodes_records)
    edges_df = pd.DataFrame(edges_records)
    graphs_df = pd.DataFrame(graphs_records)

    return nodes_df, edges_df, graphs_df, stats


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    ensure_dir(str(Path(output_path).parent))
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def print_extraction_summary(
    dataset_name: str,
    valid_csv_path: str,
    nodes_output_path: str,
    edges_output_path: str,
    graphs_output_path: str,
    stats: Dict[str, Any],
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    graphs_df: pd.DataFrame,
) -> None:
    print("=" * 80)
    print(f"[数据集] {dataset_name}")
    print(f"[输入白名单] {valid_csv_path}")
    print(f"[输出 nodes] {nodes_output_path}")
    print(f"[输出 edges] {edges_output_path}")
    print(f"[输出 graphs] {graphs_output_path}")
    print("-" * 80)
    print(f"输入 graph_id 数量: {stats['valid_graph_ids_input']}")
    print(f"graph 目录中总 json 数量: {stats['graph_json_total_indexed']}")
    print(f"扫描 graph 目录时的异常数量: {stats.get('graph_scan_error_count', 0)}")
    print(f"找到的 graph json 数量: {stats['graph_json_found']}")
    print(f"缺失的 graph json 数量: {stats['graph_json_missing']}")
    print(f"解析失败的 graph json 数量: {stats['graph_json_parse_failed']}")
    print(f"成功提取图数量: {stats['graphs_extracted']}")
    print(f"总节点数: {stats['total_nodes']}")
    print(f"总边数: {stats['total_edges']}")
    print("-" * 80)

    if stats.get("missing_examples"):
        print("前 5 个缺失样本的匹配 key：")
        for i, item in enumerate(stats["missing_examples"], start=1):
            print(f"\n[缺失样本 {i}] graph_id = {item['graph_id']}")
            print(f"repo = {item['repo']}")
            print(f"filename = {item['filename']}")
            print("尝试过的匹配 key：")
            for k, v in item["keys"].items():
                print(f"  - {k}: {v}")
        print("-" * 80)

    if stats.get("graph_scan_error_count", 0) > 0:
        print("扫描 graph 目录时存在异常路径条目，但已自动跳过。")
        print("-" * 80)

    if not graphs_df.empty:
        print("graphs_raw.csv 前 5 行示例:")
        print(graphs_df.head(5).to_string(index=False))
        print("-" * 80)

    if not nodes_df.empty:
        print("nodes_raw.csv 前 5 行示例:")
        print(nodes_df.head(5).to_string(index=False))
        print("-" * 80)

    if not edges_df.empty:
        print("edges_raw.csv 前 5 行示例:")
        print(edges_df.head(5).to_string(index=False))
        print("-" * 80)

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="从 ModelSet graph json 提取 nodes/edges/graphs 三张表")
    parser.add_argument("--dataset", type=str, required=True, choices=["ecore", "uml"], help="选择数据集：ecore 或 uml")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="default_config.yaml 路径")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    raw_data_root = config["paths"]["raw_data_root"]

    filtered_dir, extracted_dir = resolve_dataset_dirs(config, args.dataset)
    ensure_dir(extracted_dir)

    valid_csv_path = str(Path(filtered_dir) / "valid_graph_ids.csv")
    nodes_output_path = str(Path(extracted_dir) / "nodes_raw.csv")
    edges_output_path = str(Path(extracted_dir) / "edges_raw.csv")
    graphs_output_path = str(Path(extracted_dir) / "graphs_raw.csv")

    default_edge_type = config["graph_build"].get("default_edge_type", "generic_link")

    valid_df = load_valid_graphs(valid_csv_path)

    nodes_df, edges_df, graphs_df, stats = extract_graphs_to_csv(
        valid_df=valid_df,
        raw_data_root=raw_data_root,
        default_edge_type=default_edge_type,
    )

    save_csv(nodes_df, nodes_output_path)
    save_csv(edges_df, edges_output_path)
    save_csv(graphs_df, graphs_output_path)

    print_extraction_summary(
        dataset_name=args.dataset,
        valid_csv_path=valid_csv_path,
        nodes_output_path=nodes_output_path,
        edges_output_path=edges_output_path,
        graphs_output_path=graphs_output_path,
        stats=stats,
        nodes_df=nodes_df,
        edges_df=edges_df,
        graphs_df=graphs_df,
    )


if __name__ == "__main__":
    main()