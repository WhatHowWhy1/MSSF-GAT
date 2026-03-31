import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd


def parse_tags(tags_value):
    """
    兼容多种 tags 存储格式：
    1. JSON 字符串: ["tag1", "tag2"]
    2. 分隔字符串: tag1|tag2 或 tag1,tag2
    3. 空值
    """
    if pd.isna(tags_value):
        return []

    if isinstance(tags_value, list):
        return [str(x).strip() for x in tags_value if str(x).strip()]

    s = str(tags_value).strip()
    if not s:
        return []

    # 尝试按 JSON list 解析
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass

    # 尝试按常见分隔符解析
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]

    return [s]


def main():
    parser = argparse.ArgumentParser(description="统计 graphs.csv 中的 tag 数量分布")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="graphs_raw.csv 或 graphs_clean.csv 的路径"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "tags" not in df.columns:
        raise ValueError("CSV 文件中没有 tags 列")

    tag_counter = Counter()

    for _, row in df.iterrows():
        tags = parse_tags(row["tags"])
        for tag in tags:
            tag_counter[tag] += 1

    print("=" * 60)
    print(f"文件路径: {csv_path}")
    print(f"图总数: {len(df)}")
    print(f"不同 tag 总数: {len(tag_counter)}")
    print("=" * 60)

    print("各 tag 及其出现次数（按频次降序）:")
    for tag, count in tag_counter.most_common():
        print(f"{tag}: {count}")

    print("=" * 60)


if __name__ == "__main__":
    main()