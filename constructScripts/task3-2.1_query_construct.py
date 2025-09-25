import json

# 输入文件
INPUT_FILES = [
    "/data5/shaochenyang/AI_Scientist/OpenAlex/task3Result/result_DoubleNode_0826_depth5_beam8.json",
    "/data5/shaochenyang/AI_Scientist/OpenAlex/task3Result/result_DoubleNode_0826.json"
]

# 输出文件
OUTPUT_FILE = "/data5/shaochenyang/AI_Scientist/OpenAlex/Task4Evaluation/result_DoubleNode_merged.json"

processed_data = {}
key_counts = {}  # 记录重复次数

for file in INPUT_FILES:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for field, paths in data.items():
        if not paths or not isinstance(paths, list):
            continue

        # 检查 key 是否重复
        if field in processed_data:
            key_counts[field] = key_counts.get(field, 1) + 1
            new_field = f"{field}_{key_counts[field]}"
        else:
            key_counts[field] = 1
            new_field = field

        # 只取第一条路径
        first_path_entry = paths[0]

        # 路径格式是 [total_score, [list_of_papers]]
        if isinstance(first_path_entry, list) and len(first_path_entry) == 2:
            _, papers = first_path_entry
        else:
            continue

        if len(papers) < 2:
            continue

        # 提取 classic 和 recent
        classic_paper = {
            "id": papers[0].get("id"),
            "title": papers[0].get("title"),
            "cited_by_count": papers[0].get("cited_by_count"),
            "publication_year": papers[0].get("publication_year")
        }

        recent_paper = {
            "id": papers[-1].get("id"),
            "title": papers[-1].get("title"),
            "cited_by_count": papers[-1].get("cited_by_count"),
            "publication_year": papers[-1].get("publication_year")
        }

        # most_influential_path 保留全路径
        most_influential_path = [
            {
                "id": p.get("id"),
                "title": p.get("title"),
                "cited_by_count": p.get("cited_by_count"),
                "publication_year": p.get("publication_year")
            }
            for p in papers
        ]

        # 构造统一结构
        processed_data[new_field] = {
            "classic_paper": classic_paper,
            "recent_paper": recent_paper,
            "most_influential_path": most_influential_path
        }

# 保存合并结果
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

print(f"转换完成，结果保存至: {OUTPUT_FILE}")
print(f"共处理领域数量: {len(processed_data)}")
