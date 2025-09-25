import json

# 文件路径
INPUT_FILES = [
    "/data5/shaochenyang/AI_Scientist/OpenAlex/task3Result/citation_path_results_parallel_depth5.json",
    "/data5/shaochenyang/AI_Scientist/OpenAlex/task3Result/citation_path_results_parallel.json",
    "/data5/shaochenyang/AI_Scientist/OpenAlex/task3Result/citation_path_results_referenceFirs_4.json"
]
OUTPUT_QUERIES = "/data5/shaochenyang/AI_Scientist/OpenAlex/Task4Evaluation/queries_task3_paths.json"
OUTPUT_ANSWERS = "/data5/shaochenyang/AI_Scientist/OpenAlex/Task4Evaluation/answers_task3_paths.json"

merged_data = {}
key_counts = {}  # 用于记录重复次数

# 1. 合并三份 JSON
for file in INPUT_FILES:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for field, info in data.items():
            # 检查是否已有相同 key
            if field in merged_data:
                key_counts[field] = key_counts.get(field, 1) + 1
                new_field = f"{field}_{key_counts[field]}"
            else:
                key_counts[field] = 1
                new_field = field

            merged_data[new_field] = info

queries = {}
answers = {}

# 2. 构造 query/answer
for field, info in merged_data.items():
    path = info.get("most_influential_path", [])
    if len(path) <= 2:  # 筛掉路径节点 ≤ 2 的
        continue

    classic_paper = info["classic_paper"]["title"]
    recent_paper = info["recent_paper"]["title"]

    # 删除不需要的字段
    cleaned_info = {k: v for k, v in info.items() if k not in ["influence_score", "found_by_method"]}

    # 构造更明确的 query
    queries[field] = (
        f'What is the most influential citation path from "{classic_paper}" to "{recent_paper}" in the field of {field.replace("_", " ")}?'
    )

    answers[field] = cleaned_info

# 3. 保存结果
with open(OUTPUT_QUERIES, "w", encoding="utf-8") as f:
    json.dump(queries, f, ensure_ascii=False, indent=2)

with open(OUTPUT_ANSWERS, "w", encoding="utf-8") as f:
    json.dump(answers, f, ensure_ascii=False, indent=2)

print(f"生成完成：\n- Queries: {OUTPUT_QUERIES}\n- Answers: {OUTPUT_ANSWERS}")
print(f"共生成 {len(queries)} 条 query/answer 对")
