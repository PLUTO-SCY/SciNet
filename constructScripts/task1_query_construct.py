import os
import json

# 配置路径
INPUT_DIR = "/data5/shaochenyang/AI_Scientist/OpenAlex/task1Result/paperFields"
OUTPUT_NOVEL = "/data5/shaochenyang/AI_Scientist/OpenAlex/Task4Evaluation/queries_task1_novel.json"
OUTPUT_DISRUPTIVE = "/data5/shaochenyang/AI_Scientist/OpenAlex/Task4Evaluation/queries_task1_disruptive.json"

K = 5  # top-k 数值

novel_queries = {}
disruptive_queries = {}

# 遍历文件
for file_name in os.listdir(INPUT_DIR):
    if file_name.endswith(".json"):
        # 原始领域名（保持下划线不变，方便和原始文件匹配）
        field_raw = os.path.splitext(file_name)[0]
        # 用于生成句子的更自然字段（下划线转空格）
        field_clean = field_raw.replace("_", " ")

        # 构造 query
        novel_queries[field_raw] = f"What are the top {K} most novel papers in the field of {field_clean}?"
        disruptive_queries[field_raw] = f"What are the top {K} most disruptive papers in the field of {field_clean}?"

# 保存为 JSON 文件
with open(OUTPUT_NOVEL, "w", encoding="utf-8") as f:
    json.dump(novel_queries, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DISRUPTIVE, "w", encoding="utf-8") as f:
    json.dump(disruptive_queries, f, ensure_ascii=False, indent=2)

print(f"已生成 top-{K} 查询文件:\n- {OUTPUT_NOVEL}\n- {OUTPUT_DISRUPTIVE}")

